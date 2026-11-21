"""Screen the fixed RGPT Train sample against released Clean_v2 descriptors.

Uses the locally cached public DINOv2 model and the existing preprocessing.
The output is a diagnostic retrieval screen, not an independence certificate.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as functional

from drone_detection.paper_common import read_csv, write_csv
from drone_detection.scene_embeddings import preprocess, valid_patch_mask, repo_fingerprint, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    audit = args.audit.resolve()
    if not audit.is_relative_to(ROOT / "06_results"):
        raise ValueError("Use a separate audit directory")
    destination = audit / "sample_similarity.json"
    if destination.exists():
        raise FileExistsError(destination)
    dataset = ROOT / "02_datasets/DUT_Anti_UAV_Clean_v2"
    saved = json.loads((dataset / "embedding_extraction.json").read_text())
    donor = ROOT / ".cache/scene_v2/dinov2_vitb14_pretrain.pth"
    repo = ROOT / ".cache/scene_v2/torch_hub/repo/dinov2_main"
    assert sha256(donor) == saved["checkpoint_sha256"]
    assert repo_fingerprint(repo) == saved["repository_code_sha256"]
    cls_path = dataset / "embedding_vectors_cls.npy"
    patch_path = dataset / "embedding_vectors_patch.npy"
    assert sha256(cls_path) == saved["descriptors_sha256"]
    assert sha256(patch_path) == saved["patch_descriptors_sha256"]
    cls = np.load(cls_path, mmap_mode="r")
    patch = np.load(patch_path, mmap_mode="r")
    index = read_csv(dataset / "embedding_image_index.csv")
    manifest = {row["image"]: row for row in read_csv(dataset / "dataset_manifest.csv")}
    assert len(index) == len(cls) == len(patch) == 9451
    samples = read_csv(audit / "screen_sample.csv")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load(str(repo), "dinov2_vitb14", source="local", pretrained=False)
    model.load_state_dict(torch.load(donor, map_location="cpu", weights_only=True), strict=True)
    model.eval().requires_grad_(False).to(device)
    results, features_cls, features_patch = [], [], []
    with torch.inference_mode():
        for start in range(0, len(samples), 4):
            selected = samples[start:start+4]
            rows = [dict(relative_path=row["frame_file"], sha256=row["frame_sha256"],
                         width=row["width"], height=row["height"]) for row in selected]
            inputs = torch.from_numpy(np.stack([preprocess(row, ROOT) for row in rows])).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                features = model.forward_features(inputs)
                a = functional.normalize(features["x_norm_clstoken"].float(), dim=1)
                tokens = features["x_norm_patchtokens"].float()
                mask = torch.from_numpy(np.stack([valid_patch_mask(int(row["width"]), int(row["height"]))
                                                  for row in rows])).to(device).unsqueeze(-1)
                b = functional.normalize((tokens*mask).sum(1)/mask.sum(1), dim=1)
            features_cls.extend(a.cpu().numpy())
            features_patch.extend(b.cpu().numpy())
    nearest_rows = []
    for row, a, b in zip(samples, features_cls, features_patch):
        cosine_cls, cosine_patch = cls @ a, patch @ b
        strongest = np.maximum(cosine_cls, cosine_patch)
        top = np.argsort(-strongest, kind="stable")[:5]
        n = int(top[0])
        linked = int(np.count_nonzero(strongest >= .85))
        results.append(dict(sequence=row["sequence"], frame_index=int(row["frame_index"]),
                            maximum_cosine=float(strongest[n]),
                            broad_allocation_constraint_candidates=linked,
                            top_match=index[n]["image_id"],
                            current_match_split=manifest[index[n]["image_id"]]["new_split"]))
        for rank, n in enumerate(top, 1):
            nearest_rows.append(dict(sequence=row["sequence"], frame_index=row["frame_index"], rank=rank,
                                     cosine_cls=float(cosine_cls[n]), cosine_patch=float(cosine_patch[n]),
                                     old_image=index[n]["image_id"],
                                     current_image=manifest[index[n]["image_id"]]["new_relative_image"],
                                     allocation_block=manifest[index[n]["image_id"]]["allocation_block_id"]))
    write_csv(audit / "sample_nearest_neighbors.csv", nearest_rows, list(nearest_rows[0]))
    report = dict(status="DIAGNOSTIC_SAMPLE_SCREEN", device=device, samples=len(samples),
                  checkpoint_sha256=sha256(donor), repository_sha256=repo_fingerprint(repo),
                  cls_reference_sha256=sha256(cls_path), patch_reference_sha256=sha256(patch_path),
                  source_sample_sha256=sha256(audit / "screen_sample.csv"),
                  threshold=.85, threshold_semantics="Either descriptor reaching .85 proposes conservative allocation co-membership",
                  result_scope="Only 18 predeclared Train video frames. Full candidate-pool geometry and grouping remain outstanding.",
                  independent_allocation_blocks_verified=False, rows=results)
    destination.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
