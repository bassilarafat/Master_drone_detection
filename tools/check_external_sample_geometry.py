"""Check a fixed external Train sample against Clean_v2 retrieval candidates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
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
from drone_detection.scene_geometry import extract_features, _load_feature, verify_features
from drone_detection.scene_review import risk_reason


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    audit = args.audit.resolve()
    if not audit.is_relative_to(ROOT / "06_results"):
        raise ValueError("Use an external audit output")
    output = audit / "sample_geometry"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    dataset = ROOT / "02_datasets/DUT_Anti_UAV_Clean_v2"
    metadata = json.loads((dataset / "embedding_extraction.json").read_text())
    donor = ROOT / ".cache/scene_v2/dinov2_vitb14_pretrain.pth"
    repo = ROOT / ".cache/scene_v2/torch_hub/repo/dinov2_main"
    assert sha256(donor) == metadata["checkpoint_sha256"]
    assert repo_fingerprint(repo) == metadata["repository_code_sha256"]
    index = read_csv(dataset / "embedding_image_index.csv")
    manifest = {row["image"]: row for row in read_csv(dataset / "dataset_manifest.csv")}
    feature_index = {row["image_id"]: row for row in read_csv(dataset / "background_feature_index.csv")}
    cls = np.load(dataset / "embedding_vectors_cls.npy", mmap_mode="r")
    patch = np.load(dataset / "embedding_vectors_patch.npy", mmap_mode="r")
    assert sha256(dataset / "embedding_vectors_cls.npy") == metadata["descriptors_sha256"]
    assert sha256(dataset / "embedding_vectors_patch.npy") == metadata["patch_descriptors_sha256"]
    samples = read_csv(audit / "screen_sample.csv")
    phashes = np.array([int(feature_index[row["image_id"]]["phash"]) for row in index], dtype=np.uint64)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load(str(repo), "dinov2_vitb14", source="local", pretrained=False)
    model.load_state_dict(torch.load(donor, map_location="cpu", weights_only=True), strict=True)
    model.eval().requires_grad_(False).to(device)
    a_list, b_list = [], []
    with torch.inference_mode():
        for start in range(0, len(samples), 4):
            selected = samples[start:start+4]
            rows = [dict(relative_path=row["frame_file"], sha256=row["frame_sha256"],
                         width=row["width"], height=row["height"]) for row in selected]
            batch = torch.from_numpy(np.stack([preprocess(row, ROOT) for row in rows])).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                features = model.forward_features(batch)
                a = functional.normalize(features["x_norm_clstoken"].float(), dim=1)
                mask = torch.from_numpy(np.stack([valid_patch_mask(int(row["width"]), int(row["height"])) for row in rows])).to(device).unsqueeze(-1)
                tokens = features["x_norm_patchtokens"].float()
                b = functional.normalize((tokens*mask).sum(1)/mask.sum(1), dim=1)
            a_list.extend(a.cpu().numpy())
            b_list.extend(b.cpu().numpy())
    del model
    candidates, external = [], {}
    cache_hashes = {}
    for number, (row, a, b) in enumerate(zip(samples, a_list, b_list)):
        own = extract_features((ROOT / row["frame_file"]).read_bytes(), boxes=())
        external[number] = own
        cosine_a, cosine_b = cls @ a, patch @ b
        distances = np.bitwise_count(np.bitwise_xor(phashes, np.uint64(own["phash"])))
        chosen = set(np.argsort(-cosine_a, kind="stable")[:120])
        chosen.update(np.argsort(-cosine_b, kind="stable")[:120])
        chosen.update(np.argsort(distances, kind="stable")[:40])
        for n in sorted(chosen):
            ref = index[n]["image_id"]
            path = ROOT / ".cache/scene_v2/features_bg" / feature_index[ref]["feature_file"]
            if str(path) not in cache_hashes:
                with np.load(path, allow_pickle=False) as saved:
                    assert str(saved["source_sha256"]) == manifest[ref]["image_sha256"]
                    assert str(saved["label_sha256"]) == manifest[ref]["label_sha256"]
                cache_hashes[str(path.relative_to(ROOT))] = sha256(path)
            candidates.append(dict(sample_index=number, sequence=row["sequence"], frame_index=int(row["frame_index"]),
                                   image_b=ref, current_image_b=manifest[ref]["new_relative_image"],
                                   current_split_b=manifest[ref]["new_split"], block_b=manifest[ref]["allocation_block_id"],
                                   cosine_cls=float(cosine_a[n]), cosine_patch=float(cosine_b[n]),
                                   cosine_similarity=float(max(cosine_a[n], cosine_b[n])), reference_feature=str(path)))

    def check(row):
        other = _load_feature(row["reference_feature"])
        geometry = verify_features(external[row["sample_index"]], other, compute_ssim=False)
        result = dict(row, **geometry)
        result["risk_reason"] = risk_reason(result)
        return result

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(check, candidates))
    write_csv(output / "candidate_geometry.csv", results, list(results[0]))
    links = [r for r in results if r["risk_reason"] or r["geometry_decision"] == "CONFIRMED_GEOMETRY"]
    write_csv(output / "crossing_links.csv", links, list(results[0]))
    report = dict(status="SAMPLE_HAS_ALLOCATION_LINKS" if links else "NO_SAMPLE_ALLOCATION_LINKS_FOUND",
                  source_sample_frames=len(samples), evaluated_candidate_pairs=len(results),
                  candidate_rule="Cross-reference nearest 120 CLS, nearest 120 patch and nearest 40 pHash per sample; union deduplicated",
                  allocation_link_count=len(links), full_candidate_pool_screened=False,
                  independence_verified=False, reference_feature_sha256=cache_hashes,
                  meaning="A sample crossing can reject a conservatively grouped source family. Absence of sample crossings cannot approve the unscreened pool.")
    (output / "summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "reference_feature_sha256"}, indent=2))


if __name__ == "__main__":
    main()
