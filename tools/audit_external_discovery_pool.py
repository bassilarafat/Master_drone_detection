"""Audit every predeclared absent frame from the reserved external Train pool.

This applies the existing feature, retrieval and geometric risk rules to a joint
index. It performs no detector inference or training and verifies no negatives.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

from drone_detection.paper_common import read_csv, write_csv, pin_release
from drone_detection.scene_embeddings import preprocess, valid_patch_mask, repo_fingerprint, sha256
from drone_detection.scene_geometry import extract_features
from drone_detection.scene_gpu_geometry import verify_gpu_pairs
from drone_detection.scene_review import risk_reason

DATASET = ROOT / "02_datasets/DUT_Anti_UAV_Clean_v2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source_audit.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT / "06_results"):
        raise ValueError("Outputs must stay outside datasets")
    if output.exists():
        raise FileExistsError(output)
    release = pin_release(DATASET)
    original = json.loads((source / "source_audit.json").read_text())
    for path, digest in original["source_sha256"].items():
        assert sha256(ROOT / path) == digest, path
    candidates = read_csv(source / "annotation_absent_candidates.csv")
    assert len(candidates) == original["candidate_annotation_absent_frames"] == 2398
    assert {row["source_split"] for row in candidates} == {"train"}
    output.mkdir(parents=True)
    (output / "frames").mkdir()
    (output / "features").mkdir()
    progress = output / "progress.json"

    def stage(name, **details):
        progress.write_text(json.dumps(dict(stage=name, at_utc=datetime.now(timezone.utc).isoformat(), **details), indent=2)+"\n")
        print(name, details, flush=True)

    stage("EXTRACTING_EXTERNAL_TRAIN_FRAMES", expected=len(candidates))
    rows, external_features = [], []
    cv2.setNumThreads(1)
    for sequence in sorted({row["sequence"] for row in candidates}):
        selected = {int(row["frame_index"]) for row in candidates if row["sequence"] == sequence}
        video = ROOT / "02_datasets/antiuav_rgpt/train" / sequence / "visible.mp4"
        capture = cv2.VideoCapture(str(video))
        assert capture.isOpened(), video
        index = 0
        while index <= max(selected):
            ok = capture.grab()
            assert ok, (sequence, index)
            if index in selected:
                ok, frame = capture.retrieve()
                assert ok, (sequence, index)
                filename = f"{sequence}_{index:06d}.png"
                path = output / "frames" / filename
                assert cv2.imwrite(str(path), frame)
                feature = extract_features(path.read_bytes(), boxes=())
                feature_file = output / "features" / (filename+".npz")
                np.savez_compressed(feature_file, **feature)
                rows.append(dict(image_id=f"EXTERNAL/{filename}", sequence=sequence, frame_index=index,
                                 source_split="train", family=sequence.rsplit("_", 2)[0],
                                 relative_path=path.relative_to(ROOT).as_posix(), sha256=sha256(path),
                                 width=frame.shape[1], height=frame.shape[0],
                                 annotation_exist=0, verified_negative=False))
                external_features.append(dict(image_id=rows[-1]["image_id"],
                                              feature_file=str(feature_file), keypoints=len(feature["points"]),
                                              phash=int(feature["phash"])))
            index += 1
        capture.release()
        stage("EXTRACTING_EXTERNAL_TRAIN_FRAMES", processed=len(rows), expected=len(candidates))
    assert len(rows) == len(candidates)
    write_csv(output / "frame_index.csv", rows, list(rows[0]))
    saved = json.loads((DATASET / "embedding_extraction.json").read_text())
    donor = ROOT / ".cache/scene_v2/dinov2_vitb14_pretrain.pth"
    repo = ROOT / ".cache/scene_v2/torch_hub/repo/dinov2_main"
    assert sha256(donor) == saved["checkpoint_sha256"]
    assert repo_fingerprint(repo) == saved["repository_code_sha256"]
    reference_index = read_csv(DATASET / "embedding_image_index.csv")
    reference_manifest = {row["image"]: row for row in read_csv(DATASET / "dataset_manifest.csv")}
    reference_features = read_csv(DATASET / "background_feature_index.csv")
    reference_feature_map = {row["image_id"]: row for row in reference_features}
    reference_cls_path = DATASET / "embedding_vectors_cls.npy"
    reference_patch_path = DATASET / "embedding_vectors_patch.npy"
    assert sha256(reference_cls_path) == saved["descriptors_sha256"]
    assert sha256(reference_patch_path) == saved["patch_descriptors_sha256"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    stage("EXTRACTING_DINOV2", device=device, expected=len(rows))
    model = torch.hub.load(str(repo), "dinov2_vitb14", source="local", pretrained=False)
    model.load_state_dict(torch.load(donor, map_location="cpu", weights_only=True), strict=True)
    model.eval().requires_grad_(False).to(device)
    cls = np.empty((len(rows), 768), np.float32)
    patch = np.empty_like(cls)
    with torch.inference_mode():
        for start in range(0, len(rows), 8):
            selected = rows[start:start+8]
            batch = torch.from_numpy(np.stack([preprocess(row, ROOT) for row in selected])).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                features = model.forward_features(batch)
                a = functional.normalize(features["x_norm_clstoken"].float(), dim=1)
                mask = torch.from_numpy(np.stack([valid_patch_mask(row["width"], row["height"]) for row in selected])).to(device).unsqueeze(-1)
                tokens = features["x_norm_patchtokens"].float()
                b = functional.normalize((tokens*mask).sum(1)/mask.sum(1), dim=1)
            cls[start:start+len(selected)] = a.cpu().numpy()
            patch[start:start+len(selected)] = b.cpu().numpy()
            if start % 400 == 0:
                stage("EXTRACTING_DINOV2", processed=start+len(selected), expected=len(rows))
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    np.save(output / "cls.npy", cls)
    np.save(output / "patch.npy", patch)
    ref_cls, ref_patch = np.load(reference_cls_path), np.load(reference_patch_path)
    nref = len(reference_index)
    assert nref == len(ref_cls) == 9451
    all_cls = np.concatenate([ref_cls, cls])
    all_patch = np.concatenate([ref_patch, patch])
    phashes = np.array([int(reference_feature_map[row["image_id"]]["phash"]) for row in reference_index]
                      + [row["phash"] for row in external_features], dtype=np.uint64)
    all_index = reference_index + rows
    pairs = set()
    torch.backends.cuda.matmul.allow_tf32 = False

    def include(i, j):
        if (i < nref) != (j < nref):
            pairs.add((min(i, j), max(i, j)))

    stage("JOINT_GLOBAL_RETRIEVAL", images=len(all_index))
    with torch.inference_mode():
        for descriptor in (all_cls, all_patch):
            values = torch.from_numpy(descriptor).to(device)
            for start in range(0, len(descriptor), 128):
                similarity = values[start:start+128] @ values.T
                for local, global_index in enumerate(range(start, min(start+128, len(descriptor)))):
                    similarity[local, global_index] = -float("inf")
                neighbors = torch.topk(similarity, 120, dim=1, largest=True, sorted=False).indices.cpu().numpy()
                for local, neighbors_row in enumerate(neighbors):
                    for neighbor in neighbors_row:
                        include(start+local, int(neighbor))
            # Every direct cross-boundary .85 relation is included even if it
            # falls beyond the global top120 because the external pool is dense.
            for start in range(0, len(rows), 128):
                cross = values[nref+start:nref+start+128] @ values[:nref].T
                external_id, original_id = torch.where(cross >= .85)
                for i, j in zip(external_id.cpu().tolist(), original_id.cpu().tolist()):
                    include(j, nref+start+i)
            del values
    for start in range(0, len(phashes), 128):
        distances = np.bitwise_count(np.bitwise_xor(phashes[start:start+128, None], phashes[None, :])).astype(np.int16)
        for local, global_index in enumerate(range(start, min(start+128, len(phashes)))):
            distances[local, global_index] = 100
        neighbors = np.argsort(distances, axis=1, kind="stable")[:, :40]
        for local, neighbors_row in enumerate(neighbors):
            for neighbor in neighbors_row:
                include(start+local, int(neighbor))
    pairs = sorted(pairs)
    candidate_rows = []
    for i, j in pairs:
        ca, cb = float(all_cls[i] @ all_cls[j]), float(all_patch[i] @ all_patch[j])
        ref = reference_index[i]["image_id"]
        candidate_rows.append(dict(image_a=ref, image_b=rows[j-nref]["image_id"],
                                   cosine_cls=ca, cosine_patch=cb, cosine_similarity=max(ca, cb),
                                   phash_distance=int(int(phashes[i]) ^ int(phashes[j])).bit_count(),
                                   current_reference_split=reference_manifest[ref]["new_split"],
                                   reference_allocation_block=reference_manifest[ref]["allocation_block_id"]))
    fields = ["image_a", "image_b", "cosine_cls", "cosine_patch", "cosine_similarity", "phash_distance",
              "current_reference_split", "reference_allocation_block"]
    write_csv(output / "cross_candidates.csv", candidate_rows, fields)
    needed_refs = {row["image_a"] for row in candidate_rows}
    feature_index, cache_hashes = [], {}
    for row in reference_features:
        if row["image_id"] not in needed_refs:
            continue
        path = ROOT / ".cache/scene_v2/features_bg" / row["feature_file"]
        with np.load(path, allow_pickle=False) as feature:
            assert str(feature["source_sha256"]) == reference_manifest[row["image_id"]]["image_sha256"]
            assert str(feature["label_sha256"]) == reference_manifest[row["image_id"]]["label_sha256"]
        cache_hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
        feature_index.append(dict(image_id=row["image_id"], feature_file=str(path)))
    feature_index += [dict(image_id=row["image_id"], feature_file=row["feature_file"]) for row in external_features]
    write_csv(output / "geometry_feature_index.csv", feature_index, ["image_id", "feature_file"])
    stage("CROSS_BOUNDARY_GEOMETRY", candidate_pairs=len(candidate_rows))
    if candidate_rows:
        verify_gpu_pairs(output / "cross_candidates.csv", output / "geometry_feature_index.csv",
                         output / "cross_geometry.csv", batch_size=16, workers=6, device=device)
        geometry = read_csv(output / "cross_geometry.csv")
    else:
        geometry = []
    links = [dict(row, risk_reason=risk_reason(row)) for row in geometry
             if risk_reason(row) or row["geometry_decision"] == "CONFIRMED_GEOMETRY"]
    low_external = sum(row["keypoints"] < 16 for row in external_features)
    source_families = sorted({row["family"] for row in rows})
    assert len(source_families) == 1
    blocks_linked = sorted({row["reference_allocation_block"] for row in links})
    if low_external:
        low_refs = [row for row in reference_features if int(row["keypoints"]) < 16]
        blocks_linked = sorted(set(blocks_linked) | {reference_manifest[row["image_id"]]["allocation_block_id"] for row in low_refs})
    for path, digest in original["source_sha256"].items():
        assert sha256(ROOT / path) == digest, path
    assert pin_release(DATASET) == release
    write_csv(output / "crossing_links.csv", links, fields+["risk_reason"] if not links else list(links[0]))
    report = dict(status="REJECT_SOURCE_FAMILY_ALLOCATION_CROSSING" if blocks_linked else "NO_DECLARED_PROTOCOL_CROSSING_FOUND",
                  generated_at_utc=datetime.now(timezone.utc).isoformat(), source_frames=len(rows), reference_images=nref,
                  conservative_external_filename_families=source_families, low_texture_external_frames=low_external,
                  candidate_pairs=len(candidate_rows), evaluated_geometry_pairs=len(geometry), risk_or_confirmed_links=len(links),
                  linked_reference_allocation_blocks=blocks_linked,
                  allocation_separation_under_declared_screen=not blocks_linked,
                  native_recording_independence_proven=False, verified_negative_count=None,
                  detector_inference=False, training_performed=False,
                  retrieval="Joint whole-pool k120 per descriptor plus pHash40, both directions; all cross .85 similarities added",
                  geometry="Existing masked-reference RootSIFT, mutual Lowe, RANSAC and version5 allocation-risk rules",
                  caveats=["Finite retrieval does not recover all possible scene relationships",
                           "All external candidates are from one conservative filename family",
                           "Annotation absence is not independently verified UAV-free status"],
                  dataset_release=release, source_audit_sha256=sha256(source / "source_audit.json"),
                  source_input_sha256=original["source_sha256"], reference_feature_sha256=cache_hashes,
                  rules_source_sha256={name: sha256(ROOT / name) for name in ["src/drone_detection/scene_review.py",
                    "src/drone_detection/scene_geometry.py", "src/drone_detection/scene_embeddings.py",
                    "src/drone_detection/scene_gpu_geometry.py", "tools/audit_external_discovery_pool.py"]})
    (output / "pool_audit.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    stage("COMPLETED", status=report["status"], pairs=len(geometry), links=len(links), low_texture=low_external)


if __name__ == "__main__":
    main()
