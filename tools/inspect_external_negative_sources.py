"""Inventory local RGPT Train and screen a fixed absent-frame sample.

Only Train video payloads are decoded. Val/Test directory names are used to
reserve conservative filename families. This audit does not verify negatives.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from drone_detection.paper_common import read_csv, write_csv
from drone_detection.scene_geometry import extract_features

SOURCE = ROOT / "02_datasets/antiuav_rgpt"
RELEASE = ROOT / "02_datasets/DUT_Anti_UAV_Clean_v2"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def family(name):
    return "_".join(name.split("_")[:2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "06_results"):
        raise ValueError("Write outside the source datasets")
    if output.exists():
        raise FileExistsError(output)
    split_names = {part: sorted(path.name for path in (SOURCE / part).iterdir() if path.is_dir())
                   for part in ("train", "val", "test")}
    reserved = {family(name) for part in ("val", "test") for name in split_names[part]}
    inventory, candidates, source_hashes = [], [], {}
    for name in split_names["train"]:
        label = SOURCE / "train" / name / "visible.json"
        before = digest(label)
        data = json.loads(label.read_text(encoding="utf-8-sig"))
        flags, boxes = data["exist"], data["gt_rect"]
        assert len(flags) == len(boxes) and set(flags) <= {0, 1}, name
        absent = [index for index, present in enumerate(flags) if present == 0]
        kept = family(name) not in reserved
        inventory.append(dict(sequence=name, source_split="train", filename_family=family(name),
                              family_in_reserved_val_or_test=not kept,
                              label_frames=len(flags), annotated_present=sum(flags),
                              annotated_absent=len(absent), source_label_sha256=before,
                              visible_video=(SOURCE / "train" / name / "visible.mp4").relative_to(ROOT).as_posix(),
                              candidate_status="PENDING_CROSS_RELEASE_AUDIT" if kept else "RESERVED_FAMILY_EXCLUDED"))
        if kept:
            candidates.extend(dict(sequence=name, frame_index=index, source_split="train",
                                   filename_family=family(name), annotation_exist=0,
                                   annotation_gt_rect=json.dumps(boxes[index]),
                                   visual_uav_free_verification="PENDING") for index in absent)
        source_hashes[label.relative_to(ROOT).as_posix()] = before
    output.mkdir(parents=True)
    frames = output / "screen_sample"
    frames.mkdir()
    write_csv(output / "train_sequence_inventory.csv", inventory, list(inventory[0]))
    write_csv(output / "annotation_absent_candidates.csv", candidates, list(candidates[0]))
    sample = []
    for row in inventory:
        if row["family_in_reserved_val_or_test"]:
            continue
        name = row["sequence"]
        video = ROOT / row["visible_video"]
        source_hashes[video.relative_to(ROOT).as_posix()] = digest(video)
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise ValueError(f"Could not open {video}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        assert count == row["label_frames"], (name, count, row["label_frames"])
        row.update(video_frames=count, video_fps=fps, video_sha256=digest(video))
        indexes = [r["frame_index"] for r in candidates if r["sequence"] == name]
        selected = sorted({indexes[i] for i in (0, len(indexes)//2, len(indexes)-1)}) if indexes else []
        for frame_index in selected:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, bgr = capture.read()
            if not success:
                raise ValueError((name, frame_index, "decode failed"))
            filename = f"{name}_frame{frame_index:06d}.png"
            path = frames / filename
            assert cv2.imwrite(str(path), bgr)
            feature = extract_features(path.read_bytes(), boxes=(), max_side=640, nfeatures=1024)
            sample.append(dict(sequence=name, frame_index=frame_index, filename_family=family(name),
                               source_split="train", frame_file=path.relative_to(ROOT).as_posix(),
                               frame_sha256=digest(path), decoded_rgb_sha256=hashlib.sha256(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes()).hexdigest(),
                               width=bgr.shape[1], height=bgr.shape[0], keypoints=len(feature["points"]),
                               low_texture_family_link=len(feature["points"]) < 16,
                               annotation_exist=0, visual_uav_free_verification="PENDING"))
        capture.release()
    # The original Clean_v2 low-texture family is wholly inside its Train block.
    train = {row["image"]: row for row in read_csv(RELEASE / "dataset_manifest.csv") if row["new_split"] == "train"}
    low = [row for row in read_csv(RELEASE / "background_feature_index.csv")
           if row["image_id"] in train and int(row["keypoints"]) < 16]
    assert len(low) == 329
    write_csv(output / "screen_sample.csv", sample, list(sample[0]))
    all_inventory_fields = list(dict.fromkeys(key for row in inventory for key in row))
    write_csv(output / "train_sequence_inventory.csv", inventory, all_inventory_fields)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    sheet = Image.new("RGB", (1500, ((len(sample)+2)//3)*320+45), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), "RGPT Train only | annotation-absent sample | UAV-free and cross-release independence unverified", fill="black", font=font)
    for index, row in enumerate(sample):
        image = Image.open(ROOT / row["frame_file"]).convert("RGB")
        image.thumbnail((480, 265))
        x, y = (index % 3)*500+10, (index // 3)*320+45
        draw.text((x, y), f"{row['sequence']} / {row['frame_index']}\nkeypoints={row['keypoints']}", fill="black", font=font)
        sheet.paste(image, (x, y+45))
    sheet.save(output / "source_screen.png")
    for name, before in source_hashes.items():
        assert digest(ROOT / name) == before, name
    report = dict(status="SOURCE_CANDIDATE_AUDIT", generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  source_name="Local antiuav_rgpt, release identity unverified", modality="visible.mp4",
                  split_sequence_counts={part: len(names) for part, names in split_names.items()},
                  train_annotation_frames=sum(r["label_frames"] for r in inventory),
                  train_annotation_absent=sum(r["annotated_absent"] for r in inventory),
                  reserved_family_rule="Exclude Train clips sharing the first two underscore-delimited filename tokens with any Val/Test clip",
                  reserved_rule_limit="Filename-prefix grouping is conservative and does not recover native recording identities",
                  eligible_filename_family_sequences=sum(not r["family_in_reserved_val_or_test"] for r in inventory),
                  candidate_annotation_absent_frames=len(candidates),
                  sampled_frames=len(sample), sample_rule="First, middle and last annotation-absent frame per eligible clip",
                  sampled_low_texture_links=sum(r["low_texture_family_link"] for r in sample),
                  existing_train_low_texture_family_images=len(low),
                  independence_verified=False, verified_negative_count=None, detector_inference=False,
                  test_or_val_video_decoded=False, source_sha256=source_hashes,
                  next_checks=["Cross-release DINOv2 and geometry screening", "Source/video grouping adjudication",
                               "Independent review of crop and full context", "Prospective amendment of Train-only negative-source eligibility"])
    (output / "source_audit.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
