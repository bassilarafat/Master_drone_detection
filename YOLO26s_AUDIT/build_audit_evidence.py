from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "02_datasets" / "DUT_Anti_UAV"
OUT = ROOT / "YOLO26s_AUDIT"


def phash(path: Path) -> int:
    image = Image.open(path).convert("L").resize((32, 32))
    values = np.asarray(image, dtype=np.float32)
    dct = cv2.dct(values)
    low = dct[:8, :8]
    median = np.median(low[1:, 1:])
    bits = (low > median).reshape(-1)
    return sum(int(bit) << index for index, bit in enumerate(bits))


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def main() -> None:
    result: dict[str, object] = {}
    hashes: dict[str, dict[str, list[str]]] = {}
    phashes: dict[str, list[tuple[str, int]]] = {}
    for split in ("train", "val", "test"):
        image_paths = sorted((DATASET / "images" / split).glob("*.jpg"))
        label_paths = sorted((DATASET / "labels" / split).glob("*.txt"))
        image_stems = {path.stem for path in image_paths}
        label_stems = {path.stem for path in label_paths}
        image_sha: dict[str, list[str]] = {}
        split_phash: list[tuple[str, int]] = []
        objects = 0
        empty = 0
        invalid = 0
        corrupt: list[str] = []
        for image_path in image_paths:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception:
                corrupt.append(image_path.name)
                continue
            image_sha.setdefault(hashlib.sha256(image_path.read_bytes()).hexdigest(), []).append(
                image_path.name
            )
            split_phash.append((image_path.name, phash(image_path)))
            label_path = DATASET / "labels" / split / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            empty += not lines
            objects += len(lines)
            for line in lines:
                parts = line.split()
                try:
                    cls, xc, yc, width, height = map(float, parts[:5])
                    invalid += (
                        len(parts) < 5
                        or cls != 0
                        or not np.isfinite([cls, xc, yc, width, height]).all()
                        or width <= 0
                        or height <= 0
                        or xc - width / 2 < 0
                        or yc - height / 2 < 0
                        or xc + width / 2 > 1
                        or yc + height / 2 > 1
                    )
                except (ValueError, IndexError):
                    invalid += 1
        hashes[split] = image_sha
        phashes[split] = split_phash
        result[split] = {
            "images": len(image_paths),
            "label_files": len(label_paths),
            "objects_in_paired_labels": objects,
            "empty_label_files": int(empty),
            "missing_label_files": sorted(image_stems - label_stems),
            "orphan_label_files": sorted(label_stems - image_stems),
            "corrupt_images": corrupt,
            "invalid_boxes": int(invalid),
            "unique_sha256": len(image_sha),
            "exact_duplicate_groups": [group for group in image_sha.values() if len(group) > 1],
            "exact_duplicate_pairs": sum(len(group) * (len(group) - 1) // 2 for group in image_sha.values()),
        }

    cross_split_exact = []
    cross_split_near = []
    for left_split, right_split in (("train", "val"), ("train", "test"), ("val", "test")):
        left_by_sha = hashes[left_split]
        right_by_sha = hashes[right_split]
        for sha, left_names in left_by_sha.items():
            if sha in right_by_sha:
                cross_split_exact.append(
                    {"left_split": left_split, "left": left_names, "right_split": right_split, "right": right_by_sha[sha]}
                )
        right_values = phashes[right_split]
        for left_name, left_hash in phashes[left_split]:
            best_distance = min(hamming(left_hash, right_hash) for _, right_hash in right_values)
            if best_distance <= 8:
                matches = [name for name, right_hash in right_values if hamming(left_hash, right_hash) == best_distance]
                cross_split_near.append(
                    {
                        "left_split": left_split,
                        "left": left_name,
                        "right_split": right_split,
                        "right": matches[:10],
                        "hamming_distance": best_distance,
                    }
                )
    result["cross_split_exact_sha256"] = cross_split_exact
    result["cross_split_near_phash_distance_le_8"] = cross_split_near
    (OUT / "dataset_integrity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    with (OUT / "dataset_phash_near_matches.csv").open("w", encoding="utf-8") as handle:
        handle.write("left_split,left,right_split,right,hamming_distance\n")
        for row in cross_split_near:
            for right in row["right"]:
                handle.write(
                    f'{row["left_split"]},{row["left"]},{row["right_split"]},{right},{row["hamming_distance"]}\n'
                )


if __name__ == "__main__":
    main()
