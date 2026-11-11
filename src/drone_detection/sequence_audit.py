"""Read-only recovery of source sequence identities for a frozen dataset.

Flat export numbers and similarity winners never become verified frame IDs.
Only exact file/RGB identity with an inventoried original can establish a
direct content crosswalk. Reviewed scene links are separate from source IDs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .data_audit import digest, dump_csv
from .dataset_snapshot import verify_snapshot

NAMESPACE = "DUT_Anti_UAV_Tracking_V0/"
IDENTITY_KEYS = ("sequence_id", "video_id", "clip_id", "frame_number")
PROVENANCE_TAGS = {
    "DocumentName", "ImageDescription", "Software", "DateTime",
    "DateTimeOriginal", "UserComment", "XPTitle", "ImageUniqueID",
}


def _csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_original(sha256: str, pixel_sha256: str, by_sha: dict, by_pixel: dict) -> dict:
    """Return explicit ambiguity instead of choosing an arbitrary source frame."""
    byte_matches = by_sha.get(sha256, [])
    rgb_matches = by_pixel.get(pixel_sha256, [])
    # Even an exact byte match is not unique provenance if another sequence
    # contains the same decoded pixels in a differently encoded file.
    matches = list({row["original_path"]: row for row in byte_matches + rgb_matches}.values())
    method = "exact_decoded_rgb_sha256"
    if byte_matches:
        method = ("exact_file_sha256" if len(matches) == len({r["original_path"] for r in byte_matches})
                  else "exact_file_and_rgb_equivalent_sources")
    result = dict(sequence_id="", video_id="", clip_id="", frame_number="",
                  verified="no", status="UNRESOLVED", evidence_type="none",
                  frame_number_basis="", original_paths="[]", possible_source_sequences="[]")
    if not matches:
        return result
    videos = sorted({row["video_id"] for row in matches})
    result.update(evidence_type=method,
                  original_paths=json.dumps(sorted({row["original_path"] for row in matches})),
                  possible_source_sequences=json.dumps([NAMESPACE + name for name in videos]))
    if len(videos) != 1:
        result["status"] = "AMBIGUOUS_MULTIPLE_SOURCE_SEQUENCES"
        return result
    result.update(sequence_id=NAMESPACE + videos[0], video_id=NAMESPACE + videos[0], verified="yes")
    frames = sorted({int(row["frame_number"]) for row in matches})
    if len(frames) == 1:
        result.update(status="EXACT_CONTENT_CROSSWALK", frame_number=frames[0],
                      frame_number_basis="Original extracted-frame filename index; capture timestamp unknown")
    else:
        result["status"] = "VERIFIED_SEQUENCE_AMBIGUOUS_FRAME"
    return result


def image_metadata(path: Path, expected_sha: str, cached: dict | None = None) -> dict:
    from PIL import ExifTags, Image

    with Image.open(path) as image:
        width, height = image.size
        exif = image.getexif()
        fields = dict(exif)
        if 0x8769 in exif:
            fields.update(exif.get_ifd(0x8769))
        names = sorted(ExifTags.TAGS.get(key, str(key)) for key in fields)
        text = {}
        for key, value in fields.items():
            name = ExifTags.TAGS.get(key, str(key))
            if name in PROVENANCE_TAGS:
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                text[name] = str(value)[:512]
        if image.info.get("comment"):
            comment = image.info["comment"]
            text["JPEG_comment"] = (comment.decode("utf-8", errors="replace")
                                    if isinstance(comment, bytes) else str(comment))[:512]
        for key in ("xmp", "XML:com.adobe.xmp"):
            value = image.info.get(key)
            if value:
                text[key] = (value.decode("utf-8", errors="replace")
                             if isinstance(value, bytes) else str(value))[:16384]
        if cached and cached.get("sha256") == expected_sha:
            pixel_hash = cached["pixel_sha256"]
        else:
            rgb = image.convert("RGB")
            rgb.load()
            pixel_hash = hashlib.sha256(f"{width}x{height}:".encode() + rgb.tobytes()).hexdigest()
    claims = {}
    for value in text.values():
        for key in IDENTITY_KEYS:
            found = re.search(r"\b" + key + r"\s*[:=]\s*([\w./-]+)", value, re.IGNORECASE)
            if found:
                claims[key] = found.group(1)
    return dict(width=width, height=height, sha256=expected_sha, pixel_sha256=pixel_hash,
                numeric_export_index=int(path.stem) if path.stem.isdigit() else "",
                exif_present=bool(exif), exif_tag_names=json.dumps(names),
                provenance_metadata=json.dumps(text, ensure_ascii=False),
                embedded_identity_claims=json.dumps(claims),
                metadata_claims_verified="no")


def historical_candidates(dataset: Path, project: Path) -> list[dict]:
    """Preserve old hypotheses as candidates; never populate definitive IDs."""
    directory = project / "04_experiments/dataset_integrity"
    rows = []
    specs = [
        ("sequence_classification_sample.csv", "split", "current_file", "best_video", "best_frame", "best_distance", "phash256"),
        ("temporal_source_mapping_validation.csv", "split", "current_frame", "video", "best_original_frame", "ssim", "ssim_local_search"),
    ]
    for filename, split_key, image_key, video_key, frame_key, score_key, method in specs:
        path = directory / filename
        if not path.is_file():
            continue
        path_hash = digest(path)
        for row in _csv(path):
            name = row[image_key]
            if method == "ssim_local_search":
                name = f"{int(name):05d}.jpg"
            if not (dataset / "images" / row[split_key] / name).is_file():
                continue
            rows.append(dict(split=row[split_key], image=name, candidate_video_id=NAMESPACE + row[video_key],
                             candidate_original_frame=row[frame_key], method=method, score=row[score_key],
                             verified="no", status="UNVERIFIED_CANDIDATE", source=str(path),
                             source_sha256=path_hash))
    return rows


def audit_provenance(dataset, original_index, output, *, current_index=None, scene_links=None):
    dataset, original_index, output = map(lambda p: Path(p).resolve(), (dataset, original_index, output))
    if output == dataset or dataset in output.parents:
        raise ValueError("Write Gate 2 outputs outside the frozen dataset")
    if output.exists():
        raise FileExistsError(f"Preserve existing audit results: {output}")
    before = verify_snapshot(dataset)
    original_rows = _csv(original_index)
    by_sha, by_pixel = defaultdict(list), defaultdict(list)
    for row in original_rows:
        if row.get("error") or row.get("issue"):
            continue
        if not re.fullmatch(r"video\d+", row["video_id"]) or int(row["frame_number"]) < 1:
            raise ValueError("Original frame index has invalid sequence/frame identifiers")
        by_sha[row["sha256"]].append(row)
        by_pixel[row["pixel_sha256"]].append(row)
    cache = {}
    if current_index is not None:
        for line in Path(current_index).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cache[row["path"]] = row
    with (dataset / "MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
        payload = {row["path"]: row for row in csv.DictReader(handle)}
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(original_index, output / "original_frame_inventory.csv")
    project = Path(__file__).resolve().parents[2]
    candidates = historical_candidates(dataset, project)
    candidates_by_image = defaultdict(set)
    for row in candidates:
        candidates_by_image[(row["split"], row["image"])].add(row["candidate_video_id"])
    scenes = json.loads(Path(scene_links).read_text(encoding="utf-8")) if scene_links else []
    reviewed = defaultdict(set)
    for link in scenes:
        for suffix in ("a", "b"):
            split, name = link[f"split_{suffix}"], link[f"image_{suffix}"]
            if split not in ("train", "val", "test") or f"images/{split}/{name}" not in payload:
                raise ValueError("Reviewed scene link is not an image in this frozen release")
        if link.get("same_scene_confirmed") is True:
            for suffix in ("a", "b"):
                reviewed[(link[f"split_{suffix}"], link[f"image_{suffix}"])].add(link["review_scene_id"])
    metadata_rows, manifest = [], []
    for relative, entry in sorted(payload.items()):
        if not relative.startswith("images/"):
            continue
        _, split, name = relative.split("/")
        meta = image_metadata(dataset / relative, entry["sha256"], cache.get(relative))
        row = dict(split=split, image=name, **meta)
        metadata_rows.append(row)
        identity = resolve_original(meta["sha256"], meta["pixel_sha256"], by_sha, by_pixel)
        manifest.append(dict(split=split, image=name, image_sha256=entry["sha256"], **identity,
                             numeric_export_index=meta["numeric_export_index"],
                             reviewed_scene_ids=json.dumps(sorted(reviewed[(split, name)])),
                             historical_candidate_video_ids=json.dumps(sorted(candidates_by_image[(split, name)]))))
        if len(manifest) % 1000 == 0:
            print(f"Gate 2 metadata: {len(manifest)} images", flush=True)
    dump_csv(output / "image_metadata.csv", metadata_rows, list(metadata_rows[0]))
    dump_csv(output / "sequence_manifest.csv", manifest, list(manifest[0]))
    dump_csv(output / "historical_candidates.csv", candidates,
             ["split", "image", "candidate_video_id", "candidate_original_frame", "method", "score", "verified", "status", "source", "source_sha256"])
    _write_json(output / "reviewed_scene_links.json", scenes)
    sequences = defaultdict(set)
    for row in manifest:
        if row["verified"] == "yes":
            sequences[row["sequence_id"]].add(row["split"])
    shared = [dict(sequence_id=key, splits=sorted(value)) for key, value in sequences.items() if len(value) > 1]
    links = [link for link in scenes if link.get("same_scene_confirmed") is True and link["split_a"] != link["split_b"]]
    coverage = sum(row["verified"] == "yes" for row in manifest)
    after = verify_snapshot(dataset)
    if any(before[key] != after[key] for key in ("manifest_sha256", "freeze_sha256")):
        raise ValueError("Frozen dataset identity changed during Gate 2")
    blocked = bool(links or shared)
    status = "FAIL_CONFIRMED_CROSS_SPLIT_SCENE_OR_SEQUENCE" if blocked else "INDEPENDENCE_UNRESOLVED"
    report = dict(
        gate="Gate 2 / Phase 1: source sequence recovery", created_at_utc=datetime.now(timezone.utc).isoformat(),
        dataset_snapshot=before, dataset_unchanged=True, images=len(manifest),
        original_sequences=len({row["video_id"] for row in original_rows}), original_frames=len(original_rows),
        original_index_sha256=digest(original_index),
        current_index_sha256=digest(current_index) if current_index is not None else None,
        scene_links_input_sha256=digest(scene_links) if scene_links is not None else None,
        embedded_exif_images=sum(row["exif_present"] for row in metadata_rows),
        embedded_identity_claim_images=sum(row["embedded_identity_claims"] != "{}" for row in metadata_rows),
        verified_source_images=coverage, unresolved_source_images=len(manifest)-coverage,
        mapping_status_counts=dict(Counter(row["status"] for row in manifest)),
        verified_shared_source_sequences=shared, confirmed_reviewed_cross_split_scene_links=len(links),
        status=status, independence_pass=False,
        limitations=[
            "Detection and Tracking are separate official dataset components; Tracking may cover only part of Detection.",
            "Numeric Detection filenames are export indices, not source frame numbers.",
            "pHash/SSIM candidates do not verify recording identity or exact original frame index.",
            "Reviewed scene links confirm visible shared scenes, not a timestamp or native video ID.",
            "This phase is not exhaustive scene clustering; incomplete source coverage cannot establish independence.",
        ],
    )
    _write_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--original-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--current-index")
    parser.add_argument("--scene-links")
    args = parser.parse_args(argv)
    audit_provenance(args.dataset, args.original_index, args.output,
                     current_index=args.current_index, scene_links=args.scene_links)


if __name__ == "__main__":
    main()
