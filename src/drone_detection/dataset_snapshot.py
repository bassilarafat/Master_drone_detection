"""Create once, then verify a dated, independent YOLO dataset release.

Cleaning never edits the source, invents boxes, or treats ambiguous labels as
background. Exact duplicates with conflicting labels are excluded as a group.
Near-duplicate scenes and historical test exposure remain scientific limits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import stat
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

from .data_audit import IMAGE_SUFFIXES, SPLITS, digest, dump_csv

SCHEMA_VERSION = 1
LIMITS = [
    "Existing split membership is preserved; this is not a scene/video-disjoint resplit.",
    "Known shared Train/Validation scenes and unresolved Test provenance remain.",
    "Historical Test use during development is not undone by deduplication.",
    "Current source annotations are frozen, not independently human-adjudicated ground truth.",
]


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _source_files(source: Path) -> list[Path]:
    files = [source / "data.yaml"]
    for parent in ("images", "labels"):
        directory = source / parent
        files.extend(p for p in directory.rglob("*") if p.is_file())
    return sorted(files, key=lambda p: p.relative_to(source).as_posix())


def _inventory(root: Path, paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Symlinks are not allowed in a frozen release: {path}")
        rows.append(dict(path=path.relative_to(root).as_posix(),
                         size=path.stat().st_size, sha256=digest(path)))
    return rows


def _inventory_digest(rows: list[dict]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _annotations(path: Path) -> tuple[list[tuple], list[str]]:
    rows, problems = [], []
    if not path.is_file():
        return [], ["missing_label"]
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        return [], [f"unreadable_label: {exc}"]
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            values = tuple(Decimal(part) for part in line.split())
            if len(values) != 5 or not all(v.is_finite() for v in values):
                raise ValueError("malformed_row")
            cls, x, y, w, h = values
            if cls != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError("invalid_class_center_or_nonpositive_dimensions")
            tolerance = Decimal("0.000000001")
            if min(x-w/2, y-h/2) < -tolerance or max(x+w/2, y+h/2) > 1+tolerance:
                raise ValueError("box_outside_image")
            rows.append(values)
        except (ValueError, InvalidOperation) as exc:
            problems.append(f"line {number}: {exc}")
    if len(set(rows)) != len(rows):
        problems.append("duplicate_annotation_rows")
    return sorted(rows), problems


def _scan(source: Path) -> list[dict]:
    from PIL import Image

    records = []
    for split in SPLITS:
        image_dir = source / "images" / split
        if not image_dir.is_dir() or not (source / "labels" / split).is_dir():
            raise ValueError(f"Missing split directories: {split}")
        seen = set()
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f"Empty source split: {split}")
        for index, path in enumerate(images, 1):
            if path.stem.lower() in seen:
                raise ValueError(f"Ambiguous image/label stem: {path}")
            seen.add(path.stem.lower())
            label = source / "labels" / split / (path.stem + ".txt")
            annotations, issues = _annotations(label)
            pixel_hash = ""
            width = height = 0
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                    rgb.load()
                    width, height = rgb.size
                    if min(width, height) < 10:
                        raise ValueError("image_smaller_than_10_pixels")
                    pixel_hash = hashlib.sha256(
                        f"{width}x{height}:".encode() + rgb.tobytes()
                    ).hexdigest()
                if path.suffix.lower() in {".jpg", ".jpeg"}:
                    with path.open("rb") as handle:
                        handle.seek(-2, 2)
                        if handle.read() != b"\xff\xd9":
                            raise ValueError("JPEG_requires_repair")
            except (OSError, ValueError) as exc:
                issues.append(f"invalid_image: {exc}")
            records.append(dict(split=split, image=path, label=label,
                                image_sha256=digest(path), pixel_sha256=pixel_hash,
                                width=width, height=height, annotations=annotations,
                                issues=issues, reason="invalid_annotation_or_image" if issues else "",
                                duplicate_group="", kept_image=""))
            if index % 500 == 0:
                print(f"snapshot scan {split}: {index}/{len(images)}", flush=True)
    return records


def _deduplicate(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        if record["pixel_sha256"]:
            groups[record["pixel_sha256"]].append(record)
    duplicate_rows = []
    for pixel_hash, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        members.sort(key=lambda r: (r["split"], r["image"].name))
        cross_split = len({r["split"] for r in members}) > 1
        conflicting = any(r["issues"] for r in members) or any(
            r["annotations"] != members[0]["annotations"] for r in members[1:]
        )
        if cross_split:
            decision = "quarantine_cross_split_exact_duplicates"
        elif conflicting:
            decision = "quarantine_conflicting_annotations"
        else:
            decision = "keep_lexicographically_first"
        kept = members[0] if not cross_split and not conflicting else None
        for record in members:
            record["duplicate_group"] = pixel_hash
            record["kept_image"] = f"images/{kept['split']}/{kept['image'].name}" if kept else ""
            if record is not kept:
                record["reason"] = decision if kept is None else "redundant_exact_duplicate"
            duplicate_rows.append(dict(
                group=pixel_hash, image=f"images/{record['split']}/{record['image'].name}",
                byte_sha256=record["image_sha256"], decision=decision,
                retained=record is kept, kept_image=record["kept_image"],
            ))
    return duplicate_rows


def verify_snapshot(dataset: Path | str) -> dict:
    """Verify all release bytes and reject missing, additional or changed files."""
    import yaml

    dataset = Path(dataset).resolve()
    if dataset.is_file():
        dataset = dataset.parent
    freeze_path, manifest_path = dataset / "FREEZE.json", dataset / "MANIFEST.csv"
    if not freeze_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"Missing frozen dataset metadata: {dataset}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported dataset freeze schema")
    if digest(manifest_path) != freeze["manifest_sha256"]:
        raise ValueError("Frozen dataset MANIFEST.csv changed")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        entries = list(csv.DictReader(handle))
    paths = [row["path"] for row in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate paths in frozen manifest")
    for name in paths:
        candidate = Path(name)
        if candidate.is_absolute() or ".." in candidate.parts or "\\" in name:
            raise ValueError(f"Unsafe frozen manifest path: {name}")
    actual = set()
    for path in dataset.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Frozen dataset contains symlink: {path}")
        if path.is_file():
            actual.add(path.relative_to(dataset).as_posix())
    expected = set(paths) | {"FREEZE.json", "MANIFEST.csv"}
    if actual != expected:
        raise ValueError(f"Frozen dataset inventory changed: missing={sorted(expected-actual)[:8]}, "
                         f"added={sorted(actual-expected)[:8]}")
    def check_entry(row):
        path = dataset / row["path"]
        if path.stat().st_size != int(row["size"]) or digest(path) != row["sha256"]:
            raise ValueError(f"Frozen dataset file changed: {row['path']}")
    with ThreadPoolExecutor(max_workers=6) as executor:
        for _ in executor.map(check_entry, entries):
            pass
    report = json.loads((dataset / "cleaning_report.json").read_text(encoding="utf-8"))
    expected_freeze = dict(
        schema_version=SCHEMA_VERSION, dataset_name=report["dataset_name"],
        created_at_utc=report["created_at_utc"], manifest_sha256=digest(manifest_path),
        source_dataset_sha256=report["source_dataset_sha256"], counts=report["counts"],
        immutable=True, read_only_files=True,
    )
    if freeze != expected_freeze or dataset.name != freeze["dataset_name"]:
        raise ValueError("Frozen release identity differs from the hashed cleaning report")
    data = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    if Path(data["path"]).resolve() != dataset:
        raise ValueError("Frozen data.yaml must reference this release directory")
    for split in SPLITS:
        if data.get(split) != f"images/{split}":
            raise ValueError(f"Frozen data.yaml redirects {split} outside the release")
    return dict(dataset_name=freeze["dataset_name"], created_at_utc=freeze["created_at_utc"],
                manifest_sha256=freeze["manifest_sha256"], freeze_sha256=digest(freeze_path),
                files=len(entries), counts=freeze["counts"],
                source_dataset_sha256=freeze["source_dataset_sha256"],
                scientific_limits=report.get("scientific_limits", LIMITS))


def create_snapshot(source: Path | str, destination: Path | str) -> dict:
    """Build a fresh release and make its files read-only; never overwrite one."""
    import yaml

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Dataset versions are never overwritten: {destination}")
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate dataset directories")
    if not (source / "data.yaml").is_file():
        raise ValueError(f"Missing source data.yaml: {source}")
    source_yaml = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    if source_yaml.get("names") not in ({0: "UAV"}, ["UAV"]):
        raise ValueError("This release policy expects the single class 0: UAV")
    print("Recording original source hashes...", flush=True)
    source_manifest = _inventory(source, _source_files(source))
    source_fingerprint = _inventory_digest(source_manifest)
    records = _scan(source)
    duplicates = _deduplicate(records)
    selected = [r for r in records if not r["reason"]]
    counts = {}
    for split in SPLITS:
        retained = [r for r in selected if r["split"] == split]
        if not retained:
            raise ValueError(f"Cleaning would leave an empty split: {split}")
        counts[split] = dict(images=len(retained), objects=sum(len(r["annotations"]) for r in retained),
                             negative_images=sum(not r["annotations"] for r in retained))
    # This staging directory is a direct child of the explicitly resolved target
    # parent. No old directory is deleted, reused, or moved out of the workspace.
    staging = destination.parent / f".{destination.name}.building-{uuid.uuid4().hex}"
    if staging.resolve().parent != destination.parent:
        raise ValueError("Staging directory escaped the destination parent")
    staging.mkdir(parents=True, exist_ok=False)
    source_by_path = {r["path"]: r for r in source_manifest}
    for split in SPLITS:
        (staging / "images" / split).mkdir(parents=True)
        (staging / "labels" / split).mkdir(parents=True)
    for record in selected:
        for original in (record["image"], record["label"]):
            relative = original.relative_to(source)
            target = staging / relative
            shutil.copyfile(original, target)
            if digest(target) != source_by_path[relative.as_posix()]["sha256"]:
                raise ValueError(f"Source changed during copy: {original}")
    release_yaml = dict(path=destination.as_posix(),
                        **{split: f"images/{split}" for split in SPLITS}, names={0: "UAV"})
    (staging / "data.yaml").write_text(yaml.safe_dump(release_yaml, sort_keys=False), encoding="utf-8")
    exclusions = []
    for r in records:
        if r["reason"]:
            exclusions.append(dict(
                image=r["image"].relative_to(source).as_posix(),
                label=r["label"].relative_to(source).as_posix(), reason=r["reason"],
                details=json.dumps(r["issues"]), duplicate_group=r["duplicate_group"],
                kept_image=r["kept_image"],
                source_annotation=r["label"].read_text(encoding="utf-8-sig") if r["label"].is_file() else "",
            ))
    paired_labels = {r["label"].relative_to(source).as_posix() for r in records}
    paired_paths = {
        p.relative_to(source).as_posix() for r in records for p in (r["image"], r["label"])
    }
    extras = []
    for row in source_manifest:
        name = row["path"]
        if name.startswith("labels/") and name.endswith(".txt") and name not in paired_labels:
            extras.append(dict(path=name, reason="orphan_or_backup_label", sha256=row["sha256"]))
        elif name != "data.yaml" and name not in paired_paths:
            extras.append(dict(path=name, reason="non_dataset_artifact", sha256=row["sha256"]))
    dump_csv(staging / "source_manifest.csv", source_manifest, ["path", "size", "sha256"])
    dump_csv(staging / "excluded_samples.csv", exclusions,
             ["image", "label", "reason", "details", "duplicate_group", "kept_image", "source_annotation"])
    dump_csv(staging / "excluded_artifacts.csv", extras, ["path", "reason", "sha256"])
    dump_csv(staging / "duplicate_decisions.csv", duplicates,
             ["group", "image", "byte_sha256", "decision", "retained", "kept_image"])
    created = datetime.now(timezone.utc)
    history = []
    history_directory = source.parent.parent / "06_results" / "dataset_normalization"
    if history_directory.is_dir():
        (staging / "provenance").mkdir()
        for name in ("report.json", "changes.csv", "errors.csv"):
            original = history_directory / name
            if original.is_file():
                target = staging / "provenance" / f"annotation_normalization_{name}"
                shutil.copyfile(original, target)
                history.append(dict(source=str(original), sha256=digest(target),
                                    included_as=target.relative_to(staging).as_posix()))
    report = dict(
        dataset_name=destination.name, created_at_utc=created.isoformat(),
        created_at_cairo=created.astimezone(ZoneInfo("Africa/Cairo")).isoformat(),
        source=str(source), source_dataset_sha256=source_fingerprint,
        source_images={s: sum(r["split"] == s for r in records) for s in SPLITS},
        counts=counts, excluded_images=len(exclusions),
        excluded_reasons=dict(Counter(r["reason"] for r in exclusions)),
        excluded_artifacts=len(extras),
        annotation_policy="Preserve retained source label bytes; exclude whole image/label on invalid annotation.",
        duplicate_policy="Decoded RGB identity: one lexicographic copy per split only for identical numeric label multisets; quarantine conflicting or cross-split groups.",
        scientific_limits=LIMITS, annotation_history_evidence=history,
        experiments_started_at_release_creation=False,
    )
    _json(staging / "cleaning_report.json", report)
    card = f"""# {destination.name}

Frozen release created {report['created_at_cairo']} (Africa/Cairo).
No experiments have started at release creation.

## Cleaning decisions

Only paired, valid images and labels enter images/ and labels/. Retained label
bytes are unchanged from the source. Invalid annotations quarantine the entire
image/label pair; no box is guessed and no ambiguous image becomes a negative.
Excluded files remain in the untouched source and are documented in the CSVs.
Exact decoded-RGB duplicates with identical numeric label multisets retain the
lexicographically first image. Conflicting annotations or exact cross-split
groups quarantine all members. Near duplicates are not automatically removed.
Caches, backups, orphan labels, runs and visual-check artifacts are not copied.

## Frozen protocol

All release files are read-only. MANIFEST.csv hashes every payload and provenance
file; FREEZE.json binds that manifest. Verify before and after each experiment.
The project's train command verifies this release, suppresses in-place label
cache writes, and records its identity with experiment results. Direct notebook
or third-party training must obey the same checks and cache restrictions.
Never edit this version, including its metadata. Corrections require Clean_v2.
Do not run dataset-normalize --write on this directory. Do not create caches,
plots or outputs under it. OS read-only flags are an accident guard, not a
security boundary; hash verification detects changes.

## Scientific limits

""" + "\n".join(f"- {limit}" for limit in LIMITS) + "\n"
    (staging / "DATASET_CARD.md").write_text(card, encoding="utf-8")
    print("Verifying original source is unchanged...", flush=True)
    if _inventory(source, _source_files(source)) != source_manifest:
        raise ValueError(f"Source changed during release creation; incomplete staging preserved: {staging}")
    manifest = _inventory(staging, sorted(p for p in staging.rglob("*") if p.is_file()))
    dump_csv(staging / "MANIFEST.csv", manifest, ["path", "size", "sha256"])
    freeze = dict(schema_version=SCHEMA_VERSION, dataset_name=destination.name,
                  created_at_utc=report["created_at_utc"],
                  manifest_sha256=digest(staging / "MANIFEST.csv"),
                  source_dataset_sha256=source_fingerprint, counts=counts,
                  immutable=True, read_only_files=True)
    _json(staging / "FREEZE.json", freeze)
    # Both absolute targets were checked above; the final name must stay unused.
    if destination.exists() or staging.resolve().parent != destination.parent:
        raise FileExistsError(f"Destination appeared during build: {destination}")
    staging.rename(destination)
    for path in destination.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    result = verify_snapshot(destination)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Create a new immutable dataset version.")
    create.add_argument("--source", type=Path, required=True)
    create.add_argument("--destination", type=Path, required=True)
    verify = commands.add_parser("verify", help="Verify every frozen file without writing.")
    verify.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "create":
        create_snapshot(args.source, args.destination)
    else:
        print(json.dumps(verify_snapshot(args.dataset), indent=2))


if __name__ == "__main__":
    main()
