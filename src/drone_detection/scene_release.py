"""Balance whole scene groups and publish a release only after fresh Gate 2.

Confirmed scene groups and conservative allocation blocks are distinct:
allocation blocks may keep unresolved likely pairs together without claiming
that their images have a verified common scene or native video identity.
"""
from __future__ import annotations

import csv
import argparse
import json
import shutil
import stat
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from .data_audit import IMAGE_SUFFIXES, SPLITS, digest, dump_csv
from .dataset_snapshot import SCHEMA_VERSION, _annotations, _inventory, verify_snapshot

FEATURES = ("images", "objects", "tiny", "small", "medium", "large", "negative_images")
SIZE_LIMITS = (32 ** 2, 96 ** 2, 256 ** 2)
REQUIRED_GATE_CHECKS = (
    "exact_cross_split_sha256", "scene_group_cross_split",
    "confirmed_same_scene_train_val", "confirmed_same_scene_train_test",
    "confirmed_same_scene_val_test", "invalid_annotations",
    "unresolved_high_risk_cross_split",
)
SCIENTIFIC_LIMITS = [
    "Scene-group-disjoint under the declared retrieval, geometric verification and review protocol; not sequence-disjoint.",
    "Original recording IDs, capture timestamps and original video frame counters remain unresolved.",
    "Visual retrieval is not a proof that every possible scene relationship has been recovered.",
    "Annotations inherit the frozen Clean_v1 state; no boxes are invented or edited during resplitting.",
    "Historical project checkpoints may have seen images now assigned to Test; they must not initialize final paper experiments.",
]


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _old_identity(row):
    """Accept graph IDs as ``train/name.jpg`` or split plus a basename."""
    value = row.get("image_id") or row.get("image") or ""
    value = value.replace("\\", "/")
    if value.startswith("images/"):
        value = value[len("images/"):]
    parts = value.split("/")
    if len(parts) == 2 and parts[0] in SPLITS and parts[1] not in ("", ".", ".."):
        split, name = parts
    elif len(parts) == 1 and row.get("old_split") in SPLITS:
        split, name = row["old_split"], parts[0]
    else:
        raise ValueError(f"Invalid source image identity: {value!r}")
    if row.get("old_split") not in (None, "", split):
        raise ValueError(f"Conflicting old split for {value}")
    if not name or Path(name).name != name or ":" in name:
        raise ValueError(f"Invalid source image basename: {name!r}")
    return split, name


def _bin_counts(annotations, width, height):
    counts = dict(tiny=0, small=0, medium=0, large=0)
    for _, _, _, normalized_w, normalized_h in annotations:
        # These are area bins in ORIGINAL image pixels; Tiny and the extra
        # Medium/Large boundary are project extensions, not COCO AP metrics.
        area = Decimal(str(normalized_w)) * width * Decimal(str(normalized_h)) * height
        key = next((name for limit, name in zip(SIZE_LIMITS, ("tiny", "small", "medium"))
                    if area < limit), "large")
        counts[key] += 1
    return counts


def load_scene_members(source, groups_csv, *, metadata_csv=None, verify_source=True):
    """Read every source image exactly once and bind group IDs to its bytes.

    ``verify_source=False`` is only for a caller that has already verified the
    same snapshot; each copied output is still checked against its manifest.
    Cached dimensions are reused only when their image SHA matches the source.
    """
    from PIL import Image

    source = Path(source).resolve()
    if verify_source:
        verify_snapshot(source)
    payload = {r["path"]: r for r in _read_csv(source / "MANIFEST.csv")}
    expected = {path[len("images/"):] for path in payload if path.startswith("images/")}
    metadata = {}
    if metadata_csv is not None:
        for row in _read_csv(metadata_csv):
            split = row.get("old_split") or row.get("split")
            image = row.get("image") or row.get("image_id")
            if split and image:
                metadata[f"{split}/{Path(image).name}"] = row
    members, seen = [], set()
    scene_blocks = defaultdict(set)
    for row in _read_csv(groups_csv):
        split, name = _old_identity(row)
        image_id = f"{split}/{name}"
        if image_id not in expected or image_id in seen:
            raise ValueError(f"Scene groups must cover source images exactly once: {image_id}")
        seen.add(image_id)
        group = row.get("scene_group_id", "").strip()
        if not group:
            raise ValueError(f"Missing scene_group_id: {image_id}")
        block = row.get("allocation_block_id", "").strip() or group
        scene_blocks[group].add(block)
        image_rel = f"images/{image_id}"
        label_rel = f"labels/{split}/{Path(name).stem}.txt"
        if label_rel not in payload:
            raise ValueError(f"Source manifest lacks label: {label_rel}")
        annotations, issues = _annotations(source / label_rel)
        if issues:
            raise ValueError(f"Invalid source annotations: {image_id}: {issues}")
        cached = metadata.get(image_id, {})
        cached_sha = cached.get("sha256") or cached.get("image_sha256")
        if cached_sha == payload[image_rel]["sha256"] and cached.get("width") and cached.get("height"):
            width, height = int(cached["width"]), int(cached["height"])
        else:
            with Image.open(source / image_rel) as image:
                width, height = image.size
        sizes = _bin_counts(annotations, width, height)
        members.append(dict(
            image=image_id, old_split=split, old_filename=name,
            scene_group_id=group, allocation_block_id=block,
            source_relative_image=image_rel, source_relative_label=label_rel,
            image_sha256=payload[image_rel]["sha256"], label_sha256=payload[label_rel]["sha256"],
            width=width, height=height, images=1, objects=len(annotations), **sizes,
            negative_images=int(not annotations),
        ))
    if seen != expected:
        raise ValueError(f"Scene groups omit {len(expected-seen)} source images")
    if any(len(blocks) != 1 for blocks in scene_blocks.values()):
        raise ValueError("One confirmed scene group was assigned to multiple allocation blocks")
    return sorted(members, key=lambda r: r["image"])


def allocate_scene_groups(members, *, target_ratios=None, seed=17, restarts=64):
    """Deterministic multistart greedy allocation, followed by improving moves.

    Independence is a hard constraint. Balance is a soft weighted objective on
    images, objects and the four object-area bins; rare negatives are reported
    and receive only a small weight. No individual image is moved afterwards.
    """
    import numpy as np

    if not members or restarts < 1:
        raise ValueError("At least one member and one restart are required")
    by_unit = defaultdict(list)
    scene_units = defaultdict(set)
    seen = set()
    for member in members:
        if member["image"] in seen:
            raise ValueError("Duplicate member image")
        seen.add(member["image"])
        unit = member.get("allocation_block_id") or member["scene_group_id"]
        by_unit[unit].append(member)
        scene_units[member["scene_group_id"]].add(unit)
    if any(len(units) != 1 for units in scene_units.values()):
        raise ValueError("A scene group cannot span allocation units")
    if len(by_unit) < 3:
        raise ValueError("Fewer than three indivisible allocation units; three nonempty disjoint splits are impossible")
    units = sorted(by_unit)
    vectors = np.array([[sum(int(r.get(feature, 0)) for r in by_unit[unit])
                         for feature in FEATURES] for unit in units], dtype=np.float64)
    if np.any(vectors < 0) or np.any(vectors[:, 0] < 1):
        raise ValueError("Allocation feature counts must be nonnegative with positive image counts")
    totals = vectors.sum(axis=0)
    if target_ratios is None:
        old_counts = Counter(r["old_split"] for r in members)
        ratios = np.array([old_counts[split] / len(members) for split in SPLITS])
    else:
        if set(target_ratios) != set(SPLITS):
            raise ValueError("target_ratios must specify train, val and test")
        ratios = np.array([float(target_ratios[split]) for split in SPLITS])
    if np.any(~np.isfinite(ratios)) or np.any(ratios <= 0):
        raise ValueError("Every target split ratio must be positive and finite")
    ratios /= ratios.sum()
    targets = ratios[:, None] * totals[None, :]
    denominators = np.maximum(targets, 1.0)
    weights = np.array([3.0, 1.0, .65, .65, .65, .65, .02])
    weights[totals == 0] = 0

    def row_cost(load, split):
        return float(np.sum(weights * ((load-targets[split])/denominators[split]) ** 2))

    def cost(loads):
        return sum(row_cost(loads[split], split) for split in range(3))

    rng = np.random.default_rng(seed)
    best = None
    for restart in range(restarts):
        jitter = np.ones(len(units)) if restart == 0 else rng.uniform(.85, 1.15, len(units))
        order = sorted(range(len(units)), key=lambda i: (-vectors[i, 0]*jitter[i], units[i]))
        loads = np.zeros((3, len(FEATURES)), dtype=np.float64)
        assignment = np.full(len(units), -1, dtype=np.int8)
        occupancy = np.zeros(3, dtype=np.int64)
        for position, index in enumerate(order):
            empty = np.flatnonzero(occupancy == 0)
            choices = empty if len(order)-position == len(empty) else range(3)
            split = min(choices, key=lambda s: (
                row_cost(loads[s]+vectors[index], s)-row_cost(loads[s], s), int(s)))
            assignment[index] = split
            occupancy[split] += 1
            loads[split] += vectors[index]
        score = cost(loads)
        if best is None or score < best[0]-1e-12:
            best = (score, assignment.copy(), loads.copy(), occupancy.copy())
    _, assignment, loads, occupancy = best
    # Improve the best allocation without breaking scene/block membership.
    for _ in range(8):
        changed = False
        for index in rng.permutation(len(units)):
            old = int(assignment[index])
            if occupancy[old] <= 1:
                continue
            selected, best_delta = old, 0.0
            for new in range(3):
                if new == old:
                    continue
                delta = (row_cost(loads[old]-vectors[index], old) + row_cost(loads[new]+vectors[index], new)
                         - row_cost(loads[old], old) - row_cost(loads[new], new))
                if delta < best_delta-1e-12:
                    selected, best_delta = new, delta
            if selected != old:
                loads[old] -= vectors[index]
                loads[selected] += vectors[index]
                occupancy[old] -= 1
                occupancy[selected] += 1
                assignment[index] = selected
                changed = True
        if not changed:
            break
    unit_to_split = {unit: SPLITS[int(assignment[i])] for i, unit in enumerate(units)}
    rows = []
    for member in sorted(members, key=lambda r: r["image"]):
        unit = member.get("allocation_block_id") or member["scene_group_id"]
        new_split = unit_to_split[unit]
        name = f"{member['old_split']}_{member['old_filename']}"
        rows.append(dict(**member, new_split=new_split, new_filename=name,
                         new_relative_image=f"images/{new_split}/{name}",
                         new_relative_label=f"labels/{new_split}/{Path(name).stem}.txt"))
    split_summary = {}
    for split in SPLITS:
        selected = [r for r in rows if r["new_split"] == split]
        split_summary[split] = dict(
            **{feature: sum(int(r.get(feature, 0)) for r in selected) for feature in FEATURES},
            scene_groups=len({r["scene_group_id"] for r in selected}),
            allocation_blocks=len({r.get("allocation_block_id") or r["scene_group_id"] for r in selected}),
        )
    report = dict(
        seed=seed, restarts=restarts, target_ratios=dict(zip(SPLITS, map(float, ratios))),
        objective=cost(loads), objective_feature_weights=dict(zip(FEATURES, map(float, weights))),
        size_definition="Original-image pixel bbox area: Tiny<32^2; Small<96^2; Medium<256^2; otherwise Large. Project bins, not COCO AP metrics.",
        scene_groups=len(scene_units), allocation_blocks=len(units), totals=dict(zip(FEATURES, map(int, totals))),
        splits=split_summary, independence_constraint="Each entire scene group and allocation block belongs to exactly one split.",
        target_feature_counts={split: dict(zip(FEATURES, map(float, targets[i])))
                               for i, split in enumerate(SPLITS)},
        target_relative_deviations={split: dict(zip(FEATURES, map(float, (loads[i]-targets[i])/denominators[i])))
                                    for i, split in enumerate(SPLITS)},
        largest_allocation_blocks=[dict(allocation_block_id=units[i], images=int(vectors[i, 0]),
                                       new_split=unit_to_split[units[i]])
                                   for i in sorted(range(len(units)), key=lambda i: -vectors[i, 0])[:20]],
    )
    return rows, report


def write_allocation(rows, report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("split_assignment.csv", "allocation_report.json"):
        if (directory / name).exists():
            raise FileExistsError(f"Allocation output exists: {directory / name}")
    dump_csv(directory / "split_assignment.csv", rows, list(rows[0]))
    _json(directory / "allocation_report.json", report)


def _validate_assignment_rows(rows):
    scenes, blocks, paths, old_ids = defaultdict(set), defaultdict(set), set(), set()
    for row in rows:
        old_split, old_name = _old_identity(row)
        if row["image"] != f"{old_split}/{old_name}" or row["old_filename"] != old_name:
            raise ValueError("Assignment old identity is inconsistent")
        if row["source_relative_image"] != f"images/{old_split}/{old_name}" or row["source_relative_label"] != f"labels/{old_split}/{Path(old_name).stem}.txt":
            raise ValueError("Assignment source paths do not match its old image identity")
        if row["new_split"] not in SPLITS or row["image"] in old_ids:
            raise ValueError("Invalid/duplicate image assignment")
        old_ids.add(row["image"])
        scenes[row["scene_group_id"]].add(row["new_split"])
        blocks[row.get("allocation_block_id") or row["scene_group_id"]].add(row["new_split"])
        for key, prefix in (("new_relative_image", "images"), ("new_relative_label", "labels")):
            value = row[key]
            parts = value.split("/")
            if len(parts) != 3 or parts[0] != prefix or parts[1] != row["new_split"] or parts[2] in ("", ".", "..") or Path(parts[2]).name != parts[2] or ":" in parts[2]:
                raise ValueError(f"Unsafe assigned path: {value}")
            if value in paths:
                raise ValueError(f"Output filename collision: {value}")
            paths.add(value)
    if any(len(splits) != 1 for splits in scenes.values()) or any(len(splits) != 1 for splits in blocks.values()):
        raise ValueError("Scene group/allocation block crosses splits")
    if {row["new_split"] for row in rows} != set(SPLITS):
        raise ValueError("Three nonempty splits are required")


def prepare_scene_release(source, destination, groups_csv, assignment_csv, *, allocation_report=None):
    """Create a mutable staging copy for fresh Gate 2; never freeze or publish."""
    import yaml

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Release versions are never overwritten: {destination}")
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate directories")
    source_identity = verify_snapshot(source)
    rows = _read_csv(assignment_csv)
    _validate_assignment_rows(rows)
    expected = {r["path"][len("images/"):] for r in _read_csv(source / "MANIFEST.csv")
                if r["path"].startswith("images/")}
    if {row["image"] for row in rows} != expected:
        raise ValueError("Assignments must contain every frozen source image exactly once")
    source_payload = {r["path"]: r["sha256"] for r in _read_csv(source / "MANIFEST.csv")}
    group_rows = {}
    for group_row in _read_csv(groups_csv):
        old_split, old_name = _old_identity(group_row)
        key = f"{old_split}/{old_name}"
        if key in group_rows:
            raise ValueError("Duplicate image in scene group provenance")
        group_rows[key] = group_row
    if set(group_rows) != expected:
        raise ValueError("Scene group provenance must cover the frozen source exactly")
    for row in rows:
        original_group = group_rows[row["image"]]
        group = original_group["scene_group_id"]
        block = original_group.get("allocation_block_id") or group
        if row["scene_group_id"] != group or (row.get("allocation_block_id") or row["scene_group_id"]) != block:
            raise ValueError("Allocation differs from scene group provenance")
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    if staging.resolve().parent != destination.parent:
        raise ValueError("Staging escaped the destination parent")
    staging.mkdir(parents=True, exist_ok=False)
    for prefix in ("images", "labels"):
        for split in SPLITS:
            (staging / prefix / split).mkdir(parents=True)
    def copy_pair(row):
        for src_key, new_key, hash_key in (
            ("source_relative_image", "new_relative_image", "image_sha256"),
            ("source_relative_label", "new_relative_label", "label_sha256"),
        ):
            relative = row[src_key]
            if source_payload.get(relative) != row[hash_key]:
                raise ValueError(f"Assignment SHA differs from frozen source: {relative}")
            shutil.copyfile(source / relative, staging / row[new_key])
            if digest(staging / row[new_key]) != row[hash_key]:
                raise ValueError(f"Copied file mismatch: {row[new_key]}")
    with ThreadPoolExecutor(max_workers=6) as executor:
        for index, _ in enumerate(executor.map(copy_pair, rows), 1):
            if index % 500 == 0:
                print(f"scene release staging: {index}/{len(rows)} images", flush=True)
    shutil.copyfile(groups_csv, staging / "scene_groups.csv")
    shutil.copyfile(assignment_csv, staging / "split_assignment.csv")
    if allocation_report:
        shutil.copyfile(allocation_report, staging / "allocation_report.json")
    dump_csv(staging / "dataset_manifest.csv", rows, list(rows[0]))
    changes = [dict(image=r["image"], old_split=r["old_split"], new_split=r["new_split"],
                    new_image=r["new_relative_image"], new_label=r["new_relative_label"],
                    image_sha256=r["image_sha256"], label_sha256=r["label_sha256"],
                    change="move_split_and_rename" if r["old_split"] != r["new_split"] else "rename_for_unique_identity")
               for r in rows]
    dump_csv(staging / "dataset_changes.csv", changes, list(changes[0]))
    data = dict(path=destination.as_posix(), **{s: f"images/{s}" for s in SPLITS}, names={0: "UAV"})
    (staging / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    _json(staging / "BUILD.json", dict(
        event="staging_prepared", created_at_utc=datetime.now(timezone.utc).isoformat(),
        source=str(source), source_snapshot=source_identity, destination=str(destination),
        dataset_manifest_sha256=digest(staging / "dataset_manifest.csv"),
        split_assignment_sha256=digest(staging / "split_assignment.csv"),
        groups_input_sha256=digest(groups_csv), images=len(rows),
    ))
    after = verify_snapshot(source)
    if any(after[key] != source_identity[key] for key in ("manifest_sha256", "freeze_sha256")):
        raise ValueError("Frozen source identity changed during staging")
    return staging


def _validate_staged_bytes(staging):
    from PIL import Image

    rows = _read_csv(staging / "dataset_manifest.csv")
    _validate_assignment_rows(rows)
    expected = set()
    hash_splits = defaultdict(set)
    counts = {split: dict(images=0, objects=0, negative_images=0, tiny=0, small=0, medium=0, large=0) for split in SPLITS}
    for row in rows:
        split = row["new_split"]
        for path_key, hash_key in (("new_relative_image", "image_sha256"), ("new_relative_label", "label_sha256")):
            name = row[path_key]
            expected.add(name)
            path = staging / name
            if path.is_symlink() or not path.is_file() or digest(path) != row[hash_key]:
                raise ValueError(f"Staged file missing or changed: {name}")
        hash_splits[row["image_sha256"]].add(split)
        with Image.open(staging / row["new_relative_image"]) as image:
            if image.size != (int(row["width"]), int(row["height"])):
                raise ValueError("Staged image dimensions differ from manifest")
        annotations, issues = _annotations(staging / row["new_relative_label"])
        if issues:
            raise ValueError(f"Invalid staged annotation: {row['image']}: {issues}")
        if len(annotations) != int(row["objects"]):
            raise ValueError("Staged annotation count differs from manifest")
        counts[split]["images"] += 1
        counts[split]["objects"] += len(annotations)
        counts[split]["negative_images"] += int(not annotations)
        sizes = _bin_counts(annotations, int(row["width"]), int(row["height"]))
        for name, value in sizes.items():
            if int(row[name]) != value:
                raise ValueError("Staged area-bin counts differ from manifest")
            counts[split][name] += value
    actual = {p.relative_to(staging).as_posix() for folder in ("images", "labels")
              for p in (staging / folder).rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("Staged image/label inventory contains missing or unexpected files")
    if any(len(splits) > 1 for splits in hash_splits.values()):
        raise ValueError("Exact image SHA crosses splits")
    if any(path.is_symlink() for path in staging.rglob("*")):
        raise ValueError("Staging must not contain symlinks")
    return rows, counts


def finalize_scene_release(staging, destination, gate_report, *, provenance_files=None, release_record=None):
    """Publish once only when fresh Gate 2 is PASS and bound to these files."""
    staging, destination = Path(staging).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Release versions are never overwritten: {destination}")
    if staging.parent != destination.parent or staging == destination:
        raise ValueError("Checked staging and destination must be distinct siblings")
    if (staging / "FREEZE.json").exists() or (staging / "MANIFEST.csv").exists():
        raise ValueError("Staging has already been sealed; do not rewrite a frozen manifest")
    gate = json.loads(Path(gate_report).read_text(encoding="utf-8")) if not isinstance(gate_report, dict) else dict(gate_report)
    if gate.get("status") != "PASS":
        raise ValueError("Fresh Gate 2 must PASS before a scene-disjoint release is frozen")
    checks = gate.get("checks", {})
    for key in REQUIRED_GATE_CHECKS:
        if type(checks.get(key)) is not int or checks[key] != 0:
            raise ValueError(f"Gate 2 check must explicitly be integer zero: {key}")
    build = json.loads((staging / "BUILD.json").read_text(encoding="utf-8"))
    if Path(build["destination"]).resolve() != destination:
        raise ValueError("Staging was built for a different destination")
    for key, name in (("dataset_manifest_sha256", "dataset_manifest.csv"), ("split_assignment_sha256", "split_assignment.csv")):
        current = digest(staging / name)
        if current != build[key] or current != gate.get(key):
            raise ValueError(f"Gate 2 is not bound to current staged {name}")
    if release_record is not None:
        record_path = Path(release_record).resolve()
        if record_path == destination or destination in record_path.parents or record_path == staging or staging in record_path.parents:
            raise ValueError("External release record must stay outside the frozen dataset")
        if record_path.exists():
            raise FileExistsError("External release record already exists")
    rows, counts = _validate_staged_bytes(staging)
    for name, original in (provenance_files or {}).items():
        if Path(name).name != name or ":" in name or name in {"MANIFEST.csv", "FREEZE.json", "data.yaml", "BUILD.json", "dataset_manifest.csv", "split_assignment.csv", "scene_groups.csv", "cleaning_report.json", "DATASET_CARD.md", "gate2_report.json"}:
            raise ValueError(f"Unsafe/reserved provenance filename: {name}")
        if (staging / name).exists():
            raise FileExistsError(f"Provenance target already exists: {name}")
        shutil.copyfile(original, staging / name)
    _json(staging / "gate2_report.json", gate)
    created = datetime.now(timezone.utc)
    report = dict(
        dataset_name=destination.name, created_at_utc=created.isoformat(),
        created_at_cairo=created.astimezone(ZoneInfo("Africa/Cairo")).isoformat(),
        source=build["source"], source_dataset_sha256=build["source_snapshot"]["manifest_sha256"],
        source_freeze_sha256=build["source_snapshot"]["freeze_sha256"],
        counts=counts, scene_groups=len({r["scene_group_id"] for r in rows}),
        allocation_blocks=len({r.get("allocation_block_id") or r["scene_group_id"] for r in rows}),
        annotation_policy="Retain Clean_v1 image and label bytes; only filenames and split membership change.",
        scientific_limits=SCIENTIFIC_LIMITS, gate2_status="PASS", test_modified_after_release=False,
        experiments_started_at_release_creation=False,
    )
    _json(staging / "cleaning_report.json", report)
    (staging / "DATASET_CARD.md").write_text(
        f"# {destination.name}\n\nCreated {report['created_at_cairo']} (Africa/Cairo).\n\n"
        "Scene-group-disjoint release, not sequence-disjoint. Native video identities remain unknown.\n"
        "All confirmed scene groups and conservative allocation blocks belong wholly to one split.\n"
        "Image/label bytes are inherited unchanged from frozen Clean_v1; old split prefixes prevent filename collisions.\n"
        "Gate 2 PASS is bound to dataset_manifest.csv and split_assignment.csv. See gate2_report.json and provenance CSVs.\n"
        "Every file is read-only and bound by MANIFEST.csv/FREEZE.json. Never edit this release; corrections require a new version.\n"
        "No training has started at release creation. Outputs/caches must remain outside this dataset.\n\n"
        + "\n".join(f"- {limit}" for limit in SCIENTIFIC_LIMITS) + "\n", encoding="utf-8")
    payload = _inventory(staging, sorted(p for p in staging.rglob("*") if p.is_file()))
    dump_csv(staging / "MANIFEST.csv", payload, ["path", "size", "sha256"])
    _json(staging / "FREEZE.json", dict(
        schema_version=SCHEMA_VERSION, dataset_name=destination.name, created_at_utc=report["created_at_utc"],
        manifest_sha256=digest(staging / "MANIFEST.csv"), source_dataset_sha256=report["source_dataset_sha256"],
        counts=counts, immutable=True, read_only_files=True,
    ))
    # Both absolute move targets are checked siblings; no recursive deletion or
    # replacement of source, historical releases, or old outputs is performed.
    if destination.exists() or staging.resolve().parent != destination.parent:
        raise FileExistsError("Destination appeared or staging moved during finalization")
    staging.rename(destination)
    for path in destination.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    result = verify_snapshot(destination)
    result.update(scene_groups=report["scene_groups"], allocation_blocks=report["allocation_blocks"],
                  gate2_status="PASS", test_modified_after_release=False)
    if release_record is not None:
        target = Path(release_record).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    allocate = commands.add_parser("allocate", help="Balance whole scene/allocation groups without copying data.")
    allocate.add_argument("--source", required=True)
    allocate.add_argument("--groups", required=True)
    allocate.add_argument("--metadata")
    allocate.add_argument("--output", required=True)
    allocate.add_argument("--seed", type=int, default=17)
    allocate.add_argument("--restarts", type=int, default=64)
    prepare = commands.add_parser("prepare", help="Copy a mutable staging dataset; do not publish or freeze.")
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--destination", required=True)
    prepare.add_argument("--groups", required=True)
    prepare.add_argument("--assignment", required=True)
    prepare.add_argument("--allocation-report")
    finalize = commands.add_parser("finalize", help="Publish once after a fresh manifest-bound Gate 2 PASS.")
    finalize.add_argument("--staging", required=True)
    finalize.add_argument("--destination", required=True)
    finalize.add_argument("--gate-report", required=True)
    finalize.add_argument("--release-record")
    finalize.add_argument("--provenance", action="append", default=[], metavar="NAME=PATH")
    args = parser.parse_args(argv)
    if args.command == "allocate":
        members = load_scene_members(args.source, args.groups, metadata_csv=args.metadata)
        rows, report = allocate_scene_groups(members, seed=args.seed, restarts=args.restarts)
        write_allocation(rows, report, args.output)
        print(json.dumps(report, indent=2))
    elif args.command == "prepare":
        staging = prepare_scene_release(args.source, args.destination, args.groups, args.assignment,
                                        allocation_report=args.allocation_report)
        print(json.dumps(dict(staging=str(staging), published=False, frozen=False), indent=2))
    else:
        provenance = {}
        for item in args.provenance:
            if "=" not in item:
                parser.error("--provenance must be NAME=PATH")
            name, path = item.split("=", 1)
            if not name or not path or name in provenance:
                parser.error("--provenance names and paths must be nonempty and names unique")
            provenance[name] = path
        print(json.dumps(finalize_scene_release(
            args.staging, args.destination, args.gate_report,
            provenance_files=provenance, release_record=args.release_record), indent=2))


if __name__ == "__main__":
    main()
