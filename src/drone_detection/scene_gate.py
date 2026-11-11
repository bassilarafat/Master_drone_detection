"""Audit a mutable scene-release staging directory before its one-time freeze."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .data_audit import digest
from .scene_graph import read_csv
from .scene_release import _validate_staged_bytes
from .scene_review import RISK_POLICY, risk_reason


def boundary_candidates(candidates, assignment_csv, output):
    """Keep every retrieved pair crossing the NEW split, not the historical one."""
    assignment = {r["image"]: r for r in read_csv(assignment_csv)}
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    counts = Counter()
    total = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(candidates).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames + ["new_split_a", "new_split_b", "new_path_a", "new_path_b"]
        with output.open("w", newline="", encoding="utf-8") as dest:
            writer = csv.DictWriter(dest, fieldnames=fields)
            writer.writeheader()
            for pair in reader:
                a, b = assignment[pair["image_a"]], assignment[pair["image_b"]]
                if a["new_split"] == b["new_split"]:
                    continue
                pair.update(new_split_a=a["new_split"], new_split_b=b["new_split"],
                            new_path_a=a["new_relative_image"], new_path_b=b["new_relative_image"])
                writer.writerow(pair)
                counts["/".join(sorted((a["new_split"], b["new_split"])))] += 1
                total += 1
    result = dict(boundary_pairs=total, by_boundary=dict(counts),
                  assignment_sha256=digest(assignment_csv), global_candidates_sha256=digest(candidates),
                  boundary_candidates_sha256=digest(output))
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def audit_staging(staging, geometry_csv, output, *, expected_boundary_csv, manual_json=None,
                  source_review_csv=None, retrieval_report=None, feature_index=None):
    """A PASS requires explicit zero violations and complete new-boundary coverage."""
    staging, output = Path(staging), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if output.resolve() == staging.resolve() or staging.resolve() in output.resolve().parents:
        raise ValueError("Write the fresh audit outside staging before sealing it")
    rows, counts = _validate_staged_bytes(staging)
    assignment = {row["image"]: row for row in rows}
    scenes, blocks, hashes = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in rows:
        scenes[row["scene_group_id"]].add(row["new_split"])
        blocks[row.get("allocation_block_id") or row["scene_group_id"]].add(row["new_split"])
        hashes[row["image_sha256"]].add(row["new_split"])
    manual = {}
    if manual_json:
        for row in json.loads(Path(manual_json).read_text(encoding="utf-8")):
            manual[tuple(sorted((row["image_a"], row["image_b"])))]=row
    with Path(expected_boundary_csv).open(newline="", encoding="utf-8") as handle:
        expected = {tuple(sorted((r["image_a"], r["image_b"]))) for r in csv.DictReader(handle)}
    meta = json.loads(Path(geometry_csv).with_suffix(Path(geometry_csv).suffix + ".metadata.json").read_text(encoding="utf-8"))
    if meta.get("candidates_sha256") != digest(expected_boundary_csv):
        raise ValueError("Fresh geometry is not bound to the current boundary candidate file")
    if meta.get("complete") is not True and meta.get("completed_pairs") != len(expected):
        raise ValueError("Fresh geometric verification has not completed")
    feature_unverified = 1
    if feature_index is not None:
        indexed = {row["image_id"]: row for row in read_csv(feature_index)}
        if set(indexed) != set(assignment) or any(indexed[key]["sha256"] != row["image_sha256"] for key,row in assignment.items()):
            raise ValueError("Background feature identity differs from staged image bytes")
        if meta.get("feature_index_sha256") != digest(feature_index):
            raise ValueError("Fresh geometry refers to a different background feature index")
        feature_unverified = 0
    seen = set()
    confirmed = Counter()
    risky = []
    actual_confirmed = []
    with Path(geometry_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = tuple(sorted((row["image_a"], row["image_b"])))
            if key in seen:
                raise ValueError("Duplicate pair in fresh boundary audit")
            seen.add(key)
            a, b = assignment[key[0]], assignment[key[1]]
            if a["new_split"] == b["new_split"]:
                raise ValueError("Fresh boundary file contains a same-split pair")
            if row.get("direct_geometry_evaluated", "True").lower() != "true":
                raise ValueError("Fresh cross-split verification cannot skip a pair through old confirmed paths")
            override = manual.get(key, {}).get("decision")
            if override == "DIFFERENT_SCENE":
                continue
            boundary = "_".join(sorted((a["new_split"], b["new_split"]), key=("train", "val", "test").index))
            if row["geometry_decision"] == "CONFIRMED_GEOMETRY" or override == "CONFIRMED_SAME_SCENE":
                confirmed[boundary] += 1
                actual_confirmed.append(row)
            elif override == "UNCERTAIN_COLOCATE" or risk_reason(row):
                risky.append(dict(row, risk_reason=risk_reason(row) or "explicit_visual_identity_uncertainty"))
    if expected != seen:
        raise ValueError(f"Fresh audit coverage differs from new boundary candidates: missing={len(expected-seen)}, added={len(seen-expected)}")
    explicit_review_crossings = 0
    if source_review_csv:
        with Path(source_review_csv).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["review_decision"] not in {"CONFIRMED_SAME_SCENE", "UNCERTAIN_COLOCATE"}:
                    continue
                if assignment[row["image_a"]]["new_split"] != assignment[row["image_b"]]["new_split"]:
                    explicit_review_crossings += 1
    checks = dict(
        exact_cross_split_sha256=sum(len(v)>1 for v in hashes.values()),
        scene_group_cross_split=sum(len(v)>1 for v in scenes.values()),
        allocation_block_cross_split=sum(len(v)>1 for v in blocks.values()),
        confirmed_same_scene_train_val=confirmed["train_val"],
        confirmed_same_scene_train_test=confirmed["train_test"],
        confirmed_same_scene_val_test=confirmed["val_test"], invalid_annotations=0,
        unresolved_high_risk_cross_split=len(risky), source_review_constraints_cross_split=explicit_review_crossings,
        missing_boundary_pair_evaluations=len(expected-seen),
        feature_identity_unverified=feature_unverified,
    )
    status = "PASS" if all(value == 0 for value in checks.values()) else "FAIL"
    report = dict(
        status=status, created_at_utc=datetime.now(timezone.utc).isoformat(), checks=checks,
        dataset_manifest_sha256=digest(staging / "dataset_manifest.csv"),
        split_assignment_sha256=digest(staging / "split_assignment.csv"),
        fresh_boundary_candidates_sha256=digest(expected_boundary_csv),
        fresh_geometry_sha256=digest(geometry_csv), images=len(rows), counts=counts,
        scene_groups=len(scenes), allocation_blocks=len(blocks), new_boundary_pairs_checked=len(seen),
        retrieval_report_sha256=digest(retrieval_report) if retrieval_report else None,
        feature_index_sha256=digest(feature_index) if feature_index else None,
        cached_features_policy="Only byte-identical, SHA-bound image/background features may be reused; global retrieval and new-boundary geometric decisions are computed anew.",
        risk_policy=RISK_POLICY, training_performed=False, sequence_disjoint_claimed=False,
        test_release_policy="Publish Test once after PASS; hash and mark all release files read-only; later corrections require a new version.",
    )
    output.mkdir(parents=True)
    (output / "gate2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "new_confirmed_crossings.json").write_text(json.dumps(actual_confirmed, indent=2), encoding="utf-8")
    (output / "unresolved_high_risk_crossings.json").write_text(json.dumps(risky, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    boundary = sub.add_parser("boundary")
    for name in ("candidates", "assignment", "output"):
        boundary.add_argument("--" + name, required=True)
    audit = sub.add_parser("audit")
    for name in ("staging", "geometry", "boundary", "output"):
        audit.add_argument("--" + name, required=True)
    audit.add_argument("--manual")
    audit.add_argument("--source-review")
    audit.add_argument("--retrieval-report")
    audit.add_argument("--feature-index", required=True)
    args = parser.parse_args(argv)
    if args.command == "boundary":
        boundary_candidates(args.candidates, args.assignment, args.output)
    else:
        audit_staging(args.staging, args.geometry, args.output, expected_boundary_csv=args.boundary,
                      manual_json=args.manual, source_review_csv=args.source_review, retrieval_report=args.retrieval_report,
                      feature_index=args.feature_index)


if __name__ == "__main__":
    main()
