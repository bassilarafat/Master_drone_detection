"""Prepare Train-only buffered mining designs from saved scene evidence.

This diagnostic reads the frozen release and writes separate role manifests.
It never trains a model, runs detector inference or publishes verified crops.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_detection.paper_common import pin_release, read_csv, write_csv
from drone_detection.paper_negatives import safe, source_targets

DATASET = ROOT / "02_datasets/DUT_Anti_UAV_Clean_v2"
FOLDS = "ABCD"
BASE_FIELDS = ["image", "source_relative_image", "source_relative_label", "image_sha256",
               "label_sha256", "scene_group_id", "allocation_block_id", "width", "height"]


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def candidate_geometry(targets, width, height):
    """Find a geometrically GT-free 256-square without changing image bytes."""
    if min(width, height) < 256:
        return None
    xs, ys = {0, width - 256}, {0, height - 256}
    for target in targets:
        x1, y1, x2, y2 = target["box"]
        xs.update((math.floor(x1 - 16 - 256), math.ceil(x2 + 16)))
        ys.update((math.floor(y1 - 16 - 256), math.ceil(y2 + 16)))
    for x in sorted(value for value in xs if 0 <= value <= width - 256):
        for y in sorted(value for value in ys if 0 <= value <= height - 256):
            bounds = [x, y, x + 256, y + 256]
            if safe(bounds, targets, width, height):
                return bounds
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "06_results"):
        raise ValueError("Use a separate results directory")
    if output.exists():
        raise FileExistsError(output)
    release = pin_release(DATASET)
    original_folds = ROOT / "06_results/controlled_yolo26_v3_three_seed_audit_20261120/proposed_train_folds.csv"
    source_paths = [DATASET / name for name in (
        "MANIFEST.csv", "FREEZE.json", "dataset_manifest.csv", "scene_pair_candidates.csv",
        "allocation_identity_constraints.csv", "background_feature_index.csv", "scene_grouping_policy.json"
    )] + [original_folds, Path(__file__).resolve(), ROOT / "src/drone_detection/paper_negatives.py"]
    identities = {path.relative_to(ROOT).as_posix(): digest(path) for path in source_paths}
    train = {row["image"]: row for row in read_csv(DATASET / "dataset_manifest.csv") if row["new_split"] == "train"}
    assert len(train) == 6299
    groups = defaultdict(list)
    rows = {}
    for image, record in train.items():
        groups[record["scene_group_id"]].append(image)
        rows[image] = dict(image=image, source_relative_image=record["new_relative_image"],
                           source_relative_label=record["new_relative_label"],
                           **{key: record[key] for key in BASE_FIELDS[3:]})
    assert len(groups) == 1064
    assert len({r["allocation_block_id"] for r in train.values()}) == 1
    indexes = {name: index for index, name in enumerate(sorted(train))}
    direct_pairs = set()
    source_counts = Counter()
    pair_group_rows = Counter()
    confirmed_crossings = 0

    def include(a, b):
        if a not in train or b not in train:
            return
        i, j = sorted((indexes[a], indexes[b]))
        direct_pairs.add((i, j))

    import csv
    with (DATASET / "allocation_identity_constraints.csv").open(encoding="utf-8-sig", newline="") as handle:
        for edge in csv.DictReader(handle):
            a, b = edge["image_a"], edge["image_b"]
            if a in train and b in train:
                source_counts["explicit_train_constraint_rows"] += 1
                include(a, b)
    with (DATASET / "scene_pair_candidates.csv").open(encoding="utf-8-sig", newline="") as handle:
        for edge in csv.DictReader(handle):
            a, b = edge["image_a"], edge["image_b"]
            if a not in train or b not in train:
                continue
            decision = edge["review_decision"]
            source_counts["candidate_" + decision] += 1
            if decision == "UNCERTAIN_COLOCATE":
                include(a, b)
            if decision in ("CONFIRMED_SAME_SCENE", "CONFIRMED_PATH"):
                confirmed_crossings += train[a]["scene_group_id"] != train[b]["scene_group_id"]
    assert confirmed_crossings == 0
    low_texture = [row["image_id"] for row in read_csv(DATASET / "background_feature_index.csv")
                   if row["image_id"] in train and int(row["keypoints"]) < 16]
    for a, b in combinations(low_texture, 2):
        include(a, b)
    neighbors = {group: set() for group in groups}
    byindex = {index: name for name, index in indexes.items()}
    for a, b in direct_pairs:
        ga, gb = train[byindex[a]]["scene_group_id"], train[byindex[b]]["scene_group_id"]
        if ga != gb:
            neighbors[ga].add(gb)
            neighbors[gb].add(ga)
            pair_group_rows[tuple(sorted((ga, gb)))] += 1
    # Keep the previous balanced proposal as an exact reproducible baseline.
    balanced = {row["scene_group"]: row["proposed_fold"] for row in read_csv(original_folds)}
    assert set(balanced) == set(groups)
    # This second data-only proposal keeps graph communities together. It is not
    # selected using detector metrics or negative-mining yield.
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(sorted(groups))
    for (a, b), weight in sorted(pair_group_rows.items()):
        graph.add_edge(a, b, weight=weight)
    communities = nx.community.louvain_communities(graph, weight="weight", resolution=1, seed=17)
    community_folds, loads = {}, [0] * 4
    sorted_communities = sorted(communities, key=lambda c: (-sum(len(groups[g]) for g in c), min(c)))
    for community in sorted_communities:
        index = min(range(4), key=lambda i: (loads[i], i))
        for group in community:
            community_folds[group] = FOLDS[index]
        loads[index] += sum(len(groups[g]) for g in community)
    # Geometry feasibility uses the existing crop verifier's exact safety rule.
    eligibility = []
    for name, row in sorted(train.items()):
        label = DATASET / row["new_relative_label"]
        assert digest(label) == row["label_sha256"], name
        targets = source_targets(DATASET, row)
        bounds = candidate_geometry(targets, int(row["width"]), int(row["height"]))
        eligibility.append(dict(**rows[name], eligible_by_geometry=bounds is not None,
                                example_xyxy=json.dumps(bounds), verification="UNREVIEWED",
                                evidence="Geometric existence only; UAV-free status unverified"))
    eligible = {row["image"] for row in eligibility if row["eligible_by_geometry"]}
    output.mkdir(parents=True)
    write_csv(output / "source_eligibility.csv", eligibility, list(eligibility[0]))
    neighbor_rows = [dict(scene_group_a=a, scene_group_b=b, distinct_risk_image_pairs=weight,
                          provenance="Explicit identity OR uncertain candidate OR low-texture policy")
                     for (a, b), weight in sorted(pair_group_rows.items())]
    write_csv(output / "direct_risk_group_edges.csv", neighbor_rows, list(neighbor_rows[0]))
    summary_rows, manifests = [], {}
    for strategy, assignment in [("balanced_scene_groups", balanced), ("graph_communities", community_folds)]:
        target = output / strategy
        target.mkdir()
        for fold in FOLDS:
            held = {group for group in groups if assignment[group] == fold}
            buffer = set().union(*(neighbors[group] for group in held)) - held if held else set()
            retained = set(groups) - held - buffer
            # Check absence of every direct risk edge across mine/train groups.
            direct_crossings = sum(b in retained for a in held for b in neighbors[a])
            assert direct_crossings == 0
            role_rows = []
            for name, row in sorted(train.items()):
                group = row["scene_group_id"]
                role = "mining" if group in held else "buffer_excluded" if group in buffer else "discovery_train"
                role_rows.append(dict(**rows[name], proposed_fold=fold, role=role,
                                      geometric_crop_possible=name in eligible,
                                      status="DIAGNOSTIC_PROPOSAL_ONLY"))
            role_path = target / f"fold_{fold}_roles.csv"
            write_csv(role_path, role_rows, list(role_rows[0]))
            counts = Counter(row["role"] for row in role_rows)
            for role in ("mining", "buffer_excluded", "discovery_train"):
                assert counts[role] == sum(len(groups[g]) for g in
                                           (held if role == "mining" else buffer if role == "buffer_excluded" else retained))
                list_path = target / f"fold_{fold}_{role}.txt"
                selected = [row["source_relative_image"] for row in role_rows if row["role"] == role]
                list_path.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
                assert all(name.startswith("images/train/") for name in selected)
            summary = dict(strategy=strategy, fold=fold, mining_images=counts["mining"],
                           excluded_buffer_images=counts["buffer_excluded"],
                           retained_training_images=counts["discovery_train"],
                           mining_scene_groups=len(held), retained_training_scene_groups=len(retained),
                           eligible_mining_sources=sum(row["role"] == "mining" and row["geometric_crop_possible"] for row in role_rows),
                           retained_training_objects=sum(int(train[row["image"]]["objects"]) for row in role_rows if row["role"] == "discovery_train"),
                           direct_risk_crossings=direct_crossings,
                           full_allocation_block_disjoint=False,
                           manifest_sha256=digest(role_path))
            summary_rows.append(summary)
            manifests[role_path.relative_to(ROOT).as_posix()] = digest(role_path)
    write_csv(output / "fold_comparison.csv", summary_rows, list(summary_rows[0]))
    for name, before in identities.items():
        assert digest(ROOT / name) == before, (name, "source changed")
    assert pin_release(DATASET) == release
    report = dict(status="DIAGNOSTIC_DESIGN_READY_FOR_REVIEW", generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  dataset_release=release, train_images=len(train), train_scene_groups=len(groups),
                  train_allocation_blocks=1, strict_block_disjoint_retained_training_images=0,
                  low_texture_images=len(low_texture), distinct_direct_risk_image_pairs=len(direct_pairs),
                  distinct_direct_risk_group_edges=len(pair_group_rows), source_row_counts=dict(source_counts),
                  graph_communities=len(communities), networkx_version=nx.__version__, graph_seed=17,
                  graph_resolution=1, geometry_eligible_sources=len(eligible), crop_size=[256,256], margin_px=16,
                  verified_negative_count=None, detector_inference=False, training_performed=False,
                  protection="Hold out complete scene groups and remove all directly linked risk-neighbor groups, including the low-texture family",
                  limitation="Indirect allocation paths and unresolved recording identities remain. This is narrower than full allocation-block independence.",
                  strategies=summary_rows, source_sha256=identities, role_manifest_sha256=manifests)
    (output / "design_audit.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("source_sha256", "role_manifest_sha256")}, indent=2))


if __name__ == "__main__":
    main()
