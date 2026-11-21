"""Reproduce the six saved v3 Validation evaluations and paired seed statistics.

Run from the repository with its Python environment. This command reads existing
predictions and trusted project checkpoints on CPU. It performs no inference.
The destination must be new so every audit keeps its own source identities.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_detection.paper_common import bootstrap, pin_release, size_bin
from drone_detection.paper_metrics import evaluate_records
from drone_detection.yolo26_recovery import checkpoint
from drone_detection.dataset_snapshot import _annotations
from drone_detection.yolo26_validation import targets_for

SEEDS = (17, 42, 123)
ARMS = {"E0": "E0_Y26", "E1": "E1_Y26_P2"}
DATASET = ROOT / "02_datasets/DUT_Anti_UAV_Clean_v2"
BASE = ROOT / "04_experiments/controlled_yolo26_v3"
LOCK = ROOT / "configs/paper/YOLO26_LAPTOP_V3_LOCK.json"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def equal(actual, expected, location):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), location
        for key in expected:
            equal(actual[key], expected[key], f"{location}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), location
        for index, value in enumerate(expected):
            equal(actual[index], value, f"{location}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert abs(actual - expected) <= 1e-12, (location, actual, expected)
    else:
        assert actual == expected, (location, actual, expected)


def check_export(saved, reproduced, name):
    assert len(saved) == len(reproduced), (name, "row count")
    for index, (old, new) in enumerate(zip(saved, reproduced)):
        assert old.keys() == new.keys(), name
        for key, value in new.items():
            expected = old[key]
            if isinstance(value, list):
                expected = ast.literal_eval(expected)
            elif isinstance(value, bool):
                expected = expected == "True"
            elif isinstance(value, (int, float)):
                expected = float(expected)
            equal(value, expected, f"{name}[{index}].{key}")


def metric_values(summary):
    values = {key: summary[key] for key in (
        "precision", "recall", "F1", "mAP50", "mAP50_95", "TP", "FP", "FN", "FP_per_image"
    )}
    values.update({size + "_recall": entry["recall"]
                   for size, entry in summary["recall_by_size"].items()})
    values.update({"FP_" + name: value for name, value in summary["fp_subtype_counts"].items()})
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert output.is_relative_to(ROOT / "06_results"), "Use a research-results directory"
    if output.exists():
        raise FileExistsError(output)
    bootstrap()
    lock = read_json(LOCK)
    assert lock["status"] == "SEALED"
    protected = {LOCK, DATASET / "MANIFEST.csv", DATASET / "FREEZE.json",
                 DATASET / "dataset_manifest.csv", Path(__file__).resolve()}
    for name, identity in lock["hashes"].items():
        path = ROOT / name
        assert sha(path) == identity, (name, "sealed source changed")
        protected.add(path)
    release = pin_release(DATASET, lock["dataset_release"])
    manifest = read_csv(DATASET / "dataset_manifest.csv")
    validation = {row["new_filename"]: row for row in manifest if row["new_split"] == "val"}
    train = [row for row in manifest if row["new_split"] == "train"]
    assert len(validation) == 1827 and len(train) == 6299
    for row in validation.values():
        for kind in ("image", "label"):
            path = DATASET / row["new_relative_" + kind]
            assert sha(path) == row[kind + "_sha256"], path
            protected.add(path)
    sources = {}

    def remember(path):
        path = Path(path)
        sources[path.relative_to(ROOT).as_posix()] = sha(path)

    for path in protected:
        remember(path)
    runs, transitions, image_changes, fp_review = [], [], [], []
    summaries, gt_ledgers = {}, {}
    reference_settings = reference_population = None
    for arm, folder in ARMS.items():
        for seed in SEEDS:
            run = BASE / folder / f"seed_{seed}"
            directory = run / "evaluation/val"
            state = read_json(run / "state.json")
            summary = read_json(directory / "summary.json")
            evaluation = read_json(directory / "evaluation.json")
            initialization = read_json(run / "run/initialization.json")
            resolved = read_json(run / "resolved_training.json")
            identity = read_json(run / "session_identity.json")
            cp = run / "run/weights/last.pt"
            assert state["status"] == evaluation["status"] == "COMPLETED"
            assert state["last_completed_epoch"] == 100 and state["summary"] == summary
            assert state["dataset_verified_after"] and evaluation["dataset_verified_after"]
            assert not state["test_evaluated"] and not evaluation["test_evaluated"]
            assert not summary["execution_only"] and not evaluation["execution_only"]
            assert summary["arm"] == arm and summary["seed"] == seed and summary["split"] == "val"
            assert sha(cp) == summary["checkpoint_sha256"] == evaluation["checkpoint_sha256"]
            assert sha(directory / "predictions.jsonl") == evaluation["raw_sha256"]
            assert initialization["seed"] == seed and initialization["loaded_tensors"] == 240
            assert initialization["pretrained_sha256"] == lock["initialization"]["donor_sha256"]
            assert initialization["backbone_tensor_sha256"] == lock["initialization"]["source_backbone_sha256"]
            assert initialization["strides"] == ([8, 16, 32] if arm == "E0" else [4, 8, 16, 32])
            assert identity == state["identity"]
            assert identity["configuration_sha256"] == lock["configuration_sha256"]
            assert identity["protocol_sha256"] == lock["protocol_sha256"]
            assert summary["dataset_release"] == release
            for key, value in lock["exact_training_settings"].items():
                if key != "resume":
                    equal(resolved["settings"][key], value, f"{arm}-{seed}.settings.{key}")
            assert resolved["settings"]["seed"] == seed
            assert resolved["amp"] and resolved["accumulation"] == 4 and resolved["optimizer"] == "MuSGD"
            records = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
            assert len(records) == len(validation) and {r["image"] for r in records} == validation.keys()
            population = []
            for record in records:
                row = validation[record["image"]]
                assert record["split"] == "val"
                assert record["image_sha256"] == row["image_sha256"]
                assert record["label_sha256"] == row["label_sha256"]
                assert record["scene_group_id"] == row["scene_group_id"]
                assert (record["width"], record["height"]) == (int(row["width"]), int(row["height"]))
                labels, issues = _annotations(DATASET / row["new_relative_label"])
                assert not issues, (record["image"], issues)
                expected_targets = targets_for(labels, record["width"], record["height"])
                equal(record["targets"], expected_targets, f"{arm}-{seed}.{record['image']}.targets")
                population.append((record["image"], record["image_sha256"], record["label_sha256"],
                                   record["targets"], record["scene_group_id"]))
            if reference_population is None:
                reference_population, reference_settings = population, evaluation["settings"]
            assert population == reference_population and evaluation["settings"] == reference_settings
            replay, predictions, objects, images = evaluate_records(
                records, confidence=reference_settings["operating_conf"], matching_iou=reference_settings["matching_iou"])
            for key, value in replay.items():
                equal(summary[key], value, f"{arm}-{seed}.summary.{key}")
            for name, generated in [("predictions.csv", predictions), ("objects.csv", objects), ("images.csv", images)]:
                check_export(read_csv(directory / name), generated, f"{arm}-{seed}.{name}")
            check_export(read_csv(directory / "recall_by_size.csv"),
                         [dict(size=s, **v) for s, v in replay["recall_by_size"].items()], "recall_by_size")
            check_export(read_csv(directory / "fp_subtypes.csv"),
                         [dict(fp_subtype=s, count=n) for s, n in replay["fp_subtype_counts"].items()], "fp_subtypes")
            saved_cp = checkpoint(cp, identity)
            assert saved_cp["epoch"] == 99
            steps = saved_cp["controlled_optimizer_steps"]
            events = [read_json(f) for f in sorted((run / "events").glob("*.json"))]
            completed = [e for e in events if e["event"] == "checkpoint_committed"]
            assert [e["completed_epoch"] for e in completed] == list(range(1, 101))
            assert completed[-1]["checkpoint_sha256"] == summary["checkpoint_sha256"]
            history = read_csv(run / "run/results.csv")
            assert [int(r["epoch"]) for r in history] == list(range(1, 101))
            native = {key: float(history[-1]["metrics/" + key + "(B)"])
                      for key in ("precision", "recall", "mAP50", "mAP50-95")}
            runs.append(dict(arm=arm, seed=seed, **metric_values(summary),
                             checkpoint_sha256=summary["checkpoint_sha256"], epoch=100,
                             optimizer_steps=steps, resumes=sum(e["event"] == "resume" for e in events),
                             start_utc=events[0]["at_utc"], end_utc=completed[-1]["at_utc"],
                             evaluation_end_utc=evaluation["finished_at_utc"],
                             **{"native_" + k: v for k, v in native.items()},
                             source=(directory / "summary.json").relative_to(ROOT).as_posix()))
            del saved_cp
            summaries[arm, seed] = summary
            gt_ledgers[arm, seed] = {(r["image"], r["gt_id"]): r for r in objects if r["kind"] == "GT"}
            for pred in predictions:
                if pred["status"] == "FP":
                    row = validation[pred["image"]]
                    fp_review.append(dict(arm=arm, seed=seed, **pred,
                                          validation_image=row["new_relative_image"],
                                          semantic_label="UNREVIEWED", reviewer="", review_note=""))
            for path in [cp, *directory.glob("*"), run / "state.json", run / "session_identity.json",
                         run / "run/initialization.json", run / "resolved_training.json", run / "run/results.csv",
                         *run.glob("events/*.json")]:
                if path.is_file():
                    remember(path)
            print(f"Reproduced {arm} seed {seed}: {summary['TP']}/{summary['FP']}/{summary['FN']}, steps={steps}", flush=True)
    paired, aggregates, subtype_rows = [], [], []
    for seed in SEEDS:
        a, b = summaries["E0", seed], summaries["E1", seed]
        x, y = metric_values(a), metric_values(b)
        paired.append(dict(seed=seed, **{k: y[k]-x[k] for k in x}))
        for name in a["fp_subtype_counts"]:
            subtype_rows.append(dict(seed=seed, subtype=name, E0=a["fp_subtype_counts"][name],
                                     E1=b["fp_subtype_counts"][name],
                                     delta=b["fp_subtype_counts"][name]-a["fp_subtype_counts"][name]))
        left, right = gt_ledgers["E0", seed], gt_ledgers["E1", seed]
        assert left.keys() == right.keys()
        for key, target in left.items():
            other = right[key]
            assert (target["box"], target["size"]) == (other["box"], other["size"])
            change = "unchanged"
            if target["status"] == "FN" and other["status"] == "TP":
                change = "recovered"
            elif target["status"] == "TP" and other["status"] == "FN":
                change = "lost"
            transitions.append(dict(seed=seed, image=key[0], gt_id=key[1], size=target["size"],
                                    E0=target["status"], E1=other["status"], change=change))
        im0 = {r["image"]: r for r in read_csv(BASE / ARMS["E0"] / f"seed_{seed}/evaluation/val/images.csv")}
        im1 = {r["image"]: r for r in read_csv(BASE / ARMS["E1"] / f"seed_{seed}/evaluation/val/images.csv")}
        for image in im0:
            image_changes.append(dict(seed=seed, image=image, **{
                f"{arm}_{key}": int(data[image][key])
                for arm, data in [("E0", im0), ("E1", im1)] for key in ("TP", "FP", "FN")},
                delta_FP=int(im1[image]["FP"])-int(im0[image]["FP"])))
    for key in metric_values(summaries["E0", 17]):
        a = [metric_values(summaries["E0", s])[key] for s in SEEDS]
        b = [metric_values(summaries["E1", s])[key] for s in SEEDS]
        diff = [y-x for x, y in zip(a, b)]
        unit = "count" if key in ("TP", "FP", "FN") or key.startswith("FP_") and key != "FP_per_image" else "fraction"
        aggregates.append(dict(metric=key, stored_unit=unit, E0_mean=statistics.mean(a),
                               E0_sample_sd=statistics.stdev(a), E1_mean=statistics.mean(b),
                               E1_sample_sd=statistics.stdev(b), delta_mean=statistics.mean(diff),
                               delta_sample_sd=statistics.stdev(diff), n=3, ddof=1))
    train_groups = Counter(r["scene_group_id"] for r in train)
    blocks = Counter(r["allocation_block_id"] for r in train)
    for name, expected in sources.items():
        assert sha(ROOT / name) == expected, (name, "source changed during audit")
    output.mkdir(parents=True)
    for name, data in [("per_run.csv", runs), ("paired_differences.csv", paired),
                       ("aggregate_statistics.csv", aggregates), ("object_transitions.csv", transitions),
                       ("fp_subtype_changes.csv", subtype_rows), ("per_image_changes.csv", image_changes),
                       ("fp_review.csv", fp_review)]:
        save_csv(output / name, data)
    report = dict(status="VERIFIED", audited_at_utc=datetime.now(timezone.utc).isoformat(),
                  scope="Six saved v3 Validation exports, fixed last.pt epoch100",
                  evaluation_images=1827, evaluation_objects=1867, seeds=list(SEEDS),
                  numerical_tolerance=1e-12, evaluator_replay="drone_detection.paper_metrics.evaluate_records",
                  data_integrity_scope="Validation image/label bytes and release manifest/freeze identities",
                  test_inference=False, detector_inference=False, checkpoint_loading="Trusted local last.pt, CPU",
                  same_validation_targets_and_settings=True, all_source_hashes_unchanged=True,
                  units="Metric CSVs retain fractions. Multiply rates by 100 for percent or pp. Count deltas remain counts.",
                  train=dict(images=len(train), scene_groups=len(train_groups), allocation_blocks=len(blocks),
                             block_sizes=dict(blocks)),
                  runs=runs, aggregates=aggregates, source_sha256=sources)
    (output / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"Verified audit saved to {output}")


if __name__ == "__main__":
    main()
