"""Small command-line boundary around the reusable research code."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from .config import load_experiment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Drone detection research workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Run a YAML experiment configuration.")
    train.add_argument("--config", required=True)
    train.add_argument("--epochs", type=int)
    train.add_argument("--imgsz", type=int)
    train.add_argument("--batch", type=int)
    train.add_argument("--device")
    train.add_argument("--dry-run", action="store_true", help="Validate files and print the plan without loading a model.")
    subparsers.add_parser("environment", help="Print library and GPU information.")
    subparsers.add_parser("paper", add_help=False, help="Readiness audit, controlled paper dry-runs and locked evaluation.")
    dataset = subparsers.add_parser("dataset-normalize", help="Audit and optionally clip YOLO labels.")
    dataset.add_argument("--dataset", required=True)
    dataset.add_argument("--output-dir", required=True)
    dataset.add_argument("--write", action="store_true", help="Write clipped labels after making backups.")
    verify = subparsers.add_parser("dataset-verify", help="Verify every file in a frozen dataset snapshot.")
    verify.add_argument("--dataset", required=True)
    # Delegate evaluation's arguments to its own parser, including --help.
    subparsers.add_parser("evaluate", add_help=False, help="Export EXP002-compatible predictions and best IoU.")
    subparsers.add_parser("heatmap", add_help=False, help="Compare baseline and P2 detection-score heatmaps.")
    subparsers.add_parser("targeted", add_help=False, help="Audit, mine train negatives and run one P2 fine-tune.")
    subparsers.add_parser("fp-review", add_help=False, help="Export YOLO26 FPs and test one simple correction.")
    args, extra = parser.parse_known_args(argv)
    if args.command == "paper":
        from .paper_cli import main as paper_main
        paper_main(extra)
        return
    if args.command == "dataset-verify":
        from pathlib import Path

        from .dataset_snapshot import verify_snapshot

        if extra:
            parser.error(f"unrecognized arguments: {' '.join(extra)}")
        print(json.dumps(verify_snapshot(Path(args.dataset)), indent=2))
        return
    if args.command == "fp-review":
        from .fp_review import main as review_main

        review_main(extra)
        return
    if args.command == "dataset-normalize":
        from .data_audit import normalize_dataset_labels

        print(json.dumps(normalize_dataset_labels(args.dataset, args.output_dir, write=args.write),
                         indent=2))
        return
    if args.command == "targeted":
        from .targeted import main as targeted_main

        targeted_main(extra)
        return
    if args.command == "heatmap":
        from .heatmaps import main as heatmap_main

        heatmap_main(extra)
        return
    if args.command == "evaluate":
        from .evaluation import main as evaluate_main

        evaluate_main(extra)
        return
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    if args.command == "environment":
        from .environment import main as environment_main

        environment_main()
        return
    try:
        config = load_experiment(args.config)
        overrides = {key: getattr(args, key) for key in ("epochs", "imgsz", "batch", "device")
                     if getattr(args, key) is not None}
        config = replace(config, train={**config.train, **overrides})
        snapshot = config.validate_files()
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        record = config.as_dict()
        if snapshot is not None:
            record["dataset_snapshot"] = snapshot
        print(json.dumps(record, indent=2))
        return
    from .training import run_experiment

    run_experiment(config)
