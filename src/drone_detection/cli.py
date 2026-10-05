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
    # Delegate evaluation's arguments to its own parser, including --help.
    subparsers.add_parser("evaluate", add_help=False, help="Export EXP002-compatible predictions and best IoU.")
    subparsers.add_parser("heatmap", add_help=False, help="Compare baseline and P2 detection-score heatmaps.")
    args, extra = parser.parse_known_args(argv)
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
        config.validate_files()
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps(config.as_dict(), indent=2))
        return
    from .training import run_experiment

    run_experiment(config)
