"""Run Clean_v2 with the historical Standard YOLO26s settings for three seeds."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from drone_detection.config import load_experiment
from drone_detection.training import run_experiment

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    "configs/experiments/yolo26s_clean_v2_standard_seed0.yaml",
    "configs/experiments/yolo26s_clean_v2_standard_seed17.yaml",
    "configs/experiments/yolo26s_clean_v2_standard_seed42.yaml",
]
OUTPUT_ROOT = ROOT / "04_experiments" / "clean_v2_standard_protocol"


def main() -> None:
    if (OUTPUT_ROOT / 'EXPLORATORY_STOP_RECORD.json').exists():
        raise RuntimeError('This exploratory campaign was retired by researcher request; use the separate controlled YOLO26 v2 workflow.')
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_run": "04_experiments/YOLO26s_baseline-5",
        "reference_args": "04_experiments/YOLO26s_baseline-5/args.yaml",
        "controlled_changes": {
            "data": "02_datasets/DUT_Anti_UAV_Clean_v2/data.yaml",
            "seed": [0, 17, 42],
        },
        "configs": CONFIGS,
        "runs": [],
    }
    protocol_path = OUTPUT_ROOT / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    for config_path in CONFIGS:
        config = load_experiment(config_path)
        output = config.project / config.name
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite prior run: {output}")
        snapshot = config.validate_files()
        protocol["runs"].append({
            "config": config_path,
            "name": config.name,
            "dataset_snapshot_before": snapshot,
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
        try:
            run_experiment(config)
        except BaseException:
            protocol["runs"][-1]["status"] = "failed"
            protocol["runs"][-1]["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            protocol["status"] = "failed"
            protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
            raise
        protocol["runs"][-1]["status"] = "completed"
        protocol["runs"][-1]["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    protocol["status"] = "completed"
    protocol["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
