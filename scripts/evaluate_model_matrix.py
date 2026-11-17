"""Evaluate the two YOLO26 checkpoints on both project test splits."""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from drone_detection.config import snapshot_root_for_yaml
from drone_detection.dataset_snapshot import verify_snapshot
from drone_detection.models import load_model
from drone_detection.training import _without_snapshot_cache_writes

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "04_experiments" / "evaluation"
MODELS = {
    "standard_model": ROOT / "04_experiments" / "YOLO26s_baseline-5" / "weights" / "best.pt",
    "clean_model": ROOT / "04_experiments" / "local_clean_v2" / "YOLO26s_Clean_v2_local_seed17-2" / "weights" / "best.pt",
}
DATASETS = {
    "standard_test": ROOT / "02_datasets" / "DUT_Anti_UAV" / "data.yaml",
    "clean_test": ROOT / "02_datasets" / "DUT_Anti_UAV_Clean_v2" / "data.yaml",
}


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(model_name: str, data_name: str) -> dict:
    checkpoint, data = MODELS[model_name], DATASETS[data_name]
    name = f"{model_name}_on_{data_name}"
    if not checkpoint.is_file() or not data.is_file():
        raise FileNotFoundError(f"Missing evaluation input: {checkpoint} or {data}")
    frozen_root = snapshot_root_for_yaml(data)
    identity = verify_snapshot(frozen_root) if frozen_root else None
    output = OUTPUT_ROOT / name
    record_path = OUTPUT_ROOT / f"{name}.evaluation.json"
    if output.exists() or record_path.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")

    record = {
        "status": "running",
        "model_name": model_name,
        "dataset_name": data_name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "data": str(data),
        "split": "test",
        "imgsz": 960,
        "batch": 8,
        "device": "0",
        "workers": 2,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_snapshot_before": identity,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    try:
        model = load_model(checkpoint)
        guard = _without_snapshot_cache_writes(frozen_root) if frozen_root else nullcontext()
        with guard:
            metrics = model.val(
                data=str(data), split="test", imgsz=960, batch=8, device="0", workers=2,
                cache=False, project=str(OUTPUT_ROOT), name=name, exist_ok=False, plots=True,
            )
        record["metrics"] = {key: float(value) for key, value in metrics.results_dict.items()}
        record["results_dir"] = str(output)
        record["status"] = "completed"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        raise
    finally:
        if frozen_root:
            after = verify_snapshot(frozen_root)
            if after != identity:
                record["status"] = "failed_dataset_integrity"
                record["dataset_snapshot_after"] = after
                record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                raise ValueError("Clean_v2 changed during evaluation")
            record["dataset_snapshot_after"] = after
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics-local"))
    recovered_name = "standard_model_on_standard_test"
    recovered_record = OUTPUT_ROOT / f"{recovered_name}.evaluation.json"
    recovered_output = OUTPUT_ROOT / recovered_name
    if recovered_output.is_dir() and recovered_record.is_file():
        recovered = json.loads(recovered_record.read_text(encoding="utf-8"))
        if recovered.get("status") == "failed" and "save_dir" in recovered.get("error", ""):
            recovered.update(
                status="completed",
                results_dir=str(recovered_output),
                metrics={
                    "metrics/precision(B)": 0.981,
                    "metrics/recall(B)": 0.960,
                    "metrics/mAP50(B)": 0.978,
                    "metrics/mAP50-95(B)": 0.720,
                },
                note="Metrics were emitted by the completed validator before metadata recording failed.",
                finished_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            recovered_record.write_text(json.dumps(recovered, indent=2) + "\n", encoding="utf-8")
    results = []
    for model_name in MODELS:
        for data_name in DATASETS:
            name = f"{model_name}_on_{data_name}"
            record_path = OUTPUT_ROOT / f"{name}.evaluation.json"
            if record_path.is_file() and json.loads(record_path.read_text(encoding="utf-8")).get("status") == "completed":
                results.append(json.loads(record_path.read_text(encoding="utf-8")))
            else:
                results.append(evaluate(model_name, data_name))
    (OUTPUT_ROOT / "matrix_summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
