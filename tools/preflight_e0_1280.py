"""No-training memory and loss preflight for the prospective E0@1280 control."""

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "configs/paper/yolo26s-paper.yaml"


def one_backward(batch_size: int) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(MODEL)).model.to("cuda:0")
    model.args = get_cfg(overrides={"box": 7.5, "cls": 0.5, "dfl": 1.5, "amp": True})
    model.train()

    images = torch.randn(batch_size, 3, 1280, 1280, device="cuda:0")
    batch = {
        "img": images,
        "cls": torch.zeros((batch_size, 1), device="cuda:0"),
        "bboxes": torch.rand((batch_size, 4), device="cuda:0"),
        "batch_idx": torch.arange(batch_size, device="cuda:0"),
    }
    predictions = model(images)
    loss, _ = model.loss(batch, predictions)
    loss_sum = loss.sum()
    loss_sum.backward()

    report = {
        "batch": batch_size,
        "loss_finite": bool(torch.isfinite(loss_sum).item()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    del loss_sum, loss, predictions, batch, images, model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for this preflight")

    props = torch.cuda.get_device_properties(0)
    candidates = [one_backward(batch) for batch in (4, 3, 2)]
    batch4 = candidates[0]
    margin_bytes = props.total_memory - batch4["peak_allocated_bytes"]
    result = {
        "status": "PASS" if batch4["loss_finite"] else "FAIL",
        "experiment_id": "E0_1280_resolution_control",
        "protocol": "configs/paper/E0_1280_RESOLUTION_CONTROL_V1.json",
        "training_called": False,
        "real_dataset_inference": False,
        "test_inference_performed": False,
        "gpu": props.name,
        "gpu_vram_bytes": props.total_memory,
        "imgsz": 1280,
        "batch_candidates": candidates,
        "batch4_margin_bytes": margin_bytes,
        "batch4_operational_margin": "narrow" if margin_bytes < 512 * 1024 * 1024 else "adequate",
        "recommendation": "Batch 4 completed the synthetic loss backward, but memory margin is narrow; no training launch until a prospective lock and real optimizer-state preflight are completed.",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
