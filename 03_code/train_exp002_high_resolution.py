from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = PROJECT_ROOT / "02_datasets" / "DUT_Anti_UAV" / "data.yaml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "05_models" / "pretrained" / "yolov8s.pt"
EXPERIMENT_DIR = PROJECT_ROOT / "04_experiments" / "EXP002_YOLOv8s_high_resolution"


def train(weights: Path, epochs: int, image_size: int, batch_size: int, device: str) -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        project=str(EXPERIMENT_DIR),
        name="YOLOv8s_960",
        pretrained=True,
        patience=20,
        workers=4,
    )
    print(f"Training complete. Results: {EXPERIMENT_DIR / 'YOLOv8s_960'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EXP002 at high resolution.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.weights, args.epochs, args.imgsz, args.batch, args.device)
