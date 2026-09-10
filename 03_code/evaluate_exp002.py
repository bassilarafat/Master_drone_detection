from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "02_datasets" / "DUT_Anti_UAV"
TEST_IMAGES = DATASET_DIR / "images" / "test"
TEST_LABELS = DATASET_DIR / "labels" / "test"
RESULTS_DIR = PROJECT_ROOT / "06_results" / "EXP002_high_resolution"
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "notebooks"
    / "runs"
    / "04_experiments"
    / "EXP002_YOLOv8s_high_resolution"
    / "YOLOv8s_960-3"
    / "weights"
    / "best.pt"
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import calculate_iou


def load_ground_truth(image_name: str) -> list[list[float]]:
    image_path = TEST_IMAGES / f"{image_name}.jpg"
    label_path = TEST_LABELS / f"{image_name}.txt"
    with Image.open(image_path) as image:
        image_width, image_height = image.size

    boxes = []
    for line in label_path.read_text().splitlines():
        _, xc, yc, width, height = map(float, line.split()[:5])
        boxes.append([
            (xc - width / 2) * image_width,
            (yc - height / 2) * image_height,
            (xc + width / 2) * image_width,
            (yc + height / 2) * image_height,
        ])
    return boxes


def evaluate(model_path: Path, image_size: int, device: str, confidence: float) -> Path:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    image_paths = sorted(TEST_IMAGES.glob("*.jpg"))
    predictions = []

    for start in range(0, len(image_paths), 20):
        batch = image_paths[start:start + 20]
        results = model.predict(
            source=batch,
            imgsz=image_size,
            conf=confidence,
            iou=0.7,
            device=device,
            workers=0,
            verbose=False,
        )

        for image_path, result in zip(batch, results):
            for prediction_id, box in enumerate(result.boxes):
                coordinates = box.xyxy[0].cpu().numpy()
                prediction_box = [float(value) for value in coordinates]
                ground_truth = load_ground_truth(image_path.stem)
                best_iou = max(
                    (calculate_iou(prediction_box, target) for target in ground_truth),
                    default=0.0,
                )
                predictions.append({
                    "image": image_path.stem,
                    "pred_id": prediction_id,
                    "x1": prediction_box[0],
                    "y1": prediction_box[1],
                    "x2": prediction_box[2],
                    "y2": prediction_box[3],
                    "confidence": float(box.conf[0]),
                    "class": int(box.cls[0]),
                    "best_iou": best_iou,
                })

        print(f"Processed {min(start + 20, len(image_paths))}/{len(image_paths)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "exp002_test_predictions.csv"
    pd.DataFrame(predictions).to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EXP002 on the test split.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model, args.imgsz, args.device, args.conf)
