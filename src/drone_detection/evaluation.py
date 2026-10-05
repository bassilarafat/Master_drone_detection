"""Export EXP002 predictions and their best ground-truth overlap."""

from __future__ import annotations

import argparse
from pathlib import Path

from .metrics import calculate_iou
from .paths import PROJECT_ROOT, project_path

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
PREDICTION_COLUMNS = [
    "image", "pred_id", "x1", "y1", "x2", "y2", "confidence", "class", "best_iou",
]


def load_ground_truth(
    image_name: str,
    dataset_dir: Path | None = None,
    split: str = "test",
) -> list[list[float]]:
    """Convert normalized YOLO labels to pixel-coordinate bounding boxes."""
    from PIL import Image

    dataset_dir = Path(dataset_dir) if dataset_dir is not None else DATASET_DIR
    image_path = dataset_dir / "images" / split / f"{image_name}.jpg"
    label_path = dataset_dir / "labels" / split / f"{image_name}.txt"
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


def evaluate(
    model_path: Path,
    image_size: int,
    device: str,
    confidence: float,
    *,
    dataset_dir: Path | None = None,
    output_dir: Path | None = None,
    split: str = "test",
) -> Path:
    """Export predictions without changing EXP002's inference or IoU settings."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    dataset_dir = Path(dataset_dir) if dataset_dir is not None else DATASET_DIR
    output_dir = Path(output_dir) if output_dir is not None else RESULTS_DIR
    image_dir = dataset_dir / "images" / split
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No JPG images found: {image_dir}")

    import pandas as pd

    from .models import load_model

    model = load_model(model_path)
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
            ground_truth = None
            for prediction_id, box in enumerate(result.boxes):
                coordinates = box.xyxy[0].cpu().numpy()
                prediction_box = [float(value) for value in coordinates]
                if ground_truth is None:
                    ground_truth = load_ground_truth(image_path.stem, dataset_dir, split)
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

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"exp002_{split}_predictions.csv"
    pd.DataFrame(predictions, columns=PREDICTION_COLUMNS).to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EXP002 on the test split.")
    parser.add_argument("--model", type=project_path, default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--dataset-dir", type=project_path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=project_path, default=RESULTS_DIR)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    evaluate(
        args.model,
        args.imgsz,
        args.device,
        args.conf,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        split=args.split,
    )


if __name__ == "__main__":
    main()
