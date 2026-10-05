"""Compare trained detectors using per-scale, per-class detection score maps.

These maps locate classification responses on the detection grids. They are not
gradient attribution or segmentation masks. Values remain on a shared 0..1
scale; independently normalizing each model would hide confidence differences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from .paths import PROJECT_ROOT, project_path

DEFAULT_BASELINE = PROJECT_ROOT / "04_experiments/YOLO26s_baseline-5/weights/best.pt"
DEFAULT_P2 = PROJECT_ROOT / "04_experiments/YOLO26s_P2_transfer_learning/weights/best.pt"
DEFAULT_SOURCE = PROJECT_ROOT / "02_datasets/DUT_Anti_UAV/images/val"


def restore_heatmap(grid: Any, original_shape: tuple[int, int], input_shape: tuple[int, int]) -> Any:
    """Resize a score grid, remove centered letterbox padding, restore image size."""
    import cv2
    import numpy as np

    height, width = original_shape
    input_h, input_w = input_shape
    if min(height, width, input_h, input_w) <= 0:
        raise ValueError("Image dimensions must be positive")
    gain = min(input_h / height, input_w / width)
    scaled_w, scaled_h = round(width * gain), round(height * gain)
    left = round((input_w - scaled_w) / 2 - 0.1)
    top = round((input_h - scaled_h) / 2 - 0.1)
    padded = cv2.resize(np.asarray(grid, dtype=np.float32), (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    cropped = padded[top:top + scaled_h, left:left + scaled_w]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR).clip(0, 1)


class DetectorHeatmaps:
    """Collect active classifier outputs on the actual fused prediction model."""

    def __init__(self, checkpoint: Path, *, class_id: int = 0) -> None:
        from .models import load_model

        checkpoint = project_path(checkpoint)
        if checkpoint.suffix.lower() != ".pt":
            raise ValueError("Heatmap comparisons require trained .pt checkpoints, not architecture YAMLs")
        self.model = load_model(checkpoint)
        if self.model.task != "detect":
            raise ValueError("Heatmaps currently support YOLO detection checkpoints only")
        if class_id not in self.model.names:
            raise ValueError(f"Class {class_id} is not present in {checkpoint}")
        self.class_id = class_id
        self.class_name = self.model.names[class_id]
        self.strides = [int(value) for value in self.model.model.model[-1].stride.tolist()]
        self.raw_maps: dict[str, Any] = {}
        self.handles: list[Any] = []
        self.branch = ""
        self.layers: list[int] = []
        self.callback = self._attach
        self.model.add_callback("on_predict_start", self.callback)

    def _remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _attach(self, predictor: Any) -> None:
        # Ultralytics deep-copies and fuses the model during setup. Hooks must be
        # attached here, after setup, to the model that actually makes predictions.
        self._remove_hooks()
        self.raw_maps.clear()
        head = predictor.model.model.model[-1]
        self.branch = "one2one" if head.end2end else "one2many"
        classifiers = head.one2one_cv3 if head.end2end else head.cv3
        self.strides = [int(value) for value in head.stride.tolist()]
        self.layers = list(head.f)
        if classifiers is None or len(classifiers) != len(self.strides):
            raise RuntimeError("Unsupported detection head: cannot find active per-scale classifiers")
        for classifier, stride in zip(classifiers, self.strides):
            scale = f"P{int(math.log2(stride))}/{stride}"

            def capture(module: Any, inputs: Any, output: Any, *, key: str = scale) -> None:
                if output.ndim != 4 or output.shape[0] != 1:
                    raise RuntimeError("Expected one image and BCHW classification logits")
                self.raw_maps[key] = output[0, self.class_id].detach().float().sigmoid().cpu().numpy().copy()

            self.handles.append(classifier.register_forward_hook(capture))

    def extract(self, image_bgr: Any, *, image_size: int, device: str, confidence: float) -> dict[str, Any]:
        import numpy as np

        try:
            results = self.model.predict(
                source=image_bgr,
                imgsz=image_size,
                rect=False,
                device=device,
                conf=confidence,
                iou=0.7,
                classes=[self.class_id],
                nms=True,
                augment=False,
                save=False,
                verbose=False,
            )
        finally:
            self._remove_hooks()
        if len(self.raw_maps) != len(self.strides):
            raise RuntimeError("Some classifier hooks did not run; heatmaps would be incomplete")
        input_shapes = {
            (grid.shape[0] * stride, grid.shape[1] * stride)
            for grid, stride in zip(self.raw_maps.values(), self.strides)
        }
        if input_shapes != {(image_size, image_size)}:
            raise RuntimeError(f"Unexpected inference dimensions: {input_shapes}")
        maps = {
            name: restore_heatmap(grid, image_bgr.shape[:2], (image_size, image_size))
            for name, grid in self.raw_maps.items()
        }
        boxes = results[0].boxes
        predictions = [
            {"xyxy": xyxy, "confidence": score, "class_id": int(class_id)}
            for xyxy, score, class_id in zip(
                boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist(),
            )
        ]
        return {
            "heatmap": np.maximum.reduce(list(maps.values())),
            "maps": maps,
            "raw_maps": dict(self.raw_maps),
            "predictions": predictions,
            "branch": self.branch,
            "strides": self.strides,
            "layers": self.layers,
        }

    def close(self) -> None:
        self._remove_hooks()
        self.model.callbacks["on_predict_start"].remove(self.callback)

    def __enter__(self) -> DetectorHeatmaps:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def read_ground_truth(image_path: Path, shape: tuple[int, int], *, class_id: int, labels_dir: Path | None = None) -> list[list[float]]:
    """Read optional YOLO labels; infer the labels directory for images/<split>."""
    if labels_dir is None:
        if image_path.parent.parent.name != "images":
            return []
        labels_dir = image_path.parent.parent.parent / "labels" / image_path.parent.name
    label_path = labels_dir / f"{image_path.stem}.txt"
    if not label_path.is_file():
        return []
    height, width = shape
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label_class, xc, yc, bw, bh = map(float, line.split())
        if int(label_class) != class_id or bw <= 0 or bh <= 0:
            continue
        boxes.append([
            max(0.0, (xc - bw / 2) * width), max(0.0, (yc - bh / 2) * height),
            min(float(width), (xc + bw / 2) * width), min(float(height), (yc + bh / 2) * height),
        ])
    return boxes


def collect_images(sources: list[Path], limit: int) -> list[Path]:
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images: list[Path] = []
    for value in sources:
        source = project_path(value)
        if source.is_dir():
            images.extend(path for path in sorted(source.iterdir()) if path.is_file() and path.suffix.lower() in supported)
        elif source.is_file() and source.suffix.lower() in supported:
            images.append(source)
        else:
            raise FileNotFoundError(f"Image or image directory not found: {source}")
    images = list(dict.fromkeys(images))[:limit]
    if not images:
        raise ValueError("No supported images found")
    return images


def _fresh_output_dir(requested: Path) -> Path:
    candidate = requested
    index = 2
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = requested.with_name(f"{requested.name}-{index}")
            index += 1


def _summary(response: dict[str, Any]) -> dict[str, Any]:
    return {
        key: response[key] for key in ("branch", "strides", "layers", "predictions")
    } | {
        "scales": {
            name: {"grid_shape": list(grid.shape), "peak_class_score": float(grid.max())}
            for name, grid in response["raw_maps"].items()
        }
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def compare_heatmaps(
    sources: list[Path], *, baseline: Path = DEFAULT_BASELINE, p2: Path = DEFAULT_P2,
    output_dir: Path, image_size: int = 960, device: str = "cpu", confidence: float = 0.25,
    class_id: int = 0, limit: int = 4, labels_dir: Path | None = None,
) -> Path:
    import cv2
    import numpy as np
    import ultralytics

    from .heatmap_plots import render_comparison, render_scales

    if image_size <= 0 or image_size % 32:
        raise ValueError("imgsz must be a positive multiple of 32")
    if not 0 <= confidence <= 1 or limit <= 0:
        raise ValueError("conf must be between 0 and 1, and limit must be positive")
    images = collect_images(sources, limit)
    baseline, p2 = project_path(baseline), project_path(p2)
    labels_dir = project_path(labels_dir) if labels_dir is not None else None
    with ExitStack() as stack:
        baseline_model = stack.enter_context(DetectorHeatmaps(baseline, class_id=class_id))
        p2_model = stack.enter_context(DetectorHeatmaps(p2, class_id=class_id))
        if 4 in baseline_model.strides or 4 not in p2_model.strides:
            raise ValueError("Expected a baseline without stride 4 and a P2 checkpoint with stride 4")
        if baseline_model.class_name != p2_model.class_name:
            raise ValueError("The requested class has different names in the two checkpoints")
        output_dir = _fresh_output_dir(project_path(output_dir))
        manifest: dict[str, Any] = {
            "method": "sigmoid of active per-scale class logits; bilinear restore; max across scales",
            "color_scale": [0, 1], "class_id": class_id, "class_name": baseline_model.class_name,
            "imgsz": image_size, "rect": False, "nms": True, "iou": 0.7, "confidence": confidence,
            "device": device, "ultralytics": ultralytics.__version__,
            "baseline": {"path": str(baseline), "sha256": _sha256(baseline)},
            "p2": {"path": str(p2), "sha256": _sha256(p2)},
            "images": [],
        }
        report = [
            "# Baseline vs P2 detection heatmaps", "",
            f"Heatmaps show {baseline_model.class_name} class scores at detection-grid locations, on a shared 0–1 scale. "
            "The combined response is the maximum across scales. Cyan dashed boxes are ground truth; "
            "lime boxes are predictions. Both models use the same input and inference settings.", "",
            "These visualizations do not establish an accuracy gain or isolate the causal effect of P2. "
            "See manifest.json for checkpoint hashes, inference settings, per-scale peaks and predictions.", "",
        ]
        for index, image_path in enumerate(images, start=1):
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                raise ValueError(f"Could not read image: {image_path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            ground_truth = read_ground_truth(image_path, image_bgr.shape[:2], class_id=class_id, labels_dir=labels_dir)
            responses = [model.extract(image_bgr, image_size=image_size, device=device, confidence=confidence)
                         for model in (baseline_model, p2_model)]
            stem = f"{index:02d}_{image_path.stem}"
            comparison = output_dir / f"{stem}_comparison.png"
            scales = output_dir / f"{stem}_scales.png"
            render_comparison(image_rgb, *responses, ground_truth, comparison,
                              title=image_path.name, confidence=confidence, class_name=baseline_model.class_name)
            render_scales(image_rgb, *responses, ground_truth, scales,
                          title=image_path.name, class_name=baseline_model.class_name)
            np.savez_compressed(output_dir / f"{stem}_maps.npz", **{
                f"{model_name}_{name.replace('/', '_')}": grid
                for model_name, response in zip(("baseline", "p2"), responses)
                for name, grid in response["raw_maps"].items()
            })
            manifest["images"].append({
                "source": str(image_path), "shape": list(image_bgr.shape[:2]), "ground_truth": ground_truth,
                "baseline": _summary(responses[0]), "p2": _summary(responses[1]),
                "comparison": comparison.name, "scales": scales.name,
            })
            (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            report.extend([f"## {image_path.name}", "", f"![Comparison]({comparison.name})", "",
                           f"[PDF]({comparison.with_suffix('.pdf').name}) · [Per-scale heatmaps]({scales.name})", ""])
            print(f"[{index}/{len(images)}] {image_path.name}: {comparison}")
        (output_dir / "README.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Heatmap report: {output_dir / 'README.md'}")
    return output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and P2 detection-score heatmaps on the same images.")
    parser.add_argument("--source", type=project_path, nargs="+", default=[DEFAULT_SOURCE])
    parser.add_argument("--baseline", type=project_path, default=DEFAULT_BASELINE)
    parser.add_argument("--p2", type=project_path, default=DEFAULT_P2)
    parser.add_argument("--output-dir", type=project_path, default=PROJECT_ROOT / "06_results/heatmaps/baseline_vs_p2")
    parser.add_argument("--labels-dir", type=project_path)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        compare_heatmaps(args.source, baseline=args.baseline, p2=args.p2, output_dir=args.output_dir,
                         image_size=args.imgsz, device=args.device, confidence=args.conf,
                         class_id=args.class_id, limit=args.limit, labels_dir=args.labels_dir)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
