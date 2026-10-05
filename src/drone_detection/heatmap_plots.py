"""Render comparable detector class-response maps in the original image frame.

These are detector scores, not gradient-based explanations. Every panel uses
the same fixed [0, 1] color scale; maps are never normalized independently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

_BACKGROUND = "#f5f7fb"
_INK = "#17243b"
_MUTED = "#57657a"
_GROUND_TRUTH = "#32e8ee"
_PREDICTION = "#a5ff54"
_NORM = Normalize(vmin=0.0, vmax=1.0, clip=True)
_CMAP = "magma"
_SCALES = ("P2/4", "P3/8", "P4/16", "P5/32")


def _valid_boxes(
    boxes: list[list[float]], width: int, height: int,
) -> list[list[float]]:
    """Keep finite, visible boxes and clip annotations to the image boundary."""
    valid = []
    for box in boxes:
        coordinates = np.asarray(box, dtype=float)
        if coordinates.shape != (4,) or not np.isfinite(coordinates).all():
            continue
        coordinates[[0, 2]] = np.clip(coordinates[[0, 2]], 0, width)
        coordinates[[1, 3]] = np.clip(coordinates[[1, 3]], 0, height)
        x1, y1, x2, y2 = coordinates.tolist()
        if x2 > x1 and y2 > y1:
            valid.append([x1, y1, x2, y2])
    return valid


def _crop_bounds(
    boxes: list[list[float]], width: int, height: int,
) -> tuple[tuple[float, float, float, float], str]:
    if boxes:
        x1, y1, x2, y2 = min(
            boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
        )
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(120.0, 3.0 * max(x2 - x1, y2 - y1))
        label = "Same crop around the smallest annotated object"
    else:
        center_x, center_y = width / 2, height / 2
        side = max(120.0, min(width, height) / 3)
        label = "Same center crop (no ground-truth annotation)"
    side = min(side, width, height)
    left = max(0.0, min(center_x - side / 2, width - side))
    top = max(0.0, min(center_y - side / 2, height - side))
    return (left, top, left + side, top + side), label


def _check_map(heatmap: np.ndarray, image_rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(heatmap, dtype=float)
    if values.shape != image_rgb.shape[:2]:
        raise ValueError(
            f"Heatmap shape {values.shape} does not match image "
            f"shape {image_rgb.shape[:2]}."
        )
    if not np.isfinite(values).all():
        raise ValueError("Heatmaps must contain only finite values.")
    if values.min() < -1e-6 or values.max() > 1 + 1e-6:
        raise ValueError("Detector response maps must lie within [0, 1].")
    return values


def _panel(
    axis: Any,
    image_rgb: np.ndarray,
    boxes: list[list[float]],
    *,
    heatmap: np.ndarray | None = None,
    predictions: list[dict[str, Any]] | None = None,
    confidence: float = 0.25,
    crop: tuple[float, float, float, float] | None = None,
) -> None:
    height, width = image_rgb.shape[:2]
    extent = (0, width, height, 0)
    axis.imshow(image_rgb, extent=extent, interpolation="nearest", zorder=0)
    if heatmap is not None:
        axis.imshow(
            _check_map(heatmap, image_rgb), extent=extent,
            cmap=_CMAP, norm=_NORM, alpha=0.60, interpolation="nearest",
            zorder=1,
        )
    for x1, y1, x2, y2 in boxes:
        axis.add_patch(Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False,
            edgecolor=_GROUND_TRUTH, linewidth=1.65,
            linestyle=(0, (4, 2)), zorder=3,
        ))

    view = crop if crop is not None else (0, 0, width, height)
    left, top, right, bottom = view
    for prediction in predictions or []:
        score = float(prediction["confidence"])
        if score < confidence:
            continue
        visible_boxes = _valid_boxes([prediction["xyxy"]], width, height)
        if not visible_boxes:
            continue
        x1, y1, x2, y2 = visible_boxes[0]
        if x2 <= left or y2 <= top or x1 >= right or y1 >= bottom:
            continue
        axis.add_patch(Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False,
            edgecolor=_PREDICTION, linewidth=1.5, zorder=4,
        ))
        text_y = max(top + (bottom - top) * 0.04, y1)
        axis.text(
            max(x1, left), text_y, f"{score:.2f}",
            ha="left", va="bottom", fontsize=8, fontweight="bold",
            color=_PREDICTION, clip_on=True, zorder=5,
            bbox={"facecolor": "#102016", "alpha": 0.83,
                  "edgecolor": "none", "pad": 1.6},
        )
    axis.set_xlim(left, right)
    axis.set_ylim(bottom, top)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#d3dbe7")
        spine.set_linewidth(0.8)


def _colorbar(figure: Any, bounds: list[float], class_name: str) -> None:
    axis = figure.add_axes(bounds)
    bar = figure.colorbar(
        matplotlib.cm.ScalarMappable(norm=_NORM, cmap=_CMAP), cax=axis,
        ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    bar.set_label(f"{class_name} class response", fontsize=10, color=_INK, labelpad=12)
    bar.ax.tick_params(labelsize=9, colors=_MUTED)
    bar.outline.set_visible(False)


def render_comparison(
    image_rgb: np.ndarray,
    baseline: dict[str, Any],
    p2: dict[str, Any],
    ground_truth: list[list[float]],
    output_path: Path,
    *,
    title: str,
    confidence: float = 0.25,
    class_name: str = "UAV",
) -> None:
    """Write a full-image and matched-crop comparison as PNG and vector PDF.

    ``heatmap`` and ``maps`` values must be probabilities already mapped to
    original image pixels. ``predictions`` use original-image ``xyxy`` boxes.
    Ground-truth boxes use the same pixel-coordinate convention.
    """
    height, width = image_rgb.shape[:2]
    boxes = _valid_boxes(ground_truth, width, height)
    crop, crop_label = _crop_bounds(boxes, width, height)
    aspect = max(0.45, min(1.25, height / width))
    figure = plt.figure(figsize=(17.6, 6.8 + 4.5 * aspect), facecolor=_BACKGROUND)
    try:
        grid = figure.add_gridspec(
            2, 3, left=0.035, right=0.914, top=0.865, bottom=0.115,
            wspace=0.055, hspace=0.21, height_ratios=[aspect, 1],
        )
        figure.text(0.035, 0.965, title, color=_INK, fontsize=19, fontweight="bold")
        figure.text(
            0.035, 0.928,
            f"Baseline vs P2  |  {class_name} detector class response  |  "
            "Shared 0–1 scale; no per-image normalization",
            color=_MUTED, fontsize=11,
        )
        labels = ("Input + ground truth", "Baseline · P3 / P4 / P5", "P2 model · P2 / P3 / P4 / P5")
        models = (None, baseline, p2)
        for row in range(2):
            for column, (label, model) in enumerate(zip(labels, models)):
                axis = figure.add_subplot(grid[row, column])
                _panel(
                    axis, image_rgb, boxes,
                    heatmap=model["heatmap"] if model is not None else None,
                    predictions=model.get("predictions", []) if model is not None else None,
                    confidence=confidence, crop=crop if row else None,
                )
                if row == 0:
                    axis.set_title(label, fontsize=12, fontweight="bold", color=_INK, pad=11)
                else:
                    axis.set_title(
                        "Matched detail" if column == 0 else label.split(" · ")[0] + " detail",
                        fontsize=11, color=_INK, pad=9,
                    )
                if row == 0 and column == 0:
                    x1, y1, x2, y2 = crop
                    axis.add_patch(Rectangle(
                        (x1, y1), x2 - x1, y2 - y1, fill=False,
                        edgecolor="white", linewidth=1.2,
                        linestyle=(0, (2, 3)), zorder=2,
                    ))
        _colorbar(figure, [0.937, 0.245, 0.012, 0.46], class_name)
        legend_items = [
            Line2D([0], [0], color=_GROUND_TRUTH, linewidth=2,
                   linestyle="--", label="Ground truth"),
            Line2D([0], [0], color=_PREDICTION, linewidth=2,
                   label=f"Prediction (confidence ≥ {confidence:g})"),
            Line2D([0], [0], color="#8b97a8", linewidth=1.5,
                   linestyle=":", label="Detail crop boundary"),
        ]
        figure.legend(
            handles=legend_items, loc="lower left", bbox_to_anchor=(0.028, 0.057),
            ncol=3, frameon=False, fontsize=10, labelcolor=_INK,
        )
        figure.text(0.035, 0.045, crop_label, fontsize=10, color=_MUTED)
        figure.text(
            0.035, 0.021,
            "Heatmaps show detector scores, not Grad-CAM or causal attribution. "
            "Combined response = maximum across detection scales.",
            fontsize=9, color=_MUTED,
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path.with_suffix(".png"), dpi=180, facecolor=figure.get_facecolor())
        figure.savefig(output_path.with_suffix(".pdf"), dpi=180, facecolor=figure.get_facecolor())
    finally:
        plt.close(figure)


def render_scales(
    image_rgb: np.ndarray,
    baseline: dict[str, Any],
    p2: dict[str, Any],
    ground_truth: list[list[float]],
    output_path: Path,
    *,
    title: str,
    class_name: str = "UAV",
) -> None:
    """Write aligned, fixed-scale P2/P3/P4/P5 response panels as a PNG."""
    height, width = image_rgb.shape[:2]
    boxes = _valid_boxes(ground_truth, width, height)
    aspect = max(0.45, min(1.25, height / width))
    figure = plt.figure(figsize=(18.4, 3.5 + 8.2 * aspect), facecolor=_BACKGROUND)
    try:
        grid = figure.add_gridspec(
            2, 4, left=0.056, right=0.917, top=0.825, bottom=0.135,
            wspace=0.055, hspace=0.16,
        )
        figure.text(0.035, 0.957, title, color=_INK, fontsize=19, fontweight="bold")
        figure.text(
            0.035, 0.902,
            f"Response by detection scale  |  {class_name} class scores  |  "
            "Shared 0–1 scale; no per-panel normalization",
            color=_MUTED, fontsize=11,
        )
        for row, (name, model) in enumerate((("Baseline", baseline), ("P2 model", p2))):
            for column, scale in enumerate(_SCALES):
                axis = figure.add_subplot(grid[row, column])
                heatmap = model.get("maps", {}).get(scale)
                if heatmap is None:
                    axis.set_facecolor("#e7ecf3")
                    axis.set_xlim(0, width)
                    axis.set_ylim(height, 0)
                    axis.set_aspect("equal")
                    axis.set_xticks([])
                    axis.set_yticks([])
                    for spine in axis.spines.values():
                        spine.set_visible(False)
                    axis.text(
                        0.5, 0.55, "No P2 head" if scale == "P2/4" else "Head unavailable",
                        ha="center", va="center", transform=axis.transAxes,
                        fontsize=15, color=_MUTED, fontweight="bold",
                    )
                    axis.text(
                        0.5, 0.4, "This scale is absent from the architecture",
                        ha="center", va="center", transform=axis.transAxes,
                        fontsize=9, color=_MUTED,
                    )
                else:
                    _panel(axis, image_rgb, boxes, heatmap=heatmap)
                if row == 0:
                    level, stride = scale.split("/")
                    axis.set_title(
                        f"{level} · stride {stride}", fontsize=12,
                        fontweight="bold", color=_INK, pad=10,
                    )
                if column == 0:
                    axis.set_ylabel(name, fontsize=12, fontweight="bold", color=_INK, labelpad=16)
        _colorbar(figure, [0.939, 0.265, 0.011, 0.43], class_name)
        figure.legend(
            handles=[Line2D([0], [0], color=_GROUND_TRUTH, linewidth=2,
                            linestyle="--", label="Ground truth")],
            loc="lower left", bbox_to_anchor=(0.03, 0.065),
            frameon=False, fontsize=10, labelcolor=_INK,
        )
        figure.text(
            0.035, 0.031,
            "Each map is a detector class response before prediction filtering; "
            "these are not Grad-CAM maps. Color scales are identical across both models.",
            fontsize=9, color=_MUTED,
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path.with_suffix(".png"), dpi=180, facecolor=figure.get_facecolor())
    finally:
        plt.close(figure)
