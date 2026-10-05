# Baseline vs P2 heatmaps

The `heatmap` command compares the trained YOLO26s baseline and P2 checkpoints
on identical images. It exports full-image comparisons, matching close-ups,
per-scale responses, and the underlying score grids.

```powershell
python -m drone_detection heatmap --source 02_datasets/DUT_Anti_UAV/images/val/00180.jpg --device cpu
```

You can provide multiple image paths or a directory. `--limit` defaults to 4;
directory entries are taken in filename order. Relative paths resolve from the
repository root. The default checkpoints are:

- Baseline: `04_experiments/YOLO26s_baseline-5/weights/best.pt`
- P2: `04_experiments/YOLO26s_P2_transfer_learning/weights/best.pt`

Override them with `--baseline` and `--p2`. The command requires local trained
`.pt` checkpoints and checks that the P2 model has a stride-4 detection scale.
It does not train or download models. Use `--device 0` to run inference on the
first CUDA GPU; the default is CPU.

## Generated example set

The initial four examples were selected from validation using ground-truth size,
before looking at either model's predictions. Each is the first image in filename
order with a single UAV in its size category, using the project's original-image
pixel thresholds:

| Category | Image | Approximate UAV width × height |
| --- | --- | --- |
| Tiny: both dimensions <32 | `00180.jpg` | 29 × 20 px |
| Small: both <96, excluding Tiny | `00091.jpg` | 80 × 57 px |
| Medium: both <256, excluding smaller categories | `00001.jpg` | 112 × 70 px |
| Large: remaining boxes | `00002.jpg` | 327 × 140 px |

```powershell
python -m drone_detection heatmap --source 02_datasets/DUT_Anti_UAV/images/val/00180.jpg 02_datasets/DUT_Anti_UAV/images/val/00091.jpg 02_datasets/DUT_Anti_UAV/images/val/00001.jpg 02_datasets/DUT_Anti_UAV/images/val/00002.jpg --imgsz 960 --limit 4
```

## Reading the figures

- Left column: input image and ground-truth box.
- Middle: baseline score map and predicted boxes.
- Right: P2 score map and predicted boxes.
- Bottom row: the exact same crop around the smallest annotated object.
- Cyan dashed boxes: ground truth. Lime boxes and labels: predictions and scores.
- Brighter heatmap colors indicate higher UAV classification scores.

Every panel uses the same fixed 0–1 color scale. The image is blended underneath,
so consult the colorbar as a guide rather than interpreting the blended pixel
color as an exact measurement. Scores are never independently stretched to fill
the color range. `--conf` (default 0.25) filters predicted boxes, not heatmap values.

The per-scale figure aligns columns by P2/4, P3/8, P4/16 and P5/32. The baseline
has no P2/4 column. For a 960×960 input, the grids are 240×240, 120×120, 60×60 and
30×30 respectively. P2 adds a finer classification grid; its useful response
still depends on the learned model and the image.

If labels are stored under `labels/<split>` next to `images/<split>`, they are
loaded automatically. `--labels-dir` can supply another YOLO label directory.
Missing labels produce an unannotated image and a center crop. Degenerate
zero-width/height labels are skipped for plotting.

## Method

The code hooks the active per-scale classification branches of the **fused
inference model**, after Ultralytics has copied and initialized it. It applies
sigmoid to the selected class's logits, yielding the same class scores used by
the detector before selection and NMS. It supports the active one-to-many or
one-to-one branch; this comparison explicitly uses `nms=True`, hence the
one-to-many branch for both checkpoints.

Both models receive the same image, `imgsz` (default 960), square centered
letterboxing (`rect=False`), class ID (default 0/UAV), confidence threshold,
NMS IoU 0.7 and device. Maps are bilinearly resized to the inference input,
the exact integer letterbox padding is removed, and they are resized back to
the original image. The combined map is the elementwise maximum across the
restored scale maps. Raw grids retain the un-interpolated values.

These are **detection-score heatmaps**, not Grad-CAM, attention maps or object
segmentation. A hotspot marks a detection-grid location; box regression can place
the corresponding box away from that location. Different checkpoints and their
training histories can change responses for many reasons. These figures alone
do not establish improved accuracy or isolate the causal effect of adding P2.
Quantitative claims still require the same held-out evaluation protocol.

## Outputs

The default destination is `06_results/heatmaps/baseline_vs_p2/`. A subsequent
invocation creates a new suffixed directory to preserve previous output.

- `*_comparison.png` and `.pdf`: full image plus identical detail crops.
- `*_scales.png`: aligned per-scale heatmaps.
- `*_maps.npz`: raw class-score grids, keyed by model and scale.
- `manifest.json`: checkpoint hashes, image paths, boxes, scale peaks and settings.
- `README.md`: local gallery linking each figure.

New generated heatmap directories are ignored by Git. The initial four-image
`baseline_vs_p2/` report is tracked as reference results. The implementation is in
`src/drone_detection/heatmaps.py`; presentation is in `heatmap_plots.py`.

```python
from pathlib import Path
from drone_detection.heatmaps import compare_heatmaps

output = compare_heatmaps(
    [Path("02_datasets/DUT_Anti_UAV/images/val/00180.jpg")],
    output_dir=Path("06_results/heatmaps/my_comparison"),
    device="cpu",
)
```
