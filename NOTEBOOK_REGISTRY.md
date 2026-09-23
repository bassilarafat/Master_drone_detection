# Notebook Registry

Audit: 2026-09-20. Paths are repository-relative; all notebooks reside in `notebooks/`. Cell numbers below are zero-based. Status describes the evidence, not whether every cell can be rerun safely. No notebook was executed or changed. Eleven substantive research notebooks, one environment-only stub, one zero-cell placeholder and one zero-byte placeholder were found. All fourteen are listed to make the inventory complete.

## Logical research order

| # | Notebook | Purpose | Experiment | Inputs | Outputs | Main Finding | Status |
| - | -------- | ------- | ---------- | ------ | ------- | ------------ | ------ |
| 1 | `01_dataset_exploration.ipynb` | Count images/objects; normalized-area distribution | Dataset / baseline preparation | DUT `data.yaml`, image and TXT splits | Saved counts and histogram | Historical 10,109 objects; 66.86% area-defined Tiny | exploratory |
| 2 | `02_baseline_analysis.ipynb` | Not found in repository | Not found in repository | Not found in repository | Zero-byte file | No content to analyze | incomplete |
| 3 | `02_baseline_size_analysis.ipynb` | Baseline test validation, presence/IoU size analysis | EXP001 | Baseline best.pt; test images/labels | Built-in test output, tables, worst-case figures | Test mAP50 0.951 / mAP50-95 0.652; size-table join invalidates reported object recalls | superseded |
| 4 | `03_correct_size_evaluation.ipynb` | Prepare object IDs and predictions for corrected baseline size evaluation | EXP001 | Baseline best.pt; test split | In-memory GT/prediction tables and saved displays | 2,245 GT, 2,263 predictions, 2,114 images with predictions; no completed size matching | incomplete |
| 5 | `03_baseline_error_analysis.ipynb` | One-to-one matching and qualitative FP diagnosis | EXP001 | Baseline best.pt; test split | Saved TP/FP counts, top-20 visual/ratio analysis | 2,104 TP / 159 FP; shadows, background and localization examples | completed |
| 6 | `04_train_exp002_high_resolution.ipynb` | Train and validate higher-resolution baseline | EXP002 | Pretrained YOLOv8s; DUT YAML | Completed -3 run, best/last.pt, training plots/CSV, val output | 100 epochs; standalone val mAP50-95 0.65640845 | completed |
| 7 | `05_exp002_evaluation_and_error_analysis.ipynb` | Test predictions and best-IoU FP analysis | EXP002 | Completed -3 best.pt; test split | In-memory predictions and localization analysis | 2,297 predictions; 142 best-IoU FP = 98 zero-overlap + 44 localization | exploratory |
| 8 | `06_dataset_audit_and_error_driven_analysis.ipynb` | Labels/negatives audit and image-level FN exploration | EXP002 / dataset | DUT splits; completed -3 best.pt | Saved audit tables, copied negative, 141 copied FN images, val logs | Almost no negatives; early annotation problems; small-object misses | needs verification |
| 9 | `07_final_object_level_error_analysis.ipynb` | Final saved EXP002 one-to-one object-level validation | EXP002 | -3 best.pt; validation images/labels | Six CSVs under `06_results/EXP002_object_level_error_analysis/val/` | 2,433 TP / 190 FP / 188 FN; Tiny recall 86.8056% | final evaluation |
| 10 | `08_EXP003_YOLOv8s_P2.ipynb` | Construct, train and validate P2 model | EXP003 P2 | Packaged P2 YAML, YOLOv8s pretrained weights, DUT YAML | P2 run under dataset directory; built-in val results | Strides 4/8/16/32; standalone val mAP50-95 0.66139291 | completed |
| 11 | `09_EXP003_YOLOv8s_P2_Object_Level_Evaluation.ipynb` | P2 object-level validation and EXP002 comparison | EXP003 P2 | P2 best.pt; validation split; hard-coded EXP002 aggregates | Ten CSVs and two PNGs under `06_results/EXP003_object_level_evaluation/` | 2,440 TP / 176 FP / 181 FN; Tiny recall 88.6111%; proposes EXP004 | final evaluation |
| 12 | `10_EXP003_Object_Level_Evaluation.ipynb` | Not found in repository | Filename only references EXP003 | Not found in repository | Valid notebook JSON, zero cells | No evaluation content | incomplete |
| 13 | `10b_final_dataset_integrity_check.ipynb` | Dataset integrity, annotation editing and duplicate removal | Dataset / P2-assisted annotation | Current DUT files; P2 best.pt | Saved audits; historical label edit/backup and deletion operations | Historical train/test duplicate; zero-height validation label; counts change during execution | needs verification |
| 14 | `11_EXP003_Faster_RCNN_baseline.ipynb` | Environment readiness only | Intended Faster R-CNN comparator; reused EXP003 name | Torch/TorchVision imports | One cell of version/GPU output | No Faster R-CNN model, training or evaluation | incomplete |

“Final evaluation” above means the most complete saved **validation** object-level analysis, not an independent final test. Exact run paths and metric provenance are in `EXPERIMENT_REGISTRY.md`.

## Notebook analyses

### 1. `notebooks/01_dataset_exploration.ipynb`

- **Purpose:** Establish split counts and quantify apparent drone sizes.
- **Inputs:** `02_datasets/DUT_Anti_UAV/data.yaml`, split images and all label TXT files.
- **Main operations:** Count files and annotation lines; compute normalized width × height; histogram and four bins (<0.001, <0.01, <0.1, >0.1).
- **Outputs:** Seven cells with saved text and embedded histogram; no separate export found.
- **Main results:** Historical images 5,200/2,600/2,200; instances 5,243/2,621/2,245. Area min 0, max 0.70198827348, mean 0.0129696113, median 0.000510209568. Tiny/Small/Medium/Large 6,759/1,896/1,103/351.
- **Research conclusion:** Strong concentration of small normalized boxes motivates size-aware evaluation, but zero area hints at annotation invalidity. Counts precede later annotation changes.
- **Status:** exploratory. Preserve original observations without treating them as current dataset statistics.

### 2. `notebooks/02_baseline_analysis.ipynb`

- **Purpose / Inputs / Main operations / Outputs / Main results / Research conclusion:** **Not found in repository**; this file is zero bytes, not a parseable notebook.
- **Status:** incomplete. Its name provides no evidence of a baseline analysis.

### 3. `notebooks/02_baseline_size_analysis.ipynb`

- **Purpose:** Evaluate baseline test performance and find size-dependent weaknesses.
- **Inputs:** `runs/detect/04_experiments/EXP001_YOLOv8s_baseline/weights/best.pt`, 2,200 test images and 2,245 objects.
- **Main operations:** Built-in `model.val(split="test")`; predictions at confidence 0.25; initial image-presence detection proxy; pixel-box IoU; maximum overlap per GT; merge by image; worst-case visualizations.
- **Outputs:** Built-in test plots under `notebooks/runs/detect/val/`, embedded tables and images. Related baseline figures exist at `06_results/EXP001_baseline/BBOX.png` and `worst_cases.png`; exact export provenance is not established by current cells.
- **Main results:** Saved built-in P/R/mAP50/mAP50-95 = 0.966/0.927/0.951/0.652, rounded; 3.6 ms inference/image. Original test bins: Tiny 1,148, Small 555, Medium 467, Large 75. Presence proxy detected 1,096/535/454/74. Later IoU table reports counts 1,148/623/510/78 = 2,359, exceeding 2,245 GT.
- **Research conclusion:** Qualitative small-object concern is plausible, but image-only joins replicate rows for multi-object images, predictions are not assigned uniquely, and presence is not object recall. Do not quote its size recall as final evidence.
- **Status:** superseded for size metrics; retain built-in test output and historical visual analysis.

### 4. `notebooks/03_correct_size_evaluation.ipynb`

- **Purpose:** Rebuild baseline GT with per-image `gt_id` and correctly associate predictions using `result.path`.
- **Inputs:** Baseline best.pt and DUT test split.
- **Main operations:** Pixel-coordinate conversion; normalized-area categories; streamed prediction at 640, confidence 0.25, NMS IoU 0.7; count prediction-bearing images.
- **Outputs:** In-memory `gt_clean` / `pred_clean`, saved sample tables and counts; final cell scans for weight paths.
- **Main results:** 2,245 GT, 2,263 predictions, 2,114 images with predictions; test bins 1,148/555/467/75.
- **Research conclusion:** Data preparation corrects identity handling, but no completed assignment, TP/FP/FN ledger or corrected size recall is present.
- **Status:** incomplete despite “correct” in the filename.

### 5. `notebooks/03_baseline_error_analysis.ipynb`

- **Purpose:** Diagnose baseline test false positives and localization shape errors.
- **Inputs:** Baseline best.pt and test labels/images; inference confidence 0.25, image size not explicit in this notebook.
- **Main operations:** Confidence-sorted greedy matching to unmatched GT at IoU 0.5; top-20 FP selection; visual overlays; manual category notes; box-width/height ratio analysis.
- **Outputs:** Saved counts, tables and embedded visualizations; no standalone full TP/FP CSV found.
- **Main results:** 2,263 predictions, 2,104 TP, 159 FP. Top-20 subset: 9 zero-overlap, 11 localization. Manual notes on nine cases: Shadow 2, Background 3, Possible annotation issue 1, blank 3. Localization mean width ratio 1.2481803 and height ratio 1.5360993.
- **Research conclusion:** Background/shadow confusion and incorrect box extents warrant distinct remedies. FN=141 is derivable from GT−TP but no complete FN export exists. FP subtype IoU only considers unmatched GT, so duplicates may masquerade as background; manual notes include a visually overlapping UAV labelled zero-IoU.
- **Status:** completed as historical exploratory error analysis; FP semantic totals are neither exhaustive nor fully reliable.

### 6. `notebooks/04_train_exp002_high_resolution.ipynb`

- **Purpose:** Test resolution 960 versus the 640 baseline.
- **Inputs:** `05_models/pretrained/yolov8s.pt`, DUT YAML.
- **Main operations:** Train 100 epochs, batch 4, GPU 0, patience 20, workers 4; load completed -3 checkpoint; built-in validation at 960.
- **Outputs:** Actual completed run `notebooks/runs/04_experiments/EXP002_YOLOv8s_high_resolution/YOLOv8s_960-3/`; best/last weights, 100-row CSV, curves and confusion matrices; standalone validation output `notebooks/runs/detect/val-3/`.
- **Main results:** 5.626 training hours; auto AdamW lr 0.002 / momentum 0.9; standalone P 0.97314548, R 0.91250919, mAP50 0.94625813, mAP50-95 0.65640845. Training-end best-weight output is rounded 0.972/0.912/0.946/0.656 and uses a different batch context.
- **Research conclusion:** Higher-resolution baseline is trained and available; apparent improvement needs controlled comparison. Printed “Results directory” is constructed from requested path/name and does not reflect Ultralytics' actual nested, suffixed output path.
- **Status:** completed. Initial and second attempts must be tracked separately.

### 7. `notebooks/05_exp002_evaluation_and_error_analysis.ipynb`

- **Purpose:** Inspect EXP002 test predictions and localization.
- **Inputs:** Completed EXP002 -3 best.pt, test images and labels; an unused later `LABELS` variable points to validation but `get_gt_boxes` still reads test.
- **Main operations:** Batches of 20, 960, confidence 0.25, NMS 0.7; maximum IoU of each prediction against any test GT; FP split at zero overlap; localization width/height ratios.
- **Outputs:** In-memory `pred_clean_exp2`, displayed predictions, FP counts and ratios. No exported prediction CSV from this notebook was found.
- **Main results:** 2,297 predictions; 142 best-IoU FP, comprising 98 zero-overlap and 44 localization; mean width ratio 1.59559197 and height ratio 1.83596192.
- **Research conclusion:** Background/localization remain, but duplicate predictions can share a GT because this is not one-to-one assignment. The other 2,155 predictions must not automatically be promoted to unique TP, nor used to derive FN.
- **Status:** exploratory; final object-level evaluation is in notebook 07 on validation.

### 8. `notebooks/06_dataset_audit_and_error_driven_analysis.ipynb`

- **Purpose:** Audit pairing/annotation quality, inspect negatives and identify missed images.
- **Inputs:** All dataset splits, EXP002 -3 best.pt.
- **Main operations:** Pair/label checks; visualize empty-labelled images; inference on train `00724`; validation; copy annotation-defined negatives; enumerate validation images with GT but no predictions; display them; try confidence 0.01.
- **Outputs:** `06_results/no_drone_images/00639.jpg`; relative `06_results/false_negative` resolves to `notebooks/06_results/false_negative/` in the saved session (141 JPGs); built-in validation plots/logs; embedded images.
- **Main results:** Original 5,200/2,600/2,200 paired files. Three empty train labels (`00579`,`00639`,`00724`), one invalid validation label. Later one empty train label remains. Built-in EXP002 validation confirms mAP50-95 0.65640845. No-prediction images=141; displayed FN-size counts 66/65/7/3. The low-confidence trial prints detections but no GT-matched recovery metrics.
- **Research conclusion:** Tiny/Small difficulty and negative scarcity motivate improved features and future negatives. `find_negative_labels` unconditionally appends every file; its caller prints stale `result`. `fn_df` is referenced without a construction cell. These flawed exploratory cells cannot establish dataset or FN totals. Inference geometry changes in the confidence-0.01 experiment also confound a threshold-only interpretation.
- **Status:** needs verification. Some recorded operations copy data; do not execute wholesale as a read-only audit.

### 9. `notebooks/07_final_object_level_error_analysis.ipynb`

- **Purpose:** Produce complete EXP002 object-level evaluation on validation during method development.
- **Inputs:** Completed EXP002 -3 best.pt; validation images/labels, 2,600 images / 2,621 GT.
- **Main operations:** 960, confidence 0.25, NMS 0.70, batch 8; original-image pixel size bins; confidence-sorted one-to-one matching at IoU 0.50; FP classification against all GT; explicit unmatched-GT FNs; conservation checks.
- **Outputs:** `true_positives.csv`, `false_positives.csv`, `false_negatives.csv`, `ground_truth_objects.csv`, `recall_by_object_size.csv`, `summary.csv`, all in `06_results/EXP002_object_level_error_analysis/val/`.
- **Main results:** 2,623 predictions; TP 2,433 / FP 190 / FN 188; P 0.9275638582, R 0.9282716520, F1 0.9279176201. Tiny 625/720, Small 1,348/1,427, Medium 304/315, Large 156/159. FP subtypes 122/46/22.
- **Research conclusion:** Tiny recall is the weakest and geometric background is the largest FP group. Audit verified unique GT partitions and CSV consistency. Zero-height GT is included as an FN, and raw box coordinates are not exported.
- **Status:** final evaluation (validation only, with documented limitations).

### 10. `notebooks/08_EXP003_YOLOv8s_P2.ipynb`

- **Purpose:** Test a finer detection head for tiny drones.
- **Inputs:** Ultralytics `yolov8s-p2.yaml`, `yolov8s.pt`, DUT YAML.
- **Main operations:** Instantiate architecture, inspect model/strides, load pretrained weights, train at 960 for 100 epochs with batch 8, then prediction and built-in validation on val.
- **Outputs:** `02_datasets/DUT_Anti_UAV/runs/04_experiments/EXP003_YOLOv8s_P2_960/`; best/last.pt, args, results, curves/confusion matrices; standalone validation saved to `notebooks/runs/detect/val-12/`.
- **Main results:** Strides 4/8/16/32. Pre-training model: 161 layers, 10,884,336 parameters, 39.7 GFLOPs; trained fused one-class model: 91 layers, 10,626,708 parameters, 36.6 GFLOPs. Explicit load transfers 219/437 items; training stage prints 389/437. Training 10.906 h. Standalone validation P/R/mAP50/mAP50-95 = 0.9737548948 / 0.9201268175 / 0.9490919454 / 0.6613929126. Inference 12.8503 ms/image in that val call.
- **Research conclusion:** A P2 implementation is actually trained and validated. Training-epoch peak mAP50-95 0.66217 differs from standalone revalidation 0.66139; keep contexts separate. P2-only causality is not proven by this single run.
- **Status:** completed.

### 11. `notebooks/09_EXP003_YOLOv8s_P2_Object_Level_Evaluation.ipynb`

- **Purpose:** Quantify P2 object-level changes versus EXP002.
- **Inputs:** P2 best.pt, val split; EXP002 metrics, recalls and FP counts typed into comparison cells rather than loaded from CSV.
- **Main operations:** Stream directory inference at 960/confidence 0.25; zip separately sorted image paths with results; same greedy matching and original-pixel size classifier as notebook 07; summaries and comparison plots.
- **Outputs:** `EXP003_true_positives.csv`, `EXP003_false_positives.csv`, `EXP003_false_negatives.csv`, `EXP003_ground_truth.csv`, `EXP003_summary.csv`, `EXP003_recall_by_size.csv`, `EXP003_fp_analysis.csv`, `EXP002_vs_EXP003_overall.csv`, `EXP002_vs_EXP003_size_recall.csv`, `EXP002_vs_EXP003_fp_types.csv`; PNGs `EXP002_vs_EXP003_recall_by_size.png` and `EXP002_vs_EXP003_FP_types.png`. All in `06_results/EXP003_object_level_evaluation/`.
- **Main results:** 2,616 predictions on 2,478 prediction-bearing images; TP 2,440 / FP 176 / FN 181; P 0.9327217125, R 0.9309423884, F1 0.9318312011. Tiny TP 638/720, Small 1,351/1,427, Medium 297/315, Large 154/159; FP 124 Background / 40 Localization / 12 Duplicate.
- **Research conclusion:** Tiny and Small improve while Medium/Large decline; background remains unresolved. Final markdown explicitly proposes EXP004 hard-negative learning. Audit verified the underlying CSV counts independently, including the rounded hard-coded comparison values. NMS is implicit, inference batch differs, drive-less model path depends on current drive, and result ordering is assumed. Raw box coordinates/prediction IDs are not retained in final CSVs.
- **Status:** final evaluation (validation only; not a clean causal ablation).

### 12. `notebooks/10_EXP003_Object_Level_Evaluation.ipynb`

- **Purpose / Inputs / Main operations / Main results / Research conclusion:** **Not found in repository**; valid notebook metadata but zero cells.
- **Outputs:** No evaluation output.
- **Status:** incomplete. Notebook 09 contains the actual EXP003 object-level evaluation.

### 13. `notebooks/10b_final_dataset_integrity_check.ipynb`

- **Purpose:** Check pair integrity, label validity, box boundaries, corruption and exact duplicates; later cells also mutate annotations and dataset files.
- **Inputs:** DUT splits and P2 checkpoint for annotation assistance.
- **Main operations:** Counts, normalized label/boundary checks, PIL image checks, MD5 overlap by split, visual inspection; replace train `00579` label with model prediction; back up previous label; remove duplicate train `00259` image/label; rebuild hashes.
- **Outputs:** Saved tables and visualizations; historical `00579_backup_old.txt` creation and `00579.txt` replacement; historical deletion of `00259` pair. None of these operations was rerun by this audit.
- **Main results:** Before removal 5,200/2,600/2,200 images; one train/test duplicate, later zero after hash rebuild. `00991` validation height zero. Bounds overflow 16/7/10. Saved header-verification counts show zero corrupt images. Current audit confirms 5,199 train images but 5,200 TXT because the backup is counted. New `00579` label is `0 0.440292 0.462346 0.039960 0.067308`, model confidence 0.2941726744. The printed train count 5,245 includes orphan backup rows; paired count is 5,244.
- **Research conclusion:** Dataset state changed after earlier experiments; current integrity does not certify historical training/test separation. Model-generated annotation requires independent adjudication. Notebook's refreshed cross-split check does not search within-split duplication.
- **Status:** needs verification; mixed audit and destructive/editing cells make unattended execution unsuitable.

### 14. `notebooks/11_EXP003_Faster_RCNN_baseline.ipynb`

- **Purpose:** Actual content is environment verification; intended comparator is inferred only from filename.
- **Inputs:** Torch, TorchVision and common analysis libraries.
- **Main operations:** Imports and prints versions, CUDA availability and device name.
- **Outputs / Main results:** One cell: Torch 2.11.0+cu128, TorchVision 0.26.0+cu128, CUDA 12.8, RTX 5050 Laptop GPU.
- **Research conclusion:** No instantiated Faster R-CNN, pretrained-weight choice, dataset loader, training run or metrics exist. It is not a second completed EXP003.
- **Status:** incomplete.

## Preservation and interpretation

Preserve every notebook as historical evidence. Prioritize notebooks 04, 07, 08, 09 and 10b plus run YAMLs/CSVs/checkpoints and individual object tables for reproducibility. Do not substitute filename claims, embedded illustrations, old manually typed values or image-level proxies for saved object-level results. See `MASTER_PROJECT_SUMMARY.md` for severity-ranked issues and remaining work.
