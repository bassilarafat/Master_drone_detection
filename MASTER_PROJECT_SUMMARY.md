# Master’s Thesis — Drone Detection Project

Audit date: 2026-09-20. This is a repository-evidence audit, not a fresh model evaluation. Existing notebooks, datasets, weights and results were not changed or executed. Only the three requested documentation files were created. Notebook cell references below are zero-based. Unknown information is recorded as **Not found in repository**.

The strongest currently documented candidate is EXP003 YOLOv8s-P2 at 960 pixels, on **validation**, with modest overall and tiny-object improvements over EXP002. This is not yet a controlled demonstration that P2 alone caused the improvement, nor a final test or deployment result.

## 1. Research Objective

Develop and evaluate an accurate, real-time drone detector, especially for small UAVs in complex backgrounds. This objective appears in `README.md` and `01_literature/research_plan.md`. The repository supports a single-class image-detection study; completed tracking, live-video evaluation and deployment benchmarks were not found.

## 2. Research Problem

Small apparent object size, imperfect localization and background confusion limit detection. EXP002 validation misses 95/720 tiny objects, versus 3/159 large objects under the project's pixel-size definition. P2 reduces tiny misses to 82, but zero-overlap false positives remain at 124. Background-only evaluation is effectively absent, so practical false-alarm performance is unknown.

## 3. Project Directory Overview

| Directory/file | Actual contents and significance |
| --- | --- |
| `01_literature/` | `research_plan.md`; its “environment preparation” status is outdated relative to saved experiments. |
| `02_datasets/DUT_Anti_UAV/` | `data.yaml`, image/label splits, three label caches, six visual-check JPGs; also contains the completed EXP003 training run. |
| `03_code/` | Four Python files: environment check, EXP002 training, EXP002 prediction export and IoU helper. |
| `04_experiments/` | Empty nominal EXP001/EXP002 directories and `experiment_log.xlsx`; workbook sheet has no data rows. Actual runs are elsewhere. |
| `05_models/pretrained/` | `yolov8s.pt`; pretrained asset, not a trained experiment result. |
| `06_results/` | Two baseline PNGs, six EXP002 validation CSVs, ten EXP003/comparison CSVs and two comparison PNGs, one copied negative image. |
| `07_figures/` | Twelve PNG/JPEG illustrations, including `Model_arch.png` and ChatGPT/Gemini-named figures; illustrations are not experiment evidence. |
| `08_thesis/` | Empty. |
| `09_papers/` | Four PDFs: `electronics-12-03664.pdf`, `sensors-26-03424-v2.pdf`, `Vision-Based Anti-UAV Detection and Tracking.pdf`, `yoloV8pdf.pdf`. Literature assets, not proof of local implementation. |
| `notebooks/` | Fourteen notebook files: eleven substantive research notebooks, one environment-only Faster R-CNN stub, one zero-cell notebook and one zero-byte file. Also `yolov8s.pt`, `yolo26n.pt`, nested EXP002 runs, evaluation plots and 141 copied FN images. |
| `runs/detect/` | Completed EXP001 run, a validation output folder and 2,200 rendered prediction JPGs. |

Recursive inventory included ignored research/run files rather than relying on Git tracking. `.git`, virtual environments and caches were excluded from research-content enumeration; installed package metadata was read separately for environment provenance. There are five training `args.yaml` files and four training `results.csv` files. No standalone research JSON files were found; notebook JSON was parsed directly. TXT research content is predominantly YOLO labels and one saved prediction label. Eight trained checkpoint files exist: best/last for EXP001, EXP002's initial two-epoch attempt, EXP002 completed attempt, and EXP003. The EXP002 second attempt has no checkpoint.

Each completed run contains `results.png`, PR/P/R/F1 curves, raw and normalized confusion matrices, training batches and validation label/prediction mosaics. Nested `notebooks/runs/detect/val*` folders contain additional evaluation plots, with some empty attempts; do not assign an unlabelled folder to an experiment without notebook evidence. `NOTEBOOK_REGISTRY.md` gives the full notebook inventory and analyses; `EXPERIMENT_REGISTRY.md` gives exact run paths and configurations.

## 4. Dataset

### Dataset name

Local name: DUT_Anti_UAV. Download source, release identifier, immutable archive checksum, acquisition protocol and license: **Not found in repository**. Only one current dataset directory exists, but its contents have been edited over time.

### Dataset structure

`02_datasets/DUT_Anti_UAV/data.yaml` sets the absolute root `D:/master/Master_Drone_Detection/02_datasets/DUT_Anti_UAV`, with `images/train`, `images/val`, `images/test`; corresponding labels are under `labels/<split>`. One class is defined: `0: UAV`.

### Train / Validation / Test

| Current disk state | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| JPG images | 5,199 | 2,600 | 2,200 |
| TXT files, including backup | 5,200 | 2,600 | 2,200 |
| Images lacking matching label | 0 | 0 | 0 |
| Orphan TXT files | 1 | 0 | 0 |
| Empty labels | 1 | 0 | 0 |
| Annotation rows in all TXT files | 5,245 | 2,621 | 2,245 |
| Annotation rows paired to images | 5,244 | 2,621 | 2,245 |
| Boxes extending outside image bounds | 16 | 7 | 10 |

Total current images: **9,999**. `labels/train/00579_backup_old.txt` is an orphan backup, not an additional training image. Do not count its object in the paired dataset. `images/train/00259.jpg` and its matching label were removed in saved notebook history. The early exploration notebook reports 5,200 train images and 5,243 train objects; those are historical counts, not current counts.

### Number of UAV instances

Current paired annotation rows total **10,110** (5,244 + 2,621 + 2,245). This includes the invalid zero-height validation annotation in `labels/val/00991.txt`; there are 10,109 positive-width/height paired rows, which still include boundary-overflow boxes. The superficially identical historical total 10,109 in notebook 01 comes from a different dataset state and must not be substituted for this calculation.

### Annotation format

YOLO TXT rows: `class x_center y_center width height`, normalized to source-image dimensions. Current class and five-field parsing checks found one degenerate box: validation `00991.txt`, height `0.000000`. Both saved custom evaluations retain this object as Tiny and count it as FN. Boundary checks found 33 boxes whose converted corners slightly exceed the image; these require adjudication, not automatic deletion.

### Negative samples

Only train `00639.txt` is empty, and its image remains present; a copy is saved at `06_results/no_drone_images/00639.jpg`. This is **one annotation-defined negative out of 5,199 training images (~0.019%)**. Validation and test have zero empty-label images. Notebook 06 originally found three empty train labels (`00579`, `00639`, `00724`), but two were subsequently annotated. An empty label alone does not independently prove absence of a drone. Background FP detections within positive images do not replace a background-only test set.

### Dataset limitations

Dataset snapshots and label-edit provenance are not versioned with each run. Notebook `10b_final_dataset_integrity_check.ipynb` records a train/test exact duplicate (`train/00259.jpg`, `test/01374.jpg`) and its later removal. It also replaces `00579`'s label with a P2 prediction of confidence 0.2941727, leaving the old label as an orphan TXT. Manual independent validation of that pseudo-label is not recorded.

The current SHA-256 audit found **zero cross-split exact duplicates**, but substantial within-split repetition:

| Split | Unique image byte contents | Duplicate pairs | Images belonging to pairs | Redundant copies |
| --- | ---: | ---: | ---: | ---: |
| Train | 5,196 | 3 | 6 | 3 |
| Validation | 2,599 | 1 | 2 | 1 |
| Test | 1,661 | 539 | 1,078 | 539 |

All 539 test duplicate pairs have identical stripped annotation text. Thus 24.5% of the 2,200 test files are redundant copies, and 49% belong to duplicate pairs; test metrics weight these scenes repeatedly. Examples: `images/test/00378.jpg` = `images/test/01438.jpg`, and `images/test/00954.jpg` = `images/test/02200.jpg` by SHA-256. Train pairs are `00004`/`00014`, `00020`/`00158`, `00248`/`00700`; validation pair is `00002`/`00007`. Each train/validation duplicate pair has differing label text, requiring annotation review. Byte hashes do not exclude near-duplicate frames or scene/sequence leakage.

## 5. Development Environment

| Component | Evidence |
| --- | --- |
| Python | 3.11.0 in saved execution logs and notebook metadata; README says 3.11. |
| PyTorch | 2.11.0+cu128 in saved logs and installed metadata. |
| TorchVision | 0.26.0+cu128 in notebook 11 and installed metadata. |
| CUDA | 12.8 in notebook 11; GPU execution recorded in training logs. |
| Ultralytics | 8.4.124 in training/evaluation outputs and checkpoint metadata. |
| OpenCV | Installed package metadata: `opencv_python-5.0.0.93.dist-info`; historical runtime version not recorded. |
| NumPy / Pillow | Current installed metadata: 2.4.6 / 12.3.0. |
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU, logs report 8,151 MiB. |

No dependency lockfile or requirements file was found. Installed package metadata is current environment evidence, not proof that every historical run used identical dependencies. No GPU job was run during this audit.

## 6. Research Workflow

1. Research plan and environment setup; local DUT dataset arranged in YOLO format.
2. Notebook 01 counts labels and normalized box areas; small apparent objects dominate.
3. EXP001 trains YOLOv8s at 640, 100 epochs. Checkpoint metadata date: 2026-08-21.
4. Baseline test evaluation and exploratory size analysis; an image-presence proxy and an image-only many-to-many join make early size recalls unsuitable for final reporting.
5. Baseline prediction-to-GT matching and qualitative FP inspection identify localization, shadows and background structures. Corrected-size notebook builds object IDs but stops before final matching.
6. EXP002 increases resolution to 960. Initial two-epoch attempt and an empty second attempt precede the completed `YOLOv8s_960-3` run, checkpoint dated 2026-09-09.
7. EXP002 test predictions and best-IoU FP analysis; then dataset audit, annotation edits, validation image-level FN inspection and exploratory lower-confidence inference.
8. Notebook 07 creates the strongest EXP002 object-level validation evidence, with confidence-sorted one-to-one matching and explicit FP subtypes.
9. Notebook 08 constructs YOLOv8s-P2 with strides 4/8/16/32, transfers pretrained weights, trains 100 epochs at 960; checkpoint dated 2026-09-17.
10. Notebook 09 evaluates EXP003 on validation and saves object-level CSVs and comparisons; proposes EXP004 hard-negative learning.
11. Notebook 10b records further integrity checks, model-assisted annotation editing and duplicate removal. Faster R-CNN notebook contains only an environment check. No completed EXP004 or live test was found.

This is a logical reconstruction supported by dependencies and checkpoint dates. Exact cell execution dates are absent and notebook outputs are not a reliable total chronological log. Checkpoint timestamps identify saved checkpoint metadata, not necessarily training start times.

## 7. Baseline

### Model

EXP001: pretrained YOLOv8s, one UAV class. Weight: `runs/detect/04_experiments/EXP001_YOLOv8s_baseline/weights/best.pt`.

### Training configuration

100 epochs, 640 pixels, batch 8, GPU 0, workers 8, seed 0, deterministic true, optimizer auto, patience 100. Full settings are in the adjacent `args.yaml`. Configured `lr0=0.01` is not proof of the actual auto-selected learning rate.

### Results

| Protocol | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Training CSV validation, maximum mAP50-95 row, epoch 98 | 0.95610 | 0.87257 | 0.90885 | 0.57365 |
| Notebook 02 built-in **test**, rounded saved output, 2,200 images / 2,245 GT | 0.966 | 0.927 | 0.951 | 0.652 |

Notebook `03_baseline_error_analysis.ipynb` records 2,263 test predictions, 2,104 TP and 159 FP at confidence 0.25 / matching IoU 0.50. Derived from those counts: FN=141, precision=0.929739, recall=0.937194. These derived values are not a saved complete FN ledger. Baseline custom size recall is not reliably established by the flawed early notebook.

### Weaknesses

Early normalized-area tiny recall is directionally lower but its merged table has 2,359 rows for 2,245 objects, so its reported 92.16% tiny recall must not be treated as a clean baseline comparator. Corrected EXP002 validation analysis supplies stronger evidence of the tiny-object weakness. Baseline qualitative examples also show background/shadow confusion and height localization errors.

## 8. Dataset and Object-Size Analysis

Two incompatible definitions occur:

| Category | Early normalized area `w*h` | Final object-level original-image pixel dimensions |
| --- | --- | --- |
| Tiny | area < 0.001 | width < 32 AND height < 32 |
| Small | 0.001 <= area < 0.01 | otherwise width < 96 AND height < 96 |
| Medium | 0.01 <= area < 0.1 | otherwise width < 256 AND height < 256 |
| Large | area >= 0.1 | all remaining objects |

Notebook 01's last bin uses `> 0.1` rather than `>=`; the displayed counts sum to its historical total. These are project-specific definitions, not a COCO area-based evaluation. Final pixel thresholds are applied before inference resizing, using widths/heights recovered by subtracting pixel corners.

Historical notebook 01 normalized-area distribution: Tiny 6,759 (66.86%), Small 1,896 (18.76%), Medium 1,103 (10.91%), Large 351 (3.47%), total 10,109. Historical test normalized-area distribution: 1,148 / 555 / 467 / 75. Do not compare these bins numerically against the final validation pixel bins.

### Small UAVs

Under the final four-bin scheme, Tiny is the weakest category. Validation has 720 Tiny and 1,427 Small objects; EXP003 detects 638 and 1,351 respectively. Tiny+Small account for 158/181 EXP003 FNs (87.29%), but prevalence and per-category recall must both be considered.

### Medium UAVs

Validation has 315 Medium objects. TP declines from 304 to 297 under P2; recall drops 2.2222 percentage points.

### Large UAVs

Validation has 159 Large objects. TP declines from 156 to 154; recall drops 1.2579 percentage points. See section 13 for all denominators and misses.

## 9. Error Analysis

### False Negatives

Final EXP002/EXP003 validation FN totals are 188/181. Tiny misses decline 95→82; Small 79→76; Medium increase 11→18; Large 3→5. Object-ID comparison of saved TP/FN tables finds 42 EXP002 misses recovered by EXP003 and 35 formerly detected objects newly missed, yielding a net gain of seven; 146 objects are missed by both.

Notebook 06's **141 no-prediction images** are an earlier image-level subset, not the final count of unmatched objects. Its displayed size breakdown is Tiny 66, Small 65, Medium 7, Large 3, but `fn_df` construction is missing from current code. Lowering confidence to 0.01 produces predictions on many images, without a saved GT-matched recovery analysis, and changes batching/padding. Thus no verified count of confidence-caused misses is available. Counts for blur, occlusion, partial visibility or specific difficult backgrounds: **Not found in repository**. One FN in both final tables is the invalid zero-height GT box, rather than defensible detector failure.

### False Positives

Final EXP002: 190 = 122 Background + 46 Localization + 22 Duplicate. Final EXP003: 176 = 124 Background + 40 Localization + 12 Duplicate. The saved classifier defines Background as maximum IoU exactly zero against all GT, Localization as positive IoU below 0.5, and Duplicate as overlap >=0.5 with an already matched GT when no eligible unmatched GT is assigned.

### Localization Errors

Localization FP is a prediction-level event and may coexist with an FN on the same UAV; FP and FN causal groups cannot simply be added as distinct scenes. Notebook 03's top-20 FP subset has 11 localization cases, mean width ratio 1.2482 and height ratio 1.5361. Notebook 05's EXP002 test analysis has 44 localization FPs and width/height means 1.5956/1.8360. These differ in sample and matching protocol, so they do not establish worse localization after EXP002.

### Background Confusion

Zero overlap is a geometric proxy, not verified semantic ground truth. A missing annotation or severe displacement can also yield IoU zero. Baseline top-20 review labels nine cases as background and eleven as localization; among the nine, two are documented as shadows, three as background structures, one as a possible annotation issue and three remain uncategorized. That baseline matcher computes FP overlap only against unmatched GT, so duplicate detections can be mislabeled as background. Final notebooks correct this by considering all GT for FP subtype assignment. Verified bird/cloud/antenna/aircraft counts: **Not found in repository**.

## 10. EXP001

### Goal

Establish a pretrained YOLOv8s baseline for single-class drone detection.

### Configuration

640, batch 8, 100 completed epochs; DUT dataset state not frozen. See section 7 and the experiment registry.

### Results

Validation CSV peak mAP50-95 0.57365; separately, saved test built-in mAP50-95 0.652. Custom test TP/FP 2,104/159 with the qualifications above.

### Conclusion

A functioning baseline and error-analysis starting point exist. Its test-driven analysis means the test set has already influenced development, and early size tables need replacement before final comparison.

## 11. EXP002

### Goal

Test higher input resolution to retain more visual information for small drones.

### Configuration

YOLOv8s pretrained; 960, batch 4, 100 completed epochs, AdamW auto-selected at initial lr 0.002 and momentum 0.9. Completed run is `notebooks/runs/04_experiments/EXP002_YOLOv8s_high_resolution/YOLOv8s_960-3/`.

### Results

Standalone built-in validation: P 0.97314548, R 0.91250919, mAP50 0.94625813, mAP50-95 0.65640845. Custom validation: TP 2,433 / FP 190 / FN 188, precision 0.92756386, recall 0.92827165, F1 0.92791762. Earlier test notebook records 2,297 predictions and 142 best-IoU FPs, but does not enforce unique GT assignment for that analysis.

### Conclusion

Higher resolution is associated with stronger validation metrics than the baseline training CSV, while batch and other run conditions also differ. Tiny misses and zero-overlap predictions remain. The first two EXP002 attempts are incomplete and must not be mistaken for the final run.

## 12. EXP003

### Goal

Improve tiny-UAV detection by adding higher-resolution detection features.

### Architecture

Ultralytics `yolov8s-p2.yaml`, with strides `[4, 8, 16, 32]`. Pretrained YOLOv8s weights are partially transferred. The notebook logs 219/437 transferred items at explicit load and 389/437 in training setup; these are different stages, not interchangeable measures of pretrained coverage.

### Configuration

960, batch 8, 100 completed epochs, AdamW auto-selected at lr 0.002 / momentum 0.9, patience 20, seed 0. Training dataset version relative to annotation repairs is not captured. Exact run: `02_datasets/DUT_Anti_UAV/runs/04_experiments/EXP003_YOLOv8s_P2_960/`.

### Results

Standalone built-in validation: P 0.97375489, R 0.92012682, mAP50 0.94909195, mAP50-95 0.66139291. Peak training CSV validation mAP50-95 is separately 0.66217 at epoch 83. Custom validation: TP 2,440 / FP 176 / FN 181, precision 0.93272171, recall 0.93094239, F1 0.93183120.

### Conclusion

Observed validation gain in tiny-object recall and overall F1, with medium/large regressions. Background FP is not improved. No completed EXP003 test-set or live evaluation was found.

## 13. EXP002 vs EXP003

### Overall metrics

Custom rows below share validation GT, confidence 0.25, matching IoU 0.50 and nominal image size 960. EXP002 explicitly uses NMS IoU 0.70 and batches of eight; EXP003 streams a directory and does not explicitly set NMS IoU. Different batching can change rectangular padding, so this is a useful saved-result comparison with unresolved inference parity.

| Metric / protocol | EXP002 | EXP003 | Change |
| --- | ---: | ---: | ---: |
| Custom validation GT | 2,621 | 2,621 | 0 |
| Predictions | 2,623 | 2,616 | -7 |
| TP | 2,433 | 2,440 | +7 |
| FP | 190 | 176 | -14 |
| FN | 188 | 181 | -7 |
| Precision | 92.7564% | 93.2722% | +0.5158 pp |
| Recall | 92.8272% | 93.0942% | +0.2671 pp |
| F1 | 92.7918% | 93.1831% | +0.3914 pp |
| Built-in standalone validation precision | 97.3145% | 97.3755% | +0.0609 pp |
| Built-in standalone validation recall | 91.2509% | 92.0127% | +0.7618 pp |
| Built-in standalone validation mAP50 | 94.6258% | 94.9092% | +0.2834 pp |
| Built-in standalone validation mAP50-95 | 65.6408% | 66.1393% | +0.4984 pp |

Built-in P/R and AP use the validator's operating-point/curve calculations; they are not the custom fixed-confidence P/R. Custom AP: **Not found in repository**. The built-in numbers come from notebooks 04/06 and 08, not the object-level CSVs. Training-epoch CSV metrics and final revalidation also differ slightly and remain separately labelled.

### Object-size recall and false negatives

| Size | GT | EXP002 TP | EXP002 FN | EXP002 recall | EXP003 TP | EXP003 FN | EXP003 recall | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny | 720 | 625 | 95 | 86.8056% | 638 | 82 | 88.6111% | +1.8056 pp |
| Small | 1,427 | 1,348 | 79 | 94.4639% | 1,351 | 76 | 94.6741% | +0.2102 pp |
| Medium | 315 | 304 | 11 | 96.5079% | 297 | 18 | 94.2857% | -2.2222 pp |
| Large | 159 | 156 | 3 | 98.1132% | 154 | 5 | 96.8553% | -1.2579 pp |

### False positives

| Geometric subtype | EXP002 | EXP003 | Change |
| --- | ---: | ---: | ---: |
| Zero-overlap / Background | 122 | 124 | +2 |
| Localization | 46 | 40 | -6 |
| Duplicate | 22 | 12 | -10 |
| Total | 190 | 176 | -14 |

Source: individual TP/FP/FN and size CSVs under `06_results/EXP002_object_level_error_analysis/val/` and `06_results/EXP003_object_level_evaluation/`. Recomputed counts agree with summaries. Comparison CSVs use rounded hard-coded EXP002 metrics, explaining tiny rounding differences from exact count-based deltas.

## 14. Architecture Modifications

| Modification | Proposed | Implemented | Trained/evaluated | Evidence of improvement |
| --- | --- | --- | --- | --- |
| Higher resolution, EXP002 | Yes | Yes; input change, not new architecture | 100 epochs; built-in and custom evaluation | Better recorded validation metrics; not isolated from other conditions. |
| P2 head, EXP003 | Yes | Yes, packaged Ultralytics architecture | 100 epochs; built-in and custom validation | Modest observed tiny/overall gain with size trade-offs; causality/significance unproven. |
| Hard-negative learning, EXP004 | Explicitly proposed in notebook 09 final markdown | Not found in repository | Not found in repository | Not found in repository |
| Faster R-CNN comparator | Filename suggests intent | Environment imports only | Not found in repository | Not found in repository |
| GAM / other attention / SPD-Conv | No executable proposal found in research source | Not found in repository | Not found in repository | Not found in repository |

P2 extends detection to stride 4 to expose finer features. It uses the library's supplied architecture; no custom attention/SPD layer implementation or project-owned P2 YAML was found. `07_figures/Model_arch.png` displays 640 input and 10,884,336 parameters / 39.7 GFLOPs from the pre-training model summary, while actual training uses 960 and the trained fused one-class P2 summary is 10,626,708 / 36.6 GFLOPs. The illustration must be labelled appropriately before thesis use; summary FLOPs are not a measured 960-resolution deployment cost.

## 15. Important Research Findings

- Small apparent drones dominate the original area analysis; Tiny has the lowest recall in the final pixel-based evaluation.
- Higher resolution and P2 provide a coherent error-driven progression, with preserved weights and numerical outputs.
- P2 gains 13 Tiny and 3 Small TP but loses 7 Medium and 2 Large TP, net +7.
- Reduced duplicates/localization explain the FP reduction; zero-overlap FP increases slightly.
- Annotation and dataset-version issues affect both reproducibility and error interpretation.
- Final custom GT tables are identical and their object partitions are internally consistent; this supports arithmetic integrity, not full inference reproducibility.

## 16. Current Best Model

**EXP003 `02_datasets/DUT_Anti_UAV/runs/04_experiments/EXP003_YOLOv8s_P2_960/weights/best.pt`** is the leading validation candidate by saved standalone mAP50-95, custom F1 and Tiny recall. EXP002 retains better Medium/Large recall and lower recorded inference latency. Baseline test metrics cannot be used to overturn this validation ranking because the splits/protocols differ. No model is yet established as the best deployed or independently tested system.

## 17. Problems and Limitations

| Severity | Issue and evidence | Implication |
| --- | --- | --- |
| Critical | Historical train/test duplicate recorded in notebook 10b; removal happens after prior analyses, without retraining evidence. | Current clean hashes cannot retroactively validate old test results; training snapshots are needed. |
| Critical | Test examples and errors are repeatedly inspected in notebooks 02, 03 and 05 before method development. | The existing test set is development-exposed; final unbiased claims need an untouched independent set or carefully qualified protocol. |
| Important | Current test contains 539 exact duplicate pairs: 2,200 files but only 1,661 unique byte contents; train has three pairs and val one, with differing labels in those four pairs. | Test samples are repeatedly weighted and not independent; annotate/de-duplicate at a versioned dataset-design stage and assess sensitivity, without silently changing historical results. |
| Important | Unversioned annotation changes, prediction-assisted label replacement, orphan backup TXT; notebook 01 and 10b counts differ. | An architecture-only conclusion is confounded unless training data are reconstructed or rerun on a frozen version. |
| Important | Validation `00991` has height zero; both saved FN tables contain it. | At least one reported miss is annotation-invalid. |
| Important | One labelled training negative, none in val/test. | Background alarm rate and deployment generalization are not established. |
| Important | EXP002 batch 4 versus EXP003 batch 8 during training; inference batches of eight versus directory stream; implicit EXP003 NMS setting. | Not a clean P2 ablation; standardize inference and optimization conditions. |
| Important | Early size notebook uses `detections > 0` and joins object tables on image only; totals inflate to 2,359. | Early size recall is superseded, not thesis evidence. |
| Important | Notebook 05/script use best-IoU per prediction without one-to-one assignment; baseline FP typing only sees unmatched GT. | Those FP analyses cannot substitute for the final TP/FP/FN protocol. |
| Important | Final saved tables omit prediction coordinates; EXP003 also omits prediction IDs. | Cannot replay complete matching/NMS from CSVs alone or audit semantic FP categories. |
| Important | Single seed, modest net gains, no repeated runs or uncertainty intervals. | Improvement is observed, not statistically established. |
| Needs verification | Sequence/scene near-duplicate leakage; exact hashes only test identical bytes. | Dataset independence is unresolved beyond exact duplicates. |
| Needs verification | `zip(image_paths, results)` in EXP003 assumes identical order; model path begins `\master\...` without a drive. | Prediction identity and reproducibility depend on environment/order assumptions. |
| Needs verification | Notebook 06 references undefined `fn_df`; saved outputs depend on missing interactive state. | Cannot reproduce all exploratory cells from a fresh kernel. |
| Minor | Floating-point boundary at val `00378`: direct normalized width × image width is 32, corner subtraction is 31.999999999999886. | Current method assigns Tiny; mathematically exact threshold would assign Small. Preserve definition and use a documented tolerance in future work. |
| Minor | Hard-coded rounded EXP002 comparison inputs; misleading printed EXP002 run path; empty workbook/stub notebooks. | Prefer source CSVs and actual trainer `save_dir`; documentation needs maintenance. |
| Minor | Architecture illustration uses pre-training parameters and 640 input. | Illustration is not trained-model complexity evidence at 960. |

No clear error was found in the core intersection/union implementation in `03_code/metrics.py` or final notebooks 07/09 for valid boxes. The final matcher is a reasonable confidence-sorted greedy one-to-one matcher; it is not claimed identical to Ultralytics AP matching. It does not enforce class equality, harmless for the current single class but unsuitable for a future multiclass extension without revision.

## 18. Remaining Work

### Essential before thesis completion

1. Freeze and document an adjudicated dataset version, including negatives, invalid/overflow boxes, backup-file exclusion and scene/sequence separation; preserve historical data rather than silently replacing it.
2. Reconstruct training provenance or rerun controlled baselines on the same frozen data. This audit did not authorize or perform those changes/runs.
3. Standardize inference settings and use one reproducible object-level evaluator; save every raw prediction with path, coordinates, score, class and model/dataset hashes.
4. Establish independent final evaluation, with both positive and background-only data, and retain validation for threshold/model selection.
5. Report AP, fixed-point P/R/F1, size recall, semantic FP review and uncertainty; separate protocols throughout the thesis.
6. Measure comparable batch-one end-to-end latency, FPS and memory on target hardware; existing inference-only timings do not establish real-time operation.

### Recommended experiments

The repository's next hypothesis is **EXP004: P2 plus hard-negative learning**. First establish a frozen, verified P2 control. Add only independently verified training negatives, keep evaluation negatives held out, and compare FP per image/time at a matched recall target while checking Tiny recall retention. Do not mine validation/test negatives into training. Repeat across seeds. A matched standard-P3/P2 ablation at 960 and a controlled resolution ablation are also needed to separate architectural from configuration effects.

### Optional improvements

Complete the Faster R-CNN comparator; investigate other feature/attention changes only after control experiments; collect external live-video scenes and examine temporal filtering. No such improvement is established by this audit.

## 19. Thesis Contributions So Far

An assembled single-class drone-detection workflow; trained YOLOv8s resolution baselines; a trained packaged P2 variant; quantitative object-level and size-specific error analysis; separation of geometric FP subtypes; discovery of annotation/negative-data limitations. A novel attention module, novel detector architecture, validated deployment system or proven state-of-the-art result is **not** supported. Using an existing P2 configuration is an experimental contribution here, not evidence of architectural novelty.

## 20. Experiment Registry

| Experiment | Model | Major Change | ImgSz | Epochs | Dataset | Evaluation | Main Result | Status |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| EXP001 | YOLOv8s | Pretrained baseline | 640 | 100 | DUT, historical snapshot unknown | Val training CSV; test built-in/custom | Test mAP50-95 0.652, rounded | Completed; protocol caveats |
| EXP002 initial | YOLOv8s | Higher resolution, batch 8 | 960 | 2 saved / 100 requested | DUT | Training validation rows | Peak saved mAP50-95 0.38139 | Incomplete |
| EXP002 attempt -2 | YOLOv8s | Batch 4 retry | 960 | Not found in repository | DUT | Not found in repository | No weights/results CSV | Incomplete |
| EXP002 final -3 | YOLOv8s | Higher resolution, batch 4 | 960 | 100 | DUT, snapshot unknown | Built-in/custom val; exploratory test | Val custom F1 0.927918 | Completed |
| EXP003 P2 | YOLOv8s-P2 | Stride-4 head, batch 8 | 960 | 100 | DUT, snapshot unknown | Built-in/custom val | Val custom F1 0.931831 | Completed; final test missing |
| EXP003 Faster R-CNN filename | Not instantiated | Intended comparator | Not found in repository | Not found in repository | Not found in repository | Environment check only | Not found in repository | Incomplete stub; ID collision |
| EXP004 | P2 retained in proposal | Hard-negative learning | Not found in repository | Not found in repository | Not found in repository | Not found in repository | Not found in repository | Proposed only |

## 21. Important Files

| File | Purpose | Importance |
| --- | --- | --- |
| `02_datasets/DUT_Anti_UAV/data.yaml` | Split/class configuration | Preserve with dataset manifest. |
| `runs/detect/04_experiments/EXP001_YOLOv8s_baseline/{args.yaml,results.csv,weights/best.pt,weights/last.pt}` | Baseline provenance | Essential. Brace notation denotes four actual files. |
| `notebooks/runs/04_experiments/EXP002_YOLOv8s_high_resolution/YOLOv8s_960-3/{args.yaml,results.csv,weights/best.pt,weights/last.pt}` | Completed EXP002 provenance | Essential. |
| `02_datasets/DUT_Anti_UAV/runs/04_experiments/EXP003_YOLOv8s_P2_960/{args.yaml,results.csv,weights/best.pt,weights/last.pt}` | Completed P2 provenance | Essential. |
| `notebooks/07_final_object_level_error_analysis.ipynb` | EXP002 final custom evaluation | Essential code/outputs. |
| `notebooks/08_EXP003_YOLOv8s_P2.ipynb` | P2 construction/training/validation | Essential. |
| `notebooks/09_EXP003_YOLOv8s_P2_Object_Level_Evaluation.ipynb` | P2 custom evaluation/comparison | Essential. |
| `notebooks/10b_final_dataset_integrity_check.ipynb` | Audit and mutation history | Essential; do not execute as a read-only audit. |
| `06_results/EXP002_object_level_error_analysis/val/summary.csv` and `recall_by_object_size.csv` | EXP002 summary | Preserve alongside TP/FP/FN/GT tables, not alone. |
| `06_results/EXP003_object_level_evaluation/EXP003_summary.csv` and `EXP003_recall_by_size.csv` | EXP003 summary | Preserve alongside detailed tables and comparison CSVs. |
| `03_code/evaluate_exp002.py` and `03_code/metrics.py` | Prediction export / IoU | Not a replacement for final one-to-one evaluator. Expected export absent. |
| `notebooks/03_baseline_error_analysis.ipynb` | Baseline object matching and qualitative FP notes | Preserve historical evidence with limitations. |
| `notebooks/01_dataset_exploration.ipynb` | Original dataset counts | Preserve to document version drift. |
| `02_datasets/DUT_Anti_UAV/labels/train/00579_backup_old.txt` | Old annotation | Preserve as historical evidence; exclude from paired counts. |

## 22. Reproducibility Notes

All relative paths in this report are repository-relative. Saved notebooks often assume working directory `notebooks/`, while scripts resolve paths from `__file__`. Several paths are machine-specific or use `Images` rather than `images`, which works on typical Windows filesystems but is not portable.

Final custom validation: confidence 0.25; match IoU 0.50; input 960; GPU 0; original-image pixel size bins; greedy descending-score assignment to the best unmatched GT. EXP002 sets NMS IoU 0.70 and batch 8. EXP003's current notebook leaves NMS implicit and relies on stream ordering. Missing/malformed label handling can silently return/skip GT in final notebooks, so a validated input manifest is essential.

Standalone built-in evaluation uses `model.val(..., imgsz=960)` without explicit confidence; AP and reported P/R must not be interpreted as a 0.25 fixed-confidence result. The EXP003 training confusion matrix visibly records 2,449 matched UAVs, 108 background-column predictions and 172 misses, different from custom 2,440/176/181. Preserve those values as a separate built-in plot protocol, not contradictory replacements for CSV totals.

Training settings common to the runs include seed 0, deterministic true, AMP, NMS setting 0.7, max_det 300, mosaic 1.0 with close_mosaic 10, horizontal flip 0.5, translate 0.1, scale 0.5, HSV 0.015/0.7/0.4, no rotation/shear/perspective/vertical flip/mixup/copy-paste/cutmix. YAML also lists `auto_augment=randaugment` and `erasing=0.4`; their presence in a shared settings file is not proof those classification-oriented options transformed detection inputs. Full settings are retained in the experiment registry and run YAMLs.

The audit read checkpoint ZIP/pickle opcodes for date/version only, without deserializing model objects. Saved dates: EXP001 2026-08-21T13:09:05+03:00; incomplete EXP002 initial 2026-09-08T19:52:12+03:00; completed EXP002 2026-09-09T01:08:53+03:00; EXP003 2026-09-17T23:44:47+03:00. Final stripped checkpoints have `epoch=-1`, so peak CSV epochs should not be presented as recovered checkpoint metadata.

Reported timing contexts differ: baseline test 3.6 ms inference/image; EXP002 standalone val 9.3 ms in notebook 04 and 8.65 ms in notebook 06; EXP003 standalone val 12.85 ms. Training-end val logs show 4.3 ms EXP002 and 7.6 ms P2. These are not controlled end-to-end FPS measurements. Final fused summaries: YOLOv8s 11,125,971 parameters / 28.4 GFLOPs; P2 10,626,708 / 36.6 GFLOPs. The pre-training P2 summary of 10,884,336 / 39.7 is a different class-count/fusion context.

## 23. Open Questions

- Which immutable dataset and exact labels did each model train on, and did earlier checkpoints see the duplicate test frame?
- Were annotation repairs made before P2 training, and were pseudo-labels independently reviewed?
- Does the modest gain persist with identical padding/NMS/batch settings, matched training conditions and multiple seeds?
- How much of geometric “Background” is genuine background versus missing labels or displaced UAV boxes?
- What are the FP rate on background-only scenes and end-to-end latency on live video?
- What is EXP003's performance on an independent untouched test set?
- Where are the dataset release/source/license, full dependency lock and raw prediction-coordinate exports?
- Which Faster R-CNN configuration was intended, and how will its experiment ID be disambiguated?

### Audit verification appendix

The audit independently parsed all saved CSV rows, recomputed TP/FP/FN totals and FP categories, checked unique `(image, gt_id)` partitions and compared EXP002/EXP003 GT tables. Both GT tables are identical; TP and FN partitions are complete and disjoint. Every saved TP IoU is >=0.50 (minima 0.5031749 / 0.5025679). Saved validation GT dimensions and size labels also agree with current labels when computed using the notebooks' corner-subtraction method. Direct multiplication differs at the single 32-pixel boundary case `00378`, as documented above.

Current integrity checks read labels, image headers and SHA-256 file hashes without changing data. Full image decoding, perceptual duplicate search, GPU inference and retraining were not performed. Representative result/architecture plots were visually inspected; not every dataset/prediction image was manually adjudicated. Literature PDFs were inventoried as reference assets; their claims were not used to establish local experimental results.

PIL header/`verify()` checks raised no errors for the current 9,999 images. This does not prove every image fully decodes or every annotation is semantically correct. Hashing found no current cross-split matches and the within-split counts reported in section 4. Duplicate checks used image bytes, not filenames; numerical image stems repeat across splits and are not themselves leakage evidence. Duplicate label comparison used stripped TXT content. No duplicate or label was removed or edited during this audit.
