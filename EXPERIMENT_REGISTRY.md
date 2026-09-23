# Experiment Registry

Repository audit: 2026-09-20. No new training or inference was performed. Metrics are fractions unless shown as percentages. Paths are repository-relative. **Not found in repository** means no supporting artifact was located; it does not mean zero. Saved-run dates below are checkpoint metadata timestamps, not verified training start dates.

## Evidence and common protocol

Final individual object CSVs take precedence over rounded notebook comparison values. Training-epoch validation, standalone built-in validation, built-in test, custom fixed-threshold evaluation and exploratory per-prediction IoU analysis are separate protocols. `mAP50-95` is AP averaged over IoU thresholds, not AP at one threshold. Built-in P/R are not the same operating point as custom confidence 0.25 results.

All trained runs use Ultralytics 8.4.124, as confirmed by checkpoint metadata. Saved execution logs identify Python 3.11.0, Torch 2.11.0+cu128 and NVIDIA GeForce RTX 5050 Laptop GPU. Dataset configuration is `02_datasets/DUT_Anti_UAV/data.yaml`, class `0: UAV`; immutable per-run dataset versions are **Not found in repository**. Current data differ from the original exploration; see the master summary before drawing controlled experimental conclusions.

Common YAML training configuration: device 0, 100 requested epochs, pretrained, seed 0, deterministic true, AMP true, optimizer auto, configured lr0 0.01 / lrf 0.01 / momentum 0.937 / weight_decay 0.0005; warmup 3 epochs, warmup momentum 0.8, warmup_bias_lr 0.1; nbs 64; box/cls/dfl 7.5/0.5/1.5; rect false, cos_lr false, fraction 1, multi_scale 0; validation split val, conf null, iou 0.7, max_det 300. EXP001 has patience 100/workers 8; EXP002/EXP003 have patience 20/workers 4. These are configured values: auto optimizer can override lr0/momentum, as the EXP002/P2 logs explicitly demonstrate.

Common augmentation configuration: HSV h/s/v 0.015/0.7/0.4; translate 0.1; scale 0.5; horizontal flip 0.5; mosaic 1.0, close_mosaic 10; degrees, shear, perspective, vertical flip, mixup, copy_paste, cutmix and bgr are zero. Shared YAML also contains randaugment and erasing 0.4; their presence alone does not establish use in the detection pipeline. No experiment-specific attention/SPD augmentation or module was found.

Final custom size definition uses original-image pixel width and height recovered by subtraction of converted box corners: Tiny both <32; otherwise Small both <96; otherwise Medium both <256; otherwise Large. Validation GT bins are 720/1,427/315/159. This includes an invalid zero-height annotation and one floating-point boundary case. These categories differ from early normalized-area bins.

## EXP001 — YOLOv8s baseline

| Field | Verified value / interpretation |
| --- | --- |
| Objective | Establish a single-class pretrained drone detector. |
| Hypothesis | Baseline suitability; explicit formal hypothesis **Not found in repository**. |
| Model | YOLOv8s, Ultralytics 8.4.124. |
| Architecture | Standard YOLOv8s; no custom architecture code found. |
| Pretrained weights | `yolov8s.pt` in run args; exact resolved initial path **Not found in repository**. |
| Training settings | 640, batch 8, 100 requested/completed epochs; patience 100, workers 8; common settings above. Actual auto-selected optimizer/initial lr **Not found in repository**. |
| Dataset | DUT YAML; training uses train, epoch validation uses val; historical version unfrozen. |
| Model weight used | `runs/detect/04_experiments/EXP001_YOLOv8s_baseline/weights/best.pt`; last.pt also exists. |
| Saved checkpoint date | 2026-08-21T13:09:05.672258+03:00. |
| Evaluation method | Training CSV validation; notebook 02 built-in test; notebook 03 greedy one-to-one custom test. |
| Confidence threshold | Custom 0.25; built-in call does not explicitly set confidence. |
| IoU threshold | Custom matching 0.50; training args NMS 0.70; notebook 03 error-analysis predict call leaves NMS implicit. Correct-size notebook explicitly uses 0.70. |
| Precision / Recall | Built-in test 0.966 / 0.927, rounded. Custom derived from counts: 0.92973928 / 0.93719376. |
| mAP50 / mAP50-95 | Built-in test 0.951 / 0.652, rounded; custom AP **Not found in repository**. |
| TP / FP / FN | Custom test TP 2,104 / FP 159; FN **141 derived as 2,245−2,104**, not a saved FN ledger. |
| Predictions / GT | 2,263 / 2,245, test. |
| Recall by object size | Reliable final one-to-one result **Not found in repository**. Early merged size recalls are invalidated by image-only joins. |
| Inference time / FPS | Test built-in log: 3.6 ms inference/image, 1.1 ms preprocess and 0.9 ms postprocess. Measured end-to-end FPS **Not found in repository**. |
| Parameters / FLOPs | Saved fused one-class summary: 11,125,971 parameters, 28.4 GFLOPs; do not present this as independently measured deployment FLOPs. |
| Observed strengths | Strong saved test AP and functioning baseline; one-to-one prediction matching exists. |
| Observed weaknesses | Tiny-object/localization/background examples; early size analysis invalid; development used test data; historical exact train/test overlap. |
| Conclusion | Completed baseline with qualified historical metrics; final controlled baseline comparison remains necessary. |
| Status | Completed, with superseded exploratory size analysis. |

**Separate training-validation evidence:** `results.csv` has 100 rows. Maximum mAP50-95 row is epoch 98: P 0.95610, R 0.87257, mAP50 0.90885, mAP50-95 0.57365. Last epoch: 0.95367 / 0.87448 / 0.90994 / 0.57359. These are validation, not the test results above. Last CSV elapsed time 16,286.3 seconds is not a standardized inference benchmark.

**Files supporting the result:**

- `runs/detect/04_experiments/EXP001_YOLOv8s_baseline/args.yaml`
- `runs/detect/04_experiments/EXP001_YOLOv8s_baseline/results.csv`
- `runs/detect/04_experiments/EXP001_YOLOv8s_baseline/weights/best.pt`
- `notebooks/02_baseline_size_analysis.ipynb`, cells 4 and 10–17
- `notebooks/03_baseline_error_analysis.ipynb`, cells 0–2 and 8–27
- `notebooks/03_correct_size_evaluation.ipynb`, cells 0–3

## EXP002 initial attempt — `YOLOv8s_960`

| Field | Verified value / interpretation |
| --- | --- |
| Objective | Higher-resolution baseline attempt. |
| Hypothesis | More input detail may improve small-drone detection; inferred from resolution change and notebook 04's stated purpose. |
| Model / Architecture | Standard YOLOv8s, Ultralytics 8.4.124. |
| Pretrained weights | `..\05_models\pretrained\yolov8s.pt` in args. |
| Training settings | 960, batch 8, 100 requested epochs, **2 saved rows**, patience 20/workers 4; optimizer auto, configured lr0 0.01. Actual optimizer **Not found in repository**. |
| Dataset | DUT YAML, historical snapshot unknown. |
| Model weight used | Available `notebooks/runs/04_experiments/EXP002_YOLOv8s_high_resolution/YOLOv8s_960/weights/best.pt`; downstream final analyses use -3 instead. last.pt also exists. |
| Saved checkpoint date | 2026-09-08T19:52:12.485515+03:00; best checkpoint has epoch 0 metadata. |
| Evaluation method | Training-epoch built-in validation only. |
| Confidence threshold / IoU threshold | args conf null / NMS iou 0.7; custom thresholds **Not found in repository**. |
| Precision / Recall | Best mAP50-95 row, epoch 1: 0.87837 / 0.62915. |
| mAP50 / mAP50-95 | Same row: 0.70858 / 0.38139. |
| TP / FP / FN | **Not found in repository**. |
| Predictions / GT | Saved object-level totals **Not found in repository**. |
| Recall by object size | **Not found in repository**. |
| Inference time / FPS / parameters / FLOPs | Run-specific saved results **Not found in repository**. |
| Observed strengths | Checkpoints and two validation rows preserved. |
| Observed weaknesses | Only two of 100 requested epochs recorded; stop reason unknown. |
| Conclusion | Incomplete attempt, not EXP002's final trained model. |
| Status | Incomplete. |

Epoch 2 P/R/mAP50/mAP50-95 = 0.87696 / 0.66349 / 0.75266 / 0.37909. Higher epoch-2 mAP50 does not change the maximum-mAP50-95 row selection.

**Files supporting the result:** `notebooks/runs/04_experiments/EXP002_YOLOv8s_high_resolution/YOLOv8s_960/args.yaml`, `results.csv`, `weights/best.pt`, `weights/last.pt` and initial training JPGs.

## EXP002 second attempt — `YOLOv8s_960-2`

| Field | Verified value / interpretation |
| --- | --- |
| Objective / Hypothesis | Higher-resolution retry, batch reduced to 4; precise reason **Not found in repository**. |
| Model / Architecture | Standard YOLOv8s specified in args. |
| Pretrained weights | `..\05_models\pretrained\yolov8s.pt`. |
| Training settings | 960, batch 4, 100 requested epochs, patience 20/workers 4; optimizer auto, configured lr0 0.01. Completed epochs **Not found in repository**. |
| Dataset | DUT YAML; snapshot unknown. |
| Model weight used | **Not found in repository**; weights folder is empty. |
| Training date | **Not found in repository**. |
| Evaluation method | **Not found in repository**. |
| Confidence threshold / IoU threshold | args conf null / NMS iou 0.7; evaluated operating point **Not found in repository**. |
| Precision / Recall / mAP50 / mAP50-95 | **Not found in repository**. |
| TP / FP / FN / prediction count / GT count | **Not found in repository**. |
| Recall by object size | **Not found in repository**. |
| Inference time / FPS / parameters / FLOPs | **Not found in repository**. |
| Observed strengths | Configuration retained. |
| Observed weaknesses | No results.csv or checkpoint; only args.yaml and labels.jpg. |
| Conclusion / Status | Incomplete setup/attempt; no performance claim supported. |

**Files supporting the result:** `notebooks/runs/04_experiments/EXP002_YOLOv8s_high_resolution/YOLOv8s_960-2/args.yaml` and `labels.jpg`.

## EXP002 completed — `YOLOv8s_960-3`

| Field | Verified value / interpretation |
| --- | --- |
| Objective | Improve detection through higher input resolution. |
| Hypothesis | More input detail may recover small UAVs. |
| Model / Architecture | Standard YOLOv8s, Ultralytics 8.4.124; no new detection head. |
| Pretrained weights | `05_models/pretrained/yolov8s.pt`; explicit transfer log 349/355 items. |
| Training settings | 960, batch 4, 100 completed epochs, patience 20/workers 4. Log resolves auto to AdamW lr 0.002, momentum 0.9; common augmentations/settings above. |
| Dataset | DUT train/val; exact annotation snapshot **Not found in repository**. |
| Model weight used | `notebooks/runs/04_experiments/EXP002_YOLOv8s_high_resolution/YOLOv8s_960-3/weights/best.pt`; last.pt exists. |
| Saved checkpoint date | 2026-09-09T01:08:53.165104+03:00. |
| Evaluation method | Standalone built-in validation; custom one-to-one validation in notebook 07; earlier test best-IoU exploration in notebook 05. |
| Confidence threshold | Custom val/test prediction calls: 0.25. Built-in val call leaves confidence implicit. Exploratory notebook 06 also tries 0.01 without matched recovery counts. |
| IoU threshold | Custom matching 0.50; explicit NMS 0.70. |
| Precision / Recall | Custom val 0.9275638581776592 / 0.9282716520412057. Built-in val 0.9731454765538414 / 0.9125091939490722. |
| mAP50 / mAP50-95 | Built-in val 0.9462581294312052 / 0.6564084457227974; custom AP **Not found in repository**. |
| TP / FP / FN | Custom val 2,433 / 190 / 188. |
| Predictions / GT / Images | Custom val 2,623 / 2,621 / 2,600. |
| F1 | Custom val 0.9279176201372997. |
| Inference time / FPS | Standalone val 9.3 ms inference in notebook 04; 8.64536 ms in notebook 06; training-end val 4.3 ms. Controlled end-to-end FPS **Not found in repository**. |
| Parameters / FLOPs | Fused one-class saved summary 11,125,971 / 28.4 GFLOPs. |
| Observed strengths | Stronger recorded validation metrics than 640 baseline; complete object-level exports. |
| Observed weaknesses | 95 Tiny FN; 122 geometric Background FP; only one current training negative; test analysis not a complete matching ledger. |
| Conclusion | Strong high-resolution reference for P2, subject to dataset/configuration parity limitations. |
| Status | Completed; final independent evaluation still missing. |

| Validation size | GT | TP | FN | Recall |
| --- | ---: | ---: | ---: | ---: |
| Tiny | 720 | 625 | 95 | 0.8680555556 |
| Small | 1,427 | 1,348 | 79 | 0.9446391030 |
| Medium | 315 | 304 | 11 | 0.9650793651 |
| Large | 159 | 156 | 3 | 0.9811320755 |

FP subtypes: 122 zero-overlap Background, 46 Localization, 22 Duplicate. These are geometric labels, not verified counts of birds/clouds/etc.

**Separate training-validation evidence:** maximum training CSV mAP50-95 at epoch 90: P 0.97235, R 0.91225, mAP50 0.94572, mAP50-95 0.65526. Last row: 0.97190 / 0.90538 / 0.94542 / 0.65399. Training took 5.626 hours in the saved log. Training-end best revalidation rounds to 0.972/0.912/0.946/0.656. Standalone revalidation above must not be silently replaced by the peak epoch or last epoch.

**Separate exploratory test evidence:** notebook 05 predicts 2,297 boxes at 960/confidence 0.25/NMS 0.7, batches of 20. Best-IoU <0.5 yields 142 FP (98 zero overlap / 44 localization). Because unique GT assignment is absent, 2,155 remaining predictions are not verified distinct TP. Built-in test AP and full test TP/FN for this run: **Not found in repository**.

**Files supporting the result:**

- `notebooks/runs/04_experiments/EXP002_YOLOv8s_high_resolution/YOLOv8s_960-3/args.yaml`
- `notebooks/runs/04_experiments/EXP002_YOLOv8s_high_resolution/YOLOv8s_960-3/results.csv`
- `notebooks/runs/04_experiments/EXP002_YOLOv8s_high_resolution/YOLOv8s_960-3/weights/best.pt`
- `notebooks/04_train_exp002_high_resolution.ipynb`, cells 1–3
- `notebooks/05_exp002_evaluation_and_error_analysis.ipynb`, cells 1–10
- `notebooks/06_dataset_audit_and_error_driven_analysis.ipynb`, cells 16, 23–24
- `notebooks/07_final_object_level_error_analysis.ipynb`
- `06_results/EXP002_object_level_error_analysis/val/summary.csv`
- `06_results/EXP002_object_level_error_analysis/val/true_positives.csv`
- `06_results/EXP002_object_level_error_analysis/val/false_positives.csv`
- `06_results/EXP002_object_level_error_analysis/val/false_negatives.csv`
- `06_results/EXP002_object_level_error_analysis/val/ground_truth_objects.csv`
- `06_results/EXP002_object_level_error_analysis/val/recall_by_object_size.csv`

The reusable `03_code/train_exp002_high_resolution.py` defaults to matching requested settings but writes a different root-resolved experiment directory. `03_code/evaluate_exp002.py` would export `06_results/EXP002_high_resolution/exp002_test_predictions.csv`; that export was **Not found in repository**. Its best-IoU export is not the complete final matcher.

## EXP003 — YOLOv8s-P2 at 960

| Field | Verified value / interpretation |
| --- | --- |
| Objective | Recover tiny UAV detections using finer features. |
| Hypothesis | A stride-4 P2 head improves Tiny recall over standard YOLOv8s at 960. |
| Model | YOLOv8s-P2, Ultralytics 8.4.124. |
| Architecture | Packaged `yolov8s-p2.yaml`; detection strides 4/8/16/32. No project-owned attention/GAM/SPD implementation found. |
| Pretrained weights | `yolov8s.pt`; explicit load transfers 219/437 items; training setup separately logs 389/437. |
| Training settings | 960, batch 8, 100 completed epochs, patience 20/workers 4; auto AdamW lr 0.002/momentum 0.9; common settings above. |
| Dataset | DUT train/val; exact training label version **Not found in repository**. |
| Model weight used | `02_datasets/DUT_Anti_UAV/runs/04_experiments/EXP003_YOLOv8s_P2_960/weights/best.pt`; last.pt exists. |
| Saved checkpoint date | 2026-09-17T23:44:47.831905+03:00. |
| Evaluation method | Standalone built-in validation and custom one-to-one validation; no completed test/live result found. |
| Confidence threshold | Custom 0.25; built-in call leaves confidence implicit. |
| IoU threshold | Custom matching 0.50; notebook predict does not set NMS IoU explicitly (run args have 0.70). Effective inference config not separately exported. |
| Precision / Recall | Custom val 0.9327217125382263 / 0.9309423884013736. Built-in val 0.9737548948323734 / 0.9201268174887888. |
| mAP50 / mAP50-95 | Built-in val 0.9490919453879761 / 0.6613929126296838. Custom AP **Not found in repository**. |
| TP / FP / FN | Custom val 2,440 / 176 / 181. |
| Predictions / GT / Images | 2,616 / 2,621 / 2,600. Prediction-bearing images 2,478. |
| F1 | Custom val 0.9318312010693146. |
| Inference time / FPS | Standalone val inference 12.850335 ms/image, preprocess 3.03035, postprocess 0.763639; training-end val inference 7.6 ms. Measured end-to-end FPS **Not found in repository**. |
| Parameters / FLOPs | Trained fused one-class 10,626,708 / 36.6 GFLOPs. Pre-training 80-class summary 10,884,336 / 39.7; do not mix contexts or claim measured 960 cost. |
| Observed strengths | Best saved validation F1 and standalone AP among completed 960 runs; Tiny TP +13, Small TP +3; fewer duplicate/localization FPs. |
| Observed weaknesses | Tiny FN 82; background-labelled FP 124; Medium/Large recall drops; greater recorded inference cost; no independent test. |
| Conclusion | Supports an observed modest P2-associated gain, not proof of architecture-only or statistically robust improvement. |
| Status | Completed training and validation; controlled ablation/final test outstanding. |

| Validation size | GT | TP | FN | Recall | Change vs EXP002 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiny | 720 | 638 | 82 | 0.8861111111 | +1.8056 pp |
| Small | 1,427 | 1,351 | 76 | 0.9467414156 | +0.2102 pp |
| Medium | 315 | 297 | 18 | 0.9428571429 | -2.2222 pp |
| Large | 159 | 154 | 5 | 0.9685534591 | -1.2579 pp |

FP subtypes: Background 124, Localization 40, Duplicate 12. Overall differences versus EXP002: TP +7, FP −14, FN −7; P +0.515785 pp, R +0.267074 pp, F1 +0.391358 pp using full-precision source counts. Saved comparison CSVs use rounded EXP002 inputs, so their final decimals differ slightly. Matching IDs reveals 42 recovered and 35 newly missed objects, not merely seven objects changing status.

**Separate training-validation evidence:** maximum CSV mAP50-95 at epoch 83: P 0.97369, R 0.91950, mAP50 0.94912, mAP50-95 0.66217. Last row: 0.97143 / 0.92114 / 0.94730 / 0.65940. Training took 10.906 hours. Training-end best validation rounds to 0.974 / 0.920 / 0.949 / 0.662; standalone best validation rounds mAP50-95 to 0.661. Different contexts are preserved rather than selecting the largest number.

The run's saved `confusion_matrix.png` shows 2,449 UAV matches, 108 background-column predictions and 172 misses. Those are built-in confusion-matrix counts with separate settings/matching, not custom counts 2,440/176/181. The geometric Background FP count 124 is a subtype of custom FP, not the confusion matrix's entire background column.

**Files supporting the result:**

- `02_datasets/DUT_Anti_UAV/runs/04_experiments/EXP003_YOLOv8s_P2_960/args.yaml`
- `02_datasets/DUT_Anti_UAV/runs/04_experiments/EXP003_YOLOv8s_P2_960/results.csv`
- `02_datasets/DUT_Anti_UAV/runs/04_experiments/EXP003_YOLOv8s_P2_960/weights/best.pt`
- `notebooks/08_EXP003_YOLOv8s_P2.ipynb`, especially cells 4–11 and 13
- `notebooks/09_EXP003_YOLOv8s_P2_Object_Level_Evaluation.ipynb`
- `06_results/EXP003_object_level_evaluation/EXP003_summary.csv`
- `06_results/EXP003_object_level_evaluation/EXP003_true_positives.csv`
- `06_results/EXP003_object_level_evaluation/EXP003_false_positives.csv`
- `06_results/EXP003_object_level_evaluation/EXP003_false_negatives.csv`
- `06_results/EXP003_object_level_evaluation/EXP003_ground_truth.csv`
- `06_results/EXP003_object_level_evaluation/EXP003_recall_by_size.csv`
- `06_results/EXP003_object_level_evaluation/EXP003_fp_analysis.csv`
- `06_results/EXP003_object_level_evaluation/EXP002_vs_EXP003_overall.csv`
- `06_results/EXP003_object_level_evaluation/EXP002_vs_EXP003_size_recall.csv`
- `06_results/EXP003_object_level_evaluation/EXP002_vs_EXP003_fp_types.csv`

## Faster R-CNN baseline placeholder — filename reuses EXP003

| Field | Evidence |
| --- | --- |
| Objective / Hypothesis | Intended Faster R-CNN comparator suggested by filename; explicit research objective/hypothesis **Not found in repository**. |
| Model / Architecture | **Not found in repository**; no model instantiated. Backbone/FPN choice is unknown. |
| Training settings / Dataset / Model weight used | **Not found in repository**. |
| Evaluation method / Confidence threshold / IoU threshold | **Not found in repository**. |
| Precision / Recall / mAP50 / mAP50-95 | **Not found in repository**. |
| TP / FP / FN / prediction count / GT count | **Not found in repository**. |
| Recall by object size / Inference time / FPS / parameters / FLOPs | **Not found in repository**. |
| Observed strengths | TorchVision/CUDA environment availability recorded. |
| Observed weaknesses | Only one environment-check cell; experiment identifier collides with trained P2 EXP003. |
| Conclusion / Status | Incomplete placeholder; no Faster R-CNN result exists. |
| Files supporting the result | `notebooks/11_EXP003_Faster_RCNN_baseline.ipynb`. |

## EXP004 — proposed hard-negative learning

| Field | Evidence |
| --- | --- |
| Objective | Address remaining background false positives while retaining P2. |
| Hypothesis | Hard-negative learning may reduce background confusion; proposed, not tested. |
| Model / Architecture | Retain P2 according to notebook 09 final markdown; exact new configuration **Not found in repository**. |
| Training settings / Dataset / Model weight used | **Not found in repository**. |
| Evaluation method / Confidence threshold / IoU threshold | **Not found in repository**. |
| Precision / Recall / mAP50 / mAP50-95 | **Not found in repository**. |
| TP / FP / FN / prediction count / GT count | **Not found in repository**. |
| Recall by object size / Inference time / FPS / parameters / FLOPs | **Not found in repository**. |
| Observed strengths | Motivation is supported by 124 zero-overlap FP and negative scarcity. |
| Observed weaknesses | No curated negative dataset, trained model or result artifact found. |
| Conclusion / Status | Proposed only; not implemented or evaluated. |
| Files supporting the result | `notebooks/09_EXP003_YOLOv8s_P2_Object_Level_Evaluation.ipynb`, cell 26 markdown. |

## Assets and outputs that are not separate completed experiments

`notebooks/yolo26n.pt`, `notebooks/yolov8s.pt` and `05_models/pretrained/yolov8s.pt` are weight assets without additional local trained-run evidence. A filename alone does not establish a YOLO26 experiment. `notebooks/10_EXP003_Object_Level_Evaluation.ipynb` has no cells. `04_experiments/experiment_log.xlsx` contains an empty sheet. Multiple predict/val directories are evaluation artifacts, not automatically new experiments; some are empty. `07_figures/` illustrations and `09_papers/` PDFs do not establish implementation of their depicted/referenced methods.

## Comparison validity and next controlled experiment

EXP002 and EXP003 share saved validation GT tables and custom matching code structure, but their training batch sizes differ (4 vs 8), initialization transfer differs, dataset snapshots are absent, and inference batching/padding may differ. Both custom summaries include the zero-height GT. Final CSV counts were independently verified, but raw prediction coordinates are missing, preventing complete re-matching from exports alone. Single-seed results do not establish significance.

The strongest next evidence-building step is to freeze/adjudicate data and standardize evaluation before a controlled **P2 vs P2 + hard negatives** experiment, preserving an independent negative-inclusive test set. This is recommended future work only; no code/data fixes or training were performed in this audit.

The independent read-only hash audit also found **539 exact duplicate pairs within test** (2,200 files, 1,661 unique image contents; all paired label texts identical), three train duplicate pairs and one validation pair. The train/validation pairs have differing label text. Current cross-split exact overlap is zero, but that does not remove historical leakage recorded in notebook 10b. These findings further limit baseline/EXP002 test claims and the independence of validation samples; see `MASTER_PROJECT_SUMMARY.md`, section 4, for filenames and counts.
