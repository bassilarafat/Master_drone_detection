# Paper protocol: DUT Clean_v2, E0--E3

Status: **preparation only; no training, mining inference or final Test evaluation
has been performed**. Protocol date: 2026-11-10, Africa/Cairo. Machine-readable
controls are in [protocol.json](../configs/paper/protocol.json), each experiment
YAML, `environment.lock.json` and `CONTROL_LOCK.json`. Do not silently override
them; amendments require a reviewed, versioned protocol before the affected runs.

## Dataset readiness and identity

Use only `02_datasets/DUT_Anti_UAV_Clean_v2`:

- Release ID: `DUT_Anti_UAV_Clean_v2@ad46c518a7953ca3bafeeaa6d475cebe8fb196706cc078aa70b10d7e67a1a1d5`.
- MANIFEST SHA-256: `ad46c518a7953ca3bafeeaa6d475cebe8fb196706cc078aa70b10d7e67a1a1d5`.
- FREEZE SHA-256: `9d26198e83d537b96734c48de3220fc0c46646e15d81525d8036d6eb57c6c058`.
- [Gate 3 report](../06_results/paper/Gate3_20261110/gate3_report.json): PASS.
  All 9,451 images were fully decoded; every release payload was SHA-256 checked
  before and after the audit. No invalid/zero-area/out-of-bounds boxes, missing
  labels, orphan labels, duplicate annotations, crossing scene/allocation groups,
  loader caches, writable release files or changed files were found. Boundary
  tolerance is the release parser's 1e-9 normalized units; no smaller rounding
  excess was observed either. Test inspection was structural only, without a model.

The additional `runtime_preflight.json` checks JPEG end markers, EXIF orientation
and minimum dimensions without invoking an image-repair routine. This prevents
the installed loader from attempting an in-place JPEG repair and verifies the
original-pixel coordinate convention used by the unified evaluator.

| Split | Images/labels | UAV objects | Empty labels | Tiny | Small | Medium | Large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 6299 | 6317 | 1 | 3344 | 2082 | 671 | 220 |
| Validation | 1827 | 1867 | 0 | 715 | 765 | 289 | 98 |
| Test, locked | 1325 | 1377 | 0 | 577 | 544 | 192 | 64 |

There are 1,571 confirmed scene groups and 313 conservative allocation units.
The largest allocation unit contains all 6,299 Train images, comprising 1,064
confirmed scene groups. Validation has 321 scene groups/192 allocation units;
Test has 186/120. Conservative grouping takes priority over exact split/size
balance. This is scene-disjoint under the documented audit, **not native
sequence-disjoint**. Recording IDs remain unknown; labels inherit the frozen
annotation state. See [dataset methodology](dataset_clean_v2.md).

## Hypotheses and experiments

| ID | Architecture | Additional Train negatives | Main comparison |
| --- | --- | --- | --- |
| E0 | YOLOv8s, P3/P4/P5 | None | Controlled backbone-pretrained baseline |
| E1 | YOLOv8s-P2, P2/P3/P4/P5 | None | E1 minus E0: official P2 architecture variant |
| E2 | Same P2 as E1 | 512 randomly located, verified background crops | Control for adding negative data |
| E3 | Same P2 as E1 | 512 verified model-guided hard background crops | E3 minus E2: error-driven selection |

E0/E1 retain all 6,299 original Train images. E2/E3 retain them and add exactly
512 crops, making 6,811 Train samples and the same 6,317 annotated UAV objects.
Validation is identical in all arms. E2/E3 are independently trained from the
same public initialization, **not continued from E1**. E1 is used only to propose
the mining pool. This is one offline mining round, without feedback from Val/Test.

The E3--E2 contrast matches sample count, crop size, source-image frequency,
epochs, effective batch and optimizer update schedule. E2/E3 have more examples
and updates per 100 epochs than E0/E1: E3--E1 alone cannot attribute improvement
solely to hardness. No source image or positive label is removed to compensate.

## Initialization and architecture

Use the independently SHA-verified public COCO checkpoint from the
[Ultralytics repository at commit 8a9e1a55](https://huggingface.co/Ultralytics/YOLOv8/commit/8a9e1a55f987a77f9966c2ac3f80aa8aa37b3c1a):

- Local file: `05_models/pretrained/paper/yolov8s_coco_268e5bb5.pt`.
- SHA-256: `268e5bb54c640c96c3510224833bc2eeacab4135c6deb41502156e39986b562d`.
- The downloaded bytes match the publisher's LFS SHA/size. Provenance is recorded
  in `configs/paper/pretrained.provenance.json`. No existing weights are overwritten.
- Transfer only backbone modules **0--9** in every arm. Every transferred tensor
  must match in name/shape/value. Neck and detection parameters use the selected
  seed's standard initialization; the DFL projection retains its standard fixed
  bin weights. All backbones remain trainable.
- This is an explicitly **backbone-pretrained controlled baseline**, not a claim
  to reproduce default full-network COCO transfer. Index-based loading into the
  divergent P2 head is deliberately avoided.
- The [official P2 architecture](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8-p2.yaml)
  is copied locally, fixed to scale `s`, one UAV class, and SHA-pinned. E0 strides
  must be 8/16/32 and E1--E3 strides 4/8/16/32. The CPU initialization dry-run
  checks both and verifies identical pretrained backbone tensor hashes.

The official P2 variant also changes some neck/head widths; it is not a surgical
addition of one branch with every other tensor shape preserved. For example,
the final P5 C2f output is 256 channels at scale s, versus 512 in E0. The dry-run
counts 11,135,987 parameters for E0 and 10,637,236 for E1 (one UAV class).
Accordingly, E1--E0 estimates the effect of the **whole declared architecture
variant**, not the isolated causal effect of stride4. Fewer parameters do not
establish lower memory/latency cost for the dense P2 features; latency remains unmeasured. The separately authorized no-update GPU diagnostic
checks one forward/backward batch per model and is not a full training run.

Historical DUT checkpoints and previous project metrics are historical evidence;
they cannot initialize these runs or substitute for results on this release.

## Fixed training controls

| Control | Fixed value |
| --- | --- |
| Input | 960 x 960; aspect-preserving square letterbox; no multiscale |
| Epochs / stopping | 100; patience=0 disables early stopping; no resume |
| Optimizer | SGD, Nesterov implementation pinned by Ultralytics source hash |
| Learning rate | lr0=0.01; cosine decay; lrf=0.01, nominal final LR=0.0001 |
| Momentum / decay | 0.937 / 0.0005 |
| Physical / nominal effective batch | 1 image / 16 images, gradient accumulation=16 |
| Warmup | Disabled, to keep accumulation constant |
| Precision / determinism | FP32, AMP=false, TF32 disabled, deterministic=true |
| Device / loaders | Single GPU `0`, workers=0, cache=false |
| Seeds | 17, 42, 123; seed17 is the first run, never selected for best performance |
| Checkpoint used in paper | `last.pt` after epoch 100, in every arm and seed |
| Validation monitor | Native Ultralytics Val metrics only; `best.pt` is not selected |

Gradient accumulation does not turn BatchNorm's physical batch of one into a
batch of sixteen. Terminal partial batches can be smaller than the nominal
effective batch. The paired comparisons have equal sample counts/update schedules.

Augmentation is identical: HSV h/s/v=0.015/0.7/0.4, translation=0.1, scale=0.5,
horizontal flip=0.5. Vertical flip, degrees, shear, perspective, mosaic, mixup,
cutmix and copy-paste are zero; close_mosaic=0; auto_augment=None and erasing=0.
Detection loss weights are box=7.5, cls=0.5, dfl=1.5. No augmentation, optimizer,
batch or confidence search is part of this protocol. Exact remaining arguments
and installed implementation hashes are recorded in the pinned controls.

The [Ultralytics training configuration](https://docs.ultralytics.com/modes/train/)
documents `nbs`, accumulation, optimizer and warmup behavior. We explicitly choose
SGD rather than `optimizer=auto`, which can override a supplied learning rate.
The environment lock records Python/platform, package versions and the installed
trainer/metric/model source hashes. Deterministic settings do not imply identical
floating-point results across different hardware/software environments.

## Negative acquisition and verification

1. After E1 seed17 finishes, use its epoch100 `last.pt` for inference on **current
   Train only**. A basename such as `test_00310.jpg` can legitimately be in the
   new Train split; membership is checked against the frozen manifest/path, not
   the historical filename prefix. Val/Test are never mining sources.
2. Propose at most 1,024 hard candidates, highest score first, at most one per
   source image. Require confidence>=0.5 and a Background FP at matching IoU .5
   (zero IoU with all annotated targets). Localization and duplicate errors are
   not assigned negative labels.
3. Use a 256 x 256 original-pixel PNG crop around the candidate. It must be
   wholly inside the source and avoid **every GT box plus 16 original pixels**.
   Do not erase/inpaint an object, change a positive whole-image label, or make
   a partial annotated UAV disappear through cropping.
4. Explicitly inspect every selected crop and its source context, including
   zoomed inspection for tiny UAVs. An empty GT region is not proof of UAV absence.
   Mark exactly 512 selected hard rows `verified_no_uav`, with reviewer and
   timezone-aware timestamp. Reject UAV-containing or ambiguous crops.
5. Generate one uniformly sampled eligible random crop from **each approved
   hard source image**, seed17, same 256 x 256 size and GT margin. This is a
   source-matched random-background control; it does not rank locations by model
   score. Exact crop hashes must differ across controls. Inspect all 512 random
   crops independently and record the same verification fields.
6. If random candidates are rejected, preserve the old catalog and generate a
   versioned proposal with `--exclude-review`; the rejected hashes and fixed
   seed are recorded. Do not change size/count or use Val/Test to fill a shortfall.
   Insufficient verified samples block E2/E3 rather than silently weakening controls.
7. Only then publish the matched bundle. Proposal catalogs bind source/crop
   identities; review may edit verdicts/notes, not crop/provenance fields. The
   builder rechecks original source/label SHA, exact source-crop pixels, GT
   exclusion, review status, counts, sizes and source frequencies. Verified copies,
   empty crop labels, manifests and `BUNDLE_LOCK.json` are placed **outside
   Clean_v2**, marked read-only and bound by SHA. Every run records the bundle SHA.

The existing one empty Train label is inherited, not an independently verified
negative pool. Neither the random nor the hard control is automatically approved.
All positive full-image labels stay intact. The current E2/E3 templates are
therefore intentionally blocked pending E1 mining and explicit review.

## Unified evaluation, selection and Test lock

`python -m drone_detection paper evaluate` accepts only `train` or `val`.
There is no Test unlock flag. Structural Test readiness auditing is allowed;
model inference, threshold tuning, checkpoint choice and negative mining on
Test are not. A separate explicitly approved final-evaluation procedure must be
prepared after all decisions are fixed; this task does not implement or run it.

Every completed paper run is evaluated on Val using its fixed epoch100 checkpoint.
Inference uses 960 square letterbox, batch1, FP32, augment=false, rect=false,
class-aware NMS IoU=0.7, max_det=300 and raw confidence floor=0.001. "Raw" means
post-NMS predictions before the reporting threshold, not network logits.

- TP/FP/FN, micro Precision/Recall/F1: confidence>=0.25, matching IoU>=0.50,
  score-ordered one-to-one, class-consistent GT matching.
- mAP50 and mAP50-95: retain the 0.001 prediction floor, independently match at
  IoUs .50/.55/.../.95 and average the precision-envelope values at 101 recall
  points. This is a declared COCO-style AP calculation, not the full COCO
  crowd/area/maxDet100 evaluator. It need not numerically equal native trainer AP.
- Recall by size uses the same operating point and **GT area in original image
  pixels**: Tiny <32^2; Small [32^2,96^2); Medium [96^2,256^2); Large >=256^2.
  These project bins are not COCO AP-small definitions.
- Automatic FP subtypes: Background (IoU=0), Localization (0<IoU<.5), Duplicate
  (an already matched same-class GT at IoU>=.5), ClassConfusion, InvalidBox
  (finite degenerate predictions, counted as FP rather than discarded).
  Semantic classes such as bird/vegetation/building are not guessed from IoU;
  `fp_semantic_review.csv` starts UNREVIEWED and requires separate adjudication.
- FP/image = total FP divided by **all** evaluated images, including those with
  no detections. Every image occurs in the raw JSONL and image-level CSV.
- Background-only FPR = verified UAV-free images with at least one FP divided by
  all verified UAV-free images. Zero eligible images yields **null/N/A**, never
  zero. An additional annotation-empty diagnostic is explicitly distinguished
  from verified-background FPR. Val/Test have no empty labels, so a separate
  independently verified held-out negative benchmark would be needed to support
  that generalization claim. Never recycle Train mining crops as such a benchmark.

Exports: `predictions.jsonl`, `predictions.csv`, `objects.csv` (including FN),
`images.csv`, `summary.json`, `recall_by_size.csv`, `fp_subtypes.csv`,
`fp_semantic_review.csv`, and identity-bound `evaluation.json`.
Unsupported AP/recall/size/background denominators are null. Precision with no
predictions uses the explicit convention zero; F1 is 2TP/(2TP+FP+FN), or null when
that denominator is zero. No-detection images are not lost.

Report every preregistered seed, its metrics and paired contrasts; use seed means
and sample standard deviations, not the best seed. If confidence intervals are
added, resample evaluation allocation units rather than treating video-like
frames as independent. No Test numbers are produced during this stage.

## Directory layout and commands

```text
configs/paper/{protocol.json,CONTROL_LOCK.json,environment.lock.json,E0.yaml,...,E3.yaml}
04_experiments/paper_v1/E0/seed_17/PLAN.json
04_experiments/paper_v1/E0/seed_17/{runtime_data/,run/,run.json,evaluation/val/}  # future
04_experiments/paper_v1/{E1,E2,E3}/seed_{17,42,123}/
04_experiments/paper_v1/negative_pool/{verification_template.csv,STATUS.json}
04_experiments/paper_v1/negative_pool/{verified/,matched_bundle.json,BUNDLE_LOCK.json} # future
06_results/paper/{Gate3_20261110/,dry_runs/,tests/}
```

Runtime YAMLs contain Train/Val lists and **no Test entry**. All settings,
label caches, derived images, logs and outputs remain outside the frozen dataset.
Image caching is disabled; the project additionally suppresses Ultralytics label
cache writes into frozen roots. Full release SHA verification surrounds training
and evaluation, including failure paths. A real-loader fixture test checks that
this suppression works; it does not use Clean_v2 images or train a model.

Safe commands (no training):

```powershell
.\.venv\Scripts\python.exe -m drone_detection paper dry-run --config configs/paper/E0.yaml
.\.venv\Scripts\python.exe -m drone_detection paper dry-run --config configs/paper/E1.yaml
.\.venv\Scripts\python.exe -m drone_detection paper dry-run --config configs/paper/E2.yaml
.\.venv\Scripts\python.exe -m drone_detection paper dry-run --config configs/paper/E3.yaml
.\.venv\Scripts\python.exe -m drone_detection paper model-dry-run
```

**Only after explicit user approval**, start the first E0/E1 runs with:

```powershell
.\.venv\Scripts\python.exe -m drone_detection paper train --config configs/paper/E0.yaml --seed 17 --execute-training
.\.venv\Scripts\python.exe -m drone_detection paper train --config configs/paper/E1.yaml --seed 17 --execute-training
```

Those commands train 100 epochs and then run the unified Val evaluation. They
have not been executed. Repeat with seeds 42 and 123 according to the fixed
plan; the execution flag is required and is not a Test unlock. A separate no-update GPU preflight uses one real augmented Train batch,
forward and backward only, sequentially for E0 then E1. It creates no optimizer
and saves no checkpoint; it cannot establish full-run optimizer/EMA memory cost. If hardware cannot support
the fixed recipe, revise the common protocol before comparisons, rather than
changing only one arm.

After approved E1 training, future Train-only mining commands are:

```powershell
.\.venv\Scripts\python.exe -m drone_detection paper evaluate --config configs/paper/E1.yaml --checkpoint 04_experiments/paper_v1/E1/seed_17/run/weights/last.pt --split train --output 04_experiments/paper_v1/negative_pool/E1_train
.\.venv\Scripts\python.exe -m drone_detection paper mine-negatives --train-evaluation 04_experiments/paper_v1/negative_pool/E1_train --output 04_experiments/paper_v1/negative_pool/hard_v1
# Explicitly review hard_v1/review.csv before continuing.
.\.venv\Scripts\python.exe -m drone_detection paper random-negatives --hard-review 04_experiments/paper_v1/negative_pool/hard_v1/review.csv --output 04_experiments/paper_v1/negative_pool/random_v1
# Explicitly review random_v1/review.csv before continuing.
.\.venv\Scripts\python.exe -m drone_detection paper freeze-negatives --hard-review 04_experiments/paper_v1/negative_pool/hard_v1/review.csv --random-review 04_experiments/paper_v1/negative_pool/random_v1/review.csv --output 04_experiments/paper_v1/negative_pool/matched_bundle.json
```

No command in this mining sequence has been run. Required outstanding work:
user approval for E0/E1; subsequent E1-based mining and negative verification;
review of the GPU preflight and fixed recipe; semantic FP adjudication once real
predictions exist; and an independently specified final Test evaluation later.


## Pre-training review amendment (revision 2, 2026-11-10)

**Training approval is pending review of this file and E0/E1 YAMLs.** E0/E1 are
conditionally approved; E2/E3 remain blocked. Run E0 seed17 first, review its
completion and unified Validation results, and only then run E1 seed17. Never
run the two training jobs concurrently. The no-update GPU diagnostic is the
only authorized forward/backward work at this stage; it is not approval to
start either 100-epoch command. Its report is in
`06_results/paper/pretraining_review_20261110/`.

The requested final seeds are **17, 42, 123**. This changes the previous third
seed 2026 before any paper run; revision 1 controls and its unstarted plans are
preserved in `06_results/paper/controls_history/`. No hyperparameter or dataset
change accompanies this amendment. Report paired seed results and mean +/-
sample SD across all three seeds, never a selected best seed.

Tiny GT objects make up **52.94% (3344/6317)** of Train, **38.30% (715/1867)**
of Validation, and **41.90% (577/1377)** of Test. These are object proportions,
not image proportions. Preserve Clean_v2 and its scene independence; do not
resplit to equalize size bins. Interpret size-specific recall alongside this
distribution shift. Improved Tiny recall is a hypothesis, not an assumed result;
report any FP, FN, precision or other size-recall tradeoffs even if mAP increases.

The paired comparison table will contain mAP50, mAP50-95, Precision, Recall, F1,
Tiny/Small/Medium/Large recall, geometric Background/Localization/Duplicate FP,
FN, FP/image, and latency. AP uses the confidence floor 0.001; operating-point
counts use confidence 0.25 and matching IoU 0.50. Both use class-aware NMS IoU
0.70, max_det=300, and the same unified evaluator. Native Ultralytics monitor
metrics must not be substituted for any of these comparison cells. The geometric
Background count is not a visually verified background false-alarm claim.

Latency must be measured separately after training on the same GPU and same
fixed sequence of all Validation images for each model: batch1, square960,
FP32, no augmentation, prediction floor0.001, NMS0.70/max_det300. Warm up with
50 repetitions of the first Validation image, then time 3 full passes using
CUDA synchronization around each call. Report median and p95 of per-image
inference plus NMS time, excluding disk decode, CPU preprocessing, host/device
transfer and metric/export time; record runtime/hardware and exact scope. GPU
preflight timings are diagnostics and must never fill the paper latency row.
No latency inference or Validation prediction has been run in this review.

Before approving E2/E3, acquire separate **Background Validation** and
**Background Test** sets outside Clean_v2. Verify every image is UAV-free by
manual review with reviewer/timestamp; retain source provenance and image SHA.
Audit exact and near-scene overlap with all Clean_v2 splits, training/mining
negatives, and between the two new sets. Freeze their independent releases and
counts before further model selection. Background Validation may support
selection only after protocol approval; Background Test remains locked until
the final evaluation, together with the existing Clean_v2 Test. Background
sources/counts are not available yet; no independent benchmark has been made.
Never turn Train negative crops into a held-out background benchmark. The one
annotation-empty Train image is not a certified benchmark. Background-only FPR
is N/A on the current Validation/Test sets. Visual review of FP/source context
remains necessary because zero IoU can also reflect missing annotations.
