# EXP004: one targeted P2 fine-tuning experiment

Initialize from the trained EXP003 YOLOv8s-P2 checkpoint, retaining all four
detection strides (4/8/16/32). No architecture search or new layers.

Historical validation: TP 2440, FP 176, FN 181. Tiny/Small/Medium/Large FN:
82/76/18/5. These saved metrics are context; use a fresh baseline for comparison.

The initial action is preparation, a read-only audit, train mining and small
dry runs. Do not start fine-tuning while any data check is unresolved. Historical
CSVs are copied with hashes into `historical_baseline/`; the originals are never edited.

Run from the repository root with the project environment:

```powershell
$env:PYTHONPATH = "$PWD/src"
.venv/Scripts/python.exe -m drone_detection targeted audit
.venv/Scripts/python.exe -m drone_detection targeted mine --limit 64
.venv/Scripts/python.exe -m drone_detection targeted analyze --limit 32
.venv/Scripts/python.exe -m drone_detection.targeted mine
```

Analysis exports an object ledger (TP/FN with GT size; FP with predicted size),
size recall and summary. Fixed settings: 960 pixels, confidence .25, NMS .70,
matching IoU .50, max_det 300, one image per inference call. Matching is
confidence-ordered, class-aware and one-to-one. Sizes use original-image width
and height: both <32 Tiny, else both <96 Small, else both <256 Medium, else Large.
FP categories are geometric; they do not identify birds, clouds or trees.

Mining accesses train only. Background FP with confidence >=.50 become crops
with context, minimum crop dimensions 32 pixels and no intersection with any
annotated UAV **expanded by a 16-pixel safety margin**. Missing labels and malformed/zero-area labels fail explicitly;
adjudicate those labels before running. The historical validation ledger includes
a zero-height annotation, so its counts cannot be assumed identical after correction.
Do not silently drop invalid objects or change evaluation labels between runs.

Inspect the crops listed in `06_results/EXP004_targeted/mining_train/candidates.csv` and mark `approved`
as `yes` in `candidates.csv` only for verified non-UAV crops. Zero GT overlap is
insufficient to establish a true negative: an unannotated UAV can still be present.
Leave uncertain crops unapproved. Keep original train images and labels intact.
Review a sample visually for unannotated UAVs, recording crop IDs, decision and
reviewer rationale. All crops used as negatives require a positive review; a sample
alone cannot verify every crop. Directory files absent from the completed manifest
are not training candidates (an interrupted earlier mining pass can leave files).

The audit decodes every image, validates labels, hashes both file bytes and decoded
RGB pixels, and screens every cross-split pair with 63-bit pHash distance <=8,
then 256-pixel grayscale SSIM >=.95. CSVs enumerate candidates. This screen can
miss related scenes with camera motion; it cannot certify video independence.
Historical inferred video mappings are warning evidence only. Supply a verified,
complete source-video manifest (`split,image,video_id,verified`) with `verified=yes`
and source provenance for every image. Do not invent video IDs from numerical image
names, or assert independence merely because hashes differ. Shared videos across
splits block the experiment. The existing Test has historical Train–Test near
duplicate evidence and within-Test duplicates; it cannot currently be called independent.

```powershell
.venv/Scripts/python.exe -m drone_detection targeted dry-run
# After resolving label issues and split provenance, on a frozen version:
.venv/Scripts/python.exe -m drone_detection targeted audit --sequence-manifest path/to/verified_sources.csv
.venv/Scripts/python.exe -m drone_detection targeted analyze
```

`dry-run` copies eight original train images with their labels and up to eight
reviewed train negatives into a separate tiny dataset. It loads the real Ultralytics
dataset, collates batches and runs inference at 960. No optimizer or backward step
is created; the original checkpoint hash must remain unchanged. `prepare` and
`finetune --dry-run` also build the full training input without invoking training.
Use a new output root after changing the dataset or candidates; completed outputs
and the one training-run directory are protected against accidental overwrite.

```powershell
.venv/Scripts/python.exe -m drone_detection.targeted finetune
```

One run: 10 epochs, batch 4, AdamW lr .0001, lrf .1, seed 0, deterministic,
no mosaic or mixup, AMP disabled to avoid the library's separate AMP-check model
download. Add one copy of each reviewed negative to all original train
images. Empty crop labels encode background. This is a fresh fine-tune from best.pt,
not resume. Record checkpoint hash, candidate provenance and all training settings
in protocol.json. An existing experiment directory blocks accidental reruns.
This evaluates a targeted fine-tuning strategy (data plus documented fine-tuning
settings); it does not isolate the causal effect of negative data alone.

The real training command also requires `--recall-tolerance` and `--map-tolerance`
in fraction units, chosen before the run (for example `.005` means 0.5 percentage
points, only if that limit is agreed). They are saved in protocol.json and used
in the comparison's preservation checks; no implicit degradation tolerance is assumed.

Training checks that the audit passes and matches current image/label hashes, the
source manifest remains unchanged, the small dry runs completed, the real data
dry run passed with the same candidate CSV, and the original checkpoint has a
fresh complete validation ledger and AP export. It loads only the original EXP003
checkpoint and verifies stride/parameter count after training.

Compare baseline_val and finetuned_val summary.json and recall_by_size.csv:
FP reduction is the primary objective; report precision, recall, F1, FN and
Tiny recall alongside it. `comparison.json` records before/after deltas and the
historical targets: Background FP <124, Total FP <176, Tiny FN <=82. These targets
are experimental objectives, not guaranteed outcomes. Report Recall and mAP50–95
deltas; define acceptable degradation numerically before running the experiment.
No automatic tuning or repeated random trials are performed if targets fail.
Do not claim improvement unless measured. Both ledgers
must use the same adjudicated dataset and protocol. Current runtime Ultralytics
8.4.157 differs from historical training 8.4.124; rerunning the baseline controls
the evaluation runtime difference. Validation is used for checkpoint selection,
so it is not an independent final-test estimate. Test data never enter mining,
crop review or training. AP is computed by a separate built-in evaluation at
confidence .001, batch 1, NMS .70; AP sweeps confidence and is not mAP calculated
from the .25-threshold ledger. Use this identical AP protocol for both checkpoints.

After selecting the final checkpoint using validation, run the independent Test
once, only when the source-video audit is clear:

```powershell
.venv/Scripts/python.exe -m drone_detection targeted test --model 06_results/EXP004_targeted/experiment/P2_hard_negatives/weights/best.pt
```

Do not silently resplit the existing dataset or repair the zero-height validation
annotation. Those changes need a separately documented data version, and fresh
baseline/final evaluation on that same version. Keep 176/181/82 as historical
evidence rather than substituting measurements from an altered dataset.
