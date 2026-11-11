# YOLO26 FP review before fine-tuning

Evaluate the trained `04_experiments/YOLO26s_baseline-5/weights/best.pt` on
Validation, retaining the checkpoint and labels. Existing historical error CSVs
are Test-based and have empty confidence fields; they are not used to select the
new correction. No training is part of this workflow.

```powershell
$env:PYTHONPATH = "$PWD/src"
.venv/Scripts/python.exe -m drone_detection fp-review extract
```

Defaults: `imgsz=960`, confidence .25, matching IoU .50, predict IoU .70,
rectangular inference, one image per call, max_det 300. YOLO26 can use end-to-end
detection; the predict IoU parameter alone does not establish how postprocessing ran.
For the selected checkpoint in the current runtime, both model and Detect head
report `end2end=false`, despite `end2end=true` in its embedded architecture YAML.
This discrepancy is recorded as checkpoint/runtime metadata, not silently overridden.
The loader and checkpoint protocol record the actual model used. To analyze P2,
provide its checkpoint with `--model` and a new `--output` directory.

`predictions.jsonl` retains every returned prediction at the .25 operating point,
every GT box, per-image/label hashes and original prediction IDs. Matching is
confidence-ordered, class-aware and one-to-one. Every unmatched prediction is
exported to `false_positives.csv` with image, box corners, confidence and best IoU.
All FP crops and full-image context views appear under `review/`. Contact sheets
are `review_page_*.png`; red outlines mark FP boxes and green outlines mark GT.

Review each FP using both crop and context. Fill `semantic_type`, `confirmed`
(`yes` or `no`), `reviewer` and `notes`. Suggested labels: bird, vegetation,
background_structure, localization, duplicate, unannotated_UAV, uncertain. Keep
blurred or tiny flying objects uncertain when a UAV cannot be excluded. Geometric
Background/Localization/Duplicate are separate columns and are not semantic labels.
Codex visual review is recorded as such; it is not human ground truth. Human
confirmation is appropriate before using semantic counts as thesis annotations.

Rank confirmed categories separately from the uncertain count. Do not assume
the most frequent confirmed category covers all unreviewed/uncertain errors.
Annotation errors remain FPs under the fixed evaluation ledger; do not silently
turn them into model TPs by modifying labels. Zero-area GT is flagged and retained
as an unmatchable FN, preserving the object denominator. Out-of-image GT is also
flagged and retained. This is a diagnostic protocol, not a certified clean dataset.

After review, choose exactly one correction and record its rationale before
testing. Examples (choose one, no sweep):

```powershell
# Low-confidence distractors: one higher operating threshold.
.venv/Scripts/python.exe -m drone_detection fp-review correct --confidence 0.50
# Alternatively, confirmed repeated boxes: explicit extra class-aware NMS.
.venv/Scripts/python.exe -m drone_detection fp-review correct --extra-nms 0.50
# Alternatively, confirmed overly large boxes: one uniform 10% shrink.
.venv/Scripts/python.exe -m drone_detection fp-review correct --box-scale 0.90
```

The correction runs on cached predictions with no training. It rematches every
image and reports TP, FP, FN, precision, recall, F1 and GT size groups, including
lost true positives. It refuses a second correction result in the same output
directory. Keeping only the old FP rows would hide recall loss; this workflow
uses the complete prediction ledger instead.
The box shrink is applied to every prediction around its center, without using
GT to pick which boxes to change. Confidences and checkpoint weights stay fixed.
Recovered and newly missed GT IDs are recorded separately; net counts alone can
hide regression on previously correct detections. Corrected FP boxes are exported
to `corrected_false_positives.csv`.

Confidence filtering changes the operating point; it does not demonstrate a
model or AP improvement. The corrected result is exploratory on Validation.
The known cross-split scene overlap means it is not independent Test evidence.
Use an independently sourced Test only after the correction has been fixed.
If the correction fails, report why and investigate the confirmed cause before
considering one justified fine-tuning experiment.
