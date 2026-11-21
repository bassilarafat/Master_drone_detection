# Controlled YOLO26: seed 17 Validation comparison

Checkpoint: last.pt at epoch 100, as prescribed by the v3 checkpoint policy.
Saved-export audit only; no new inference or Test access. Both runs and evaluations completed.
Verified current checkpoint hashes, raw prediction hashes, identical image/label/target/scene identities and evaluation settings.
Population: 1,827 Validation images, 1,867 annotated objects. Input 960; NMS IoU 0.70; maxDet 300; operating confidence 0.25; matching IoU 0.50; AP prediction floor 0.001.

| Metric | E0 | E1 | E1 minus E0 |
|---|---:|---:|---:|
| mAP50 (%) | 80.53 | 82.46 | +1.93 pp |
| mAP50_95 (%) | 52.02 | 54.01 | +1.99 pp |
| precision (%) | 87.29 | 87.63 | +0.33 pp |
| recall (%) | 76.16 | 78.15 | +1.98 pp |
| F1 (%) | 81.35 | 82.62 | +1.27 pp |
| TP | 1422 | 1459 | +37 |
| FP | 207 | 206 | -1 |
| FN | 445 | 408 | -37 |

| Size | GT | E0 recall (%) | E1 recall (%) | Difference (pp) | Recovered | Lost |
|---|---:|---:|---:|---:|---:|---:|
| Tiny | 715 | 64.20 | 69.51 | +5.31 | 55 | 17 |
| Small | 765 | 86.01 | 86.14 | +0.13 | 20 | 19 |
| Medium | 289 | 78.89 | 77.51 | -1.38 | 9 | 13 |
| Large | 98 | 78.57 | 80.61 | +2.04 | 4 | 2 |

## False positive categories

| Category | E0 | E1 |
|---|---:|---:|
| Background | 144 | 146 |
| Localization | 54 | 53 |
| Duplicate | 9 | 7 |
| ClassConfusion | 0 | 0 |
| InvalidBox | 0 | 0 |

## Interpretation and next steps

E1 improves overall recall and AP for seed 17, driven mostly by Tiny objects. Medium recall decreases. Object transitions are paired by image and GT index; recovered/lost counts are changes on individual targets, not just net totals.
False-positive categories are geometric. Semantic FP review remains unreviewed; there are no independently verified negative images, so background-only FPR is unavailable.
AP uses the pinned custom 101-point evaluator and is not full COCO crowd/area evaluation. These scores differ from Ultralytics training-log metrics because the evaluator differs.
One seed cannot establish training stability or statistical significance. Complete E0/E1 seeds 42 and 123 regardless of seed-17 outcome, then report paired differences and mean/sample SD (ddof=1).
Keep Test locked. The approved study records prior descriptive Test exposure; independent final confirmation needs the prospective external protocol.
Source exports: 04_experiments/controlled_yolo26_v3/{E0_Y26,E1_Y26_P2}/seed_17/evaluation/val/.
