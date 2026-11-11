# YOLO26 paper milestone — 2026-11-11

## Dataset evolution

- **Original:** DUT Anti-UAV before the reproducibility cleaning and split
  controls.
- **Clean_v1:** integrity cleaning, invalid annotation removal, Test duplicate
  removal, and SHA-256 freezing.
- **Clean_v2:** the immutable paper release, preserving Clean_v1 content while
  allocating scene groups across Train/Validation/Test.

## Clean_v2 status

| Split | Images | UAV objects |
|---|---:|---:|
| Train | 6,299 | 6,317 |
| Validation | 1,827 | 1,867 |
| Test | 1,325 | 1,377 |
| **Total** | **9,451** | **9,561** |

Gate 2 and Gate 3 are **PASS** for the pinned
`DUT_Anti_UAV_Clean_v2` release. The protocol is scene-group-disjoint. Native
sequence IDs and frame numbers are unavailable, so native sequence-level
independence is not claimed. The conservative 8,177 image statistic is
described as scene-linked/affected under the audit graph, not as independently
proven native-sequence leakage.

Clean_v1 and Clean_v2 are frozen and must not be modified. No Clean_v3 is
created by this milestone.

## Historical experiments

Previous YOLOv8 and exploratory YOLO26 results are historical motivation only.
They predate the frozen Clean_v2 paper protocol and are not final comparison
results.

## Current paper matrix

| ID | Configuration | Purpose |
|---|---|---|
| E0_Y26 | YOLO26s @ 960 | Official baseline |
| E1_Y26_P2 | YOLO26s-P2 @ 960 | Test the Tiny-UAV hypothesis |
| E2_Y26_P2 | E1 plus 512 matched random verified negatives | Negative-data control |
| E3_Y26_P2 | E1 plus 512 error-driven verified hard negatives | Proposed method |

All arms use Clean_v2, 100 epochs, physical batch 1, nominal batch 16,
MuSGD, standard NMS inference, and seeds 17/42/123. Test is locked. E2/E3
remain blocked until their negative pools are verified.

## YOLO26 preflight

The pinned preflight passed without loading real dataset images:

- E0 outputs strides 8/16/32.
- E1 outputs strides 4/8/16/32.
- E0 and E1 transfer the same YOLO26s backbone modules 0 through 10.
- MuSGD construction, synthetic forward/backward, one optimizer update, and
  EMA update passed.
- Diagnostic peak GPU memory was approximately 1.25 GB for E0 and 1.96 GB for
  E1-P2 on the local RTX 5050 Laptop GPU.
- `training_called=false` and `test_evaluated=false`.

The machine-readable result is
[`preflight.json`](../06_results/paper_yolo26/preflight.json), and the
protocol is [`paper_protocol_yolo26.md`](paper_protocol_yolo26.md).

## Next authorized action

The first and only authorized paper run is **E0_Y26 Seed 17**. Do not start
E1, E2, E3, or Test automatically after E0; stop for review.
