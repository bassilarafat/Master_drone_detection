# Negative-mining design review

Updated 21 November 2026. Evidence level is DIAGNOSTIC for design and source audits. Candidate-yield checks are EXPLORATORY. E2/E3 efficacy remains PLANNED.

[Research dashboard](RESEARCH_PROGRESS_DASHBOARD.md) · [Verified controlled results](CONTROLLED_P2_THREE_SEED_RESULTS.md)

## Researcher decision

The researcher requires **full allocation-block separation** between discovery-model training and the negative-discovery population. The direct-neighbor-buffered internal design was rejected on this ground. The original E0/E1 endpoint comparison remains the completed study described in its own report.

Clean_v2 Train contains 6,299 images and 1,064 confirmed scene groups inside one conservative allocation block. Holding out any part of that block while excluding the entire block leaves **zero** Train images for the discovery model. Internal four-fold mining cannot satisfy the selected requirement on this release.

[Decision record](../configs/paper/negative_mining_design_decision_20261121.json) · [Retired buffered draft](../configs/paper/negative_mining_buffered_oof_v1.draft.json)

## Completed internal feasibility audit

The audit combined explicit identity constraints, uncertain candidate links and the full low-texture family of 329 images. The resulting direct-risk graph contains 327,232 distinct image pairs and 61,688 distinct cross-scene group edges. A diagnostic graph-community assignment used Louvain with seed 17 and resolution 1, followed by whole-community image-count balancing. Detection scores played no role in constructing either assignment.

| Proposed mining fold | Balanced assignment retained Train images | Graph-community assignment retained Train images | Graph-community mining images |
| --- | --- | --- | --- |
| A | 1,495 | 3,829 | 1,576 |
| B | 1,435 | 3,562 | 1,575 |
| C | 1,472 | 3,523 | 1,574 |
| D | 1,114 | 4,105 | 1,574 |

These retained-image counts follow direct-neighbor buffer exclusion. Every proposed boundary has zero direct risk edges, while all images still belong to the same original allocation block. Both internal designs are therefore ineligible under the researcher’s full-block requirement. The files remain diagnostic records.

Of all 6,299 Train images, 6,214 geometrically admit at least one 256×256 crop outside annotated boxes and the 16-pixel margin. This screen establishes geometric space. Visual UAV-free verification remains a separate requirement.

[Design audit and source hashes](../06_results/negative_mining_design_20261120/design_audit.json) · [Fold comparison](../06_results/negative_mining_design_20261120/fold_comparison.csv) · [Geometric source eligibility](../06_results/negative_mining_design_20261120/source_eligibility.csv)

## External source investigation

The local antiuav_rgpt directory contains 160 Train, 67 Validation and 91 Test clip directories. Only Train video payloads enter the source investigation. Validation/Test directory names reserve potentially related filename families. Grouping by the first two underscore-delimited tokens is a conservative filename heuristic and does not establish native recording identity.

The Train annotation files contain 149,528 frame entries, of which 7,336 have existence flag 0. Excluding every Train clip whose filename family also occurs in Validation or Test leaves eight clips in family 20190926_193515. These clips contain 7,919 video frames and 2,398 annotation-absent frame entries. Only those predeclared absent frames enter the candidate pool. These numbers describe annotation flags and metadata, with independent UAV-free verification still pending.

The inspected sample uses the first, middle and last annotation-absent frame from each of the six remaining clips that contain absence entries. All 18 frames are visible-modality nighttime imagery, often with camera overlays. Source diversity is limited to one conservative filename family. This source’s ability to address the building-background errors observed on Clean_v2 remains an empirical question.

The official Anti-UAV repository describes the multimodal benchmark and publishes modality-specific annotation start-frame corrections. Its metadata is a reference for provenance checking. The local archive identity and equivalence to a specific corrected release remain unverified. [Official Anti-UAV README](https://github.com/ucas-vg/Anti-UAV/blob/main/README.md)

[Sequence inventory](../06_results/external_negative_source_audit_20261120/train_sequence_inventory.csv) · [Source audit](../06_results/external_negative_source_audit_20261120/source_audit.json) · [Candidate frame indices](../06_results/external_negative_source_audit_20261120/annotation_absent_candidates.csv) · [Visual source sample](../06_results/external_negative_source_audit_20261120/source_screen.png)

## Sample overlap checks

The cached public DINOv2 ViT-B/14 checkpoint, preprocessing and released descriptor identities were checked against the existing extraction record. Across the 18-frame sample, the highest either-descriptor cosine against the 9,451-image Clean_v2 reference pool was 0.507627. No sampled cosine reached the conservative 0.85 allocation threshold.

The geometric sample check evaluated 4,404 distinct cross-reference pairs proposed by CLS120, patch120 and pHash40 per sample frame. Existing RootSIFT, Lowe matching, RANSAC and version-5 risk rules produced zero confirmed or risk-allocation links in this sample. All sample frames yielded at least 16 keypoints. Camera-overlay features contribute to these counts.

These are partial screening results. Full-pool coverage and independent negative review determine whether the source can progress.

[Sample retrieval](../06_results/external_negative_source_audit_20261120/sample_similarity.json) · [Sample geometry](../06_results/external_negative_source_audit_20261120/sample_geometry/summary.json)

## Full-pool audit and yield check

The full-pool audit completed all 2,398 predeclared candidate frames. It constructs a joint index with the 9,451 released images, recomputes global k120 retrieval per descriptor and pHash40 in both directions, adds every cross-boundary cosine relation reaching 0.85, and directly evaluates all resulting boundary candidates. The declared low-texture family rule is applied as well. Existing allocation blocks are retained for checking new source crossings.

The full audit has completed. Its result establishes compliance with the finite declared screen. Native recording independence requires further source provenance.

If the source passes that allocation screen, a frozen exploratory diagnostic uses E1 seed 17, epoch-100 last.pt, confidence 0.50, input 960, NMS 0.70 and a maximum of one proposed 256×256 crop per source frame. Candidate confidence ranks proposals, with a cap of 1,024. The count of independently verified negatives remains unknown until review. An insufficient yield is retained under the recorded rules.

[Full-pool progress](../06_results/external_discovery_pool_audit_20261121/progress.json) · [Frozen exploratory yield plan](../configs/paper/external_negative_yield_check_v1.json)

## Completed full-pool findings

The joint index contains 11,849 images. Retrieval produced **10,431 cross-boundary candidate pairs**, all evaluated by the existing geometric pipeline. It produced **0 confirmed or policy-risk links** and **0 external low-texture frames**. The candidate family therefore occupies a separate component under the declared allocation screen. This is a protocol-qualified claim over the indexed pool.

Freshly decoded external pixels had **zero exact overlaps** with the released reference images, using decoded-pixel hashes reused by matching original image-byte identities. A CPU/GPU parity sample checked 128 pair decisions with zero differences.

The geometric pipeline found 9,623 pairs with too few mutual matches and 808 pairs with insufficient reference texture. No pair supplied enough support for a RANSAC fit. These outcomes supply bounded screening evidence. They do not prove that all possible relationships were recovered or establish native recording identities.

[Full audit](../06_results/external_discovery_pool_audit_20261121/pool_audit.json) · [Explicit block witness](../06_results/external_discovery_pool_audit_20261121/allocation_separation_witness.json) · [Exact-overlap check](../06_results/external_discovery_pool_audit_20261121/exact_overlap_check.json) · [Frame provenance](../06_results/external_discovery_pool_audit_20261121/frame_provenance_check.json) · [CPU/GPU parity](../06_results/external_discovery_pool_audit_20261121/cpu_gpu_parity.json)

## Completed exploratory yield check

At the frozen confidence threshold 0.50, E1 seed 17 returned **1 detection(s)** over the 2,398 predeclared external Train frames. The one-crop-per-source rule produced **1 candidate crop(s)** for review. A matched study requires 512 accepted hard crops per arm, so the current source is **insufficient at the declared settings**.

The single crop shows a dim blue luminous patch at the frame edge. Its identity remains uncertain after assistant inspection of the crop and full context. Independent UAV-free adjudication is pending. The detection count is a proposal-yield diagnostic. Ground-truth negative status and an operational false-positive rate were not established by this check. The source is one nighttime filename family, with limited diversity relative to the Clean_v2 building-background errors.

The result is preserved under its predeclared threshold. Further source or threshold changes require a new prospective version with this exposure history. The tested cohort is insufficient under version 1. Additional source development must retain this result and declare subsequent changes prospectively.

[Frozen rules](../configs/paper/external_negative_yield_check_v1.json) · [Yield outcome](../06_results/external_negative_yield_check_20261121/progress.json) · [Complete raw predictions](../06_results/external_negative_yield_check_20261121/predictions.jsonl) · [Candidate review](../06_results/external_negative_yield_check_20261121/review.csv)

## Operating-threshold yield check, version 2

Version 2 was frozen after observing version 1 and before generating version-2 predictions. It changes confidence from 0.50 to the existing official operating threshold 0.25. The model, 2,398-frame source cohort and crop rules stay fixed. This is exploratory source development, with prior exposure recorded in the plan.

| Version | Confidence | Source frames | Detections | Proposed crops | Accepted independently verified negatives |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.50 | 2,398 | 1 | 1 | Not established |
| 2 | 0.25 | 2,398 | 4 | 4 | Not established |

All four version-2 crops remain **UNCERTAIN** after assistant inspection of crops and full contexts. Two are neighboring frames 659/660 showing the same blue edge feature; the others show a dim red point and a luminous streak. These are four frame proposals, with correlated observations, and cannot be called four independent negative events. Independent review is pending. Both checks fall below the required 512 hard crops even before verification.

The filename-family reservation is an engineering choice that protects potential external evaluation families. The inventory contains 27 Train filename families; another 4,938 absence-flag entries were excluded by that reservation. Their allocation separation and candidate yield have not been established. The present result applies to the tested cohort at the two declared thresholds. It does not establish that all local or external sources are unusable. A broader source policy would need a new prospective version, exposure accounting and explicit effects on the reserved external evaluation population.

[Version-2 frozen rules](../configs/paper/external_negative_yield_check_v2.json) · [Outcome](../06_results/external_negative_yield_check_v2_20261121/progress.json) · [Raw predictions](../06_results/external_negative_yield_check_v2_20261121/predictions.jsonl) · [Candidate review](../06_results/external_negative_yield_check_v2_20261121/review.csv) · [Context sheet](../06_results/external_negative_yield_check_v2_20261121/context_review_sheet.png) · [All Train source families](../06_results/external_negative_source_audit_20261120/source_family_options.csv)

## Dependencies before E2/E3

1. Establish a sufficient allocation-separated discovery cohort through a prospectively declared source policy. The tested nighttime cohort passed the finite boundary screen but supplied only one/four proposals at confidence 0.50/0.25.
2. Review candidate crops and full contexts independently, with source/video exposure and uncertain cases recorded.
3. Prepare 512 accepted hard crops and 512 verified random controls matched by source-frame frequencies and dimensions.
4. Prospectively amend source eligibility if external discovery images are adopted. The approved analysis plan currently specifies current Clean_v2 Train sources.
5. Integrate the verified bundle with a reviewed executable E2/E3 protocol, including actual optimizer-update parity.
6. Preserve an independent negative-only evaluation population and a separately locked final external-validation protocol.

The investigative outputs support preparation and traceability. They do not supply an E2/E3 performance measurement.

## Reproducibility

The tools analyze_negative_mining_folds.py, inspect_external_negative_sources.py, screen_negative_source_similarity.py, check_external_sample_geometry.py and audit_external_discovery_pool.py reproduce the design and source checks into new output directories. The separately frozen version-1 and version-2 plans drive check_external_negative_yield.py. Full-pool inputs, source code and derived files have delivery hashes. Candidate review decisions are editable records with their own provenance.
