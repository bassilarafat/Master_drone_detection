# Negative-study preparation and readiness

Evidence level DIAGNOSTIC for feasibility and PLANNED for the proposed model studies. Reviewed 20 November 2026.

## Existing design and implementation

E2 adds 512 verified source-matched random background crops to the E1 architecture. E3 adds 512 verified source-matched error-driven crops. Each arm starts from matched public initialization.

The existing crop helper uses 256×256 crops, excludes annotated UAV boxes plus a 16-pixel margin and requires explicit review before publishing. It checks source-image frequencies, crop count and byte identities. Its provenance contract is bound to legacy protocol.json, E1.yaml and seed-17 Train inference. The v3 entry point blocks E2/E3. Cross-fitted discovery and v3 integration need a reviewed implementation and prospective amendment.

[Approved design](../../configs/paper/analysis_plan_yolo26_v1.json) · [Existing crop helper](../../src/drone_detection/paper_negatives.py) · [v3 entry point](../../src/drone_detection/yolo26_workflow.py)

## Out-of-fold feasibility

Train contains 6,299 images, 1,064 confirmed scene groups and one conservative allocation block. The diagnostic assignment sorts groups by descending image count, ties by group ID, and places each whole group into the smallest current fold, ties A through D.

| Proposed fold | Train images | Scene groups |
| --- | --- | --- |
| A | 1575 | 265 |
| B | 1575 | 266 |
| C | 1575 | 267 |
| D | 1574 | 266 |

No confirmed scene group crosses the proposed folds. The explicit allocation-identity file contains 267,493 constraint rows connecting current Train images. **66,782 of these rows cross the proposed folds.** These rows include conservative similarity/source ambiguity. They do not count independently verified native videos, physical scenes or every possible cross-fold relation.

Four folds preserving the entire single allocation block are infeasible on this release. The diagnostic grouping does not establish recording independence. The prospective design must state how crossing conservative links are handled, whether eligible sources require restriction or whether an independent discovery pool is needed.

[Membership](proposed_train_folds.csv) · [Crossing examples](cross_fold_constraint_examples.csv) · [Machine-readable findings](negative_study_readiness.json)

## Before fitting E2/E3

1. Specify the discovery fold policy, allowed dependence, training budget, candidate score rule, crop geometry, exclusion margin and review policy in a new prospective version. Record exposure to the current P2 results.
2. Verify discovery models excluded each mined fold from their training. Preserve Train-only raw predictions, training-source lists and checkpoint hashes.
3. Review every proposed negative with full-image context. Record reviewer, time and uncertainty. Independent UAV-free verification remains outstanding.
4. Prepare 512 accepted hard samples and 512 verified random controls. Match source-image frequencies and crop dimensions. Reject duplicated crops and annotation-overlap violations.
5. Freeze the common bundle and hashes before E2/E3. Retain a fixed bundle across seeds.
6. Integrate E2/E3 in a new reviewed v3-compatible version. Check matched positives, augmentation, optimizer, schedule, nominal accumulation and actual update budget.
7. Evaluate every seed with the pinned Validation evaluator and report E3−E2 first. Secondary E2−E1 and E3−E1 can include added-data and update effects.
8. Acquire an independent verified negative-only evaluation population. Final external validation needs its own locked protocol.

The 36 visual FP examples belong to current Validation and are ineligible mining inputs. Historical basename prefixes do not change membership.

Current readiness is **blocked pending discovery-design resolution, verified crop bundles and integration**. Available preparation includes the blank review template, diagnostic folds and quantified constraints. Training efficacy and verified sample counts remain unmeasured.
