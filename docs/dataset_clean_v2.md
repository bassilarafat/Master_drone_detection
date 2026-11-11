# DUT_Anti_UAV_Clean_v2

Release status: fresh Gate 2 **PASS**, before first publication. No detector
training was performed. The version is scene-group-disjoint under the declared
protocol, **not sequence-disjoint**. Native video/clip IDs, timestamps and original
frame numbers remain unresolved. Numeric source filenames are export indices.

## Source and unchanged annotations

All 9,451 images and 9,561 objects are retained from frozen Clean_v1. Image and
label bytes are unchanged; filenames receive old-split prefixes and whole groups
are resplit. `dataset_manifest.csv` and `dataset_changes.csv` retain the mapping.
Clean_v1 manifest SHA-256: `d2225406e2ce663121d878552ccadd63593c49ffb9a3c90ad0251d2ba805b408`.
Its complete snapshot was independently checked before publication.

The 539 redundant Test copies, conflicting duplicate-label cases, zero-height
annotation case and orphan/backup cleanup were handled in Clean_v1. This release
does not reintroduce removed images or edit the inherited labels. Retained-label
quality is inherited, not newly human adjudicated.

## Whole-pool retrieval and background verification

Official DINOv2 ViT-B/14 inference covers every image without fine-tuning. Code
commit: `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; checkpoint SHA-256:
`0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`.
Images retain aspect ratio and are letterboxed to 518 by 518 with ImageNet-mean
padding. CLS768 and valid-region mean-patch768 are independently L2-normalized.

Construction uses exact global cosine kNN (60 neighbors per descriptor), pHash
neighbors and documented prior links: 459,118 pairs. The fresh audit recomputes
global kNN at 120 neighbors per descriptor plus pHash40 over all 9,451 images,
producing 1,106,070 pairs. pHash and background SSIM are auxiliary signals.

Background RootSIFT masks GT UAV boxes with an eight-pixel feature-scale margin,
uses a maximum long side of 640 and up to 1,024 keypoints. Automatic scene edges
require mutual Lowe .75 matches, RANSAC3px, at least 24 inliers, inlier ratio .55,
hull coverage .10 in both images, six cells in both 4x4 grids, median error at
most 1.5px, p95 at most 3.5px and a plausible non-reflected homography. Full
settings, feature-index hashes and software provenance accompany the release.

Previously confirmed paths are explicitly transitive, not fabricated direct
edges. Every pair crossing the **final new boundaries** has direct geometric
evaluation. Unchanged image/label SHA identities allow local-feature and direct
pair-metric reuse. Newly exposed pairs are evaluated before the gate. The GPU
matcher refines exact L2 candidates and falls back to direct distances near
uncertain floating-point margins; saved CPU/GPU comparisons found no decision
differences in 48,480 checked pairs.

The unified candidate/review ledgers include every expanded k=120 retrieval,
including within-split pairs. Before the final grouping, all remaining global
candidates were evaluated. Every global candidate has direct geometry or an
explicitly confirmed transitive path. No unexamined expanded candidate is labeled
as a geometric negative. Final boundary candidates all have direct evaluation.

Targeted visual inspection of filename neighborhoods added one source-context
pair outside the global retrieval pool; it also received direct geometry.
The final unified ledger therefore contains 1,106,071 pairs. Filename adjacency
only proposed images for inspection; it did not supply a video/sequence ID or
establish a physical scene. The targeted review sheets and rationale are saved.

## Review, repair and conservative allocation

Codex inspected component representatives and selected borderline/cross-boundary
pairs. These are assistant visual decisions, not independent human ground truth.
Sheets, selections and pair rationales are saved. The first automatic PASS was
superseded by an actual visual FAIL before publication. Subsequent failures and
repairs remain in `release_audit_history.json` with their original reports.

Policy version 5 was fixed before the final gate and before any training. It
co-allocates pairs with either DINO cosine at least .85; weak local patches
with at least six inliers, ratio .50 and low reprojection error; partial distributed
geometry; and declared low-texture ambiguity. Repeated logos may indicate shared
source context but **do not confirm a physical scene**. They impose conservative
allocation constraints only. All 329 images with fewer than 16 background
keypoints share one conservative allocation family. This does not assert a
common scene. No images are deleted to improve the gate.

The broad .85 retrieval constraint deliberately also co-allocates semantically
similar but physically distinct scenes. It favors independence conservatively
and can reduce scene diversity across splits. It is not a scene-confirmation
threshold or a recovered native-video label. The allocation report quantifies
the resulting split and object-size distribution changes.

Confirmed edges form **1,571 scene groups**. Conservative links
produce **313 allocation blocks**, each wholly within one
split. Largest block: 6299 images. A block may
contain several confirmed scenes. No failed gate is overwritten or hidden.

## Final allocation

Seed 17; initial multistart whole-block balancing followed by minimum-change
whole-block repairs. Independence takes priority over exact image targets.
Balancing considers images, UAV objects and original-pixel object areas:
Tiny <32 squared; Small [32 squared,96 squared); Medium [96 squared,256 squared);
Large at least 256 squared. These project bins are not COCO AP-small metrics.

| Split | Images | UAV objects | Tiny | Small | Medium | Large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 6299 | 6317 | 3344 | 2082 | 671 | 220 |
| Validation | 1827 | 1867 | 715 | 765 | 289 | 98 |
| Test | 1325 | 1377 | 577 | 544 | 192 | 64 |
| Total | 9451 | 9561 | 4636 | 3391 | 1152 | 382 |

One annotation-defined negative is retained; Validation/Test are not a dedicated
background-only false-alarm benchmark. Full deviations are in the allocation
report.

## Final Gate 2 and freeze

Final directly checked boundary pairs: **282,293**.
All gate checks are zero: exact SHA overlap; crossing scene groups/allocation
blocks; confirmed Train-Val, Train-Test and Val-Test links; invalid labels;
unresolved high-risk crossings; crossing source-review constraints; missing pair
evaluations; and unbound feature identities. The gate is bound to the current
dataset manifest and split assignment SHA-256 values.

After this PASS and final visual review, files are sealed with `MANIFEST.csv` and
`FREEZE.json`, marked read-only and published once. The external identity is
`configs/datasets/DUT_Anti_UAV_Clean_v2.release.json`. Test is not edited after
publication; later corrections require a new version. Clean_v1 is unchanged.

Visual retrieval cannot prove that every possible unseen scene relation was
recovered. This is a protocol-qualified scene-disjoint claim, not proof of native
sequence independence. Historical project checkpoints may already have seen
images now in Test and **must not initialize paper experiments**. Future training
must start from public pretrained weights and use only this new Train split.
Experiment protocols/configurations are a separate later step.
