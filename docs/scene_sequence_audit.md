# Gate 2: Scene/Sequence Independence Audit — Phase 1

The question is whether the same scene or recording contributes images to
Train and Validation/Test. Gate 1 exact deduplication cannot answer it.
This audit reads `DUT_Anti_UAV_Clean_v1` and writes all outputs outside that
frozen dataset. It does not change split membership or start training.

## Source recovery findings

The local Tracking source contains 20 folders (`video01` through `video20`)
and 24,804 extracted JPEG frames. Each folder has a matching GT file with
the same number of annotation rows. Original numbered filenames are contiguous
and one-based. Their integers identify extracted JPEG files, not a recovered
capture timestamp or a verified frame counter in an original video container.

The official [DUT Anti-UAV repository](https://github.com/wangdongdut/DUT-Anti-UAV)
distributes Detection as separate train/val/test archives and Tracking as
separate frames/GT archives. The [original paper, section III-A](https://arxiv.org/html/2205.10851#S3.SS1)
reports 5,200/2,600/2,200 Detection images and 20 Tracking sequences. The inspected
documentation does not supply a Detection-image-to-sequence crosswalk or a
scene-disjoint Detection splitting protocol. This is not proof that no such
mapping exists elsewhere, and does not mean all Detection images originated
in the 20 local Tracking sequences.

A fresh complete inventory found **zero byte-identical and zero decoded-RGB
identical matches** between the 24,804 originals and the 9,451 frozen images.
This rules out a direct exact-content crosswalk to this local source; it does
**not** rule out different frames from the same video or scene.

The flat Detection numbers are recorded as `numeric_export_index`. They are
never promoted to `frame_number`. Definitive `sequence_id`, `video_id`,
`clip_id` and `frame_number` remain blank for unresolved images. A pHash/SSIM
nearest neighbor is stored separately as an unverified candidate.
All 9,451 frozen image headers were inspected: none contains EXIF, and none
has an explicit sequence/video/clip/frame identity claim in the inspected
provenance metadata. The verified source-identity coverage is therefore 0/9,451.

The targeted Downloads and relevant local-folder search found correction
exports for `00579` and `00724`, but no original DUT Detection archives or VOC
XML collection within those searched scopes. Their XML/COCO paths, flat
filenames and upload/export dates do not identify original sequences. The
AntiUAV-RGBT conversion notebook addresses a different source path and must
not be treated as provenance for DUT Detection.

## Scene evidence in the frozen release

Fresh visual inspection confirms shared distinctive building scenes across
both split boundaries below. The files have different SHA-256 hashes.

| Train | Other split | Evidence |
| --- | --- | --- |
| `01858.jpg` | Val `00930.jpg` | Same red-brick courtyard, cylindrical stair tower, walkway and viewpoint. |
| `01025.jpg` | Test `01502.jpg` | Same orange facade, trees, courtyard and light pole; UAV location differs. |
| `00966.jpg` | Test `00393.jpg` | Same striped-brick building, glass stairwell, roof beams and viewpoint. |
| `00946.jpg` | Test `00392.jpg` | Same building scene as the previous pair. |
| `01230.jpg` | Test `00440.jpg` | Same tall residential building, neighboring low building and viewpoint. |

These are Codex visual-review decisions, not independently adjudicated human
ground truth. The full image comparisons and review notes are saved with the
audit. The four `review_scene_*` identifiers are local review labels; they are
not native video IDs. They are assigned only to the reviewed images, without
propagating guesses to nearby filenames or entire similarity components.

SSIM is recomputed for the reviewed pairs using OpenCV grayscale, resize to
256×256 with `INTER_AREA`, and `data_range=255`. It supplements visual evidence;
no SSIM threshold alone establishes a source frame number or recording identity.

The current split therefore **fails scene independence as defined by Gate 2**.
The native recording IDs remain unresolved. The five reviewed links are not
an estimate of total scene overlap; this phase is not exhaustive scene grouping.

## Artifacts and reproduction

Results are in `06_results/Gate2_scene_sequence_audit_v1_20261109/`:

- `report.json`: coverage, status and frozen-dataset identity before/after.
- `sequence_manifest.csv`: one row per retained image, with unresolved IDs explicit.
- `image_metadata.csv`: complete EXIF/provenance metadata scan, hashes and export indices.
- `original_frame_inventory.csv` and `sequence_inventory.csv`: originals and available source IDs.
- `historical_candidates.csv`: old similarity-based suggestions, all `verified=no`.
- `reviewed_scene_links.json` and image sheets: reviewed cross-split evidence.
- `source_discovery.json` and `official_sources.json`: searched scopes and external provenance.

The reusable audit is `python -m drone_detection.sequence_audit --help`.
Original full-frame hashing/decoding is recorded in the source inventory;
cached Clean pixel hashes are reused only when the corresponding byte SHA
matches the frozen manifest. Audit outputs are never written inside Clean_v1.

Do not relabel guessed videos as verified or use Detection filename counters
as chronological frame numbers. Further independence work needs a source
crosswalk or carefully validated scene/sequence grouping and a separately
versioned split. Keep Clean_v1 and historical results intact.
