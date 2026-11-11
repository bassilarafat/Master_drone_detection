# DUT_Anti_UAV_Clean_v1

This release freezes the current source annotation state; it does not rewrite
historical experiments. Original images, labels, checkpoints and run outputs
remain untouched. The exact creation time in UTC and Africa/Cairo is recorded
in the release's `cleaning_report.json` and `FREEZE.json`.

## Release decisions

| Decision | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Source images | 5,199 | 2,600 | 2,200 |
| Exclude conflicting-label exact duplicates | 6 | 2 | 0 |
| Exclude whole image with invalid annotation | 0 | 1 | 0 |
| Exclude redundant, identically labelled exact copies | 0 | 0 | 539 |
| Frozen images | **5,193** | **2,597** | **1,661** |
| Frozen UAV objects | **5,238** | **2,617** | **1,706** |

The invalid sample is Validation `00991.jpg`. Its label contains one zero-height
box and one valid box. Both image and label are excluded from active splits
pending human adjudication. Deleting just the invalid row could leave an
unlabelled target; no replacement box is guessed.

The conflicting exact-duplicate groups are Train `00004/00014`, `00020/00158`,
`00248/00700`, and Validation `00002/00007`. All eight members are excluded.
Annotations are not merged or selected using model predictions.

For the 539 Test pairs, both byte hashes and decoded RGB pixel hashes match,
and numeric annotation multisets match. One copy, first by filename, is kept.
Whitespace, numeric formatting and row order do not create a false conflict;
row multiplicity is preserved during comparison. The general builder also
excludes every member of an exact cross-split group; none were found here.

`labels/train/00579_backup_old.txt`, all label caches, and original `runs/` and
`visual_check/` artifacts are omitted. Only eligible image/label pairs are
copied. Files retained in the release preserve source bytes. Historical
normalization evidence (33 clipped boxes, one unresolved invalid sample) is
copied under `provenance/` and included in the release manifest.

## Create and verify

Creation is deliberately separate from training and refuses an existing target:

```powershell
.\.venv\Scripts\python.exe -m drone_detection.dataset_snapshot create --source 02_datasets/DUT_Anti_UAV --destination 02_datasets/DUT_Anti_UAV_Clean_v1
```

After creation, this command reads all frozen files and verifies their SHA-256
hashes. It does not modify the release:

```powershell
.\.venv\Scripts\python.exe -m drone_detection dataset-verify --dataset 02_datasets/DUT_Anti_UAV_Clean_v1
```

Each release includes:

- `data.yaml`: the new version's own dataset definition.
- `source_manifest.csv`: hashes of source image/label assets and their caches.
- `excluded_samples.csv`: each excluded image/label, reason and original label text.
- `excluded_artifacts.csv` and `duplicate_decisions.csv`: orphan/cache and duplicate decisions.
- `cleaning_report.json` and `DATASET_CARD.md`: dated decisions and scientific limits.
- `MANIFEST.csv` and `FREEZE.json`: payload hashes and the immutable release identity.

The builder copies files independently (no hardlinks), verifies every copy,
and rehashes the source before publishing. All released files are then marked
read-only. Changing metadata, removing a file or adding a cache fails verification.
Read-only flags protect against accidental file writes; they are not a security
boundary. Do not edit or regenerate this version. Future corrections require a
new directory/version such as `DUT_Anti_UAV_Clean_v2`.

The compact [external release record](../configs/datasets/DUT_Anti_UAV_Clean_v1.release.json)
also records the manifest/freeze hashes, creation time, counts and independent
verification results outside the ignored dataset directory.

## Experiments

Existing experiment configurations and historical results are unchanged.
New experiment configurations must explicitly use:

```yaml
data: 02_datasets/DUT_Anti_UAV_Clean_v1/data.yaml
```

Use the project `python -m drone_detection train --config ...` workflow. It
verifies the release before loading a model and after training, records the
identity in the run's `experiment.json`, forces image caching off, and suppresses
Ultralytics label-cache writes inside the release. A dry run also verifies the
release. Outputs must be outside the dataset; multi-device training is rejected
because a fresh distributed process would bypass the in-memory cache guard.

Direct notebook calls to Ultralytics `.train()` or `.val()` bypass these project
guards. Do not use them on this release without the same before/after checks
and cache-write suppression. No training is started by creating this release.
`dataset-normalize --write` is blocked for frozen versions.

## Scientific interpretation

This cleanup preserves existing split membership. Exact deduplication does not
resolve known shared Train/Validation scenes, recover complete source-video
identities, or undo historical Test use during development. This release alone
must not be described as a scene-independent benchmark. A future disjoint
split or unseen-video evaluation needs its own immutable dataset version.
Historical metrics used older labels and repeated Test images; recompute all
new comparisons on the same release and identify the version in the paper.
