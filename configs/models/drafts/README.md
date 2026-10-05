# Incomplete model drafts

These files preserve the existing ideas and are **not runnable architectures**.

- `yolo26s_p2_mamba.yaml` is the original empty placeholder. No Mamba layer,
  registration, or architecture exists in this project yet.
- `yolo26s_p2_hepan.yaml` has no backbone layers, references absent layer
  indices, and attempts to concatenate feature maps from different resolutions
  without resizing them. Its proposed fusion output also is not connected to
  Detect. A complete architecture and tensor-shape checks are needed before
  enabling this experiment.

The original files are retained at their previous locations for historical
notebook references. Do not select these drafts for training.
