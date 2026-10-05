# Model definitions

Model YAML belongs here; downloaded weights remain in `05_models/pretrained/`
and training outputs remain in `04_experiments/`.

| Definition | Status |
| --- | --- |
| `yolo26s_p2.yaml` | Exact copy of the existing P2 definition; P2/P3/P4/P5 detection. |
| `yolo26s_gam.yaml` | New, untrained correction: Detect consumes the GAM output at layer 23. |
| `legacy/yolo26s_gam.yaml` | Historical graph: Detect consumes layers 16, 19, 22 and bypasses GAM. |
| `drafts/` | Preserved incomplete ideas; excluded from runnable experiment configurations. |

Construct custom models through `drone_detection.models.load_model()` so GAM
registration works on an unmodified Ultralytics installation. The adapter also
loads existing checkpoints that reference `ultralytics.nn.modules.gam.GAM`.
GAM's constructor receives the explicit YAML channel count without automatic
scaling; the current definition is intended for YOLO26 **s** at 512 channels.

The original `YOLO26s_GAM` results belong to the historical graph. Loading its
checkpoint preserves that graph: changing a YAML on disk does not rewrite a
saved checkpoint. The corrected definition needs a separate run and fresh
evaluation. Inserting GAM moves Detect from layer 23 to layer 24, so loading
baseline weights only transfers compatible keys; the Detect weights are not
automatically remapped. The attention implementation itself is preserved for
checkpoint compatibility and is not a claim of equivalence to a paper's GAM.
