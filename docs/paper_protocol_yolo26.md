# YOLO26 paper protocol: Clean_v2, E0--E3

Status: **preflight only**. No YOLO26 training or Test evaluation is authorized
by this revision. The YOLOv8 protocol remains unchanged and is historical
context; these experiments use the immutable `DUT_Anti_UAV_Clean_v2` release.

## Experiments

| ID | Architecture | Purpose |
| --- | --- | --- |
| E0_Y26 | YOLO26s, P3/P4/P5 | Official baseline |
| E1_Y26_P2 | YOLO26s-P2, P2/P3/P4/P5 | Test the Tiny-UAV hypothesis |
| E2_Y26_P2 | E1 plus 512 random negatives | Negative-data control |
| E3_Y26_P2 | E1 plus 512 error-driven hard negatives | Proposed method |

E2 and E3 are blocked until E0/E1 are completed and the negative pool is
prepared and independently reviewed. Test remains locked for every arm.

## Initialization fairness

Both E0 and E1 construct their local architecture from the same seed and
transfer only YOLO26s COCO backbone modules 0 through 10 inclusive from
`yolo26s.pt`. Neck and detection-head parameters are initialized from that same
seed in both arms. No YOLO26 P2 pretrained checkpoint is assumed.

The official topology gives E0 strides 8/16/32 and E1 strides 4/8/16/32.
The local YAMLs and checkpoint are SHA-pinned in
`configs/paper/protocol_yolo26.json`.

## Fixed controls

- Dataset: `DUT_Anti_UAV_Clean_v2`
- Input: 960, physical batch 1, nominal batch 16
- Epochs: 100
- Seeds: 17, 42, 123; seed 17 is first
- Optimizer: MuSGD, `lr0=0.01`, momentum `0.937`, weight decay `0.0005`
- Standard Ultralytics NMS inference path for all primary metrics
- NMS-free inference is a separate deployment/latency experiment
- No training, negative mining, or Test inference before preflight approval

MuSGD support was verified in the installed Ultralytics 8.4.157 trainer. The
preflight still constructs the optimizer and performs one synthetic update
before any real dataset or training command can be authorized.

## Required preflight

The preflight must pass model construction, backbone transfer, expected
strides, synthetic forward/backward, MuSGD construction, one optimizer update,
EMA update, and memory accounting for both architectures. It must report
`training_called=false` and `test_evaluated=false`.

Run it with:

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m drone_detection.paper_cli yolo26-preflight
```

The completed preflight report is stored at
`06_results/paper_yolo26/preflight.json`. It passed for both E0 and E1; no
real dataset images were loaded.
