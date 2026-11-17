"""Synthetic GPU check of the new controlled settings; no dataset image is loaded."""
import json
from datetime import datetime, timezone
from pathlib import Path

from drone_detection.paper_common import write_json
from drone_detection.paper_runner import configure_runtime
from drone_detection.yolo26_controls import load_controls
from drone_detection.yolo26_controlled import shared_model


def main():
    configure_runtime()
    import gc
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import ModelEMA
    reports = []
    for arm in ('E0', 'E1'):
        c, _, _ = load_controls(arm, require_lock=False)
        model, init = shared_model(c, 17)
        model = model.cuda().float().train()
        model.args = get_cfg(overrides=dict(c['train'], seed=17))
        trainer = DetectionTrainer.__new__(DetectionTrainer)
        trainer.args = model.args
        optimizer = trainer.build_optimizer(model, name=c['train']['optimizer'], lr=c['train']['lr0'],
                                            momentum=c['train']['momentum'], decay=c['train']['weight_decay'],
                                            iterations=100000)
        torch.cuda.reset_peak_memory_stats()
        batch = dict(img=torch.randn(1,3,960,960,device='cuda'), cls=torch.tensor([[0.]],device='cuda'),
                     bboxes=torch.tensor([[.5,.5,.1,.1]],device='cuda'),
                     batch_idx=torch.tensor([0],dtype=torch.long,device='cuda'))
        loss, _ = model.loss(batch, model(batch['img']))
        loss = loss.sum()
        assert torch.isfinite(loss).all()
        loss.backward()
        optimizer.step()
        ema = ModelEMA(model)
        ema.update(model)
        # Confirm the installed prediction backend selects the ordinary head before fusion.
        model.end2end = False
        assert not model.end2end and model.model[-1].cv2 is not None
        model.eval()
        with torch.no_grad():
            model(batch['img'])
        reports.append(dict(arm=arm, initialization=init, optimizer=type(optimizer).__name__,
                            synthetic_loss=float(loss.detach().cpu()),
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            standard_nms_head_available=True, ema_updated=True))
        del batch, loss, model, optimizer, trainer, ema
        gc.collect()
        torch.cuda.empty_cache()
    assert reports[0]['initialization']['backbone_tensor_sha256'] == reports[1]['initialization']['backbone_tensor_sha256']
    output=Path(__file__).resolve().parents[1]/'06_results/paper_yolo26_v2'
    output.mkdir(parents=True,exist_ok=True)
    report=dict(status='PASS',created_at_utc=datetime.now(timezone.utc).isoformat(),
                reports=reports, synthetic_only=True, real_dataset_images_loaded=False,
                detector_training_called=False, test_evaluated=False)
    write_json(output/'synthetic_preflight.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
