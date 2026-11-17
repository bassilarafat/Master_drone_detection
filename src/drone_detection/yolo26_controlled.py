"""Versioned YOLO26 controlled training; validation only, never final Test."""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from .data_audit import digest
from .dataset_snapshot import _annotations
from .paper_common import project_path, pin_release, read_csv, size_bin, write_json
from .paper_metrics import export_metrics
from .paper_runner import configure_runtime, training_view
from .training import _without_snapshot_cache_writes
from .yolo26_controls import load_controls


def _request(arm, seed):
    if arm not in ('E0', 'E1'):
        raise ValueError('Only controlled E0 and E1 are implemented; Test and E2/E3 are locked')
    if type(seed) is not int or seed not in (17, 42, 123):
        raise ValueError('Seed must be one of the fixed plan: 17, 42, 123')


def now():
    return datetime.now(timezone.utc).isoformat()


def dry_run(arm='E0', seed=17):
    _request(arm, seed)
    c, p, path = load_controls(arm)
    identity = pin_release(project_path(c['dataset']['path']), c['dataset'], full=True)
    base = project_path(c['output_root']) / f'seed_{seed}'
    if base.exists():
        raise FileExistsError(f'Never overwrite or resume a controlled run: {base}')
    return dict(status='READY_FOR_EXPLICIT_EXECUTION', arm=arm, seed=seed,
                config_sha256=digest(path), protocol_sha256=c['protocol_sha256'],
                dataset_release=identity, training=c['train'], evaluation=c['evaluation'],
                checkpoint_selection='last.pt after completed epoch 100; no outcome selection',
                output=str(base), training_called=False, test_evaluated=False)


def shared_model(c, seed):
    """Same public backbone values; seed-initialized neck/heads in each arm."""
    import hashlib
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import init_seeds
    from .models import transfer_backbone_weights

    init_seeds(seed, deterministic=True)
    target = DetectionModel(str(project_path(c['model']['path'])), ch=3, nc=1, verbose=False)
    donor = YOLO(str(project_path(c['initialization']['path'])))
    if donor.model.model[-1].nc != 80 or donor.names.get(0) != 'person':
        raise ValueError('Donor must be the hash-pinned public COCO80 checkpoint')
    keys = transfer_backbone_weights(target, donor.model.float(), last_module_exclusive=11)
    hasher = hashlib.sha256()
    for key in sorted(keys):
        hasher.update(key.encode())
        hasher.update(target.state_dict()[key].detach().cpu().numpy().tobytes())
    strides = [int(x) for x in target.stride.tolist()]
    expected = [8, 16, 32] if c['architecture'] == 'E0' else [4, 8, 16, 32]
    if strides != expected:
        raise ValueError(f'Unexpected model strides: {strides}')
    report = dict(seed=seed, policy='public COCO backbone modules 0..10 only',
                  pretrained_sha256=c['initialization']['sha256'], loaded_keys=keys,
                  loaded_tensors=len(keys), backbone_tensor_sha256=hasher.hexdigest(),
                  strides=strides, training_called=False)
    return target, report


def _fit(c, seed, base, data, record):
    import torch
    from ultralytics import YOLO
    from ultralytics.data import build_yolo_dataset
    from ultralytics.models.yolo.detect import DetectionTrainer

    class ControlledTrainer(DetectionTrainer):
        def _build_train_pipeline(self):
            # The pinned library's OOM retry is inline and rebuilds this pipeline.
            if getattr(self, 'controlled_pipeline_builds', 0):
                raise RuntimeError('Controlled pipeline cannot be rebuilt for an automatic OOM retry')
            self.controlled_pipeline_builds = 1
            return super()._build_train_pipeline()

        def get_model(self, cfg=None, weights=None, verbose=True):
            model, report = shared_model(c, seed)
            write_json(Path(self.save_dir) / 'initialization.json', report)
            return model

        def build_dataset(self, img_path, mode='train', batch=None):
            return build_yolo_dataset(self.args, img_path, batch, self.data,
                                      mode=mode, rect=False, stride=32)

        def _setup_train(self):
            super()._setup_train()
            resolved = vars(self.args)
            for key, expected in c['train'].items():
                if resolved.get(key) != expected:
                    raise ValueError(f'Trainer changed controlled setting {key}: {resolved.get(key)!r}')
            if self.accumulate != 16 or self.amp:
                raise ValueError('Expected fixed accumulation16 and AMP disabled')
            if type(self.optimizer).__name__ != 'MuSGD':
                raise ValueError('Optimizer did not resolve to MuSGD')
            if self.args.nms is not True or self.args.resume:
                raise ValueError('Standard NMS and no resume are required')
            self.controlled_updates = 0
            report = dict(settings=resolved, optimizer=type(self.optimizer).__name__,
                          gradient_accumulation=self.accumulate,
                          train_images=len(self.train_loader.dataset),
                          validation_images=len(self.test_loader.dataset),
                          validation_batch=self.test_loader.batch_size,
                          training_dual_head=hasattr(self.model.model[-1], 'one2one_cv2'),
                          scheduler=type(self.scheduler).__name__,
                          parameter_group_lrs=[g['lr'] for g in self.optimizer.param_groups])
            if report['train_images'] != 6299 or report['validation_images'] != 1827:
                raise ValueError('Training/validation image counts differ from pinned release')
            write_json(base / 'resolved_training.json', report)

        def optimizer_step(self):
            if not torch.isfinite(self.loss).all():
                raise FloatingPointError('Nonfinite loss; controlled runs fail instead of silently restarting')
            super().optimizer_step()
            self.controlled_updates += 1

        def _handle_nan_recovery(self, *args, **kwargs):
            loss = getattr(self, 'loss', None)
            fitness = getattr(self, 'fitness', None)
            if loss is not None and not torch.isfinite(loss).all():
                raise FloatingPointError('Nonfinite loss; no automatic checkpoint recovery')
            if fitness is not None and not torch.isfinite(torch.tensor(fitness)):
                raise FloatingPointError('Nonfinite validation fitness; no automatic recovery')
            return False

        def final_eval(self):
            # Preserve raw epoch100 last.pt; no second best.pt selection/evaluation.
            return None

    model = YOLO(str(project_path(c['model']['path'])))
    observed_batches = 0

    def started(trainer):
        record.update(status='TRAINING', last_completed_epoch=0, optimizer_updates=0,
                      updated_at_utc=now())
        write_json(base / 'run.json', record)

    def batch_progress(trainer):
        nonlocal observed_batches
        observed_batches += 1
        if not torch.isfinite(trainer.loss).all():
            raise FloatingPointError('Nonfinite batch loss; no automatic checkpoint recovery')
        if observed_batches in (1, 16) or observed_batches % 100 == 0:
            record.update(status='TRAINING', observed_batches=observed_batches,
                          optimizer_updates=getattr(trainer, 'controlled_updates', 0),
                          current_training_loss=float(trainer.loss.detach().cpu()),
                          updated_at_utc=now())
            write_json(base / 'run.json', record)

    def progress(trainer):
        record.update(status='TRAINING', last_completed_epoch=int(trainer.epoch) + 1,
                      optimizer_updates=getattr(trainer, 'controlled_updates', 0), updated_at_utc=now())
        write_json(base / 'run.json', record)

    model.add_callback('on_train_start', started)
    model.add_callback('on_train_batch_end', batch_progress)
    model.add_callback('on_fit_epoch_end', progress)
    with _without_snapshot_cache_writes(project_path(c['dataset']['path'])), \
            patch('ultralytics.utils.checks.check_pip_update_available', lambda: None):
        model.train(trainer=ControlledTrainer, **dict(c['train'], seed=seed, data=str(data),
                    project=str(base), name='run', exist_ok=False))
    rows = list(csv.DictReader((base / 'run/results.csv').open(encoding='utf-8-sig')))
    if len(rows) != 100 or int(float(rows[-1]['epoch'])) != 100:
        raise ValueError('Controlled run did not complete exactly 100 epochs')
    checkpoint = base / 'run/weights/last.pt'
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if ckpt.get('epoch') != 99:
        raise ValueError('Checkpoint is not the unstripped epoch100 last.pt')
    record['optimizer_updates'] = model.trainer.controlled_updates
    return checkpoint


def _evaluate(c, checkpoint, base, record):
    from ultralytics import YOLO
    dataset = project_path(c['dataset']['path'])
    identity = pin_release(dataset, c['dataset'], full=True)
    output = base / 'evaluation/val'
    output.mkdir(parents=True, exist_ok=False)
    settings = c['evaluation']
    predictions = []
    state = dict(status='EVALUATING', split='val', test_evaluated=False,
                 dataset_verified_after=False, checkpoint_sha256=digest(checkpoint),
                 settings=settings, started_at_utc=now())
    write_json(output / 'evaluation.json', state)
    try:
        model = YOLO(str(checkpoint))
        rows = [r for r in read_csv(dataset / 'dataset_manifest.csv') if r['new_split'] == 'val']
        for row in rows:
            image = dataset / row['new_relative_image']
            width, height = int(row['width']), int(row['height'])
            parsed, issues = _annotations(dataset / row['new_relative_label'])
            if issues:
                raise ValueError(issues)
            targets = [dict(box=[(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height],
                       size=size_bin(w*width*h*height), **{'class': int(cls)})
                       for cls, x, y, w, h in parsed]
            result = model.predict(str(image), imgsz=settings['imgsz'],
                                   conf=settings['prediction_conf'], iou=settings['nms_iou'],
                                   max_det=settings['max_det'], batch=settings['batch'],
                                   rect=settings['rect'], augment=settings['augment'],
                                   quantize=settings['quantize'],
                                   agnostic_nms=settings['agnostic_nms'], nms=settings['nms'],
                                   device=settings['device'], save=False,
                                   save_txt=False, save_crop=False, verbose=False)[0]
            if model.predictor.model.end2end:
                raise ValueError('Inference resolved to NMS-free path despite nms=True')
            boxes = result.boxes
            pred = [dict(box=list(map(float, b)), confidence=float(s), **{'class': int(k)})
                    for b, s, k in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(),
                                       boxes.cls.cpu().tolist())]
            predictions.append(dict(image=image.name, split='val', width=width, height=height,
                                    targets=targets, predictions=pred,
                                    image_sha256=row['image_sha256'], label_sha256=row['label_sha256'],
                                    scene_group_id=row['scene_group_id'], negative_verified=False))
        raw = output / 'predictions.jsonl'
        with raw.open('w', encoding='utf-8') as handle:
            for item in predictions:
                handle.write(json.dumps(item, allow_nan=False) + '\n')
        summary = export_metrics(output, predictions, confidence=settings['operating_conf'],
                                 matching_iou=settings['matching_iou'])
        summary.update(arm=c['architecture'], seed=record['seed'], split='val',
                       checkpoint_sha256=digest(checkpoint), dataset_release=identity)
        write_json(output / 'summary.json', summary)
        state.update(status='PENDING_INTEGRITY_CHECK', prediction_records=len(predictions),
                     raw_sha256=digest(raw), effective_end2end=False, nms=True)
    except BaseException as exc:
        state.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try:
            after = pin_release(dataset, c['dataset'], full=True)
            if after != identity:
                raise ValueError('Dataset identity changed during validation')
            state['dataset_verified_after'] = True
            if state['status'] == 'PENDING_INTEGRITY_CHECK':
                state['status'] = 'COMPLETED'
        except BaseException:
            state.update(status='FAILED_DATASET_INTEGRITY', dataset_verified_after=False)
            raise
        finally:
            state['finished_at_utc'] = now()
            write_json(output / 'evaluation.json', state)
    return summary


def train_controlled(arm='E0', seed=17, *, execute=False):
    _request(arm, seed)
    if not execute:
        raise ValueError('Training not started; explicit execute=True is required')
    plan = dry_run(arm, seed)
    c, p, path = load_controls(arm)
    configure_runtime()
    dataset = project_path(c['dataset']['path'])
    base = project_path(c['output_root']) / f'seed_{seed}'
    # Atomic reservation prevents two launchers from using the same run path.
    base.mkdir(parents=True, exist_ok=False)
    record = dict(status='STARTING', arm=arm, experiment_id=c['experiment_id'], seed=seed,
                  started_at_utc=now(), config_sha256=digest(path), protocol_sha256=c['protocol_sha256'],
                  dataset_release=plan['dataset_release'], dataset_verified_after=False,
                  test_evaluated=False, training_settings=c['train'], evaluation_settings=c['evaluation'])
    write_json(base / 'run.json', record)
    started = time.monotonic()
    try:
        view, bundle = training_view(dict(c, negative_mode='none'), seed)
        checkpoint = _fit(c, seed, base, view, record)
        record.update(status='TRAINING_COMPLETED', checkpoint=str(checkpoint),
                      checkpoint_sha256=digest(checkpoint), training_seconds=time.monotonic()-started)
        write_json(base / 'run.json', record)
    except BaseException as exc:
        record.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try:
            after = pin_release(dataset, c['dataset'], full=True)
            if after != plan['dataset_release']:
                raise ValueError('Frozen dataset identity changed')
            record['dataset_verified_after'] = True
        except BaseException:
            record.update(status='FAILED_DATASET_INTEGRITY', dataset_verified_after=False)
            raise
        finally:
            record['updated_at_utc'] = now()
            write_json(base / 'run.json', record)
    try:
        # Recheck code/configuration identities before interpreting the result.
        load_controls(arm)
        summary = _evaluate(c, checkpoint, base, record)
    except BaseException as exc:
        record.update(status='VALIDATION_FAILED', error=f'{type(exc).__name__}: {exc}')
        write_json(base / 'run.json', record)
        raise
    record.update(status='COMPLETED', finished_at_utc=now(), summary=summary)
    write_json(base / 'run.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['E0', 'E1'], default='E0')
    parser.add_argument('--seed', type=int, choices=[17, 42, 123], default=17)
    parser.add_argument('--execute-training', action='store_true')
    args = parser.parse_args()
    result = train_controlled(args.arm, args.seed, execute=True) if args.execute_training else dry_run(args.arm, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
