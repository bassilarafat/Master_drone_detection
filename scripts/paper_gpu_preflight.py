"""One real Train batch per model, sequential forward/backward; never optimize or save weights.

This diagnostic is deliberately separate from the paper training entry point.
Run from the project root. Its own SHA and sealed config SHAs are recorded.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

from drone_detection.paper_common import bootstrap, pin_release, project_path, read_csv, write_json, outside_dataset
from drone_detection.data_audit import digest
from drone_detection.paper_runner import configure_runtime, load_config, shared_backbone_model


def parity(a, b):
    differences = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    allowed = ['experiment_id', 'model', 'output_root']
    if differences != allowed:
        raise ValueError(f'Unexpected E0/E1 differences: {differences}')
    return dict(status='PASS', differing_fields=differences, training_equal=a['train']==b['train'],
                evaluation_equal=a['evaluation']==b['evaluation'], initialization_equal=a['initialization']==b['initialization'],
                seeds_equal=a['seeds']==b['seeds'], physical_batch=a['train']['batch'],
                accumulation_steps=round(a['train']['nbs']/a['train']['batch']), nominal_effective_batch=a['train']['nbs'])


def tensor_hash(items):
    h = hashlib.sha256()
    for name, value in items:
        h.update(name.encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default='06_results/paper/pretraining_review_20261110/gpu_preflight.json')
    args = ap.parse_args()
    bootstrap(); configure_runtime()
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.utils.torch_utils import init_seeds
    from drone_detection.training import _without_snapshot_cache_writes
    torch.set_num_threads(4)
    a, p, a_path = load_config('configs/paper/E0.yaml')
    b, _, b_path = load_config('configs/paper/E1.yaml')
    dataset = project_path(a['dataset']['path'])
    output = outside_dataset(project_path(args.output), dataset)
    if output.exists(): raise FileExistsError('Preserve previous diagnostics; use a new output path')
    if list(project_path('04_experiments/paper_v1').glob('*/seed_*/run')):
        raise ValueError('This pre-training diagnostic must precede real paper runs')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='RUNNING', diagnostic_only=True, config_parity=parity(a,b),
                  script_sha256=digest(Path(__file__)), protocol_sha256=a['protocol_sha256'],
                  config_sha256={'E0':digest(a_path), 'E1':digest(b_path)}, seed=17,
                  optimizer_created=False, optimizer_updates=0, checkpoints_saved=0,
                  training_entrypoint_called=False, validation_inference=False, test_inference=False,
                  models_run_sequentially=True, models=[],
                  limitation='One forward/backward feasibility check, not full optimizer/EMA or 100-epoch feasibility. Timings are not paper latency.')
    write_json(output, report)
    try:
        report['dataset_before'] = pin_release(dataset, a['dataset'], full=True)
        report['frozen_file_count_before'] = sum(x.is_file() for x in dataset.rglob('*'))
        report['frozen_numpy_or_cache_payloads_before'] = sorted(x.relative_to(dataset).as_posix() for x in dataset.rglob('*') if x.suffix in ('.cache','.npy'))
        if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU unavailable; no CPU fallback')
        device = torch.device('cuda:0')
        prop = torch.cuda.get_device_properties(device)
        report['hardware'] = dict(name=prop.name, total_vram_bytes=prop.total_memory,
                                  torch=torch.__version__, cuda=torch.version.cuda,
                                  amp=False, tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                                  tf32_cudnn=torch.backends.cudnn.allow_tf32)
        rows = [r for r in read_csv(dataset/'dataset_manifest.csv') if r['new_split']=='train' and int(r['objects'])>0]
        row = sorted(rows, key=lambda r:(-int(r['objects']),r['new_relative_image']))[0]
        image = dataset/row['new_relative_image']
        image_list = output.parent/'preflight_train_one_image.txt'
        if image_list.exists(): raise FileExistsError(image_list)
        image_list.write_text(str(image)+'\n', encoding='utf-8')
        init_seeds(17, deterministic=True)
        with _without_snapshot_cache_writes(dataset):
            ds = build_yolo_dataset(get_cfg(overrides=a['train']), str(image_list), 1,
                                    data={'nc':1,'names':{0:'UAV'}}, mode='train', rect=False, stride=32)
            cpu_batch = YOLODataset.collate_fn([ds[0]])
        if list(cpu_batch['img'].shape) != [1,3,960,960] or len(cpu_batch['cls']) == 0:
            raise ValueError('Actual augmented batch must be 1x3x960x960 with targets')
        report['batch'] = dict(source=row['new_relative_image'], split='train', image_sha256=digest(image),
                               label_sha256=digest(dataset/row['new_relative_label']),
                               original_target_count=int(row['objects']), augmented_target_count=len(cpu_batch['cls']),
                               shape=list(cpu_batch['img'].shape), actual_training_augmentations=True,
                               tensor_sha256=tensor_hash((k,v) for k,v in sorted(cpu_batch.items()) if torch.is_tensor(v)),
                               identical_tensor_for_both_models=True)
        del ds
        for c in (a,b):
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            free_before,total = torch.cuda.mem_get_info(device)
            record = dict(experiment_id=c['experiment_id'], status='RUNNING', forward_calls=0, backward_calls=0,
                          optimizer_updates=0, checkpoints_saved=0, free_before_bytes=free_before)
            report['models'].append(record); write_json(output,report)
            model, initialization = shared_backbone_model(project_path(c['model']['path']), project_path(c['initialization']['path']),17)
            record['initialization'] = initialization
            record['backbone_sha256'] = tensor_hash((k,v) for k,v in model.state_dict().items() if k in initialization['loaded_keys'])
            record['parameter_count'] = sum(v.numel() for v in model.parameters())
            before = tensor_hash(model.named_parameters())
            model.args = get_cfg(overrides=c['train'])
            model = model.to(device).float()
            model.train()  # nn.Module mode only; not YOLO.train() or a trainer.
            batch = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in cpu_batch.items()}
            batch['img'] = batch['img'].float()/255.0
            torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast('cuda',enabled=False):
                loss, components = model(batch)
                record['forward_calls'] = 1
                scalar_loss = loss.sum()
            if not torch.isfinite(scalar_loss).item(): raise ValueError('Non-finite loss')
            scalar_loss.backward(); record['backward_calls'] = 1
            torch.cuda.synchronize()
            record.update(forward_backward_seconds=time.perf_counter()-start,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                          free_after_backward_bytes=torch.cuda.mem_get_info(device)[0],
                          loss=float(scalar_loss.detach()),
                          loss_components=({k:float(v.detach().cpu()) for k,v in components.items()}
                                           if isinstance(components,dict) else components.detach().cpu().tolist()))
            gradients=[v.grad for v in model.parameters() if v.grad is not None]
            record['gradient_tensors'] = len(gradients)
            record['gradients_finite'] = bool(gradients) and all(torch.isfinite(v).all().item() for v in gradients)
            record['nonzero_gradient_tensors'] = sum(torch.count_nonzero(v).item()>0 for v in gradients)
            after = tensor_hash(model.named_parameters())
            record.update(parameters_sha256_before=before, parameters_sha256_after=after,
                          parameters_unchanged=before==after, batchnorm_buffers='Updated in ephemeral train-mode memory only, then discarded')
            if before != after or not record['gradients_finite']: raise ValueError('Parameter mutation or invalid gradients')
            record['status']='PASS'
            del gradients, batch, scalar_loss, components, loss, model
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            record['allocated_after_cleanup_bytes'] = torch.cuda.memory_allocated(device)
            write_json(output,report)
        if report['models'][0]['backbone_sha256'] != report['models'][1]['backbone_sha256']:
            raise ValueError('Initial pretrained backbone bytes differ')
        report['status']='PASS'
    except BaseException as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try:
            report['dataset_after'] = pin_release(dataset,a['dataset'],full=True)
            report['frozen_file_count_after'] = sum(x.is_file() for x in dataset.rglob('*'))
            report['dataset_integrity_verified_after']=True
            report['frozen_numpy_or_cache_payloads_after']=sorted(x.relative_to(dataset).as_posix() for x in dataset.rglob('*') if x.suffix in ('.cache','.npy'))
            report['new_cache_files']=sorted(set(report['frozen_numpy_or_cache_payloads_after'])-set(report.get('frozen_numpy_or_cache_payloads_before',[])))
            if report['frozen_file_count_after'] != report.get('frozen_file_count_before') or report['new_cache_files']:
                raise ValueError('Unexpected frozen dataset files')
            report['paper_training_runs_created']=len(list(project_path('04_experiments/paper_v1').glob('*/seed_*/run')))
            if report['paper_training_runs_created']: raise ValueError('A paper training run exists')
        except BaseException:
            report['status']='FAIL_DATASET_OR_RUN_INTEGRITY'
            raise
        finally: write_json(output,report)
    print(json.dumps({k:report[k] for k in ('status','hardware','config_parity','models')},indent=2))


if __name__=='__main__': main()
