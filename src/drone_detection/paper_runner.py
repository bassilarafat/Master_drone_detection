"""Controlled paper dry-runs, explicitly opted-in future training and locked evaluation."""
from __future__ import annotations
import importlib.metadata as metadata
import json
from pathlib import Path
from decimal import Decimal
import yaml
from .paper_common import (PROJECT_ROOT, bootstrap, project_path, pin_release, outside_dataset,
    allow_split, read_csv, size_bin, write_json,source_fingerprints)
from .data_audit import digest

def load_config(path):
    path=project_path(path);c=yaml.safe_load(path.read_text(encoding='utf-8'))
    protocol_path=project_path(c['protocol'])
    if digest(protocol_path)!=c['protocol_sha256']:raise ValueError('Paper protocol changed; do not silently override preregistered controls')
    p=json.loads(protocol_path.read_text());arm=c['experiment_id']
    if p.get('implementation_sha256') and p['implementation_sha256']!=source_fingerprints():
        raise ValueError('Paper implementation changed after protocol sealing; version/review the protocol before training')
    if arm not in ('E0','E1','E2','E3'):raise ValueError('Unknown paper experiment')
    for key in ('dataset','initialization','seeds','default_seed','train','evaluation'):
        if c[key]!=p[key]:raise ValueError(f'Experiment changed a common controlled parameter: {key}')
    if c['model']!=p['topologies']['E0' if arm=='E0' else 'E1']:raise ValueError('Architecture differs from pinned experiment')
    if c['negative_mode']!={'E0':'none','E1':'none','E2':'random','E3':'hard'}[arm] or c['negative_count']!=(0 if arm in ('E0','E1') else 512):
        raise ValueError('Negative control assignment/count differs')
    if c['negative_bundle']!=p['negatives']['bundle']:raise ValueError('Negative bundle differs')
    dataset=project_path(c['dataset']['path']);pin_release(dataset,c['dataset'])
    for value,key in [(c['model'],'path'),(c['initialization'],'path')]:
        if digest(project_path(value[key]))!=value['sha256']:raise ValueError(f'Pinned asset changed: {value[key]}')
    release=project_path(c['dataset']['release_record'])
    if digest(release)!=c['dataset']['release_record_sha256']:raise ValueError('External release record changed')
    lock_path=project_path('configs/paper/environment.lock.json')
    if digest(lock_path)!=p['environment_lock_sha256']:raise ValueError('Environment lock changed')
    lock=json.loads(lock_path.read_text())
    for name,version in lock['packages'].items():
        if metadata.version(name)!=version:raise ValueError(f'Pinned package version differs: {name}')
    package=Path(metadata.distribution('ultralytics').locate_file('ultralytics'))
    for name,sha in lock['implementation_sha256'].items():
        if digest(package/name)!=sha:raise ValueError(f'Pinned Ultralytics implementation changed: {name}')
    gate_path=project_path(p['gate3']['path']);gate=json.loads(gate_path.read_text())
    if digest(gate_path)!=p['gate3']['sha256'] or gate['status']!='PASS' or any(gate['checks'].values()):raise ValueError('Gate 3 has not passed with the pinned audit')
    if gate['dataset_release']['release_id']!=c['dataset']['release_id']:raise ValueError('Gate 3 refers to another dataset')
    if p['gate3'].get('runtime_preflight'):
        preflight=project_path(p['gate3']['runtime_preflight']);runtime=json.loads(preflight.read_text())
        if digest(preflight)!=p['gate3']['runtime_preflight_sha256'] or runtime['status']!='PASS' or any(runtime['checks'].values()):
            raise ValueError('Dataset runtime safety preflight is not the pinned PASS')
    if c['output_root']!=f'04_experiments/paper_v1/{arm}':raise ValueError('Experiment output root is not the controlled paper directory')
    outside_dataset(project_path(c['output_root']),dataset)
    lock_file=project_path('configs/paper/CONTROL_LOCK.json')
    if lock_file.exists():
        controls=json.loads(lock_file.read_text())
        if controls['protocol_sha256']!=digest(protocol_path) or controls['config_sha256'][arm]!=digest(path):
            raise ValueError('Configuration changed after paper controls were sealed')
    return c,p,path

def dry_run(config,seed=None):
    c,p,path=load_config(config);seed=c['default_seed'] if seed is None else seed
    if seed not in c['seeds']:raise ValueError('Seed is outside the fixed repeated-run plan')
    dataset=project_path(c['dataset']['path']);identity=pin_release(dataset,c['dataset'],full=True)
    blockers=[];negative_samples=0
    if c['negative_mode']!='none':
        bundle=project_path(c['negative_bundle'])
        if not bundle.is_file():blockers.append('512 verified hard AND 512 matched verified random negatives have not been published')
        else:
            from .paper_negatives import validate_bundle
            negative_samples=len(validate_bundle(bundle,dataset)[c['negative_mode']])
    report=dict(status='BLOCKED_NEGATIVE_REVIEW' if blockers else 'READY_FOR_APPROVED_TRAINING',experiment_id=c['experiment_id'],seed=seed,
        config_sha256=digest(path),protocol_sha256=c['protocol_sha256'],dataset_release=identity,
        physical_batch=1,gradient_accumulation=16,nominal_effective_batch=16,epochs=100,imgsz=960,
        checkpoint_selection='last.pt epoch100',negative_samples=negative_samples,blockers=blockers,
        implementation_sha256=source_fingerprints(),model_loaded=False,training_called=False,test_evaluated=False,frozen_dataset_writes=0)
    pin_release(dataset,c['dataset'],full=True)
    return report

def configure_runtime():
    bootstrap()
    from ultralytics import settings
    keys=('sync','wandb','mlflow','comet','tensorboard','hub','clearml','neptune','raytune','dvc')
    settings.update({k:False for k in keys if k in settings})
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')

def shared_backbone_model(model_yaml,weights,seed,backbone_last_module_exclusive=10):
    """Load only semantically identical backbone tensors; never index-match divergent heads."""
    configure_runtime()
    from .paper_weights import SHA as PUBLIC_SHA
    if digest(weights)!=PUBLIC_SHA:raise ValueError('Pretrained initialization must match the independently verified official public SHA-256')
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(seed,deterministic=True)
    target=DetectionModel(str(model_yaml),ch=3,nc=1,verbose=False)
    donor=YOLO(str(weights))
    if donor.model.model[-1].nc!=80 or donor.names.get(0)!='person' or donor.ckpt.get('train_args',{}).get('data') not in ('coco.yaml','coco2017.yaml'):
        raise ValueError('Initialization must be the pinned public COCO80 pretrained model, not a historical DUT checkpoint')
    src=donor.model.float()
    from .models import transfer_backbone_weights
    keys=transfer_backbone_weights(target,src,last_module_exclusive=backbone_last_module_exclusive)
    report=dict(policy=f'shared COCO backbone modules 0..{backbone_last_module_exclusive - 1} only',
        loaded_tensors=len(keys),loaded_keys=keys,
        pretrained_sha256=digest(weights),seed=seed,pretrained_classes=80,target_classes=1,
        donor_training_data=donor.ckpt.get('train_args',{}).get('data'),donor_epoch=donor.ckpt.get('epoch'),
        remote_digest_independently_verified=True,train_called=False)
    return target,report

def training_view(c,seed):
    dataset=project_path(c['dataset']['path']);base=project_path(c['output_root'])/f'seed_{seed}'
    view=base/'runtime_data'
    if view.exists():raise FileExistsError('A training view already exists; never overwrite a paper run')
    view.mkdir(parents=True)
    rows=read_csv(dataset/'dataset_manifest.csv')
    train=[str(dataset/r['new_relative_image']) for r in rows if r['new_split']=='train']
    val=[str(dataset/r['new_relative_image']) for r in rows if r['new_split']=='val']
    bundle_sha=None
    if c['negative_mode']!='none':
        from .paper_negatives import validate_bundle
        bundle=project_path(c['negative_bundle']);selected=validate_bundle(bundle,dataset)[c['negative_mode']]
        train.extend(str(project_path(r['crop_path'])) for r in selected);bundle_sha=digest(bundle)
    for split,paths in [('train',train),('val',val)]:
        (view/f'{split}.txt').write_text('\n'.join(paths)+'\n',encoding='utf-8')
    data=dict(path=str(view),train=str(view/'train.txt'),val=str(view/'val.txt'),nc=1,names={0:'UAV'})
    yaml_path=view/'data.yaml';yaml_path.write_text(yaml.safe_dump(data),encoding='utf-8')
    return yaml_path,bundle_sha

def train(config,seed=None,*,execute=False):
    if not execute:raise ValueError('Training is not started: --execute-training is required only after user approval')
    plan=dry_run(config,seed)
    if plan['blockers']:raise ValueError(plan['blockers'])
    c,p,path=load_config(config);seed=plan['seed'];dataset=project_path(c['dataset']['path'])
    if not p.get('implementation_sha256'):raise ValueError('Paper protocol implementation must be sealed before training')
    configure_runtime()
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from .training import _without_snapshot_cache_writes
    from unittest.mock import patch
    base=project_path(c['output_root'])/f'seed_{seed}';run=base/'run'
    if run.exists():raise FileExistsError('Never resume/overwrite a paper comparison run')
    data,bundle_sha=training_view(c,seed)
    record=dict(experiment_id=c['experiment_id'],seed=seed,config_sha256=digest(path),protocol_sha256=c['protocol_sha256'],
        dataset_release=pin_release(dataset,c['dataset']),negative_bundle_sha256=bundle_sha,status='STARTING',
        final_test_locked=True,implementation_sha256=source_fingerprints(),training_options=dict(c['train'],seed=seed))
    write_json(base/'run.json',record)
    class PaperTrainer(DetectionTrainer):
        def get_model(self,cfg=None,weights=None,verbose=True):
            model,report=shared_backbone_model(project_path(c['model']['path']),project_path(c['initialization']['path']),seed)
            write_json(Path(self.save_dir)/'initialization.json',report)
            return model
        def build_dataset(self,img_path,mode='train',batch=None):
            # Use identical square letterboxing at validation and unified evaluation.
            from ultralytics.data import build_yolo_dataset
            return build_yolo_dataset(self.args,img_path,batch,self.data,mode=mode,rect=False,stride=32)
    try:
        model=YOLO(str(project_path(c['model']['path'])))
        with _without_snapshot_cache_writes(dataset),patch('ultralytics.utils.checks.check_pip_update_available',lambda:None):
            model.train(trainer=PaperTrainer,**dict(c['train'],seed=seed,data=str(data),project=str(base),name='run',exist_ok=False))
        checkpoint=run/'weights/last.pt'
        record.update(status='COMPLETED',checkpoint_sha256=digest(checkpoint),checkpoint=str(checkpoint))
    except BaseException:
        record['status']='FAILED';raise
    finally:
        try:pin_release(dataset,c['dataset'],full=True);record['dataset_verified_after']=True
        except BaseException:record.update(status='FAILED_DATASET_INTEGRITY',dataset_verified_after=False);raise
        finally:write_json(base/'run.json',record)
    return evaluate(config,checkpoint,base/'evaluation/val',split='val')

def evaluate(config,checkpoint,output,*,split='val',dry=False):
    allow_split(split)  # Reject final Test before opening model, images or labels.
    c,p,path=load_config(config);dataset=project_path(c['dataset']['path']);output=outside_dataset(output,dataset)
    checkpoint=project_path(checkpoint)
    if dry:return dict(status='EVALUATION_PLAN_ONLY',split=split,checkpoint=str(checkpoint),checkpoint_exists=checkpoint.is_file(),
        evaluation=c['evaluation'],model_loaded=False,test_locked=True)
    if output.exists():raise FileExistsError(output)
    # Only a completed, identity-bound paper run can enter the unified evaluator.
    run_meta_path=checkpoint.parent.parent.parent/'run.json'
    run=json.loads(run_meta_path.read_text())
    if run['status']!='COMPLETED' or run['experiment_id']!=c['experiment_id'] or run['config_sha256']!=digest(path) or run['protocol_sha256']!=c['protocol_sha256'] or run['checkpoint_sha256']!=digest(checkpoint):
        raise ValueError('Evaluation requires the completed pinned paper epoch100 last.pt, never historical DUT weights')
    if checkpoint.name!='last.pt':raise ValueError('Paper selection is fixed to epoch100 last.pt')
    identity=pin_release(dataset,c['dataset'],full=True);configure_runtime()
    from ultralytics import YOLO
    from PIL import Image
    from .dataset_snapshot import _annotations
    from .paper_metrics import export_metrics
    rows=[r for r in read_csv(dataset/'dataset_manifest.csv') if r['new_split']==split]
    model=YOLO(str(checkpoint));records=[];evaluation_record=None
    try:
        for r in rows:
            image=dataset/r['new_relative_image'];width,height=int(r['width']),int(r['height'])
            parsed,issues=_annotations(dataset/r['new_relative_label'])
            if issues:raise ValueError(issues)
            targets=[dict(box=[float((x-w/2)*width),float((y-h/2)*height),float((x+w/2)*width),float((y+h/2)*height)],
                size=size_bin(w*width*h*height),**{'class':int(cls)}) for cls,x,y,w,h in parsed]
            result=model.predict(str(image),imgsz=960,conf=.001,iou=.7,max_det=300,batch=1,rect=False,augment=False,
                half=False,agnostic_nms=False,device=c['train']['device'],save=False,save_txt=False,save_crop=False,verbose=False)[0]
            preds=[dict(box=list(map(float,b)),confidence=float(s),**{'class':int(k)}) for b,s,k in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist())]
            records.append(dict(image=image.name,split=split,width=width,height=height,targets=targets,predictions=preds,
                image_sha256=r['image_sha256'],label_sha256=r['label_sha256'],scene_group_id=r['scene_group_id'],
                negative_verified=False))
        output.mkdir(parents=True)
        raw=output/'predictions.jsonl'
        with raw.open('w',encoding='utf-8') as f:
            for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
        summary=export_metrics(output,records,confidence=.25,matching_iou=.5)
        summary.update(experiment_id=c['experiment_id'],seed=run['seed'],split=split,dataset_release_id=identity['release_id'],
            checkpoint_sha256=digest(checkpoint),protocol_sha256=c['protocol_sha256'])
        write_json(output/'summary.json',summary)
        evaluation_record=dict(experiment_id=c['experiment_id'],seed=run['seed'],split=split,
            dataset_release=identity,config_sha256=digest(path),protocol_sha256=c['protocol_sha256'],checkpoint_sha256=digest(checkpoint),
            raw_records_sha256=digest(raw),settings=dict(c['evaluation'],split=split),implementation_sha256=source_fingerprints(),
            status='PENDING_DATASET_VERIFICATION',dataset_verified_after=False,test_evaluated=False,records=len(records))
        write_json(output/'evaluation.json',evaluation_record)
    finally:
        try:pin_release(dataset,c['dataset'],full=True)
        except BaseException:
            if evaluation_record is not None:
                evaluation_record.update(status='FAILED_DATASET_INTEGRITY',dataset_verified_after=False)
                write_json(output/'evaluation.json',evaluation_record)
            raise
        else:
            if evaluation_record is not None:
                evaluation_record.update(status='COMPLETED',dataset_verified_after=True)
                write_json(output/'evaluation.json',evaluation_record)
    return summary
