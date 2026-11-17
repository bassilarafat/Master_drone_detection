"""Controlled v3 lifecycle: validate, seal, start, pause, recover; no Test inference."""
import argparse,hashlib,json,os,platform,subprocess,sys,uuid
from pathlib import Path
from datetime import datetime,timezone
import yaml
from .paper_common import PROJECT_ROOT,project_path,pin_release,write_json,allow_split
from .paper_runner import configure_runtime,training_view
from .training import _without_snapshot_cache_writes
from .yolo26_laptop import load,CONFIG,SEAL
from .yolo26_recovery import event,latest,trainer_class,atomic_bytes,digest,validate_resume


def environment():
    from .paper_common import bootstrap
    bootstrap()
    import torch,ultralytics
    def git(*args):
        p=subprocess.run(['git',*args],cwd=PROJECT_ROOT,capture_output=True,text=True)
        return p.stdout.strip() if p.returncode==0 else None
    props=torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    status=git('status','--porcelain')
    return dict(python=sys.version,pytorch=torch.__version__,ultralytics=ultralytics.__version__,
                cuda=torch.version.cuda,gpu=props.name if props else None,
                gpu_vram_bytes=props.total_memory if props else None,os=platform.platform(),
                git_commit=git('rev-parse','HEAD'),dirty_working_tree=bool(status) if status is not None else None,
                git_status=status,command=[sys.executable,*sys.argv])


def identity(c,arm,seed):
    return dict(arm=arm,seed=seed,configuration_sha256=digest(project_path(CONFIG)),
        protocol_sha256=digest(project_path('configs/paper/protocol_yolo26_v2.json')),
        dataset_release=c['dataset']['release_id'],model_architecture_sha256=c['model']['sha256'],
        initialization_policy='pinned_public_backbone_transfer_modules0..10',
        initialization_donor_sha256=c['initialization']['sha256'])


def verify_seal():
    lock=json.loads(project_path(SEAL).read_text())
    for path,value in lock['hashes'].items():
        if digest(project_path(path))!=value:raise ValueError('Sealed protocol/implementation mismatch: '+path)
    return lock


def check(arm,seed,split='val'):
    allow_split(split)
    cfg,c,p=load(arm,seed)
    from .yolo26_controls import _check_environment
    _check_environment(p)
    if cfg['train']['save_period']!=1:raise ValueError('Every epoch checkpoint must be retained')
    return cfg,c,p


def seal():
    cfg,c,p=check('E0',17);check('E1',17)
    report_path=project_path('06_results/paper_yolo26_v3/hardware_preflight.json')
    report=json.loads(report_path.read_text())
    if report['status']!='PASS' or report['configuration_sha256']!=digest(project_path(CONFIG)):
        raise ValueError('Exact configuration preflight did not pass')
    required=('dataset_integrity_before','dataset_integrity_after','E0_batch4_amp_resume',
              'E1_batch4_amp_resume','test_guard','initialization','checkpoint_save_reload',
              'simulated_resume','corrupt_fallback','protocol_mismatch_rejected')
    if any(report.get('checks',{}).get(name) is not True for name in required):
        raise ValueError('A required preflight check did not pass or is missing')
    implementation=report.get('implementation_sha256',{})
    for name in ('src/drone_detection/yolo26_workflow.py','src/drone_detection/yolo26_recovery.py',
                 'src/drone_detection/yolo26_laptop.py','src/drone_detection/yolo26_validation.py','tools/preflight_controlled.py'):
        if implementation.get(name)!=digest(project_path(name)):raise ValueError('Implementation changed since preflight: '+name)
    pin_release(project_path(c['dataset']['path']),c['dataset'],full=True)
    files=[CONFIG,'configs/paper/protocol_yolo26_v2.json','configs/paper/environment.lock.json',
        'src/drone_detection/yolo26_workflow.py','src/drone_detection/yolo26_recovery.py',
        'src/drone_detection/yolo26_laptop.py','src/drone_detection/yolo26_controlled.py','src/drone_detection/yolo26_validation.py',
        'src/drone_detection/paper_runner.py','src/drone_detection/paper_common.py',
        'src/drone_detection/paper_metrics.py','src/drone_detection/training.py',
        'src/drone_detection/dataset_snapshot.py','src/drone_detection/models/loader.py',
        'src/drone_detection/yolo26_controls.py','tools/run_controlled.py',
        'tools/preflight_controlled.py','06_results/paper_yolo26_v3/hardware_preflight.json']
    value=dict(status='SEALED',sealed_at_utc=datetime.now(timezone.utc).isoformat(),
        configuration_sha256=digest(project_path(CONFIG)),protocol_sha256=c['base_protocol_sha256'] if 'base_protocol_sha256' in c else digest(project_path('configs/paper/protocol_yolo26_v2.json')),
        dataset_release=c['dataset'],initialization=cfg['initialization'],environment=environment(),
        exact_training_settings=c['train'],hashes={name:digest(project_path(name)) for name in files})
    with project_path(SEAL).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
    return value


def _execute_locked(arm,seed,resume=False,split='val'):
    allow_split(split);cfg,c,p=check(arm,seed);verify_seal();configure_runtime()
    import torch
    from ultralytics import YOLO
    dataset=project_path(c['dataset']['path']);before=pin_release(dataset,c['dataset'],full=True)
    base=project_path(c['output_root'])/f'seed_{seed}';ident=identity(c,arm,seed)
    cp=None;state=None
    if resume:
        if json.loads((base/'session_identity.json').read_text())!=ident:raise ValueError('Run protocol mismatch')
        init=json.loads((base/'run/initialization.json').read_text())
        if init['backbone_tensor_sha256']!=cfg['initialization']['source_backbone_sha256'] or init['loaded_tensors']!=240:
            raise ValueError('Original initialization provenance mismatch')
        cp,state,errors=latest(base,ident)
        validate_resume(state,c['train'])
        event(base,'resume',previous_last_completed_epoch=state['epoch']+1,checkpoint=str(cp),
            checkpoint_sha256=digest(cp),resumed_at_utc=datetime.now(timezone.utc).isoformat(),
            reason='user_restart_or_interruption',invalid_candidates=errors)
        canonical=base/'run/weights/last.pt'
        if cp!=canonical:
            atomic_bytes(canonical,cp.read_bytes())
            event(base,'canonical_checkpoint_repaired',source=str(cp),checkpoint_sha256=digest(canonical))
        data=base/'runtime_data/data.yaml'
        # Recover CSV from the committed checkpoint if a crash left a newer/partial row.
        import pandas as pd
        if state.get('train_results'):
            csv_path=base/'run/results.csv'
            if csv_path.exists():atomic_bytes(base/('results.before_resume_'+uuid.uuid4().hex+'.csv'),csv_path.read_bytes())
            atomic_bytes(csv_path,pd.DataFrame(state['train_results']).to_csv(index=False).encode())
    else:
        data,_=training_view(c,seed)
        write_json(base/'session_identity.json',ident);event(base,'fresh_start',identity=ident,environment=environment())
    request=base/'PAUSE_REQUESTED'
    if request.exists():request.unlink()
    result=dict(status='STARTING',identity=ident,dataset_verified_after=False,test_evaluated=False)
    model=YOLO(str(cp) if resume else str(project_path(c['model']['path'])))
    def progress(trainer):
        result.update(status='TRAINING',last_completed_epoch=trainer.epoch+1)
        write_json(base/'state.json',result)
        if request.exists():trainer.stop=True
    model.add_callback('on_fit_epoch_end',progress)
    try:
        with _without_snapshot_cache_writes(dataset):
            model.train(trainer=trainer_class(c,base,ident,resume,state),
                **dict(c['train'],seed=seed,data=str(data),project=str(base),name='run',exist_ok=resume,
                       resume=str(cp) if resume else False))
        result['status']='COMPLETED' if result.get('last_completed_epoch')==100 else 'PAUSED'
        event(base,result['status'].lower(),last_completed_epoch=result.get('last_completed_epoch'))
        if result['status']=='COMPLETED':
            from .yolo26_validation import evaluate_validation
            result['summary']=evaluate_validation(c,base/'run/weights/last.pt',base,dict(seed=seed))
    except BaseException as exc:
        result.update(status='INTERRUPTED_OR_FAILED',error=str(exc));event(base,'interruption',error=str(exc));raise
    finally:
        try:
            if pin_release(dataset,c['dataset'],full=True)!=before:raise ValueError('Dataset changed')
            result['dataset_verified_after']=True
        except BaseException:
            result.update(status='FAILED_DATASET_INTEGRITY',dataset_verified_after=False);raise
        finally:
            write_json(base/'state.json',result)
    return result


def execute(arm,seed,resume=False,split='val'):
    allow_split(split)
    _,c,_=check(arm,seed);verify_seal()
    base=project_path(c['output_root'])/f'seed_{seed}'
    if not resume:base.mkdir(parents=True,exist_ok=False)
    if not base.is_dir():raise FileNotFoundError('No exact run directory exists for resume')
    import msvcrt
    handle=(base/'process.lock').open('a+b')
    if handle.tell()==0:handle.write(b'0');handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        return _execute_locked(arm,seed,resume,split)
    finally:handle.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=['E0','E1','E2','E3'],default='E0')
    parser.add_argument('--seed',type=int,choices=[17,42,123],default=17)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--pause',action='store_true');parser.add_argument('--seal',action='store_true')
    parser.add_argument('--split',default='val')
    args=parser.parse_args();allow_split(args.split)
    if args.arm in ('E2','E3'):parser.error('E2/E3 blocked pending independently verified matched negatives; definition preserved')
    if args.seal:result=seal()
    elif args.pause:
        _,c,_=check(args.arm,args.seed);base=project_path(c['output_root'])/f'seed_{args.seed}'
        if not base.is_dir():raise ValueError('No run to pause')
        (base/'PAUSE_REQUESTED').touch();result={'status':'PAUSE_REQUESTED_AT_EPOCH_BOUNDARY'}
    elif args.dry_run:
        cfg,c,_=check(args.arm,args.seed);verify_seal()
        result=dict(status='VALIDATED_DRY_RUN',identity=identity(c,args.arm,args.seed),training_started=False)
        if args.resume:
            stored=json.loads((project_path(c['output_root'])/f'seed_{args.seed}'/'session_identity.json').read_text())
            if stored!=result['identity']:raise ValueError('Run identity mismatch')
            cp,state,errors=latest(project_path(c['output_root'])/f'seed_{args.seed}',result['identity'])
            result.update(checkpoint=str(cp),completed_epoch=state['epoch']+1,invalid_candidates=errors)
            if state['epoch']>=99:result['status']='FINISHED_NO_RETRAINING'
            else:result['next_epoch']=validate_resume(state,c['train'])+1
    else:result=execute(args.arm,args.seed,args.resume,args.split)
    print(json.dumps(result,indent=2))
