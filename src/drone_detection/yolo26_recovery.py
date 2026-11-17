"""Atomic, identity-bound training checkpoints and append-only event provenance."""
import hashlib,json,os,uuid
from pathlib import Path
from datetime import datetime,timezone


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def event(base,kind,**fields):
    folder=Path(base)/'events';folder.mkdir(exist_ok=True,parents=True)
    record=dict(event=kind,at_utc=datetime.now(timezone.utc).isoformat(),**fields)
    p=folder/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'_'+uuid.uuid4().hex+'.json')
    with p.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2);f.flush();os.fsync(f.fileno())
    return record


def checkpoint(path,identity=None):
    import torch
    p=Path(path)
    value=torch.load(p,map_location='cpu',weights_only=False)
    if value.get('epoch',-1)<0 or value.get('optimizer') is None:raise ValueError('Not a resumable training checkpoint')
    if identity is not None and value.get('controlled_identity')!=identity:raise ValueError('Checkpoint protocol/arm/seed/dataset/architecture/initialization mismatch')
    for key in ('controlled_scheduler','scaler','controlled_live_model_state'):
        if identity is not None and key not in value:raise ValueError('Required training state missing: '+key)
    tensors=value.get('controlled_live_model_state',{})
    if any(not torch.isfinite(x).all() for x in tensors.values() if x.is_floating_point()):raise ValueError('Nonfinite model state')
    return value


def latest(base,identity=None):
    folder=Path(base)/'run/weights';valid=[];errors=[];mismatches=[]
    files=list(folder.glob('epoch*.pt'))+[folder/'last.pt',folder/'last_verified.pt',folder/'previous_verified.pt']
    for p in dict.fromkeys(files):
        if not p.is_file():continue
        try:
            c=checkpoint(p,identity)
            valid.append((int(c['epoch']),p.name=='last.pt',p))
            del c
        except Exception as exc:
            errors.append(dict(path=str(p),error=str(exc)))
            if 'mismatch' in str(exc):mismatches.append(str(p))
    if mismatches:raise ValueError('Protocol mismatch in checkpoint candidates: '+', '.join(mismatches))
    if not valid:raise ValueError('No intact resumable checkpoint: '+str(errors))
    _,_,p=max(valid,key=lambda x:(x[0],x[1]))
    return p,checkpoint(p,identity),errors


def validate_resume(state,training):
    if state.get('epoch',-1)>=99:raise ValueError('FINISHED: epoch100 already completed; no retraining permitted')
    if state.get('epoch',-1)<0:raise ValueError('Invalid checkpoint epoch')
    for key,value in training.items():
        if key!='resume' and state.get('train_args',{}).get(key)!=value:
            raise ValueError('Checkpoint training configuration mismatch: '+key)
    return state['epoch']+1


def trainer_class(c,base,identity,resuming=False,resume_state=None):
    import io,copy,tempfile
    import torch
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.data import build_yolo_dataset
    from .yolo26_controlled import shared_model
    class RecoverableTrainer(DetectionTrainer):
        def build_optimizer(self,*args,**kwargs):
            optimizer=super().build_optimizer(*args,**kwargs)
            self.controlled_optimizer_steps=(resume_state or {}).get('controlled_optimizer_steps',0)
            def stepped(*unused):self.controlled_optimizer_steps+=1
            optimizer.register_step_post_hook(stepped)
            return optimizer
        def get_model(self,cfg=None,weights=None,verbose=True):
            nonlocal resume_state
            if resuming:
                if resume_state is None:resume_state=checkpoint(self.args.model,identity)
                model=super().get_model(cfg,weights,verbose)
                model.load_state_dict(resume_state['controlled_live_model_state'],strict=True)
                return model
            model,report=shared_model(c,identity['seed'])
            expected='b928e4ec27534c8a27f038005eb76ae934b7858af32a9f8f60a0192e988823a3'
            if report['backbone_tensor_sha256']!=expected or report['loaded_tensors']!=240:raise ValueError('Initialization audit failed')
            (Path(self.save_dir)/'initialization.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            return model
        def build_dataset(self,img_path,mode='train',batch=None):
            return build_yolo_dataset(self.args,img_path,batch,self.data,mode=mode,rect=False,stride=32)
        def _build_train_pipeline(self):
            if getattr(self,'built_once',False):raise RuntimeError('OOM automatic retry disabled; batch4 must not change')
            self.built_once=True
            return super()._build_train_pipeline()
        def _setup_train(self):
            # Synthetic AMP preflight tests the actual autocast/scaler path; avoid hidden auxiliary-model inference.
            from unittest.mock import patch
            with patch('ultralytics.engine.trainer.check_amp',lambda model:True):super()._setup_train()
            for k,v in c['train'].items():
                if k!='resume' and getattr(self.args,k)!=v:raise ValueError('Resolved controlled setting differs: '+k)
            if not self.amp or self.accumulate!=4 or type(self.optimizer).__name__!='MuSGD':raise ValueError('AMP/accumulation/optimizer mismatch')
            if resuming:
                self.scheduler.load_state_dict(resume_state['controlled_scheduler'])
                if self.start_epoch!=resume_state['epoch']+1:raise ValueError('Resume epoch mismatch')
                event(base,'training_state_restored',start_epoch=self.start_epoch+1,
                      optimizer_state_entries=len(self.optimizer.state),scheduler=self.scheduler.state_dict(),
                      scaler=self.scaler.state_dict(),ema_updates=self.ema.updates)
            (Path(base)/'resolved_training.json').write_text(json.dumps(dict(settings=vars(self.args),
                accumulation=self.accumulate,optimizer=type(self.optimizer).__name__,amp=bool(self.amp),
                start_epoch=self.start_epoch),indent=2),encoding='utf-8')
        def save_model(self):
            # Let the pinned framework serialize all native state into a staging directory.
            original=(self.wdir,self.last,self.best)
            with tempfile.TemporaryDirectory(prefix='checkpoint_stage_',dir=Path(base)) as d:
                self.wdir=Path(d);self.last=self.wdir/'last.pt';self.best=self.wdir/'best.pt'
                try:
                    super().save_model()
                    value=torch.load(self.last,map_location='cpu',weights_only=False)
                    live=self.model._orig_mod if hasattr(self.model,'_orig_mod') else self.model
                    state={k:v.detach().cpu().clone() for k,v in live.state_dict().items()}
                    if any(not torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):raise FloatingPointError('Nonfinite checkpoint')
                    value.update(controlled_identity=identity,controlled_live_model_state=state,
                                 controlled_scheduler=self.scheduler.state_dict(),
                                 controlled_optimizer_steps=self.controlled_optimizer_steps,
                                 controlled_resume_semantics='Native epoch-boundary continuation; not bitwise equivalence')
                    buf=io.BytesIO();torch.save(value,buf);data=buf.getvalue()
                    # Publish immutable epoch checkpoint first; canonical last is an atomic alias of it.
                    epoch_path=original[0]/f'epoch{self.epoch}.pt'
                    if epoch_path.exists():raise FileExistsError('Never overwrite a committed epoch checkpoint')
                    atomic_bytes(epoch_path,data)
                    atomic_bytes(original[1],data)
                    if self.best_fitness==self.fitness:atomic_bytes(original[2],data)
                    event(base,'checkpoint_committed',completed_epoch=self.epoch+1,
                          checkpoint=str(epoch_path),checkpoint_sha256=digest(epoch_path))
                finally:self.wdir,self.last,self.best=original
            return True
        def optimizer_step(self):
            if not torch.isfinite(self.loss).all():raise FloatingPointError('NaN/Inf loss')
            return super().optimizer_step()
        def _handle_nan_recovery(self,*a,**kw):
            if not torch.isfinite(self.loss).all():raise FloatingPointError('NaN recovery disabled')
            if self.fitness is not None and not torch.isfinite(torch.tensor(self.fitness)):raise FloatingPointError('Nonfinite validation')
            return False
        def final_eval(self):return None
    return RecoverableTrainer
