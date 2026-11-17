"""Execution-only E0/E1 check on a fixed Train/Val subset; never Test."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import argparse,gc,json,shutil
from datetime import datetime,timezone
import yaml
from PIL import Image,ImageDraw
from drone_detection.paper_runner import configure_runtime
from drone_detection.paper_common import pin_release,write_json,read_csv
from drone_detection.yolo26_workflow import check,identity,CONFIG
from drone_detection.yolo26_recovery import latest,trainer_class,digest,checkpoint,atomic_bytes


def main():
    configure_runtime()
    import torch
    from ultralytics import YOLO
    from drone_detection.paper_metrics import export_metrics
    out=ROOT/'06_results/paper_yolo26_v3';out.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    attempt=out/('attempt_'+stamp);attempt.mkdir()
    if (out/'hardware_preflight.json').exists():shutil.copy2(out/'hardware_preflight.json',attempt/'previous_preflight_report.json')
    report=dict(status='RUNNING',configuration_sha256=digest(ROOT/CONFIG),arms=['E0','E1'],
                execution_only=True,official_training_started=False,test_inference_performed=False,checks={},details={})
    report['implementation_sha256']={name:digest(ROOT/name) for name in (
        'src/drone_detection/yolo26_workflow.py','src/drone_detection/yolo26_recovery.py',
        'src/drone_detection/yolo26_laptop.py','src/drone_detection/yolo26_validation.py','tools/preflight_controlled.py')}
    write_json(out/'hardware_preflight.json',report)
    try:
        cfg,c,p=check('E0',17)
        pin_release(ROOT/c['dataset']['path'],c['dataset'],full=True)
        report['checks']['dataset_integrity_before']=True
        dataset=ROOT/c['dataset']['path']
        index=read_csv(dataset/'dataset_manifest.csv')
        for arm in ('E0','E1'):
            try:
                _,c,_=check(arm,17);c['train']=dict(c['train'],epochs=2)
                base=attempt/('synthetic_'+arm)
                base.mkdir(exist_ok=False)
                fixture_rows={}
                for split,n in [('train',512),('val',4)]:
                    images=base/'fixture/images'/split;labels=base/'fixture/labels'/split
                    images.mkdir(parents=True);labels.mkdir(parents=True)
                    selected=sorted([r for r in index if r['new_split']==split],key=lambda r:r['new_relative_image'])[:n]
                    for i,r in enumerate(selected):
                        shutil.copyfile(dataset/r['new_relative_image'],images/f'{i:03}.jpg')
                        shutil.copyfile(dataset/r['new_relative_label'],labels/f'{i:03}.txt')
                    fixture_rows[split]=selected
                write_json(base/'fixture/provenance.json',dict(selection='first512 Train and first4 Val sorted by new_relative_image; fixed without inspecting scores',
                    rows=fixture_rows,test_images_loaded=False,execution_only=True))
                data=base/'fixture/data.yaml'
                data.write_text(yaml.safe_dump(dict(path=str(base/'fixture'),train='images/train',val='images/val',nc=1,names={0:'UAV'})))
                ident=identity(c,arm,17);ident['dataset_release']='EXECUTION_PREFLIGHT_SUBSET_ONLY'
                torch.cuda.reset_peak_memory_stats()
                seen=[]
                model=YOLO(str(ROOT/c['model']['path']))
                def stop_after_one(t):seen.append(t.epoch+1);t.stop=True
                model.add_callback('on_fit_epoch_end',stop_after_one)
                model.train(trainer=trainer_class(c,base,ident),**dict(c['train'],seed=17,data=str(data),project=str(base),name='run',exist_ok=False))
                cp,state,_=latest(base,ident)
                assert state['epoch']==0 and state['optimizer']['state'] and state['scaler'] and state['controlled_optimizer_steps']>0
                init=json.loads((base/'run/initialization.json').read_text())
                assert init['backbone_tensor_sha256']==cfg['initialization']['source_backbone_sha256']
                del model;gc.collect();torch.cuda.empty_cache()
                # Simulate a process interruption with a truncated canonical checkpoint.
                atomic_bytes(base/'run/weights/last.pt',b'intentionally corrupted synthetic checkpoint')
                recovered,state,errors=latest(base,ident)
                assert recovered.name=='epoch0.pt' and errors
                assert checkpoint(recovered,ident)['controlled_scheduler']
                wrong=dict(ident,seed=42)
                try:latest(base,wrong)
                except ValueError:pass
                else:raise AssertionError('Protocol mismatch accepted')
                model=YOLO(str(recovered));finished=[]
                model.add_callback('on_fit_epoch_end',lambda t:finished.append(t.epoch+1))
                model.train(trainer=trainer_class(c,base,ident,True,state),**dict(c['train'],seed=17,
                    resume=str(recovered),data=str(data),project=str(base),name='run',exist_ok=True))
                cp,final,_=latest(base,ident)
                assert final['epoch']==1 and finished==[2],finished
                restored=list((base/'events').glob('*.json'))
                events=[json.loads(x.read_text()) for x in restored]
                assert any(x['event']=='training_state_restored' and x['start_epoch']==2 for x in events)
                assert all(torch.isfinite(x).all() for x in final['controlled_live_model_state'].values() if x.is_floating_point())
                # Exercise the same final Validation exporter used by official v3 runs.
                del model;gc.collect();torch.cuda.empty_cache()
                from drone_detection.yolo26_validation import evaluate_validation
                evaluate_validation(c,cp,base,{'seed':17},
                    validation_subset=[r['new_relative_image'] for r in fixture_rows['val']],execution_only=True)
                report['details'][arm]=dict(strides=init['strides'],initialized_backbone_sha256=init['backbone_tensor_sha256'],
                    peak_gpu_bytes=torch.cuda.max_memory_allocated(),first_epochs=seen,resumed_epochs=finished,
                    checkpoint_reloaded=True,optimizer_state_restored=True,scaler_state_restored=True,scheduler_state_restored=True,
                    corrupted_canonical_fallback=True,validation_evaluator_invoked=True,finite_model=True,
                    result_scope='Fixed Train/Val execution subset only; no research performance inference')
                report['checks'][arm+'_batch4_amp_resume']=True
                del state,final;gc.collect();torch.cuda.empty_cache()
                write_json(out/'hardware_preflight.json',report)
            except Exception as exc:
                report['details'][arm]=dict(status='FAIL',error=f'{type(exc).__name__}: {exc}')
                report['checks'][arm+'_batch4_amp_resume']=False
                try:del model
                except UnboundLocalError:pass
                gc.collect();torch.cuda.empty_cache()
                write_json(out/'hardware_preflight.json',report)
        from drone_detection.paper_common import allow_split
        try:allow_split('test')
        except ValueError:report['checks']['test_guard']=True
        else:raise AssertionError('Test was not blocked')
        pin_release(ROOT/c['dataset']['path'],c['dataset'],full=True)
        both=all(report['checks'].get(a+'_batch4_amp_resume') is True for a in ('E0','E1'))
        report['checks'].update(dataset_integrity_after=True,initialization=both,checkpoint_save_reload=both,
                               simulated_resume=both,corrupt_fallback=both,protocol_mismatch_rejected=both)
        report['status']='PASS' if both else 'FAIL'
    except BaseException as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['finished_at_utc']=datetime.now(timezone.utc).isoformat();write_json(out/'hardware_preflight.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
