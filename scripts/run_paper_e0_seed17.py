"""Durable E0-only launcher: exact approved command, logs, progress, and final artifacts.

Never starts E1/E2/E3, resumes a failed run, changes controls, or unlocks Test.
"""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

from drone_detection.paper_runner import load_config
from drone_detection.paper_common import project_path
from drone_detection.data_audit import digest


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def progress(base):
    values = {}
    result = base/'run/results.csv'
    if result.is_file():
        with result.open(newline='',encoding='utf-8-sig') as f:
            rows = [{k.strip():v for k,v in row.items() if k} for row in csv.DictReader(f)]
        rows = [r for r in rows if r.get('epoch','').strip()]
        if rows:
            values.update(completed_epochs=len(rows), last_epoch=int(float(rows[-1]['epoch'])),
                          epoch_csv_elapsed_seconds=float(rows[-1]['time']))
    meta = base/'run.json'
    if meta.is_file():
        values['paper_run_status']=json.loads(meta.read_text())['status']
    return values


def main():
    c,p,path = load_config('configs/paper/E0.yaml')
    assert c['experiment_id']=='E0' and c['train']['epochs']==100 and c['train']['imgsz']==960
    assert c['evaluation']['test_locked'] and c['default_seed']==17
    root=project_path('.');base=project_path(c['output_root'])/'seed_17'
    assert not (base/'run').exists() and not (base/'runtime_data').exists(), 'Never overwrite/resume a paper run'
    assert not list(project_path('04_experiments/paper_v1').glob('*/seed_*/run')), 'No concurrent paper runs'
    gpu_path=project_path('06_results/paper/pretraining_review_20261110/gpu_preflight.json')
    gpu=json.loads(gpu_path.read_text())
    assert gpu['status']=='PASS' and gpu['config_sha256']['E0']==digest(path) and gpu['protocol_sha256']==c['protocol_sha256']
    state=base/'EXECUTION.json'
    command=[sys.executable,'-m','drone_detection','paper','train','--config','configs/paper/E0.yaml','--seed','17','--execute-training']
    record=dict(status='LAUNCHING',experiment_id='E0',seed=17,epochs_requested=100,
                authorization='User explicitly requested E0 only, then review results before any E1; E2/E3 not authorized.',
                command=command,working_directory=str(root),launcher_pid=os.getpid(),launcher_sha256=digest(Path(__file__)),
                started_at_utc=now(),config_sha256=digest(path),protocol_sha256=c['protocol_sha256'],
                dataset_release=c['dataset'],gpu_preflight_sha256=digest(gpu_path),
                final_test_locked=True,e1_e2_e3_started=False,automatic_retry=False,completed_epochs=0)
    with state.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
    start=time.monotonic();child=None
    try:
        env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONUTF8='1')
        with (base/'console.log').open('xb') as log:
            child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            record.update(status='RUNNING',training_process_pid=child.pid);save(state,record)
            while child.poll() is None:
                try:record.update(progress(base))
                except (ValueError,OSError,json.JSONDecodeError):pass  # An epoch write may be in flight.
                record.update(last_checked_at_utc=now(),elapsed_wall_seconds=time.monotonic()-start)
                if record.get('paper_run_status')=='COMPLETED':
                    record.setdefault('training_completed_observed_at_utc',now())
                    record['phase']='UNIFIED_VALIDATION_OR_POSTCHECK'
                else:record['phase']='TRAINING_OR_INITIALIZATION'
                save(state,record);time.sleep(15)
        record.update(exit_code=child.returncode,**progress(base))
        if child.returncode!=0:raise RuntimeError(f'E0 command exited {child.returncode}; inspect console.log. No automatic retry.')
        evaluation=base/'evaluation/val';meta=json.loads((evaluation/'evaluation.json').read_text())
        assert record['completed_epochs']==100 and record['last_epoch']==100
        assert meta['status']=='COMPLETED' and meta['dataset_verified_after'] and meta['split']=='val' and not meta['test_evaluated']
        expected={'results.csv':base/'run/results.csv','summary.json':evaluation/'summary.json',
                  'recall_by_size.csv':evaluation/'recall_by_size.csv','fp_subtypes.csv':evaluation/'fp_subtypes.csv'}
        record['artifacts']={name:dict(path=str(f.relative_to(root)),sha256=digest(f)) for name,f in expected.items()}
        record.update(status='COMPLETED',phase='FINISHED',dataset_integrity_verified_after=True)
    except BaseException as exc:
        record.update(status='FAILED',phase='STOPPED',error=f'{type(exc).__name__}: {exc}')
        (base/'launcher_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        if child is not None:record['exit_code']=child.poll()
    finally:
        record.update(finished_at_utc=now(),elapsed_wall_seconds=time.monotonic()-start)
        save(state,record)
    if record['status']=='COMPLETED':
        packet=base/'E0_seed17_results.zip'
        with zipfile.ZipFile(packet,'x',zipfile.ZIP_DEFLATED) as z:
            for name,item in record['artifacts'].items():z.write(root/item['path'],name)
            for f in (state,base/'run.json',evaluation/'evaluation.json'):z.write(f,f.name)
        print(json.dumps(dict(status='COMPLETED',packet=str(packet),elapsed_wall_seconds=record['elapsed_wall_seconds']),indent=2))
    else:
        print(json.dumps(record,indent=2));return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
