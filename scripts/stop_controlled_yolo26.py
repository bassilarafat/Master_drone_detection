"""Stop the verified v2 process tree and preserve its last completed checkpoint."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import psutil

root=Path(__file__).resolve().parents[1]
campaign=root/'04_experiments/controlled_yolo26_v2'
parent=psutil.Process(int((campaign/'logs/active_training.pid').read_text().strip()))
assert 'drone_detection.yolo26_controlled' in parent.cmdline()
children=parent.children(recursive=True)
trainers=[p for p in children if 'drone_detection.yolo26_controlled' in p.cmdline()]
assert len(trainers)==1
trainer=trainers[0]
identities=[dict(pid=p.pid,created_at=p.create_time(),cmdline=p.cmdline()) for p in [parent,*children]]
trainer.suspend();parent.suspend()
for p in children:
    if p.pid!=trainer.pid:
        try:p.suspend()
        except psutil.NoSuchProcess:pass
for p in [trainer,*[p for p in children if p.pid!=trainer.pid],parent]:
    try:p.terminate()
    except psutil.NoSuchProcess:pass
_,alive=psutil.wait_procs([parent,*children],timeout=10)
assert not alive
base=campaign/'E0_Y26/seed_17'
record_path=base/'run.json'
previous=record_path.read_bytes()
(base/'run.before_user_stop.json').write_bytes(previous)
rows=list(csv.DictReader((base/'run/results.csv').open()))
files=[base/'run/results.csv',*sorted((base/'run/weights').glob('*.pt'))]
record=dict(status='STOPPED_BY_USER_FOR_PROSPECTIVE_REVISION',stopped_at_utc=datetime.now(timezone.utc).isoformat(),
    last_completed_epoch=int(float(rows[-1]['epoch'])),completed_epochs=len(rows),
    classification='Stopped v2 attempt; do not combine with future v3 runs',
    reason='Researcher requested faster settings, review before launch, and interruption/resume support',
    stopped_processes=identities,artifacts_preserved=[dict(path=p.relative_to(root).as_posix(),
    bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files],
    new_training_started=False)
(base/'USER_STOP_RECORD.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
run=json.loads(previous);run.update(status=record['status'],stopped_at_utc=record['stopped_at_utc'],
    stop_record=str((base/'USER_STOP_RECORD.json').relative_to(root)),excluded_from_future_v3_aggregates=True)
record_path.write_text(json.dumps(run,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':record['status'],'saved_through_epoch':record['last_completed_epoch'],
    'preserved':[p.relative_to(root).as_posix() for p in files]},indent=2))
