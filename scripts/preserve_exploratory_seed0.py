"""Stop only the verified historical-settings campaign; preserve its artifacts."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import psutil


def main():
    root = Path(__file__).resolve().parents[1]
    campaign = root / '04_experiments/clean_v2_standard_protocol'
    parent = psutil.Process(int((campaign / 'logs/training.pid').read_text().strip()))
    expected = 'scripts/train_clean_v2_standard_seeds.py'
    assert expected in [x.replace('\\', '/') for x in parent.cmdline()], 'Unexpected launcher PID'
    children = parent.children(recursive=True)
    trainers = [p for p in children if expected in [x.replace('\\', '/') for x in p.cmdline()]]
    assert len(trainers) == 1, 'Expected one verified training child'
    trainer = trainers[0]
    identities = [{'pid': p.pid, 'created_at': p.create_time(), 'cmdline': p.cmdline()}
                  for p in [parent, *children]]
    trainer.suspend()
    parent.suspend()
    for p in children:
        if p.pid != trainer.pid:
            try:
                p.suspend()
            except psutil.NoSuchProcess:
                pass
    for p in [trainer, *[p for p in children if p.pid != trainer.pid], parent]:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs([parent, *children], timeout=10)
    assert not alive, 'Verified training processes remain alive'
    run = campaign / 'YOLO26s_Clean_v2_standard_seed0'
    rows = list(csv.DictReader((run / 'results.csv').open(encoding='utf-8-sig')))
    artifacts = []
    for p in [run / 'results.csv', run / 'args.yaml', *sorted((run / 'weights').glob('*.pt'))]:
        artifacts.append(dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size,
                              sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    protocol = campaign / 'protocol.json'
    archive = campaign / 'protocol.before_exploratory_stop.json'
    if archive.exists():
        raise FileExistsError(archive)
    archive.write_bytes(protocol.read_bytes())
    now = datetime.now(timezone.utc).isoformat()
    record = dict(status='STOPPED_BY_RESEARCHER_REQUEST',
                  classification='Exploratory E0 / seed0; excluded from controlled E0-E3 seed aggregates',
                  stopped_at_utc=now,
                  reason='Seed0 and auto-selected AdamW do not match approved controlled seeds and explicit MuSGD',
                  completed_epochs=len(rows), last_completed_epoch=int(float(rows[-1]['epoch'])),
                  last_validation_row=rows[-1], partial_current_epoch='Not saved beyond the last checkpoint',
                  processes_stopped=identities, artifacts_preserved_in_place=artifacts,
                  later_campaign_seeds_started=False, test_evaluated_by_this_action=False)
    (campaign / 'EXPLORATORY_STOP_RECORD.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    p = json.loads(protocol.read_text())
    p.update(status='stopped_exploratory', finished_at_utc=now, classification=record['classification'],
             stop_record='04_experiments/clean_v2_standard_protocol/EXPLORATORY_STOP_RECORD.json')
    p['runs'][-1].update(status='stopped_exploratory', finished_at_utc=now, completed_epochs=len(rows))
    protocol.write_text(json.dumps(p, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=record['status'], saved_through_epoch=record['last_completed_epoch'],
                          weights_preserved=[a['path'] for a in artifacts if a['path'].endswith('.pt')],
                          stopped_pids=[p['pid'] for p in identities]), indent=2))


if __name__ == '__main__':
    main()
