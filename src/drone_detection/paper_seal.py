"""Bind the reviewable, not-yet-run paper protocol to public weights and implementation bytes."""
import json
from pathlib import Path
import yaml
from .paper_common import PROJECT_ROOT,source_fingerprints,write_json
from .data_audit import digest
from .paper_weights import RELATIVE,SHA,URL,COMMIT

def seal():
    root=PROJECT_ROOT;folder=root/'configs/paper'
    if (folder/'CONTROL_LOCK.json').exists():raise FileExistsError('Paper controls already sealed; preserve the previous protocol when revising')
    if list((root/'04_experiments/paper_v1').glob('*/seed_*/run')):raise ValueError('Never revise controls after a paper run has started')
    p=json.loads((folder/'protocol.json').read_text())
    if digest(root/RELATIVE)!=SHA:raise ValueError('Public pretrained checkpoint does not match its official hash')
    p['initialization'].update(path=RELATIVE,sha256=SHA,source_url=URL,repository_commit=COMMIT,
        remote_digest_independently_verified=True,
        policy='COCO-pretrained shared backbone modules 0..9 only; neck/detection parameters use the run-seeded standard initialization, including fixed DFL bin projection')
    p['negatives'].update(max_hard_proposals=1024,max_hard_crops_per_source=1,
        review_catalog_identity_preserved=True,random_rejection_policy='Seed17 sampling with explicitly recorded rejected-crop hash exclusions; all new proposals require review')
    preflight=root/'06_results/paper/Gate3_20261110/runtime_preflight.json'
    report=json.loads(preflight.read_text())
    if report['status']!='PASS' or any(report['checks'].values()):raise ValueError('Ultralytics runtime safety preflight has not passed')
    p['gate3'].update(runtime_preflight=preflight.relative_to(root).as_posix(),runtime_preflight_sha256=digest(preflight))
    p['implementation_sha256']=source_fingerprints()
    write_json(folder/'protocol.json',p);sha=digest(folder/'protocol.json')
    provenance=json.loads((root/RELATIVE).parent.joinpath('provenance.json').read_text())
    write_json(folder/'pretrained.provenance.json',provenance)
    configs={}
    for arm in ('E0','E1','E2','E3'):
        path=folder/f'{arm}.yaml';c=yaml.safe_load(path.read_text())
        c.update(initialization=p['initialization'],protocol_sha256=sha)
        path.write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8');configs[arm]=digest(path)
        for seed in p['seeds']:
            plan=root/f'04_experiments/paper_v1/{arm}/seed_{seed}/PLAN.json'
            record=json.loads(plan.read_text());record.update(config_sha256=configs[arm],protocol_sha256=sha)
            write_json(plan,record)
    write_json(folder/'CONTROL_LOCK.json',dict(protocol_sha256=sha,config_sha256=configs,
        dataset_release=p['dataset'],training_started=False,test_unlocked=False))
    return dict(protocol_sha256=sha,configs=configs)

if __name__=='__main__':print(json.dumps(seal(),indent=2))
