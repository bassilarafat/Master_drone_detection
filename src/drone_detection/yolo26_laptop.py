"""Laptop-ready draft: explicit start/resume, epoch-boundary pause, atomic backups."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime,timezone
import yaml
from .paper_common import PROJECT_ROOT,project_path,pin_release,write_json
from .paper_runner import configure_runtime,training_view
from .training import _without_snapshot_cache_writes
from .yolo26_controlled import shared_model

CONFIG='configs/paper/yolo26_laptop_v3.review.yaml'
SEAL='configs/paper/YOLO26_LAPTOP_V3_LOCK.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_copy(source,destination):
    """Previous published backup remains intact until replacement is complete."""
    source,destination=Path(source),Path(destination)
    temporary=destination.with_suffix(destination.suffix+'.tmp')
    with source.open('rb') as src,temporary.open('wb') as dst:
        while block:=src.read(1024*1024):dst.write(block)
        dst.flush();os.fsync(dst.fileno())
    os.replace(temporary,destination)


def load(arm,seed,locked=False):
    if arm not in ('E0','E1') or type(seed) is not int or seed not in (17,42,123):
        raise ValueError('Only E0/E1 and seeds17/42/123 are valid')
    cfg=yaml.safe_load(project_path(CONFIG).read_text(encoding='utf-8'))
    reference=cfg['base_protocol']
    if sha(project_path(reference['path']))!=reference['sha256']:raise ValueError('Base protocol changed')
    p=json.loads(project_path(reference['path']).read_text())
    if locked:
        seal=json.loads(project_path(SEAL).read_text())
        for name,value in seal['hashes'].items():
            if sha(project_path(name))!=value:raise ValueError(f'Sealed input changed: {name}')
    for item in (p['models'][arm],p['initialization']):
        if sha(project_path(item['path']))!=item['sha256']:raise ValueError('Pinned model/weights changed')
    c=dict(architecture=arm,experiment_id=p['experiments'][arm]['experiment_id'],
           dataset=p['dataset'],model=p['models'][arm],initialization=p['initialization'],
           train=cfg['train'],evaluation=p['evaluation'],negative_mode='none',
           output_root=cfg['output_root']+'/'+p['experiments'][arm]['experiment_id'])
    return cfg,c,p


def choose_checkpoint(base):
    from .yolo26_recovery import latest
    p,c,errors=latest(base)
    return p,c


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['check','seal','start','resume','pause'],default='check')
    parser.add_argument('--arm',choices=['E0','E1'],default='E0')
    parser.add_argument('--seed',type=int,choices=[17,42,123],default=17)
    parser.add_argument('--execute-training',action='store_true')
    args=parser.parse_args()
    from . import yolo26_workflow as w
    if args.mode=='check':
        cfg,c,p=w.check(args.arm,args.seed)
        print(json.dumps(dict(status=cfg['status'],training_started=False,train=c['train']),indent=2))
    elif args.mode=='seal':print(json.dumps(w.seal(),indent=2))
    elif args.mode=='pause':
        _,c,_=w.check(args.arm,args.seed)
        (project_path(c['output_root'])/f'seed_{args.seed}'/'PAUSE_REQUESTED').touch()
    else:
        if not args.execute_training:parser.error('Explicit execution required')
        w.execute(args.arm,args.seed,resume=args.mode=='resume')

if __name__=='__main__':main()
