"""Pinned paper identities and paths; importing this module never loads a model."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
from .paths import PROJECT_ROOT, project_path
from .data_audit import digest

DATASET = 'DUT_Anti_UAV_Clean_v2'
MANIFEST_SHA = 'ad46c518a7953ca3bafeeaa6d475cebe8fb196706cc078aa70b10d7e67a1a1d5'
FREEZE_SHA = '9d26198e83d537b96734c48de3220fc0c46646e15d81525d8036d6eb57c6c058'
RELEASE_ID = DATASET + '@' + MANIFEST_SHA
SIZES = ('Tiny', 'Small', 'Medium', 'Large')

def bootstrap():
    # Ultralytics must not use a user-profile settings directory or dataset caches.
    config = PROJECT_ROOT / '.cache/paper/ultralytics'
    config.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(config)
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['COMET_MODE'] = 'DISABLED'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')

def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def outside_dataset(path, dataset):
    path, dataset = Path(path).resolve(), Path(dataset).resolve()
    if path == dataset or path.is_relative_to(dataset):
        raise ValueError('All outputs/caches must stay outside the frozen dataset')
    if any((p/'FREEZE.json').is_file() for p in (path,*path.parents)):
        raise ValueError('Outputs/caches may not be placed inside any frozen dataset release')
    return path

def pin_release(dataset, expected=None, *, full=False):
    dataset = Path(dataset).resolve()
    expected = expected or dict(release_id=RELEASE_ID, manifest_sha256=MANIFEST_SHA, freeze_sha256=FREEZE_SHA)
    actual = dict(release_id=dataset.name+'@'+digest(dataset/'MANIFEST.csv'),
                  manifest_sha256=digest(dataset/'MANIFEST.csv'), freeze_sha256=digest(dataset/'FREEZE.json'))
    if dataset.name != DATASET or any(actual[k] != expected[k] for k in actual):
        raise ValueError('Paper dataset release identity differs from the pinned Clean_v2 release')
    if full:
        from .dataset_snapshot import verify_snapshot
        verify_snapshot(dataset)
    return actual

def size_bin(area):
    # Caller may pass Decimal to retain exact documented pixel-area boundaries.
    return SIZES[0] if area < 32**2 else SIZES[1] if area < 96**2 else SIZES[2] if area < 256**2 else SIZES[3]

def allow_split(split):
    if split not in ('train', 'val'):
        raise ValueError('Final Test is locked: paper inference/model selection accepts train or val only')

def load_protocol(path='configs/paper/protocol.json'):
    return json.loads(project_path(path).read_text(encoding='utf-8'))

def source_fingerprints():
    paths=list((PROJECT_ROOT/'src/drone_detection').glob('paper_*.py'))
    paths.extend(PROJECT_ROOT/'src/drone_detection'/name for name in
                 ('training.py','cli.py','metrics.py','dataset_snapshot.py','data_audit.py','paths.py','__main__.py'))
    return {p.relative_to(PROJECT_ROOT).as_posix():digest(p) for p in sorted(paths)}
