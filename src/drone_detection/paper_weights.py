"""Explicit public-weight preparation; never called implicitly by training or dry-runs."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import urllib.request
from .paper_common import PROJECT_ROOT,write_json

COMMIT='8a9e1a55f987a77f9966c2ac3f80aa8aa37b3c1a'
SHA='268e5bb54c640c96c3510224833bc2eeacab4135c6deb41502156e39986b562d'
RELATIVE='05_models/pretrained/paper/yolov8s_coco_268e5bb5.pt'
URL=f'https://huggingface.co/Ultralytics/YOLOv8/resolve/{COMMIT}/yolov8s.pt'

def prepare_weights():
    target=PROJECT_ROOT/RELATIVE;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest()!=SHA:raise ValueError('Existing paper pretrained file has changed; it is never silently replaced')
    else:
        partial=target.with_suffix('.pt.part');h=hashlib.sha256();size=0
        with urllib.request.urlopen(URL,timeout=60) as response,partial.open('wb') as f:
            while block:=response.read(1024*1024):f.write(block);h.update(block);size+=len(block)
        if h.hexdigest()!=SHA or size!=22573363:raise ValueError('Official pretrained artifact does not match the published LFS SHA/size')
        partial.rename(target)
    provenance=dict(path=RELATIVE,sha256=SHA,bytes=target.stat().st_size,source_url=URL,
        publisher='Ultralytics',repository_commit=COMMIT,published_digest_url=f'https://huggingface.co/Ultralytics/YOLOv8/raw/{COMMIT}/yolov8s.pt',
        downloaded_at_utc=datetime.now(timezone.utc).isoformat(),remote_digest_independently_verified=True,training_performed=False)
    write_json(target.parent/'provenance.json',provenance);return provenance

if __name__=='__main__':print(json.dumps(prepare_weights(),indent=2))
