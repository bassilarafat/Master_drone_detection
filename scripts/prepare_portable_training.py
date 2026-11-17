"""Prepare the reviewable laptop training configuration; never starts training."""
if __name__ == '__main__':
    raise RuntimeError('Archived draft generator: use the existing v3 configuration and tools/preflight_controlled.py; do not regenerate over reviewed metadata.')
import hashlib,json
from pathlib import Path
import yaml
root=Path(__file__).resolve().parents[1]
p=root/'configs/paper/protocol_yolo26_v2.json'
v=json.loads(p.read_text())
train=dict(v['train'],batch=4,workers=2,amp=True,save_period=-1)
proposal=dict(schema_version=3,status='DRAFT_FOR_RESEARCHER_REVIEW_NOT_BENCHMARKED',
    version='controlled_yolo26_laptop_v3',
    base_protocol=dict(path=p.relative_to(root).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()),
    seeds=[17,42,123],train=train,
    output_root='04_experiments/controlled_yolo26_v3',
    checkpoint_policy=dict(final='last.pt epoch100',save='every completed epoch',
        resume='same arm, seed, configuration and verified checkpoint only',
        power_loss='resume from last intact completed epoch; incomplete epoch repeats',
        exact_equivalence='Interrupted training is recorded separately; bitwise equivalence to uninterrupted training is not claimed'),
    requirements_before_launch=['Researcher reviews the files','Validate batch4/AMP on both E0 and E1 without using Test or outcome-based selection','Seal the reviewed configuration and implementation before starting'],
    test_inference_allowed=False,automatic_followup=False)
out=root/'configs/paper/yolo26_laptop_v3.review.yaml'
if out.exists():raise FileExistsError(out)
out.write_text(yaml.safe_dump(proposal,sort_keys=False,allow_unicode=True),encoding='utf-8')
print(out)
