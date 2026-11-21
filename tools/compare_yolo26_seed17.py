"""Audit and compare saved controlled E0/E1 Validation exports; no inference."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '04_experiments/controlled_yolo26_v3'
OUT = ROOT / '06_results/controlled_yolo26_v3_seed17_comparison'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    summaries, evaluations, identities, objects = [], [], [], []
    for arm in ('E0_Y26', 'E1_Y26_P2'):
        run = BASE / arm / 'seed_17'
        directory = run / 'evaluation/val'
        summary = json.loads((directory / 'summary.json').read_text())
        evaluation = json.loads((directory / 'evaluation.json').read_text())
        state = json.loads((run / 'state.json').read_text())
        assert state['status'] == evaluation['status'] == 'COMPLETED'
        assert state['last_completed_epoch'] == 100
        assert evaluation['dataset_verified_after'] and not evaluation['test_evaluated']
        assert sha(run / 'run/weights/last.pt') == summary['checkpoint_sha256'] == evaluation['checkpoint_sha256']
        assert sha(directory / 'predictions.jsonl') == evaluation['raw_sha256']
        records = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines()]
        identities.append({r['image']: (r['image_sha256'], r['label_sha256'], r['targets'], r['scene_group_id']) for r in records})
        assert len(identities[-1]) == summary['images'] == len(records)
        with (directory / 'objects.csv').open(newline='') as f:
            gt = {(r['image'], r['gt_id']): r for r in csv.DictReader(f) if r['kind'] == 'GT'}
        assert len(gt) == summary['GT']
        assert sum(r['status'] == 'TP' for r in gt.values()) == summary['TP']
        objects.append(gt)
        summaries.append(summary)
        evaluations.append(evaluation)
    assert evaluations[0]['settings'] == evaluations[1]['settings']
    assert identities[0] == identities[1]
    assert summaries[0]['dataset_release'] == summaries[1]['dataset_release']
    assert objects[0].keys() == objects[1].keys()
    OUT.mkdir(parents=True, exist_ok=True)
    transitions = []
    for key, a in objects[0].items():
        b = objects[1][key]
        assert (a['box'], a['size']) == (b['box'], b['size'])
        transitions.append(dict(image=key[0], gt_id=key[1], size=a['size'], E0=a['status'], E1=b['status'],
                                change='recovered' if a['status'] == 'FN' and b['status'] == 'TP' else
                                'lost' if a['status'] == 'TP' and b['status'] == 'FN' else 'unchanged'))
    with (OUT / 'object_transitions.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(transitions[0])); w.writeheader(); w.writerows(transitions)
    a, b = summaries
    lines = ['# Controlled YOLO26: seed 17 Validation comparison', '',
             'Checkpoint: last.pt at epoch 100, as prescribed by the v3 checkpoint policy.',
             'Saved-export audit only; no new inference or Test access. Both runs and evaluations completed.',
             'Verified current checkpoint hashes, raw prediction hashes, identical image/label/target/scene identities and evaluation settings.',
             'Population: 1,827 Validation images, 1,867 annotated objects. Input 960; NMS IoU 0.70; maxDet 300; operating confidence 0.25; matching IoU 0.50; AP prediction floor 0.001.', '',
             '| Metric | E0 | E1 | E1 minus E0 |', '|---|---:|---:|---:|']
    for k in ('mAP50', 'mAP50_95', 'precision', 'recall', 'F1'):
        lines.append(f'| {k} (%) | {a[k]*100:.2f} | {b[k]*100:.2f} | {(b[k]-a[k])*100:+.2f} pp |')
    for k in ('TP', 'FP', 'FN'):
        lines.append(f'| {k} | {a[k]} | {b[k]} | {b[k]-a[k]:+d} |')
    lines += ['', '| Size | GT | E0 recall (%) | E1 recall (%) | Difference (pp) | Recovered | Lost |', '|---|---:|---:|---:|---:|---:|---:|']
    for size in a['recall_by_size']:
        x, y = a['recall_by_size'][size], b['recall_by_size'][size]
        recovered = sum(t['size'] == size and t['change'] == 'recovered' for t in transitions)
        lost = sum(t['size'] == size and t['change'] == 'lost' for t in transitions)
        assert recovered-lost == y['TP']-x['TP']
        lines.append(f'| {size} | {x["GT"]} | {x["recall"]*100:.2f} | {y["recall"]*100:.2f} | {(y["recall"]-x["recall"])*100:+.2f} | {recovered} | {lost} |')
    lines += ['', '## False positive categories', '', '| Category | E0 | E1 |', '|---|---:|---:|']
    for k in a['fp_subtype_counts']:
        lines.append(f'| {k} | {a["fp_subtype_counts"][k]} | {b["fp_subtype_counts"][k]} |')
    lines += ['', '## Interpretation and next steps', '',
              'E1 improves overall recall and AP for seed 17, driven mostly by Tiny objects. Medium recall decreases. Object transitions are paired by image and GT index; recovered/lost counts are changes on individual targets, not just net totals.',
              'False-positive categories are geometric. Semantic FP review remains unreviewed; there are no independently verified negative images, so background-only FPR is unavailable.',
              'AP uses the pinned custom 101-point evaluator and is not full COCO crowd/area evaluation. These scores differ from Ultralytics training-log metrics because the evaluator differs.',
              'One seed cannot establish training stability or statistical significance. Complete E0/E1 seeds 42 and 123 regardless of seed-17 outcome, then report paired differences and mean/sample SD (ddof=1).',
              'Keep Test locked. The approved study records prior descriptive Test exposure; independent final confirmation needs the prospective external protocol.',
              'Source exports: 04_experiments/controlled_yolo26_v3/{E0_Y26,E1_Y26_P2}/seed_17/evaluation/val/.', '']
    (OUT / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
