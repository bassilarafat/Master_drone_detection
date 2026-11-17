"""Read-only dataset review; write derived evidence outside the frozen release."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '02_datasets/DUT_Anti_UAV_Clean_v2'
OUT = ROOT / '06_results/dataset_review_clean_v2_20261114'
OUT.mkdir(parents=True, exist_ok=True)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

rows = list(csv.DictReader((DATA / 'dataset_manifest.csv').open(encoding='utf-8-sig', newline='')))
result = {'review_date': '2026-11-14', 'source': str(DATA), 'splits': {}, 'issues': [], 'hash_mismatches': [], 'image_dimensions': Counter(), 'objects_per_image': Counter()}
scene_splits, block_splits = defaultdict(set), defaultdict(set)
transitions = Counter()
image_hashes = defaultdict(list)
for split in ('train', 'val', 'test'):
    selected = [r for r in rows if r['new_split'] == split]
    expected_images = {r['new_relative_image'] for r in selected}
    expected_labels = {r['new_relative_label'] for r in selected}
    actual_images = {p.relative_to(DATA).as_posix() for p in (DATA / 'images' / split).iterdir() if p.is_file()}
    actual_labels = {p.relative_to(DATA).as_posix() for p in (DATA / 'labels' / split).glob('*.txt')}
    for kind, expected, actual in [('images', expected_images, actual_images), ('labels', expected_labels, actual_labels)]:
        for name in sorted(expected - actual):
            result['issues'].append({'missing_' + kind: name})
        for name in sorted(actual - expected):
            result['issues'].append({'extra_' + kind: name})
    counts = Counter(images=len(actual_images), labels=len(actual_labels), objects=0, negative_images=0, tiny=0, small=0, medium=0, large=0)
    for r in selected:
        scene_splits[r['scene_group_id']].add(split)
        block_splits[r['allocation_block_id']].add(split)
        transitions[r['old_split'] + ' -> ' + split] += 1
        image_hashes[r['image_sha256']].append(r['new_relative_image'])
        result['image_dimensions'][r['width'] + 'x' + r['height']] += 1
        for key, column in [('image', 'new_relative_image'), ('label', 'new_relative_label')]:
            p = DATA / r[column]
            if p.exists() and sha(p) != r[key + '_sha256']:
                result['hash_mismatches'].append(r[column])
        label = DATA / r['new_relative_label']
        if not label.exists():
            continue
        lines = [s for s in label.read_text().splitlines() if s.strip()]
        result['objects_per_image'][str(len(lines))] += 1
        counts['negative_images'] += not lines
        local = Counter(objects=len(lines), tiny=0, small=0, medium=0, large=0)
        for line_number, line in enumerate(lines, 1):
            try:
                values = [float(v) for v in line.split()]
                assert len(values) == 5 and all(math.isfinite(v) for v in values)
                cls, x, y, w, h = values
                assert cls == 0 and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
                assert x - w / 2 >= -1e-6 and y - h / 2 >= -1e-6 and x + w / 2 <= 1 + 1e-6 and y + h / 2 <= 1 + 1e-6
                area = w * int(r['width']) * h * int(r['height'])
                local['tiny' if area < 32**2 else 'small' if area < 96**2 else 'medium' if area < 256**2 else 'large'] += 1
            except (ValueError, AssertionError):
                result['issues'].append({'invalid_annotation': r['new_relative_label'], 'line': line_number})
        for k, v in local.items():
            counts[k] += v
            if v != int(r[k]):
                result['issues'].append({'manifest_count_mismatch': r['new_relative_label'], 'field': k, 'actual': v, 'manifest': int(r[k])})
    counts['scene_groups'] = len({r['scene_group_id'] for r in selected})
    counts['allocation_blocks'] = len({r['allocation_block_id'] for r in selected})
    counts['image_percentage'] = round(100 * len(actual_images) / len(rows), 4)
    counts['size_percentages'] = {k: round(100 * counts[k] / counts['objects'], 4) for k in ('tiny', 'small', 'medium', 'large')}
    result['splits'][split] = dict(counts)

result['scene_groups'] = len(scene_splits)
result['allocation_blocks'] = len(block_splits)
result['crossing_scene_groups'] = sum(len(v) > 1 for v in scene_splits.values())
result['crossing_allocation_blocks'] = sum(len(v) > 1 for v in block_splits.values())
result['duplicate_image_hash_groups'] = [v for v in image_hashes.values() if len(v) > 1]
result['old_to_new_splits'] = dict(transitions)
result['images_moved_from_old_split'] = sum(r['old_split'] != r['new_split'] for r in rows)
result['total_bytes_images_labels'] = sum((DATA / r[k]).stat().st_size for r in rows for k in ('new_relative_image', 'new_relative_label'))
gate = json.loads((DATA / 'gate2_report.json').read_text())
release = json.loads((ROOT / 'configs/datasets/DUT_Anti_UAV_Clean_v2.release.json').read_text())
result['identity_checks'] = {
    'gate_dataset_manifest': sha(DATA / 'dataset_manifest.csv') == gate['dataset_manifest_sha256'],
    'gate_split_assignment': sha(DATA / 'split_assignment.csv') == gate['split_assignment_sha256'],
    'release_manifest': sha(DATA / 'MANIFEST.csv') == release['manifest_sha256'],
    'release_freeze': sha(DATA / 'FREEZE.json') == release['freeze_sha256'],
    'release_counts': all(all(result['splits'][s][k] == v for k, v in counts.items()) for s, counts in release['counts'].items()),
}
result['scope'] = 'All current image and label bytes verified against dataset_manifest; all labels parsed. Dimensions taken from manifest, not decoded. Historical scene retrieval and geometry not rerun; auxiliary evidence files not fully rehashed.'
(OUT / 'review_statistics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(result, indent=2, ensure_ascii=False))
