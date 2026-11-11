"""Dataset integrity audit and safe YOLO label normalization."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from .paths import PROJECT_ROOT

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}
SPLITS = ('train', 'val', 'test')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_digest(dataset):
    """Bind a report to every current image, label and the dataset YAML."""
    result = hashlib.sha256()
    paths = [dataset / 'data.yaml']
    for split in ('train', 'val', 'test'):
        paths.extend(sorted((dataset / 'images' / split).glob('*')))
        paths.extend(sorted((dataset / 'labels' / split).glob('*.txt')))
    for path in paths:
        if path.is_file():
            result.update(str(path.relative_to(dataset)).encode())
            result.update(bytes.fromhex(digest(path)))
    return result.hexdigest()


def dump_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _normalized_box_to_pixels(values, width, height):
    """Convert a YOLO ``xywh`` row to pixel ``xyxy`` coordinates."""
    _, x, y, box_width, box_height = values
    return (
        (x - box_width / 2) * width,
        (y - box_height / 2) * height,
        (x + box_width / 2) * width,
        (y + box_height / 2) * height,
    )


def _pixels_to_normalized_box(class_id, corners, width, height):
    x1, y1, x2, y2 = corners
    return (
        class_id,
        ((x1 + x2) / 2) / width,
        ((y1 + y2) / 2) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def normalize_dataset_labels(dataset, output, *, write=False):
    """Audit and optionally clip YOLO boxes using each image's pixel dimensions.

    Labels are interpreted as ``class x_center y_center width height``. Only
    boxes whose pixel corners change after clipping are rewritten. When
    ``write`` is true, original files are copied to ``output/backups`` first.
    """
    from PIL import Image
    import shutil

    dataset = Path(dataset)
    output = Path(output)
    resolved_dataset = dataset.resolve()
    if (resolved_dataset / 'FREEZE.json').exists() or '_Clean_v' in resolved_dataset.name:
        if write:
            raise ValueError('Frozen dataset versions cannot be normalized; create a new version.')
        if output.resolve() == resolved_dataset or resolved_dataset in output.resolve().parents:
            raise ValueError('Write audit reports outside the frozen dataset.')
    output.mkdir(parents=True, exist_ok=True)
    changes, errors = [], []
    backups = output / 'backups'

    for split in SPLITS:
        image_dir = dataset / 'images' / split
        label_dir = dataset / 'labels' / split
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = label_dir / f'{image_path.stem}.txt'
            if not label_path.is_file():
                errors.append(dict(split=split, image=image_path.name, issue='Missing label'))
                continue
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                rows = []
                file_changes = []
                for line_number, line in enumerate(
                    label_path.read_text(encoding='utf-8').splitlines(), 1
                ):
                    if not line.strip():
                        rows.append(line)
                        continue
                    values = list(map(float, line.split()))
                    if len(values) != 5 or not all(math.isfinite(value) for value in values):
                        raise ValueError(f'{label_path.name}:{line_number}: malformed YOLO xywh row')
                    class_id, x, y, box_width, box_height = values
                    if class_id != int(class_id) or box_width <= 0 or box_height <= 0:
                        raise ValueError(f'{label_path.name}:{line_number}: invalid class or dimensions')
                    corners = _normalized_box_to_pixels(values, width, height)
                    clipped = tuple(max(0.0, min(limit, value))
                                    for value, limit in zip(corners, (width, height, width, height)))
                    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                        raise ValueError(f'{label_path.name}:{line_number}: clipping removes box')
                    normalized = _pixels_to_normalized_box(int(class_id), clipped, width, height)
                    changed = any(abs(a - b) > 1e-12 for a, b in zip(corners, clipped))
                    if changed:
                        normalized_text = f'{int(normalized[0])} ' + ' '.join(
                            f'{value:.12f}' for value in normalized[1:]
                        )
                        rows.append(normalized_text)
                        file_changes.append(dict(
                            split=split, image=image_path.name, label=label_path.name,
                            line=line_number, width=width, height=height,
                            original=line, normalized=normalized_text,
                        ))
                    else:
                        rows.append(line)
                if file_changes:
                    changes.extend(file_changes)
                    if write:
                        backup_path = backups / split / label_path.name
                        backup_path.parent.mkdir(parents=True, exist_ok=True)
                        if not backup_path.exists():
                            shutil.copy2(label_path, backup_path)
                        label_path.write_text('\n'.join(rows) + '\n', encoding='utf-8')
            except (OSError, ValueError) as exc:
                errors.append(dict(split=split, image=image_path.name, issue=str(exc)))

    dump_csv(output / 'changes.csv', changes,
             ['split', 'image', 'label', 'line', 'width', 'height', 'original', 'normalized'])
    dump_csv(output / 'errors.csv', errors, ['split', 'image', 'issue'])
    report = dict(dataset=str(dataset.resolve()), write=write, changed_boxes=len(changes),
                  changed_files=len({(row['split'], row['label']) for row in changes}),
                  errors=len(errors), format='class x_center y_center width height',
                  coordinate_space='pixel corners clipped to image bounds, then normalized')
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def audit_dataset(dataset, output, sequence_manifest=None):
    import cv2
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity

    output.mkdir(parents=True, exist_ok=True)
    fingerprint = dataset_digest(dataset)
    records, errors, thumbnails = [], [], []
    for split in SPLITS:
        paths = sorted(p for p in (dataset/'images'/split).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            errors.append(dict(split=split, image='', issue='Empty split'))
        for number, path in enumerate(paths):
            try:
                with Image.open(path) as image:
                    rgb = image.convert('RGB')
                    rgb.load()
                    width, height = rgb.size
                    gray = np.asarray(rgb.convert('L').resize((256, 256)), dtype=np.uint8)
                    tiny = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
                    frequencies = cv2.dct(tiny)[:8, :8].flatten()[1:]
                    phash = sum(int(bit) << i for i, bit in enumerate(frequencies > np.median(frequencies)))
                    pixel_hash = hashlib.sha256(f'{width}x{height}:'.encode()+rgb.tobytes()).hexdigest()
                label = dataset/'labels'/split/(path.stem+'.txt')
                if not label.is_file():
                    errors.append(dict(split=split, image=path.name, issue='Missing label'))
                else:
                    for line_number, line in enumerate(label.read_text().splitlines(), 1):
                        if not line.strip():
                            continue
                        try:
                            values = list(map(float, line.split()))
                            if len(values) != 5 or not all(math.isfinite(v) for v in values):
                                raise ValueError('Malformed detection label')
                            cls, x, y, w, h = values
                            if cls != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                                raise ValueError('Invalid class/center/zero-area dimensions')
                            # Pixel boxes outside the image need explicit adjudication too.
                            if min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
                                raise ValueError('Box outside image')
                        except ValueError as exc:
                            errors.append(dict(split=split, image=path.name, issue=f'{label.name}:{line_number}: {exc}'))
                records.append(dict(split=split, image=path.name, sha256=digest(path),
                                    pixel_sha256=pixel_hash, phash=phash, width=width, height=height))
                thumbnails.append(gray)
            except (OSError, ValueError) as exc:
                errors.append(dict(split=split, image=path.name, issue=str(exc)))
            if (number+1) % 500 == 0:
                print(f'audit {split}: {number+1}/{len(paths)}', flush=True)

    exact = []
    for field in ('sha256', 'pixel_sha256'):
        groups = defaultdict(list)
        for r in records:
            groups[r[field]].append(r)
        for group in groups.values():
            for i, a in enumerate(group):
                for b in group[i+1:]:
                    exact.append(dict(split_a=a['split'], image_a=a['image'], split_b=b['split'],
                                      image_b=b['image'], method=field))
    near = []
    # Persist the expensive decode/label checks before pair screening starts.
    dump_csv(output/'images.csv', records, ['split','image','sha256','pixel_sha256','phash','width','height'])
    dump_csv(output/'annotation_issues.csv', errors, ['split','image','issue'])
    dump_csv(output/'exact_duplicates.csv', exact, ['split_a','image_a','split_b','image_b','method'])
    lookup = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    for split_a, split_b in (('train','val'), ('train','test'), ('val','test')):
        left = [i for i,r in enumerate(records) if r['split'] == split_a]
        right = [i for i,r in enumerate(records) if r['split'] == split_b]
        right_hashes = np.array([records[i]['phash'] for i in right], dtype=np.uint64)
        for i in left:
            xor = np.bitwise_xor(np.uint64(records[i]['phash']), right_hashes)
            distances = lookup[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)
            for position in np.flatnonzero(distances <= 8):
                j = right[int(position)]
                similarity = float(structural_similarity(thumbnails[i], thumbnails[j], data_range=255))
                if similarity >= .95:
                    near.append(dict(split_a=split_a, image_a=records[i]['image'], split_b=split_b,
                                     image_b=records[j]['image'], phash_distance=int(distances[position]), ssim=similarity))
        dump_csv(output/'near_duplicates.csv', near, ['split_a','image_a','split_b','image_b','phash_distance','ssim'])
        print(f'audit similarity {split_a}/{split_b}: complete; '
              f'{sum(r["split_a"]==split_a and r["split_b"]==split_b for r in near)} candidates', flush=True)

    # Historical temporal mapping is a warning, never verified source metadata.
    inferred = []
    temporal = PROJECT_ROOT/'04_experiments/dataset_integrity/temporal_source_mapping_validation.csv'
    if temporal.exists():
        with temporal.open() as handle:
            for row in csv.DictReader(handle):
                if float(row['ssim']) >= .95:
                    inferred.append(row)
        dump_csv(output/'historical_inferred_sources.csv', inferred,
                 ['split','current_frame','video','best_original_frame','ssim','offset'])

    coverage = False
    shared_sources = []
    sequence_errors = []
    sequence_hash = None
    if sequence_manifest is not None:
        sequence_hash = digest(sequence_manifest)
        with Path(sequence_manifest).open(encoding='utf-8') as handle:
            mapping = list(csv.DictReader(handle))
        expected = {(r['split'], r['image']) for r in records}
        actual = {(r.get('split'), r.get('image')) for r in mapping}
        if actual != expected or len(mapping) != len(expected):
            sequence_errors.append('Manifest must cover each image exactly once')
        if any(r.get('verified','').lower() != 'yes' or not r.get('video_id','').strip() for r in mapping):
            sequence_errors.append('Every video_id needs verified=yes and real source provenance')
        sources = defaultdict(set)
        for row in mapping:
            sources[row.get('video_id','')].add(row.get('split',''))
        shared_sources = [dict(video_id=k, splits=sorted(v)) for k,v in sources.items() if len(v)>1]
        coverage = not sequence_errors
    else:
        sequence_errors.append('No verified complete source-video manifest; independence unresolved')

    dump_csv(output/'images.csv', records, ['split','image','sha256','pixel_sha256','phash','width','height'])
    dump_csv(output/'annotation_issues.csv', errors, ['split','image','issue'])
    dump_csv(output/'exact_duplicates.csv', exact, ['split_a','image_a','split_b','image_b','method'])
    dump_csv(output/'near_duplicates.csv', near, ['split_a','image_a','split_b','image_b','phash_distance','ssim'])
    dump_csv(output/'sequence_manifest_template.csv',
             [dict(split=r['split'],image=r['image'],video_id='',verified='') for r in records],
             ['split','image','video_id','verified'])
    cross_exact = [r for r in exact if r['split_a'] != r['split_b']]
    stable = fingerprint == dataset_digest(dataset)
    report = dict(dataset=str(dataset.resolve()), dataset_sha256=fingerprint, stable_during_audit=stable,
                  images={s:sum(r['split']==s for r in records) for s in ('train','val','test')},
                  annotation_issues=len(errors), cross_split_exact=len(cross_exact),
                  within_split_exact_byte_pairs=sum(r['split_a']==r['split_b'] and r['method']=='sha256' for r in exact),
                  near_duplicate_pairs=len(near), near_duplicate_pairs_by_splits={
                      f'{a}/{b}':sum(r['split_a']==a and r['split_b']==b for r in near)
                      for a,b in (('train','val'),('train','test'),('val','test'))},
                  verified_source_coverage=coverage, shared_video_sources=shared_sources,
                  sequence_errors=sequence_errors, sequence_manifest=str(sequence_manifest) if sequence_manifest else None,
                  sequence_manifest_sha256=sequence_hash,
                  limitation='pHash<=8 + SSIM>=.95 is a candidate screen, not proof of scene independence',
                  ready_for_experiment=bool(stable and not errors and not cross_exact and not near and coverage and not shared_sources))
    (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return report
