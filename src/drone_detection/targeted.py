"""One fixed P2 fine-tuning experiment with training-only negative crops."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from .metrics import calculate_iou
from .paths import PROJECT_ROOT, project_path

DEFAULT_MODEL = PROJECT_ROOT / '02_datasets/DUT_Anti_UAV/runs/04_experiments/EXP003_YOLOv8s_P2_960/weights/best.pt'


def safe_crop(box, targets, dimensions, margin=16):
    """Keep context and a 16-pixel exclusion zone around every GT box."""
    x1, y1, x2, y2 = box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    half = max(64, x2-x1, y2-y1)
    width, height = dimensions
    crop = [max(0, math.floor(cx-half)), max(0, math.floor(cy-half)),
            min(width, math.ceil(cx+half)), min(height, math.ceil(cy+half))]
    if crop[2]-crop[0] < 32 or crop[3]-crop[1] < 32:
        return None
    if any(crop[0] <= t['box'][2]+margin and crop[2] >= t['box'][0]-margin and
           crop[1] <= t['box'][3]+margin and crop[3] >= t['box'][1]-margin for t in targets):
        return None
    return crop


def size_bin(box):
    width, height = box[2] - box[0], box[3] - box[1]
    return next((name for limit, name in ((32, 'Tiny'), (96, 'Small'), (256, 'Medium'))
                 if width < limit and height < limit), 'Large')


def match_objects(predictions, targets, threshold=0.5):
    """Confidence-ordered, class-aware one-to-one assignment."""
    used, matches, false_positives = set(), [], []
    for index in sorted(range(len(predictions)), key=lambda i: -predictions[i]['confidence']):
        prediction = predictions[index]
        candidates = [(calculate_iou(prediction['box'], target['box']), j)
                      for j, target in enumerate(targets)
                      if j not in used and target['class'] == prediction['class']]
        overlap, target_index = max(candidates, default=(0, -1))
        if overlap >= threshold:
            used.add(target_index)
            matches.append((index, target_index, overlap))
        else:
            overlaps = [calculate_iou(prediction['box'], target['box']) for target in targets]
            best = max(overlaps, default=0)
            kind = 'Background' if best == 0 else ('Duplicate' if best >= threshold else 'Localization')
            false_positives.append((index, kind, best))
    return matches, false_positives, [j for j in range(len(targets)) if j not in used]


def read_targets(image_path, dataset, split, dimensions):
    label = dataset / 'labels' / split / (image_path.stem + '.txt')
    if not label.is_file():
        raise FileNotFoundError(f'Missing label; cannot assume background: {label}')
    width, height = dimensions
    targets = []
    for line_number, line in enumerate(label.read_text().splitlines(), 1):
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5:
            raise ValueError(f'Invalid detection label: {label}:{line_number}')
        cls, x, y, w, h = values
        if not all(math.isfinite(v) for v in values) or cls != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f'Invalid annotation: {label}:{line_number}; adjudicate before running')
        targets.append({'class': int(cls), 'box': [(x-w/2)*width, (y-h/2)*height,
                                                  (x+w/2)*width, (y+h/2)*height]})
    return targets


def write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def analyze(model, dataset, split, output, device, mine=False, limit=None):
    from PIL import Image
    from .data_audit import dataset_digest, digest
    fingerprint = dataset_digest(dataset)
    output.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in (dataset / 'images' / split).iterdir()
                    if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'))
    if not images:
        raise ValueError(f'No images in {split}')
    if mine and split != 'train':
        raise ValueError('Mining is restricted to train')
    if limit is not None:
        if limit <= 0:
            raise ValueError('Sample size must be positive')
        images = images[:limit]
    if (output/'summary.json').exists():
        raise FileExistsError(f'Preserve existing results; choose a new output directory: {output}')
    rows, counts = [], {name: {'GT': 0, 'TP': 0, 'FN': 0} for name in ('Tiny', 'Small', 'Medium', 'Large')}
    tp = fp = fn = 0
    negatives = []
    for number, path in enumerate(images):
        with Image.open(path) as source:
            targets = read_targets(path, dataset, split, source.size)
            result = model.predict(str(path), imgsz=960, conf=0.25, iou=0.7,
                                   device=device, max_det=300, verbose=False)[0]
            predictions = [{'box': box, 'confidence': score, 'class': int(cls)}
                           for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(),
                                                     result.boxes.conf.cpu().tolist(),
                                                     result.boxes.cls.cpu().tolist())]
            matches, errors, misses = match_objects(predictions, targets)
            tp += len(matches); fp += len(errors); fn += len(misses)
            for target in targets:
                counts[size_bin(target['box'])]['GT'] += 1
            matched = {j: (i, overlap) for i, j, overlap in matches}
            for j, target in enumerate(targets):
                size = size_bin(target['box'])
                status = 'TP' if j in matched else 'FN'
                counts[size][status] += 1
                i, overlap = matched.get(j, (-1, 0))
                rows.append(dict(image=path.name, status=status, object_id=j, size=size,
                                 box=json.dumps(target['box']), confidence=predictions[i]['confidence'] if i >= 0 else '',
                                 iou=overlap, fp_type=''))
            for i, kind, overlap in errors:
                prediction = predictions[i]
                rows.append(dict(image=path.name, status='FP', object_id=i, size=size_bin(prediction['box']),
                                 box=json.dumps(prediction['box']), confidence=prediction['confidence'], iou=overlap, fp_type=kind))
                # Keep context, but never erase or include annotated targets.
                if mine and kind == 'Background' and prediction['confidence'] >= 0.5:
                    crop = safe_crop(prediction['box'], targets, source.size)
                    if crop is None:
                        continue
                    name = f'hn_{path.stem}_{i}.png'
                    crop_dir = output / 'candidates'
                    crop_dir.mkdir(exist_ok=True)
                    source.convert('RGB').crop(crop).save(crop_dir / name)
                    negatives.append(dict(source=str(path.resolve()), crop=str((crop_dir/name).resolve()),
                                          confidence=prediction['confidence'], crop_box=json.dumps(crop),
                                          source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                          crop_sha256=hashlib.sha256((crop_dir/name).read_bytes()).hexdigest(), approved=''))
        if (number+1) % 100 == 0:
            print(f'{split}: {number+1}/{len(images)}', flush=True)
    write_csv(output/'objects.csv', rows, ['image', 'status', 'object_id', 'size', 'box', 'confidence', 'iou', 'fp_type'])
    write_csv(output/'recall_by_size.csv', [dict(size=k, **v, recall=v['TP']/v['GT'] if v['GT'] else 0) for k,v in counts.items()], ['size','GT','TP','FN','recall'])
    summary = dict(images=len(images), TP=tp, FP=fp, FN=fn, precision=tp/(tp+fp) if tp+fp else 0,
                   recall=tp/(tp+fn) if tp+fn else 0, F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                   Background_FP=sum(r['status']=='FP' and r['fp_type']=='Background' for r in rows),
                   sample_only=limit is not None, split=split, imgsz=960, confidence=.25,
                   matching_iou=.5, nms_iou=.7, safety_margin_pixels=16 if mine else None,
                   dataset_sha256=fingerprint, checkpoint_sha256=digest(model.ckpt_path))
    if dataset_digest(dataset) != fingerprint:
        raise ValueError('Dataset changed during inference; results cannot be used')
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    if mine:
        write_csv(output/'candidates.csv', negatives, ['source','crop','confidence','crop_box','source_sha256','crop_sha256','approved'])
    return summary


def preserve_baseline(output):
    import shutil
    source = PROJECT_ROOT/'06_results/EXP003_object_level_evaluation'
    destination = output/'historical_baseline'
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ('EXP003_summary.csv','EXP003_recall_by_size.csv','EXP003_fp_analysis.csv'):
        original = source/name
        target = destination/name
        content_hash = hashlib.sha256(original.read_bytes()).hexdigest()
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != content_hash:
            raise ValueError('Historical baseline changed; do not overwrite')
        if not target.exists():
            shutil.copyfile(original, target)
        hashes[name] = content_hash
    (destination/'hashes.json').write_text(json.dumps(hashes, indent=2))


def require_clean_audit(args):
    from .data_audit import dataset_digest, digest
    report = json.loads((args.output/'audit/report.json').read_text())
    if not report['ready_for_experiment']:
        raise ValueError('Data audit is unresolved: review audit/report.json before training')
    if report['dataset_sha256'] != dataset_digest(args.dataset):
        raise ValueError('Data changed after audit; rerun audit')
    if digest(report['sequence_manifest']) != report['sequence_manifest_sha256']:
        raise ValueError('Source-video manifest changed after audit')


def builtin_metrics(model, dataset, split, output, device):
    """AP is measured separately from the fixed-confidence object ledger."""
    result = model.val(data=str(dataset/'data.yaml'), split=split, imgsz=960, conf=.001,
                       iou=.7, max_det=300, batch=1, workers=0, device=device,
                       project=str(output), name='builtin_ap', plots=False)
    values = {k:float(v) for k,v in result.results_dict.items()}
    (output/'ap.json').write_text(json.dumps(values, indent=2))
    return values


def dry_run_data(args, model):
    """Exercise real Ultralytics collation and inference, never an optimizer."""
    import shutil
    import torch
    from ultralytics.data.dataset import YOLODataset
    from .data_audit import digest, dataset_digest, IMAGE_SUFFIXES

    output = args.output/'dry_run_data'
    if (output/'report.json').exists():
        raise FileExistsError('Preserve the completed dry run; choose a new output root')
    output.mkdir(parents=True, exist_ok=True)
    image_dir, label_dir = output/'images/train', output/'labels/train'
    image_dir.mkdir(parents=True, exist_ok=True); label_dir.mkdir(parents=True, exist_ok=True)
    originals = sorted(p for p in (args.dataset/'images/train').iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[:8]
    for source in originals:
        shutil.copyfile(source, image_dir/source.name)
        shutil.copyfile(args.dataset/'labels/train'/(source.stem+'.txt'), label_dir/(source.stem+'.txt'))
    manifest = args.output/'mining_train/candidates.csv'
    with manifest.open(encoding='utf-8') as handle:
        selected = [r for r in csv.DictReader(handle) if r['approved'].strip().lower() == 'yes'][:8]
    if not selected:
        raise ValueError('Dry-run data needs at least one manually verified train negative')
    for row in selected:
        crop = Path(row['crop'])
        if digest(crop) != row['crop_sha256'] or digest(row['source']) != row['source_sha256']:
            raise ValueError('Candidate changed since mining')
        if Path(row['source']).resolve().parent != (args.dataset/'images/train').resolve():
            raise ValueError('Dry-run negative must come from train')
        shutil.copyfile(crop, image_dir/crop.name)
        (label_dir/(crop.stem+'.txt')).write_text('')
    dataset = YOLODataset(img_path=str(image_dir), data={'names':{0:'UAV'}, 'nc':1},
                          imgsz=960, augment=False, rect=False, batch_size=2, stride=32, cache=False)
    if len(dataset) != len(originals)+len(selected):
        raise ValueError('Ultralytics excluded some dry-run samples')
    negative_count = sum(len(label['cls']) == 0 for label in dataset.labels)
    if negative_count < len(selected):
        raise ValueError('Empty negative labels did not load correctly')
    checksum = digest(args.model)
    for start in range(0, len(dataset), 2):
        batch = dataset.collate_fn([dataset[i] for i in range(start, min(start+2,len(dataset)))])
        with torch.inference_mode():
            model.predict(batch['img'].float()/255, imgsz=960, conf=.25, iou=.7,
                          device=args.device, verbose=False)
    if checksum != digest(args.model):
        raise ValueError('Original checkpoint changed during dry run')
    report = dict(passed=True, samples=len(dataset), negative_samples=negative_count,
                  optimizer_steps=0, checkpoint_sha256=checksum, dataset_sha256=dataset_digest(args.dataset),
                  candidates_sha256=digest(manifest), imgsz=960, batch=2,
                  torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    (output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def compare_results(output):
    before = json.loads((output/'baseline_val/summary.json').read_text())
    after = json.loads((output/'finetuned_val/summary.json').read_text())
    for key in ('dataset_sha256','imgsz','confidence','matching_iou','nms_iou','split'):
        if before[key] != after[key]:
            raise ValueError(f'Incomparable evaluation protocols: {key}')
    sizes = list(csv.DictReader((output/'finetuned_val/recall_by_size.csv').open()))
    tiny_fn = int(next(r['FN'] for r in sizes if r['size']=='Tiny'))
    before_ap = json.loads((output/'baseline_val/ap.json').read_text())
    after_ap = json.loads((output/'finetuned_val/ap.json').read_text())
    tolerances = json.loads((output/'experiment/protocol.json').read_text())['degradation_tolerances']
    ap_key = 'metrics/mAP50-95(B)'
    guarded = dict(recall_preserved=after['recall'] >= before['recall']-tolerances['recall'],
                   map50_95_preserved=after_ap[ap_key] >= before_ap[ap_key]-tolerances['map50_95'])
    report = dict(before=before, after=after,
                  delta={key:after[key]-before[key] for key in ('TP','FP','FN','Background_FP','precision','recall','F1')},
                  ap_before=before_ap, ap_after=after_ap,
                  ap_delta={k:after_ap[k]-v for k,v in before_ap.items() if k in after_ap},
                  historical_targets=dict(background_fp_below_124=after['Background_FP']<124,
                      total_fp_below_176=after['FP']<176, tiny_fn_at_most_82=tiny_fn<=82),
                  degradation_tolerances=tolerances, preservation_checks=guarded,
                  conclusion='Report deltas against the fresh baseline and historical targets. No automatic reruns.')
    (output/'comparison.json').write_text(json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['audit', 'analyze', 'mine', 'dry-run', 'prepare', 'finetune', 'compare', 'test'])
    parser.add_argument('--model', type=project_path, default=DEFAULT_MODEL)
    parser.add_argument('--dataset', type=project_path, default=PROJECT_ROOT/'02_datasets/DUT_Anti_UAV')
    parser.add_argument('--output', type=project_path, default=PROJECT_ROOT/'06_results/EXP004_targeted')
    parser.add_argument('--device', default='0')
    parser.add_argument('--limit', type=int, help='Small inference sample; never a training limit')
    parser.add_argument('--sequence-manifest', type=project_path)
    parser.add_argument('--recall-tolerance', type=float, help='Prespecified maximum Recall drop, fraction e.g. .005')
    parser.add_argument('--map-tolerance', type=float, help='Prespecified maximum mAP50-95 drop, fraction e.g. .005')
    parser.add_argument('--dry-run', action='store_true', help='Prepare and check data without training')
    args = parser.parse_args(argv)
    preserve_baseline(args.output)
    if args.stage == 'audit':
        from .data_audit import audit_dataset
        audit_dataset(args.dataset, args.output/'audit', args.sequence_manifest)
        return
    if args.stage == 'compare':
        print(json.dumps(compare_results(args.output), indent=2))
        return
    if args.limit is not None and args.stage not in ('analyze','mine'):
        parser.error('--limit is only valid for analyze or mine')
    from .models import load_model
    model = load_model(args.model)
    if list(model.model.stride.cpu().tolist()) != [4, 8, 16, 32]:
        raise ValueError('Expected the existing P2 checkpoint with strides 4/8/16/32')
    if args.stage == 'analyze':
        dest = args.output/('dry_run_val' if args.limit else 'baseline_val')
        analyze(model, args.dataset, 'val', dest, args.device, limit=args.limit)
        if args.limit is None:
            builtin_metrics(model, args.dataset, 'val', dest, args.device)
    elif args.stage == 'mine':
        dest = args.output/('dry_run_train' if args.limit else 'mining_train')
        analyze(model, args.dataset, 'train', dest, args.device, mine=True, limit=args.limit)
    elif args.stage == 'test':
        require_clean_audit(args)
        if args.model == DEFAULT_MODEL:
            raise ValueError('Provide the final fine-tuned checkpoint with --model')
        analyze(model, args.dataset, 'test', args.output/'final_test', args.device)
        builtin_metrics(model, args.dataset, 'test', args.output/'final_test', args.device)
    elif args.stage == 'dry-run':
        dry_run_data(args, model)
    else:
        import shutil
        import yaml
        from .data_audit import dataset_digest, digest
        if args.stage == 'finetune' and not args.dry_run:
            require_clean_audit(args)
            if any(v is None or not 0 <= v <= 1 for v in (args.recall_tolerance, args.map_tolerance)):
                raise ValueError('Specify accepted --recall-tolerance and --map-tolerance before the one training run')
            current_fingerprint = dataset_digest(args.dataset)
            for stage in ('dry_run_train','dry_run_val','baseline_val'):
                summary = json.loads((args.output/stage/'summary.json').read_text())
                if summary['images'] <= 0 or summary['dataset_sha256'] != current_fingerprint or summary['checkpoint_sha256'] != digest(DEFAULT_MODEL):
                    raise ValueError('Missing completed dry run or fresh baseline')
            dry_run = json.loads((args.output/'dry_run_data/report.json').read_text())
            if not dry_run['passed'] or dry_run['dataset_sha256'] != current_fingerprint:
                raise ValueError('A successful current dry-run data check is required')
            if not (args.output/'baseline_val/ap.json').exists():
                raise ValueError('Fresh baseline AP evaluation is required')
            if args.model != DEFAULT_MODEL:
                raise ValueError('This experiment must start from the original EXP003 checkpoint')
        manifest = args.output/'mining_train/candidates.csv'
        if args.stage == 'finetune' and not args.dry_run and digest(manifest) != dry_run['candidates_sha256']:
            raise ValueError('Reviewed candidates changed after the dry run')
        candidates = list(csv.DictReader(manifest.open(encoding='utf-8')))
        approved = [r for r in candidates if r['approved'].strip().lower() == 'yes']
        if not approved:
            raise ValueError('Review candidates.csv and mark verified non-UAV crops approved=yes first')
        run = args.output/('prepared_data' if args.stage == 'prepare' or args.dry_run else 'experiment')
        run.mkdir(parents=True, exist_ok=False)
        image_dir, label_dir = run/'images/train', run/'labels/train'
        image_dir.mkdir(parents=True); label_dir.mkdir(parents=True)
        original = sorted(p.resolve() for p in (args.dataset/'images/train').iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'))
        added = []
        for row in approved:
            source = Path(row['source']).resolve()
            crop = Path(row['crop']).resolve()
            if source not in original or not crop.is_relative_to((args.output/'mining_train/candidates').resolve()):
                raise ValueError('Candidate does not originate from this training split')
            if hashlib.sha256(source.read_bytes()).hexdigest() != row['source_sha256']:
                raise ValueError('Training source changed since mining')
            if hashlib.sha256(crop.read_bytes()).hexdigest() != row['crop_sha256']:
                raise ValueError('Negative crop changed since review')
            from PIL import Image
            with Image.open(source) as image:
                targets = read_targets(source, args.dataset, 'train', image.size)
                crop_box = json.loads(row['crop_box'])
                # Verify the safety margin against current labels again.
                if any(crop_box[0] <= t['box'][2]+16 and crop_box[2] >= t['box'][0]-16 and
                       crop_box[1] <= t['box'][3]+16 and crop_box[3] >= t['box'][1]-16 for t in targets):
                    raise ValueError('Negative crop intersects an annotated target safety margin')
            destination = image_dir/crop.name
            shutil.copyfile(crop, destination)
            (label_dir/(crop.stem+'.txt')).write_text('')
            added.append(str(destination.resolve()))
        # One copy of each approved negative; all original positive labels remain intact.
        (run/'train.txt').write_text('\n'.join(map(str, original))+ '\n'+'\n'.join(added)+'\n')
        data = yaml.safe_load((args.dataset/'data.yaml').read_text())
        data.update(path=str(args.dataset.resolve()), train=str((run/'train.txt').resolve()),
                    val=str((args.dataset/'images/val').resolve()))
        data.pop('test', None)
        (run/'data.yaml').write_text(yaml.safe_dump(data))
        settings = dict(data=str(run/'data.yaml'), epochs=10, imgsz=960, batch=4, device=args.device,
                        optimizer='AdamW', lr0=0.0001, lrf=0.1, seed=0, deterministic=True,
                        workers=0, mosaic=0, mixup=0, close_mosaic=0, patience=10,
                        amp=False, project=str(run), name='P2_hard_negatives', exist_ok=False, resume=False)
        (run/'protocol.json').write_text(json.dumps(dict(checkpoint=str(args.model),
            checkpoint_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(), settings=settings,
            approved_candidates=approved, architecture=model.model.yaml,
            dataset_sha256=dataset_digest(args.dataset),
            degradation_tolerances={'recall':args.recall_tolerance, 'map50_95':args.map_tolerance}), indent=2, default=str))
        if args.stage == 'prepare' or args.dry_run:
            print(f'Data prepared without training: {run}', flush=True)
            return
        original_parameters = sum(p.numel() for p in model.model.parameters())
        model.train(**settings)
        best = load_model(Path(model.trainer.save_dir)/'weights/best.pt')
        if best.model.stride.tolist() != [4,8,16,32] or sum(p.numel() for p in best.model.parameters()) != original_parameters:
            raise ValueError('Fine-tuned architecture differs from the original checkpoint')
        analyze(best, args.dataset, 'val', args.output/'finetuned_val', args.device)
        builtin_metrics(best, args.dataset, 'val', args.output/'finetuned_val', args.device)
        compare_results(args.output)


if __name__ == '__main__':
    main()
