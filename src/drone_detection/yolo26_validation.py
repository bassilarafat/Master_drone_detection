"""V3 Validation export with explicit original-pixel float coordinates."""
from datetime import datetime,timezone
import json
from .paper_common import project_path,pin_release,read_csv,size_bin,write_json
from .data_audit import digest
from .dataset_snapshot import _annotations
from .paper_metrics import export_metrics

def now():return datetime.now(timezone.utc).isoformat()

def targets_for(parsed,width,height):
    return [dict(box=list(map(float,[(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height])),
                 size=size_bin(w*width*h*height),**{'class':int(cls)})
            for cls,x,y,w,h in parsed]

def evaluate_validation(c, checkpoint, base, record, *, split="val", validation_subset=None, execution_only=False):
    if split != "val": raise ValueError("Test is locked; this evaluator accepts Validation only")
    if validation_subset is not None and not execution_only: raise ValueError("Subset evaluation is execution-preflight only")
    from ultralytics import YOLO
    dataset = project_path(c['dataset']['path'])
    identity = pin_release(dataset, c['dataset'], full=True)
    output = base / 'evaluation/val'
    output.mkdir(parents=True, exist_ok=False)
    settings = c['evaluation']
    predictions = []
    state = dict(execution_only=execution_only, status='EVALUATING', split='val', test_evaluated=False,
                 dataset_verified_after=False, checkpoint_sha256=digest(checkpoint),
                 settings=settings, started_at_utc=now())
    write_json(output / 'evaluation.json', state)
    try:
        model = YOLO(str(checkpoint))
        rows = [r for r in read_csv(dataset / 'dataset_manifest.csv') if r['new_split'] == 'val']
        if validation_subset is not None:
            selected=set(validation_subset)
            rows=[r for r in rows if r['new_relative_image'] in selected]
            if {r['new_relative_image'] for r in rows} != selected:
                raise ValueError('Subset contains an image outside frozen Validation')
        for row in rows:
            image = dataset / row['new_relative_image']
            width, height = int(row['width']), int(row['height'])
            parsed, issues = _annotations(dataset / row['new_relative_label'])
            if issues:
                raise ValueError(issues)
            targets = targets_for(parsed,width,height)
            result = model.predict(str(image), imgsz=settings['imgsz'],
                                   conf=settings['prediction_conf'], iou=settings['nms_iou'],
                                   max_det=settings['max_det'], batch=settings['batch'],
                                   rect=settings['rect'], augment=settings['augment'],
                                   quantize=settings['quantize'],
                                   agnostic_nms=settings['agnostic_nms'], nms=settings['nms'],
                                   device=settings['device'], save=False,
                                   save_txt=False, save_crop=False, verbose=False)[0]
            if model.predictor.model.end2end:
                raise ValueError('Inference resolved to NMS-free path despite nms=True')
            boxes = result.boxes
            pred = [dict(box=list(map(float, b)), confidence=float(s), **{'class': int(k)})
                    for b, s, k in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(),
                                       boxes.cls.cpu().tolist())]
            predictions.append(dict(image=image.name, split='val', width=width, height=height,
                                    targets=targets, predictions=pred,
                                    image_sha256=row['image_sha256'], label_sha256=row['label_sha256'],
                                    scene_group_id=row['scene_group_id'], negative_verified=False))
        raw = output / 'predictions.jsonl'
        with raw.open('w', encoding='utf-8') as handle:
            for item in predictions:
                handle.write(json.dumps(item, allow_nan=False) + '\n')
        summary = export_metrics(output, predictions, confidence=settings['operating_conf'],
                                 matching_iou=settings['matching_iou'])
        summary.update(execution_only=execution_only, arm=c['architecture'], seed=record['seed'], split='val',
                       checkpoint_sha256=digest(checkpoint), dataset_release=identity)
        write_json(output / 'summary.json', summary)
        state.update(status='PENDING_INTEGRITY_CHECK', prediction_records=len(predictions),
                     raw_sha256=digest(raw), effective_end2end=False, nms=True)
    except BaseException as exc:
        state.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try:
            after = pin_release(dataset, c['dataset'], full=True)
            if after != identity:
                raise ValueError('Dataset identity changed during validation')
            state['dataset_verified_after'] = True
            if state['status'] == 'PENDING_INTEGRITY_CHECK':
                state['status'] = 'COMPLETED'
        except BaseException:
            state.update(status='FAILED_DATASET_INTEGRITY', dataset_verified_after=False)
            raise
        finally:
            state['finished_at_utc'] = now()
            write_json(output / 'evaluation.json', state)
    return summary

