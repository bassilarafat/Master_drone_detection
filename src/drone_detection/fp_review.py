"""Export and review YOLO26 false positives, then test one postprocessing change."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from .data_audit import IMAGE_SUFFIXES, digest
from .metrics import calculate_iou
from .paths import PROJECT_ROOT, project_path
from .targeted import match_objects, size_bin, write_csv

DEFAULT_MODEL = PROJECT_ROOT/'04_experiments/YOLO26s_baseline-5/weights/best.pt'
DEFAULT_OUTPUT = PROJECT_ROOT/'06_results/YOLO26s_baseline_fp_review_20261011'


def targets_for(path, dataset, split, dimensions):
    label = dataset/'labels'/split/(path.stem+'.txt')
    width, height = dimensions
    targets, issues = [], []
    for line_number, line in enumerate(label.read_text().splitlines(), 1):
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError(f'Malformed label: {label}:{line_number}')
        cls, x, y, w, h = values
        if cls != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            raise ValueError(f'Invalid label: {label}:{line_number}')
        box = [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]
        if w == 0 or h == 0:
            issues.append({'image':path.name, 'line':line_number, 'issue':'zero-area GT retained as unmatchable FN'})
        elif min(box[:2]) < 0 or box[2] > width or box[3] > height:
            issues.append({'image':path.name, 'line':line_number, 'issue':'out-of-image GT retained unchanged'})
        targets.append({'class':int(cls), 'box':box})
    return targets, issues


def summarize(records, confidence=.25, extra_nms=None, box_scale=1):
    tp = fp = fn = 0
    kinds = Counter()
    sizes = {name:dict(GT=0, TP=0, FN=0) for name in ('Tiny','Small','Medium','Large')}
    errors = []
    for record in records:
        predictions = adjusted_predictions(record['predictions'],confidence,extra_nms,box_scale)
        matches, false_positives, misses = match_objects(predictions, record['targets'])
        tp += len(matches); fp += len(false_positives); fn += len(misses)
        for j,target in enumerate(record['targets']):
            group = sizes[size_bin(target['box'])]
            group['GT'] += 1
            group['FN' if j in misses else 'TP'] += 1
        for i,kind,overlap in false_positives:
            prediction = predictions[i]
            kinds[kind] += 1
            errors.append(dict(fp_id=f'{Path(record["image"]).stem}_{prediction["pred_id"]}',
                image=record['image'], image_path=record['image_path'], pred_id=prediction['pred_id'],
                **dict(zip(('x1','y1','x2','y2'),prediction['box'])), confidence=prediction['confidence'],
                pred_class=prediction['class'], geometric_type=kind, best_iou=overlap,
                semantic_type='', confirmed='', reviewer='', notes=''))
    summary = dict(images=len(records), TP=tp, FP=fp, FN=fn, Background_FP=kinds['Background'],
        Duplicate_FP=kinds['Duplicate'], Localization_FP=kinds['Localization'],
        precision=tp/(tp+fp) if tp+fp else 0, recall=tp/(tp+fn) if tp+fn else 0,
        F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0, confidence=confidence,
        extra_nms_iou=extra_nms, box_scale=box_scale, recall_by_size=sizes)
    return summary, errors


def adjusted_predictions(predictions,confidence=.25,extra_nms=None,box_scale=1):
    result=[p for p in predictions if p['confidence'] >= confidence]
    if box_scale != 1:
        result=shrink_boxes(result,box_scale)
    if extra_nms is not None:
        result=suppress_duplicates(result,extra_nms)
    return result


def object_transitions(records,confidence=.25,extra_nms=None,box_scale=1):
    changes=[]
    for record in records:
        before={j for i,j,v in match_objects(record['predictions'],record['targets'])[0]}
        after={j for i,j,v in match_objects(adjusted_predictions(record['predictions'],confidence,extra_nms,box_scale),record['targets'])[0]}
        for j in sorted(before ^ after):
            changes.append(dict(image=record['image'],gt_id=j,
                transition='recovered' if j in after else 'newly_missed',size=size_bin(record['targets'][j]['box'])))
    recovered=sum(r['transition']=='recovered' for r in changes)
    missed=sum(r['transition']=='newly_missed' for r in changes)
    return dict(recovered_gt=recovered,newly_missed_gt=missed,net_tp=recovered-missed,changed_objects=changes)


def suppress_duplicates(predictions, threshold):
    kept = []
    for p in sorted(predictions, key=lambda r:-r['confidence']):
        if not any(p['class']==q['class'] and calculate_iou(p['box'],q['box']) > threshold for q in kept):
            kept.append(p)
    return kept


def shrink_boxes(predictions, factor):
    if not 0 < factor <= 1:
        raise ValueError('Box shrink factor must be in (0,1]')
    result=[]
    for p in predictions:
        x1,y1,x2,y2=p['box']
        cx,cy=(x1+x2)/2,(y1+y2)/2
        half_w,half_h=(x2-x1)*factor/2,(y2-y1)*factor/2
        result.append({**p,'box':[cx-half_w,cy-half_h,cx+half_w,cy+half_h]})
    return result


def extract(args):
    from PIL import Image
    from .models import load_model
    import ultralytics
    import torch

    args.output.mkdir(parents=True, exist_ok=False)
    model = load_model(args.model)
    fingerprint = digest(args.model)
    records, issues = [], []
    source_hash = hashlib.sha256()
    image_paths = sorted(p for p in (args.dataset/'images'/args.split).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise ValueError('No evaluation images')
    for number,path in enumerate(image_paths):
        image_hash = digest(path)
        label_hash = digest(args.dataset/'labels'/args.split/(path.stem+'.txt'))
        source_hash.update((path.name+image_hash+label_hash).encode())
        with Image.open(path) as image:
            targets, findings = targets_for(path,args.dataset,args.split,image.size)
        issues.extend(findings)
        result = model.predict(str(path), imgsz=960, conf=.25, iou=.7, rect=True,
                               device=args.device, max_det=300, verbose=False)[0]
        predictions = [dict(pred_id=i,box=box,confidence=score,**{'class':int(cls)})
                       for i,(box,score,cls) in enumerate(zip(result.boxes.xyxy.cpu().tolist(),
                                    result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()))]
        records.append(dict(image=path.name,image_path=str(path.resolve()),image_sha256=image_hash,
                            label_sha256=label_hash, targets=targets,predictions=predictions))
        if (number+1)%100 == 0:
            print(f'{args.split}: {number+1}/{len(image_paths)}',flush=True)
    summary, errors = summarize(records)
    for record in records:
        path = Path(record['image_path'])
        if digest(path) != record['image_sha256'] or digest(args.dataset/'labels'/args.split/(path.stem+'.txt')) != record['label_sha256']:
            raise ValueError('Evaluation data changed during inference')
    with (args.output/'predictions.jsonl').open('w',encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record)+'\n')
    fields = ['fp_id','image','image_path','pred_id','x1','y1','x2','y2','confidence',
              'pred_class','geometric_type','best_iou','semantic_type','confirmed','reviewer','notes']
    write_csv(args.output/'false_positives.csv',errors,fields)
    write_csv(args.output/'annotation_findings.csv',issues,['image','line','issue'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.output/'protocol.json').write_text(json.dumps(dict(checkpoint=str(args.model),
        checkpoint_sha256=fingerprint, dataset=str(args.dataset),split=args.split,
        split_sha256=source_hash.hexdigest(), imgsz=960,confidence=.25,matching_iou=.5,
        predict_iou=.7,rect=True,max_det=300, device=args.device, ultralytics=ultralytics.__version__,
        torch=torch.__version__,end2end=bool(getattr(model.model,'end2end',False)),
        note='YOLO26 end-to-end predictions may not apply NMS; geometric and semantic FP categories differ.'),indent=2))
    if digest(args.model) != fingerprint:
        raise ValueError('Checkpoint changed during evaluation')
    print(json.dumps(summary,indent=2),flush=True)
    gallery(args.output)


def gallery(output):
    """Every FP gets a contextual crop and a full-image view with annotations."""
    from PIL import Image, ImageDraw
    rows = list(csv.DictReader((output/'false_positives.csv').open()))
    records = {r['image']:r for r in read_records(output)}
    crops = output/'review'
    crops.mkdir(exist_ok=True)
    for start in range(0,len(rows),12):
        page_rows = rows[start:start+12]
        page = Image.new('RGB',(1600,380*math.ceil(len(page_rows)/4)),'white')
        draw = ImageDraw.Draw(page)
        for j,row in enumerate(page_rows):
            record = records[row['image']]
            with Image.open(row['image_path']) as original:
                image = original.convert('RGB')
            x1,y1,x2,y2 = [float(row[k]) for k in ('x1','y1','x2','y2')]
            cx,cy=(x1+x2)/2,(y1+y2)/2
            half=max(64,x2-x1,y2-y1)
            bounds=(max(0,math.floor(cx-half)),max(0,math.floor(cy-half)),
                    min(image.width,math.ceil(cx+half)),min(image.height,math.ceil(cy+half)))
            crop=image.crop(bounds)
            painter=ImageDraw.Draw(crop)
            for target in record['targets']:
                box=target['box']
                painter.rectangle([box[0]-bounds[0],box[1]-bounds[1],box[2]-bounds[0],box[3]-bounds[1]],outline='lime',width=1)
            painter.rectangle([x1-bounds[0],y1-bounds[1],x2-bounds[0],y2-bounds[1]],outline='red',width=1)
            crop.save(crops/(row['fp_id']+'.png'))
            full=ImageDraw.Draw(image)
            for t in record['targets']:
                full.rectangle(t['box'],outline='lime',width=2)
            full.rectangle([x1,y1,x2,y2],outline='red',width=3)
            image.thumbnail((1280,720))
            image.save(crops/(row['fp_id']+'_context.jpg'))
            scale=min(380/crop.width,325/crop.height)
            crop=crop.resize((round(crop.width*scale),round(crop.height*scale)))
            x,y=(j%4)*400,(j//4)*380
            page.paste(crop,(x+(400-crop.width)//2,y+45))
            draw.text((x+8,y+5),f'{row["fp_id"]} conf={float(row["confidence"]):.3f}',fill='black')
            draw.text((x+8,y+22),f'{row["geometric_type"]} IoU={float(row["best_iou"]):.3f}',fill='black')
        page.save(output/f'review_page_{start//12+1:03}.png')
    print(f'Exported {len(rows)} FP review crops and context views',flush=True)


def read_records(output):
    with (output/'predictions.jsonl').open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]


def test_correction(args):
    records = read_records(args.output)
    reviews = list(csv.DictReader((args.output/'false_positives.csv').open()))
    _, original_errors = summarize(records)
    if len(reviews) != len(original_errors) or {r['fp_id'] for r in reviews} != {r['fp_id'] for r in original_errors}:
        raise ValueError('The review must include every original FP exactly once')
    confirmed = Counter(r['semantic_type'] for r in reviews if r['confirmed']=='yes')
    if any(not r['semantic_type'] for r in reviews):
        raise ValueError('Classify every FP, using uncertain for unresolved cases')
    if any(r['confirmed'] not in ('yes','no') or not r['reviewer'].strip() or not r['notes'].strip() for r in reviews):
        raise ValueError('Record a confirmation decision, reviewer and rationale for every FP')
    if not confirmed:
        raise ValueError('No confirmed failure mode')
    changes=sum((args.confidence != .25, args.extra_nms is not None, args.box_scale != 1))
    if changes == 0:
        raise ValueError('Specify one correction: --confidence, --extra-nms or --box-scale')
    if changes > 1:
        raise ValueError('Test only one change at a time')
    if (args.output/'correction.json').exists():
        raise FileExistsError('One correction was already tested; retain the result')
    # Parameter is declared explicitly once, not optimized with a threshold sweep.
    before,_ = summarize(records)
    after,errors = summarize(records,confidence=args.confidence,extra_nms=args.extra_nms,box_scale=args.box_scale)
    retained = {r['fp_id'] for r in errors}
    by_type = {kind:dict(before=count, remaining=sum(r['confirmed']=='yes' and r['semantic_type']==kind and r['fp_id'] in retained for r in reviews))
               for kind,count in confirmed.items()}
    report = dict(most_frequent_confirmed=confirmed.most_common(),reviewer='Codex visual review; not human ground truth',
        protocol=json.loads((args.output/'protocol.json').read_text()),
        uncertain=sum(r['confirmed']!='yes' for r in reviews), before=before,after=after,
        delta={k:after[k]-before[k] for k in ('TP','FP','FN','precision','recall','F1')},
        confirmed_types_after_correction=by_type,
        retained_fp_ids=[r['fp_id'] for r in errors],
        warning='Exploratory validation correction, not independent test evidence; no training or architecture change.')
    transitions=object_transitions(records,args.confidence,args.extra_nms,args.box_scale)
    report['recovered_gt']=transitions['recovered_gt']
    report['newly_missed_gt']=transitions['newly_missed_gt']
    (args.output/'correction_object_transitions.json').write_text(json.dumps(transitions,indent=2))
    (args.output/'correction.json').write_text(json.dumps(report,indent=2))
    if errors:
        write_csv(args.output/'corrected_false_positives.csv',errors,list(errors[0]))
    print(json.dumps(report,indent=2),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['extract','gallery','correct'])
    parser.add_argument('--model',type=project_path,default=DEFAULT_MODEL)
    parser.add_argument('--dataset',type=project_path,default=PROJECT_ROOT/'02_datasets/DUT_Anti_UAV')
    parser.add_argument('--split',choices=['val','train'],default='val')
    parser.add_argument('--output',type=project_path,default=DEFAULT_OUTPUT)
    parser.add_argument('--device',default='0')
    parser.add_argument('--confidence',type=float,default=.25)
    parser.add_argument('--extra-nms',type=float)
    parser.add_argument('--box-scale',type=float,default=1,help='One global box shrink factor in (0,1]')
    args=parser.parse_args(argv)
    if not .25 <= args.confidence <= 1 or (args.extra_nms is not None and not 0 < args.extra_nms < 1) or not 0 < args.box_scale <= 1:
        parser.error('Confidence must be .25..1; NMS IoU must be 0..1; box scale must be (0,1]')
    if args.stage != 'correct' and (args.confidence != .25 or args.extra_nms is not None or args.box_scale != 1):
        parser.error('Correction parameters only apply to the correct stage')
    if args.stage=='extract': extract(args)
    elif args.stage=='gallery': gallery(args.output)
    else: test_correction(args)


if __name__=='__main__':
    main()
