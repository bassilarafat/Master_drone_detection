"""One score-ordered evaluator: original-pixel IoU, operating metrics and 101-point AP.

Inputs are post-NMS predictions at conf>=0.001, not a thresholded TP-only export.
No detector, training API or filesystem dataset is touched by this pure scorer.
"""
from __future__ import annotations
from collections import Counter
import math
from .metrics import calculate_iou
from .paper_common import SIZES, size_bin, write_csv, write_json

FP_TYPES=('Background','Localization','Duplicate','ClassConfusion','InvalidBox')

def match(predictions,targets,threshold=.5):
    used=set();matched={};fp={}
    for i in sorted(range(len(predictions)),key=lambda i:(-predictions[i]['confidence'],i)):
        p=predictions[i]
        if p['box'][0]>=p['box'][2] or p['box'][1]>=p['box'][3]:
            fp[i]=('InvalidBox',0.);continue
        choices=[(calculate_iou(p['box'],g['box']),j) for j,g in enumerate(targets) if j not in used and p['class']==g['class']]
        overlap,j=max(choices,key=lambda t:(t[0],-t[1]),default=(0,-1))
        if j>=0 and overlap>=threshold:
            used.add(j);matched[i]=(j,overlap);continue
        same=max((calculate_iou(p['box'],g['box']) for g in targets if g['class']==p['class']),default=0)
        any_class=max((calculate_iou(p['box'],g['box']) for g in targets),default=0)
        kind='Duplicate' if same>=threshold else 'ClassConfusion' if any_class>=threshold else 'Localization' if any_class>0 else 'Background'
        fp[i]=(kind,any_class)
    return matched,fp,[j for j in range(len(targets)) if j not in used]

def average_precision(records,iou,category=0):
    total=sum(g['class']==category for r in records for g in r['targets'])
    if not total:return None
    ranked=[]
    for number,r in enumerate(records):
        targets=[g for g in r['targets'] if g['class']==category]
        predictions=[p for p in r['predictions'] if p['class']==category]
        hits,_,_=match(predictions,targets,iou)
        ranked.extend((p['confidence'],number,i,int(i in hits)) for i,p in enumerate(predictions))
    ranked.sort(key=lambda p:(-p[0],p[1],p[2]))
    tp=0;points=[]
    for n,(_,_,_,hit) in enumerate(ranked,1):
        tp+=hit;points.append((tp/total,tp/n))
    # COCO-style sampled precision envelope, explicitly maxDet=300 and no crowd/area ignore rules.
    return sum(max((p for r,p in points if r>=step/100),default=0) for step in range(101))/101

def validate_records(records, prediction_floor=.001):
    names=set()
    for r in records:
        if r['image'] in names:raise ValueError('Every image, including no-detection images, must occur exactly once')
        names.add(r['image'])
        if r.get('negative_verified') and r['targets']:raise ValueError('A verified negative cannot contain an annotated UAV')
        for kind in ('targets','predictions'):
            if kind=='predictions' and len(r[kind])>300:raise ValueError('More than the protocol maxDet=300 post-NMS predictions')
            for obj in r[kind]:
                b=obj['box'];c=obj['class']
                if len(b)!=4 or not all(math.isfinite(x) for x in b):raise ValueError('Invalid original-pixel box')
                if kind=='targets' and (b[0]>=b[2] or b[1]>=b[3]):raise ValueError('Invalid ground-truth box')
                if type(c) is not int or c<0:raise ValueError('Class must be a nonnegative integer')
                if kind=='predictions' and (not math.isfinite(obj['confidence']) or not prediction_floor<=obj['confidence']<=1):
                    raise ValueError('Raw predictions must retain the declared AP confidence floor')

def evaluate_records(records, *, confidence=.25, matching_iou=.5):
    validate_records(records)
    sizes={s:dict(GT=0,TP=0,FN=0) for s in SIZES}
    tp=fp=fn=0;types=Counter();raw=[];objects=[];per_image=[];empty=verified=empty_fp=verified_fp=0
    for r in records:
        predictions=r['predictions'];targets=r['targets']
        selected=[i for i,p in enumerate(predictions) if p['confidence']>=confidence]
        operating=[predictions[i] for i in selected]
        hits,errors,misses=match(operating,targets,matching_iou)
        gt_to_pred={j:(i,v) for i,(j,v) in hits.items()}
        tp+=len(hits);fp+=len(errors);fn+=len(misses)
        for j,g in enumerate(targets):
            size=g.get('size') or size_bin((g['box'][2]-g['box'][0])*(g['box'][3]-g['box'][1]))
            sizes[size]['GT']+=1;sizes[size]['TP' if j in gt_to_pred else 'FN']+=1
            i,overlap=gt_to_pred.get(j,(-1,0))
            objects.append(dict(image=r['image'],kind='GT',status='TP' if i>=0 else 'FN',gt_id=j,
                pred_id=selected[i] if i>=0 else '',size=size,fp_subtype='',iou=overlap,
                confidence=operating[i]['confidence'] if i>=0 else '',box=g['box']))
        lookup={original:i for i,original in enumerate(selected)}
        for i,p in enumerate(predictions):
            k=lookup.get(i);status='BELOW_OPERATING_THRESHOLD';subtype='';overlap=0;gt=''
            if k in hits:status='TP';gt,overlap=hits[k]
            elif k in errors:
                status='FP';subtype,overlap=errors[k];types[subtype]+=1
                objects.append(dict(image=r['image'],kind='PRED',status='FP',gt_id='',pred_id=i,
                    size='',fp_subtype=subtype,iou=overlap,confidence=p['confidence'],box=p['box']))
            raw.append(dict(image=r['image'],pred_id=i,**dict(zip(('x1','y1','x2','y2'),p['box'])),
                confidence=p['confidence'],pred_class=p['class'],status=status,matched_gt=gt,iou=overlap,fp_subtype=subtype))
        no_gt=not targets;has_fp=bool(errors)
        empty+=no_gt;empty_fp+=no_gt and has_fp
        verified+=no_gt and bool(r.get('negative_verified'));verified_fp+=no_gt and bool(r.get('negative_verified')) and has_fp
        per_image.append(dict(image=r['image'],GT=len(targets),raw_predictions=len(predictions),TP=len(hits),FP=len(errors),FN=len(misses),
            annotation_empty=no_gt,negative_verified=bool(r.get('negative_verified'))))
    categories=sorted({g['class'] for r in records for g in r['targets']})
    ap={f'{i/100:.2f}':{str(c):average_precision(records,i/100,c) for c in categories} for i in range(50,96,5)}
    maps={t:sum(v.values())/len(v) if v else None for t,v in ap.items()}
    summary=dict(images=len(records),GT=tp+fn,TP=tp,FP=fp,FN=fn,
        precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn) if tp+fn else None,
        F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        mAP50=maps['0.50'],mAP50_95=sum(maps.values())/10 if categories else None,
        AP_by_iou=maps,AP_by_class_and_iou=ap,
        fp_subtype_counts={k:types[k] for k in FP_TYPES},FP_per_image=fp/len(records) if records else None,
        recall_by_size={s:dict(**v,recall=v['TP']/v['GT'] if v['GT'] else None) for s,v in sizes.items()},
        background_only=dict(annotation_empty_images=empty,verified_negative_images=verified,
            verified_negative_images_with_fp=verified_fp,false_positive_rate=verified_fp/verified if verified else None,
            annotation_empty_fp_image_rate=empty_fp/empty if empty else None,
            reason=None if verified else 'No independently verified background-only images in this evaluation set'),
        operating_confidence=confidence,matching_iou=matching_iou,AP_prediction_floor=.001,
        AP_definition='101-recall-point precision-envelope mean; IoU .50:.05:.95; score-ordered one-to-one matching; maxDet=300 post-NMS; not full COCO crowd/area evaluation',
        semantic_fp_counts=dict(unreviewed=fp))
    return summary,raw,objects,per_image

def export_metrics(output,records,**options):
    summary,raw,objects,images=evaluate_records(records,**options)
    write_json(output/'summary.json',summary)
    write_csv(output/'predictions.csv',raw,['image','pred_id','x1','y1','x2','y2','confidence','pred_class','status','matched_gt','iou','fp_subtype'])
    write_csv(output/'objects.csv',objects,['image','kind','status','gt_id','pred_id','size','fp_subtype','iou','confidence','box'])
    write_csv(output/'images.csv',images,['image','GT','raw_predictions','TP','FP','FN','annotation_empty','negative_verified'])
    write_csv(output/'recall_by_size.csv',[dict(size=s,**v) for s,v in summary['recall_by_size'].items()],['size','GT','TP','FN','recall'])
    write_csv(output/'fp_subtypes.csv',[dict(fp_subtype=s,count=n) for s,n in summary['fp_subtype_counts'].items()],['fp_subtype','count'])
    review=[dict(fp_id=f"{r['image']}::{r['pred_id']}",image=r['image'],pred_id=r['pred_id'],geometric_subtype=r['fp_subtype'],semantic_subtype='UNREVIEWED',reviewer='',verified='',notes='') for r in raw if r['status']=='FP']
    write_csv(output/'fp_semantic_review.csv',review,['fp_id','image','pred_id','geometric_subtype','semantic_subtype','reviewer','verified','notes'])
    return summary
