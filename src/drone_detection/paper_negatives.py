"""Train-only negative proposals, explicit visual verification and matched controls."""
from __future__ import annotations
from collections import Counter
from datetime import datetime
from pathlib import Path
import json
import random
from PIL import Image
from .data_audit import digest
from .dataset_snapshot import _annotations
from .paper_common import outside_dataset,pin_release,read_csv,write_csv,write_json,project_path
from .paper_metrics import match

FIELDS=['sample_id','method','source_relative_image','source_image_sha256','source_label_sha256',
        'crop_path','crop_sha256','crop_xyxy','width','height','verification','reviewer','reviewed_at_utc',
        'notes','mining_checkpoint_sha256']

def source_map(dataset):
    return {r['new_relative_image']:r for r in read_csv(Path(dataset)/'dataset_manifest.csv') if r['new_split']=='train'}

def source_targets(dataset,row):
    path=Path(dataset)/row['new_relative_label']
    if digest(path)!=row['label_sha256']:raise ValueError('Source annotation changed')
    parsed,issues=_annotations(path)
    if issues:raise ValueError(issues)
    width,height=int(row['width']),int(row['height'])
    return [dict(box=[float((x-w/2)*width),float((y-h/2)*height),float((x+w/2)*width),float((y+h/2)*height)],**{'class':int(c)}) for c,x,y,w,h in parsed]

def safe(bounds,targets,width,height,margin=16):
    x1,y1,x2,y2=bounds
    if not(0<=x1<x2<=width and 0<=y1<y2<=height):return False
    return all(x2<=g['box'][0]-margin or x1>=g['box'][2]+margin or y2<=g['box'][1]-margin or y1>=g['box'][3]+margin for g in targets)

def verify_row(row,dataset,index,*,review_required=True,size=(256,256)):
    source=row['source_relative_image'];dataset=Path(dataset).resolve()
    # Current split membership, not historical basename prefixes, governs eligibility.
    if source not in index or not (dataset/source).resolve().is_relative_to(dataset/'images/train'):
        raise ValueError('Negative sources must belong to the CURRENT frozen Train split only')
    original=index[source]
    if row['source_image_sha256']!=original['image_sha256'] or row['source_label_sha256']!=original['label_sha256']:
        raise ValueError('Negative source identity differs from Clean_v2')
    bounds=json.loads(row['crop_xyxy'])
    if len(bounds)!=4 or any(type(x) is not int for x in bounds):raise ValueError('Crop coordinates must be integer original pixels')
    if (bounds[2]-bounds[0],bounds[3]-bounds[1])!=tuple(size):raise ValueError('Negative crop sizes must match the fixed control')
    if not safe(bounds,source_targets(dataset,original),int(original['width']),int(original['height'])):
        raise ValueError('An empty-label negative crop may not intersect an annotated UAV or its 16px safety margin')
    crop=project_path(row['crop_path']);outside_dataset(crop,dataset)
    if digest(crop)!=row['crop_sha256'] or digest(dataset/source)!=row['source_image_sha256']:raise ValueError('Reviewed negative image/source bytes changed')
    with Image.open(crop) as im,Image.open(dataset/source) as src:
        if im.size!=tuple(size) or im.convert('RGB').tobytes()!=src.convert('RGB').crop(bounds).tobytes():
            raise ValueError('Negative must be the exact declared source crop, without label-erasing edits')
    if (int(row['width']),int(row['height']))!=tuple(size):raise ValueError('Crop dimension metadata differs')
    if review_required:
        if row['verification']!='verified_no_uav' or not row['reviewer'].strip():raise ValueError('Every negative needs explicit UAV-free visual verification')
        date=datetime.fromisoformat(row['reviewed_at_utc'].replace('Z','+00:00'))
        if date.utcoffset() is None:raise ValueError('Review timestamp must include UTC offset')
    return crop

def verified_rows(path,dataset,index,method,count=512):
    rows=[r for r in read_csv(path) if r['verification']=='verified_no_uav']
    if len(rows)!=count:raise ValueError(f'Need exactly {count} explicitly verified {method} negatives, found {len(rows)}')
    if len({r['crop_sha256'] for r in rows})!=count or len({r['sample_id'] for r in rows})!=count:raise ValueError('Duplicate negative samples')
    for row in rows:
        if row['method']!=method:raise ValueError('Negative method differs from experiment arm')
        verify_row(row,dataset,index)
    return rows

def verify_catalog(review_path,method):
    """Review edits may change verdicts/notes, never the proposed crop or mining provenance."""
    review_path=Path(review_path);proposal=json.loads((review_path.parent/'proposal.json').read_text())
    catalog=review_path.parent/'candidates_manifest.csv'
    if digest(catalog)!=proposal['candidate_catalog_sha256']:raise ValueError('Negative candidate catalog changed')
    original={r['sample_id']:r for r in read_csv(catalog)}
    immutable=[k for k in FIELDS if k not in ('verification','reviewer','reviewed_at_utc','notes')]
    for r in read_csv(review_path):
        if r['sample_id'] not in original or any(r[k]!=original[r['sample_id']][k] for k in immutable):
            raise ValueError('Visual review changed crop identity or proposal method/provenance')
    if method=='hard':
        path=Path(proposal['source_evaluation'])/'evaluation.json'
        meta=json.loads(path.read_text())
        if digest(path)!=proposal['source_evaluation_sha256'] or meta['split']!='train' or meta['experiment_id']!='E1' or meta['seed']!=17:
            raise ValueError('Hard negatives lack the pinned E1 Train-only mining provenance')
    return proposal

def propose_hard(dataset,evaluation_dir,output,*,limit=1024):
    dataset=Path(dataset).resolve();output=outside_dataset(output,dataset)
    if output.exists():raise FileExistsError(output)
    pin_release(dataset,full=True)
    evaluation_dir=Path(evaluation_dir);meta=json.loads((evaluation_dir/'evaluation.json').read_text())
    if meta.get('status')!='COMPLETED' or meta.get('dataset_verified_after') is not True:
        raise ValueError('Mining requires a completed, integrity-verified Train-only evaluation')
    if meta['split']!='train' or meta['experiment_id']!='E1' or meta['seed']!=17:
        raise ValueError('Hard negatives must come from the fixed E1 seed17 Train-only inference')
    if meta['protocol_sha256']!=digest(project_path('configs/paper/protocol.json')) or meta['config_sha256']!=digest(project_path('configs/paper/E1.yaml')):
        raise ValueError('Mining inference differs from the controlled E1 protocol')
    pin_release(dataset,meta['dataset_release'])
    records_path=evaluation_dir/'predictions.jsonl'
    if digest(records_path)!=meta['raw_records_sha256']:raise ValueError('Mining predictions changed')
    index=source_map(dataset);candidates=[];seen_images=set();rows=[];seen_hash=set()
    with records_path.open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);source='images/train/'+r['image']
            if source not in index or r['split']!='train':raise ValueError('Validation/Test or unknown source encountered during mining')
            target=source_targets(dataset,index[source]);pred=[p for p in r['predictions'] if p['confidence']>=.5]
            _,errors,_=match(pred,target,.5)
            for i,(kind,_) in errors.items():
                if kind=='Background':candidates.append((pred[i]['confidence'],source,i,pred[i]['box']))
    output.mkdir(parents=True);(output/'candidates').mkdir()
    for score,source,i,box in sorted(candidates,key=lambda p:(-p[0],p[1],p[2])):
        if source in seen_images:continue  # Prevent one source image dominating the pool.
        original=index[source];width,height=int(original['width']),int(original['height'])
        if width<256 or height<256:continue
        x=max(0,min(width-256,round((box[0]+box[2])/2)-128));y=max(0,min(height-256,round((box[1]+box[3])/2)-128))
        bounds=[x,y,x+256,y+256]
        if not safe(bounds,source_targets(dataset,original),width,height):continue
        name=f'hard_{Path(source).stem}_{i}';path=output/'candidates'/(name+'.png')
        with Image.open(dataset/source) as im:im.convert('RGB').crop(bounds).save(path)
        sha=digest(path)
        if sha in seen_hash:continue
        seen_images.add(source);seen_hash.add(sha)
        rows.append(dict(sample_id=name,method='hard',source_relative_image=source,source_image_sha256=original['image_sha256'],
            source_label_sha256=original['label_sha256'],crop_path=str(path.resolve()),crop_sha256=sha,crop_xyxy=json.dumps(bounds),
            width=256,height=256,verification='pending',reviewer='',reviewed_at_utc='',
            notes=f'Model-guided Background FP confidence={score}; GT exclusion is NOT visual UAV-free verification',mining_checkpoint_sha256=meta['checkpoint_sha256']))
        if len(rows)>=limit:break
    write_csv(output/'review.csv',rows,FIELDS);write_csv(output/'candidates_manifest.csv',rows,FIELDS)
    write_json(output/'proposal.json',dict(source_evaluation_sha256=digest(evaluation_dir/'evaluation.json'),proposed=len(rows),
        source_evaluation=str(evaluation_dir.resolve()),candidate_catalog_sha256=digest(output/'candidates_manifest.csv'),
        max_per_source_image=1,crop_size=[256,256],review_required=True,training_performed=False))
    pin_release(dataset,full=True)
    return dict(proposed=len(rows),review=str(output/'review.csv'))

def propose_random(dataset,hard_review,output,*,exclude_review=None):
    dataset=Path(dataset).resolve();output=outside_dataset(output,dataset)
    if output.exists():raise FileExistsError(output)
    pin_release(dataset,full=True);index=source_map(dataset)
    verify_catalog(hard_review,'hard')
    hard=verified_rows(hard_review,dataset,index,'hard');rng=random.Random(17)
    excluded=set()
    if exclude_review is not None:
        verify_catalog(exclude_review,'random')
        excluded={r['crop_sha256'] for r in read_csv(exclude_review) if r['verification'].startswith('rejected')}
    output.mkdir(parents=True);(output/'candidates').mkdir();rows=[];used={r['crop_sha256'] for r in hard}|excluded
    for n,h in enumerate(sorted(hard,key=lambda r:r['sample_id'])):
        original=index[h['source_relative_image']];width,height=int(original['width']),int(original['height'])
        targets=source_targets(dataset,original);found=False;path=output/'candidates'/f'random_{n:04d}.png'
        with Image.open(dataset/h['source_relative_image']) as source:
            source=source.convert('RGB')
            for _ in range(5000):
                x=rng.randint(0,width-256);y=rng.randint(0,height-256);bounds=[x,y,x+256,y+256]
                if not safe(bounds,targets,width,height):continue
                source.crop(bounds).save(path);sha=digest(path)
                if sha in used:continue
                used.add(sha);found=True;break
        if not found:raise ValueError('Insufficient distinct source-matched random crops; do not change size/count silently')
        rows.append(dict(h,sample_id=f'random_{n:04d}',method='random',crop_path=str(path.resolve()),crop_sha256=sha,
            crop_xyxy=json.dumps(bounds),verification='pending',reviewer='',reviewed_at_utc='',notes='Uniform source-matched GT-free crop; independent visual verification required'))
    write_csv(output/'review.csv',rows,FIELDS);write_csv(output/'candidates_manifest.csv',rows,FIELDS)
    write_json(output/'proposal.json',dict(random_seed=17,hard_review_sha256=digest(hard_review),samples=len(rows),
        candidate_catalog_sha256=digest(output/'candidates_manifest.csv'),source_image_frequency_matched=True,crop_size=[256,256],review_required=True))
    if exclude_review is not None:
        report=json.loads((output/'proposal.json').read_text());report['excluded_review_sha256']=digest(exclude_review);report['excluded_crops']=len(excluded)
        write_json(output/'proposal.json',report)
    pin_release(dataset,full=True)
    return dict(proposed=len(rows),review=str(output/'review.csv'))

def publish_bundle(dataset,hard_review,random_review,output):
    import shutil
    import stat
    dataset=Path(dataset).resolve();output=outside_dataset(output,dataset)
    if output.exists():raise FileExistsError(output)
    lock_path=output.parent/'BUNDLE_LOCK.json'
    if lock_path.exists():raise FileExistsError('A negative-control bundle has already been frozen')
    identity=pin_release(dataset,full=True);index=source_map(dataset)
    verify_catalog(hard_review,'hard');control=verify_catalog(random_review,'random')
    if control['hard_review_sha256']!=digest(hard_review):raise ValueError('Hard review changed after constructing the random control')
    hard=verified_rows(hard_review,dataset,index,'hard');random_rows=verified_rows(random_review,dataset,index,'random')
    if Counter(r['source_relative_image'] for r in hard)!=Counter(r['source_relative_image'] for r in random_rows):raise ValueError('Random and hard controls must match source-image frequency')
    if set(r['crop_sha256'] for r in hard)&set(r['crop_sha256'] for r in random_rows):raise ValueError('Identical crops across controls are not allowed')
    checkpoint_hashes={r['mining_checkpoint_sha256'] for r in hard+random_rows}
    if len(checkpoint_hashes)!=1:raise ValueError('Matched control rows refer to different mining checkpoints')
    base=output.parent/'verified';base.mkdir(parents=True,exist_ok=False)
    manifests={}
    for method,rows in [('hard',hard),('random',random_rows)]:
        images=base/method/'images/train';labels=base/method/'labels/train';images.mkdir(parents=True);labels.mkdir(parents=True)
        for r in rows:
            target=images/(r['sample_id']+'.png');shutil.copyfile(project_path(r['crop_path']),target)
            label=labels/(r['sample_id']+'.txt');label.write_text('',encoding='utf-8')
            r.update(crop_path=str(target.resolve()),label_path=str(label.resolve()))
        path=base/method/'verified_manifest.csv';write_csv(path,rows,[*FIELDS,'label_path'])
        manifests[method]=dict(path=str(path.resolve()),sha256=digest(path),count=len(rows))
    result=dict(schema_version=1,dataset_release=identity,count_per_arm=512,crop_size=[256,256],source_split='train',
        source_image_frequency_matched=True,manifests=manifests,mining_checkpoint_sha256=next(iter(checkpoint_hashes)),hard_review_sha256=digest(hard_review),random_review_sha256=digest(random_review))
    write_json(output,result)
    write_json(lock_path,dict(bundle_sha256=digest(output),dataset_release=identity,immutable=True,count_per_arm=512,crop_size=[256,256]))
    for path in [*base.rglob('*'),output,lock_path]:
        if path.is_file():path.chmod(stat.S_IREAD|stat.S_IRGRP|stat.S_IROTH)
    pin_release(dataset,full=True)
    return result

def validate_bundle(path,dataset,expected_count=512):
    path=Path(path);bundle=json.loads(path.read_text());pin_release(dataset,bundle['dataset_release'])
    lock=json.loads((path.parent/'BUNDLE_LOCK.json').read_text())
    if lock['bundle_sha256']!=digest(path) or lock.get('immutable') is not True:raise ValueError('Negative bundle differs from its frozen identity')
    if bundle['source_split']!='train' or bundle['count_per_arm']!=expected_count or bundle['crop_size']!=[256,256]:raise ValueError('Negative bundle differs from the controlled protocol')
    index=source_map(dataset);result={}
    for method in ('hard','random'):
        manifest=project_path(bundle['manifests'][method]['path'])
        if digest(manifest)!=bundle['manifests'][method]['sha256']:raise ValueError('Verified manifest changed')
        result[method]=verified_rows(manifest,dataset,index,method,expected_count)
        for r in result[method]:
            if r['mining_checkpoint_sha256']!=bundle['mining_checkpoint_sha256']:raise ValueError('Negative mining checkpoint provenance differs')
            label=project_path(r['label_path']);outside_dataset(label,dataset)
            if label.read_bytes()!=b'':raise ValueError('Verified negative label must be exactly empty')
    if Counter(r['source_relative_image'] for r in result['hard'])!=Counter(r['source_relative_image'] for r in result['random']):raise ValueError('Matched negative source frequencies differ')
    if set(r['crop_sha256'] for r in result['hard'])&set(r['crop_sha256'] for r in result['random']):raise ValueError('Controls share identical crops')
    return result
