"""Gate 3: read-only complete readiness audit, including metadata-only Test inspection."""
from __future__ import annotations
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import json
import stat
from PIL import Image
from .data_audit import IMAGE_SUFFIXES, digest
from .paper_common import SIZES, outside_dataset, pin_release, read_csv, size_bin, write_csv, write_json

TOLERANCE = Decimal('0.000000001')  # Same 1e-9 release parser policy; report smaller rounding excess separately.

def inspect_pair(item):
    image, label, split = item
    issues=[]; counts=Counter(); rounding=0
    try:
        with Image.open(image) as im:
            im.load(); width,height=im.size
    except (OSError,ValueError) as exc:
        return dict(image=image.name,split=split,objects=0,empty=False,issues=[('unreadable_image',str(exc))],sizes={},rounding=0)
    if not label.is_file():
        return dict(image=image.name,split=split,objects=0,empty=False,issues=[('missing_label',label.name)],sizes={},rounding=0)
    seen=set(); objects=0
    try: lines=label.read_text(encoding='utf-8-sig').splitlines()
    except (OSError,UnicodeError) as exc: lines=[];issues.append(('unreadable_label',str(exc)))
    for n,line in enumerate(lines,1):
        if not line.strip():continue
        try:
            values=tuple(Decimal(x) for x in line.split())
            if len(values)!=5 or not all(x.is_finite() for x in values): raise ValueError('malformed_or_nonfinite')
            cls,x,y,w,h=values
            if cls!=0:issues.append(('invalid_class',str(n)))
            if w==0 or h==0:issues.append(('zero_width_or_height',str(n)))
            if not(0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1):issues.append(('invalid_normalized_coordinates',str(n)))
            excess=max(Decimal(0),-(x-w/2),-(y-h/2),x+w/2-1,y+h/2-1)
            if excess>TOLERANCE:issues.append(('boundary_violation',f'{n}: {excess}'))
            elif excess>0:rounding+=1
            if values in seen:issues.append(('duplicate_annotation',str(n)))
            seen.add(values);objects+=1
            if w>0 and h>0:counts[size_bin(w*width*h*height)]+=1
        except (InvalidOperation,ValueError) as exc:issues.append(('malformed_annotation',f'{n}: {exc}'))
    return dict(image=image.name,split=split,objects=objects,empty=not any(x.strip() for x in lines),
                issues=issues,sizes=dict(counts),rounding=rounding,width=width,height=height)

def audit(dataset, output):
    dataset=Path(dataset).resolve();output=outside_dataset(output,dataset)
    if output.exists():raise FileExistsError(output)
    identity=pin_release(dataset,full=True)
    before={p.relative_to(dataset).as_posix():(p.stat().st_size,p.stat().st_mtime_ns) for p in dataset.rglob('*') if p.is_file()}
    manifest=read_csv(dataset/'dataset_manifest.csv'); keyed={r['new_relative_image']:r for r in manifest}
    scene_rows=read_csv(dataset/'scene_groups.csv'); scene_map={r['image']:r for r in scene_rows}
    assignment=read_csv(dataset/'split_assignment.csv')
    if assignment != manifest:raise ValueError('Split assignment and dataset manifest disagree')
    counters=Counter();findings=[];summary={};items=[]
    for split in ('train','val','test'):
        images=[p for p in (dataset/'images'/split).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
        labels=list((dataset/'labels'/split).glob('*.txt'))
        stems=Counter(p.stem for p in images)
        counters['duplicate_image_stems']+=sum(v>1 for v in stems.values())
        for label in labels:
            if label.stem not in stems:counters['orphan_labels']+=1;findings.append(dict(split=split,image=label.name,issue='orphan_label',detail=''))
        items.extend((p,dataset/'labels'/split/(p.stem+'.txt'),split) for p in sorted(images))
        summary[split]=dict(images=len(images),labels=len(labels),objects=0,empty_labels=0,**{k:0 for k in SIZES})
    with ThreadPoolExecutor(max_workers=6) as pool:
        results=list(pool.map(inspect_pair,items))
    for r in results:
        s=summary[r['split']];s['objects']+=r['objects'];s['empty_labels']+=int(r['empty'])
        for k,n in r['sizes'].items():s[k]+=n
        counters['coordinate_rounding_excess_within_1e_9']+=r['rounding']
        for kind,detail in r['issues']:
            counters[kind]+=1;findings.append(dict(split=r['split'],image=r['image'],issue=kind,detail=detail))
        expected=keyed.get(f"images/{r['split']}/{r['image']}")
        if expected is None or int(expected['objects'])!=r['objects'] or any(int(expected[k.lower()])!=r['sizes'].get(k,0) for k in SIZES):
            counters['manifest_count_mismatch']+=1
    scenes=defaultdict(set);blocks=defaultdict(set);hashes=defaultdict(set)
    for r in manifest:
        split=r['new_split'];scenes[r['scene_group_id']].add(split);blocks[r['allocation_block_id']].add(split);hashes[r['image_sha256']].add(split)
        g=scene_map.get(r['image'])
        if not g or any(g[k]!=r[k] for k in ('scene_group_id','allocation_block_id','image_sha256')):counters['scene_metadata_mismatch']+=1
    counters['scene_group_cross_split']=sum(len(x)>1 for x in scenes.values())
    counters['allocation_block_cross_split']=sum(len(x)>1 for x in blocks.values())
    counters['exact_image_sha_cross_split']=sum(len(x)>1 for x in hashes.values())
    counters['manifest_image_coverage_mismatch']=len(set(keyed) ^ {f'images/{s}/{p.name}' for p,_,s in items})
    counters['scene_coverage_mismatch']=len(set(scene_map) ^ {r['image'] for r in manifest})
    files=[p for p in dataset.rglob('*') if p.is_file()]
    counters['cache_files']=sum(p.suffix.lower() in ('.cache','.npy','.tmp','.bak') or '__pycache__' in p.parts for p in files if p.parent.name in ('train','val','test') or p.suffix=='.cache')
    # Root NPY descriptor provenance is intentional frozen evidence, not image-loader caches.
    counters['writable_release_files']=sum(bool(p.stat().st_mode & stat.S_IWRITE) for p in files)
    pin_release(dataset,identity,full=True)
    after={p.relative_to(dataset).as_posix():(p.stat().st_size,p.stat().st_mtime_ns) for p in files}
    counters['files_changed_during_audit']=sum(before.get(k)!=v for k,v in after.items())+len(set(before)-set(after))
    expected_counts=json.loads((dataset/'FREEZE.json').read_text())['counts']
    counters['release_count_mismatch']=sum(summary[s]['images']!=expected_counts[s]['images'] or summary[s]['objects']!=expected_counts[s]['objects'] or summary[s]['empty_labels']!=expected_counts[s]['negative_images'] for s in summary)
    keys=['missing_label','orphan_labels','malformed_annotation','unreadable_label','unreadable_image','invalid_class','invalid_normalized_coordinates','zero_width_or_height','boundary_violation','duplicate_annotation','duplicate_image_stems','manifest_count_mismatch','scene_metadata_mismatch','scene_group_cross_split','allocation_block_cross_split','exact_image_sha_cross_split','manifest_image_coverage_mismatch','scene_coverage_mismatch','cache_files','writable_release_files','files_changed_during_audit','release_count_mismatch']
    checks={k:counters[k] for k in keys}
    report=dict(status='PASS' if not any(checks.values()) else 'FAIL',created_at_utc=datetime.now(timezone.utc).isoformat(),
        dataset_release=identity,checks=checks,counts=summary,scene_groups=len(scenes),allocation_blocks=len(blocks),
        boundary_tolerance=str(TOLERANCE),rounding_excess_within_tolerance=counters['coordinate_rounding_excess_within_1e_9'],
        images_fully_decoded=len(results),sha256_verification='All release payload files, before and after audit',
        frozen_dataset_writes=0,model_loaded=False,training_performed=False,test_model_evaluation_performed=False,
        limitations=['Validation/Test contain no annotation-defined empty-label images: background-only FPR is undefined there.',
          'The single empty Train label is not independently certified UAV-free by this structural audit.',
          'Scene-disjoint under the declared Gate 2 protocol, not native sequence-disjoint.',
          'Large conservative Train allocation block constrains object-size/split balance.'])
    output.mkdir(parents=True)
    write_json(output/'gate3_report.json',report)
    write_csv(output/'annotation_issues.csv',findings,['split','image','issue','detail'])
    write_csv(output/'split_distribution.csv',[dict(split=s,**v) for s,v in summary.items()],['split','images','labels','objects','empty_labels',*SIZES])
    return report

def runtime_preflight(dataset,output):
    """Reject images that Ultralytics would try to repair in-place or transpose."""
    dataset=Path(dataset).resolve();output=outside_dataset(output,dataset)
    identity=pin_release(dataset,full=True)
    images=[dataset/r['new_relative_image'] for r in read_csv(dataset/'dataset_manifest.csv')]
    def check(path):
        result=Counter()
        with Image.open(path) as im:
            result['unsupported_exif_orientation']=int(im.getexif().get(274,1) not in (None,1))
            result['image_dimension_below_10']=int(min(im.size)<10)
            jpeg=im.format.lower() in ('jpg','jpeg')
        if jpeg:
            with path.open('rb') as f:f.seek(-2,2);result['jpeg_missing_eoi']=int(f.read()!=b'\xff\xd9')
        return result
    counts=Counter()
    with ThreadPoolExecutor(max_workers=6) as pool:
        for r in pool.map(check,images):counts.update(r)
    report=dict(status='PASS' if not any(counts.values()) else 'FAIL',dataset_release=identity,images_checked=len(images),
        checks=dict(counts),source_image_repairs_requested=False,source_files_written=0,
        reason='Ultralytics check_image can save repaired JPEGs; all source JPEG end markers and EXIF coordinate conventions must already be safe')
    pin_release(dataset,identity,full=True);output.parent.mkdir(parents=True,exist_ok=True);write_json(output,report)
    return report
