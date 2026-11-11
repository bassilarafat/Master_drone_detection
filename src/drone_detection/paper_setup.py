"""Create pinned E0--E3 templates; no model training or dataset mutation."""
from __future__ import annotations
from pathlib import Path
import importlib.metadata as metadata
import json
import platform
import sys
import yaml
from .paper_common import PROJECT_ROOT, DATASET, RELEASE_ID, MANIFEST_SHA, FREEZE_SHA, bootstrap, write_json
from .data_audit import digest

def prepare():
    bootstrap()
    root=PROJECT_ROOT;folder=root/'configs/paper'
    if (folder/'protocol.json').exists():raise FileExistsError('Preserve the protocol; changes require an explicitly versioned replacement')
    folder.mkdir(parents=True,exist_ok=True)
    package=Path(metadata.distribution('ultralytics').locate_file('ultralytics'))
    topology={}
    for arm,source_name,target in [('E0','yolov8.yaml','yolov8s-paper.yaml'),('E1','yolov8-p2.yaml','yolov8s-p2-paper.yaml')]:
        source=package/'cfg/models/v8'/source_name
        values=yaml.safe_load(source.read_text(encoding='utf-8'));values.update(nc=1,scale='s')
        path=folder/target
        path.write_text('# Derived from Ultralytics official architecture; AGPL-3.0, https://ultralytics.com/license\n'+yaml.safe_dump(values,sort_keys=False),encoding='utf-8')
        topology[arm]=dict(path=path.relative_to(root).as_posix(),sha256=digest(path),source_sha256=digest(source),
            source_url='https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/'+source_name,
            package_version=metadata.version('ultralytics'))
    from .paper_weights import RELATIVE,SHA,URL,COMMIT
    weights=root/RELATIVE
    if not weights.is_file() or digest(weights)!=SHA:raise FileNotFoundError('Run the explicit paper_weights preparation for the verified public artifact; setup never downloads weights')
    environment=dict(python=sys.version,platform=platform.platform(),packages={p:metadata.version(p) for p in
        ['ultralytics','torch','torchvision','numpy','Pillow','PyYAML','opencv-python','scipy']},
        implementation_sha256={str(p.relative_to(package)):digest(p) for p in [package/'engine/trainer.py',package/'utils/metrics.py',package/'nn/tasks.py',package/'cfg/default.yaml']})
    write_json(folder/'environment.lock.json',environment)
    protocol=dict(schema_version=1,protocol_id='DUT_Clean_v2_paper_v1',created_date='2026-11-10',
        dataset=dict(path='02_datasets/'+DATASET,release_id=RELEASE_ID,manifest_sha256=MANIFEST_SHA,freeze_sha256=FREEZE_SHA,
            release_record='configs/datasets/DUT_Anti_UAV_Clean_v2.release.json',
            release_record_sha256=digest(root/'configs/datasets/DUT_Anti_UAV_Clean_v2.release.json')),
        gate3=dict(path='06_results/paper/Gate3_20261110/gate3_report.json',sha256=digest(root/'06_results/paper/Gate3_20261110/gate3_report.json')),
        initialization=dict(path=weights.relative_to(root).as_posix(),sha256=digest(weights),
            source_url=URL,repository_commit=COMMIT,
            policy='COCO-pretrained shared backbone modules 0..9 only; all neck/head layers randomly initialized with run seed in every arm',
            remote_digest_independently_verified=True),
        topologies=topology,environment_lock_sha256=digest(folder/'environment.lock.json'),
        seeds=[17,42,123],default_seed=17,
        train=dict(epochs=100,imgsz=960,batch=1,nbs=16,optimizer='SGD',lr0=.01,lrf=.01,momentum=.937,
            weight_decay=.0005,cos_lr=True,warmup_epochs=0.,warmup_momentum=.8,warmup_bias_lr=.1,
            patience=0,device='0',workers=0,amp=False,cache=False,rect=False,multi_scale=0.,
            deterministic=True,resume=False,pretrained=False,freeze=None,fraction=1.,val=True,split='val',
            conf=.001,iou=.7,max_det=300,agnostic_nms=False,single_cls=False,
            hsv_h=.015,hsv_s=.7,hsv_v=.4,degrees=0.,translate=.1,scale=.5,shear=0.,perspective=0.,
            fliplr=.5,flipud=0.,mosaic=0.,mixup=0.,cutmix=0.,copy_paste=0.,close_mosaic=0,
            erasing=0.,auto_augment=None,dropout=0.,box=7.5,cls=.5,dfl=1.5,
            save=True,save_period=-1,plots=False,verbose=False,compile=False),
        optimization=dict(nominal_effective_batch=16,accumulation_steps=16,warmup_disabled_to_keep_accumulation_constant=True,
            tail_batches='Partial terminal batches can be smaller; paired arms have the same sample count and update schedule'),
        checkpoint=dict(selection='last.pt after epoch 100 for all arms/seeds; best.pt is not used for paper selection',
            validation_monitor='Ultralytics native validation only; final paper metrics use the unified evaluator'),
        evaluation=dict(split='val',test_locked=True,imgsz=960,prediction_conf=.001,operating_conf=.25,
            matching_iou=.5,nms_iou=.7,max_det=300,batch=1,rect=False,augment=False,half=False,
            agnostic_nms=False,AP_iou_thresholds=[i/100 for i in range(50,96,5)],
            size_definitions=dict(Tiny='area<1024',Small='1024<=area<9216',Medium='9216<=area<65536',Large='area>=65536'),
            size_units='GT box area in original image pixels, not resized network-input pixels',
            evaluator='python -m drone_detection paper evaluate'),
        negatives=dict(samples_per_arm=512,crop_size=[256,256],gt_margin_pixels=16,source_split='train',
            mining_arm='E1',mining_seed=17,mining_checkpoint='last.pt after epoch 100',mining_confidence=.5,
            random_seed=17,random_control='Uniform eligible background crops from the same source images as approved hard negatives',
            verification='Every crop must be explicitly reviewed verified_no_uav with reviewer and UTC timestamp; geometric exclusion alone is insufficient',
            bundle='04_experiments/paper_v1/negative_pool/matched_bundle.json',
            no_validation_or_test_mining=True,hard_and_random_counts_sizes_source_frequency_matched=True),
        execution=dict(training_started=False,test_evaluated=False,test_unlock_implemented=False))
    write_json(folder/'protocol.json',protocol)
    protocol_sha=digest(folder/'protocol.json')
    for arm in ('E0','E1','E2','E3'):
        cfg=dict(schema_version=1,experiment_id=arm,protocol='configs/paper/protocol.json',protocol_sha256=protocol_sha,
            dataset=protocol['dataset'],model=topology['E0' if arm=='E0' else 'E1'],initialization=protocol['initialization'],
            seeds=protocol['seeds'],default_seed=17,train=protocol['train'],evaluation=protocol['evaluation'],
            negative_mode={'E0':'none','E1':'none','E2':'random','E3':'hard'}[arm],
            negative_count=0 if arm in ('E0','E1') else 512,negative_bundle=protocol['negatives']['bundle'],
            output_root='04_experiments/paper_v1/'+arm)
        (folder/f'{arm}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
        for seed in protocol['seeds']:
            d=root/f'04_experiments/paper_v1/{arm}/seed_{seed}'
            d.mkdir(parents=True,exist_ok=True)
            write_json(d/'PLAN.json',dict(experiment_id=arm,seed=seed,config=f'configs/paper/{arm}.yaml',
                config_sha256=digest(folder/f'{arm}.yaml'),dataset_release=protocol['dataset'],
                status='NOT_STARTED',checkpoint='run/weights/last.pt'))
    pool=root/'04_experiments/paper_v1/negative_pool';pool.mkdir(parents=True,exist_ok=True)
    fields=['sample_id','method','source_relative_image','source_image_sha256','source_label_sha256','crop_path','crop_sha256',
        'crop_xyxy','width','height','verification','reviewer','reviewed_at_utc','notes','mining_checkpoint_sha256']
    from .paper_common import write_csv
    write_csv(pool/'verification_template.csv',[],fields)
    write_json(pool/'STATUS.json',dict(status='BLOCKED_PENDING_TRAIN_ONLY_MINING_AND_EXPLICIT_REVIEW',
        required_verified_hard=512,required_verified_random=512,training_performed=False))
    return dict(configurations=[f'configs/paper/{a}.yaml' for a in ('E0','E1','E2','E3')],protocol_sha256=protocol_sha)
