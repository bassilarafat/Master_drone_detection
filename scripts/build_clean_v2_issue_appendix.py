"""Evidence-backed historical issue appendix for the Arabic Word report."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def append(c):
    root, data, assets = c['ROOT'], c['DATA'], c['ASSETS']
    p, title, table, pic = (c[k] for k in ('p', 'title', 'table', 'pic'))
    src = root / '02_datasets/DUT_Anti_UAV'
    v1 = root / '02_datasets/DUT_Anti_UAV_Clean_v1'
    index = {r['image']: r for r in c['ROWS']}
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 25)
    sf = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 21)
    evidence = []

    def source(note):
        p('الدليل: ' + note, size=18, color='647B88')

    def load(identity):
        return Image.open(src / 'images' / identity).convert('RGB')

    def box(im, values):
        _, x, y, w, h = map(float, values.split())
        return ((x-w/2)*im.width, (y-h/2)*im.height, (x+w/2)*im.width, (y+h/2)*im.height)

    def labelled(identity, label=None, color='#00FF77'):
        im = load(identity)
        label = label if label is not None else (src/'labels'/Path(identity).with_suffix('.txt')).read_text()
        d = ImageDraw.Draw(im)
        for line in label.splitlines():
            if not line.strip(): continue
            b = box(im,line)
            d.rectangle(b,outline=color,width=max(2,im.width//500))
        return im

    def pair(name, items, height=570):
        canvas = Image.new('RGB',(1600,height),'#EFF4F8')
        d = ImageDraw.Draw(canvas)
        for j,(im,heading,detail) in enumerate(items):
            x=j*800
            d.text((x+18,16),heading,font=font,fill='#16364A')
            d.text((x+18,55),detail,font=sf,fill='#176B87')
            c['fit'](canvas,im,(x+15,100,770,height-115))
        canvas.save(assets/name,quality=94)
        return name

    # 1. True pre-normalization labels are retained in provenance/backups.
    norm = v1/'provenance/annotation_normalization_changes.csv'
    changes = list(csv.DictReader(norm.open(encoding='utf-8-sig')))
    change = next(r for r in changes if r['split']=='train' and r['image']=='05000.jpg')
    identity = change['split']+'/'+change['image']
    im=load(identity)
    before, after=box(im,change['original']),box(im,change['normalized'])
    # Show the image's actual edge with padded space, then zoom the ~1px change.
    panels=[]
    for b,color,label in [(before,'#F08035','BEFORE | recorded original box'),(after,'#00C878','AFTER | recorded clipped box')]:
        pad=100
        framed=Image.new('RGB',(im.width+pad*2,im.height+pad*2),'#DCE4EA')
        framed.paste(im,(pad,pad))
        d=ImageDraw.Draw(framed)
        d.rectangle((pad,pad,pad+im.width-1,pad+im.height-1),outline='#176B87',width=1)
        d.rectangle(tuple(t+pad for t in b),outline=color,width=1)
        edge=max([(max(0,-before[0]),'left'),(max(0,-before[1]),'top'),(max(0,before[2]-im.width),'right'),(max(0,before[3]-im.height),'bottom')])[1]
        cx=(before[0]+before[2])/2+pad;cy=(before[1]+before[3])/2+pad
        if edge=='bottom':cy=im.height+pad
        elif edge=='top':cy=pad
        elif edge=='right':cx=im.width+pad
        else:cx=pad
        crop=framed.crop((int(cx-70),int(cy-40),int(cx+70),int(cy+40))).resize((700,400),Image.Resampling.NEAREST)
        panels.append((crop,label,'train/05000.jpg | image edge magnified x5'))
    pair('08_clipping_before_after.jpg',panels)
    evidence.append({'issue':'out_of_bounds_boxes','file':str(norm.relative_to(root)), 'example':identity,'original':change['original'],'corrected':change['normalized']})

    # 2. A real exact-duplicate decision with matching annotations.
    dup_rows=list(csv.DictReader((v1/'duplicate_decisions.csv').open()))
    removed=next(r for r in dup_rows if r['retained']=='False' and r['image'].startswith('images/test/') and 'keep_lexicographically' in r['decision'])
    keep=removed['kept_image'].removeprefix('images/')
    drop=removed['image'].removeprefix('images/')
    h=lambda x: hashlib.sha256((src/'images'/x).read_bytes()).hexdigest()
    assert h(keep)==h(drop)==removed['byte_sha256']
    assert list(load(keep).getdata())==list(load(drop).getdata())
    numeric=lambda x: sorted(tuple(map(float,s.split())) for s in (src/'labels'/Path(x).with_suffix('.txt')).read_text().splitlines() if s.strip())
    assert numeric(keep)==numeric(drop)
    pair('09_exact_duplicate.jpg',[(labelled(keep),'KEPT | '+keep,'Byte-identical + same numeric labels'),(labelled(drop),'EXCLUDED COPY | '+drop,'Duplicate within original Test split')])
    evidence.append({'issue':'exact_duplicates','retained':keep,'excluded':drop,'sha256':h(keep),'numeric_labels_identical':True})

    # 3. Visually identical source frames with conflicting annotation rows.
    excluded=list(csv.DictReader((v1/'excluded_samples.csv').open()))
    conflict=[next(r for r in excluded if r['image']==f'images/train/{name}.jpg') for name in ('00004','00014')]
    a,b='train/00004.jpg','train/00014.jpg'
    assert load(a).tobytes()==load(b).tobytes()
    assert conflict[0]['source_annotation'] != conflict[1]['source_annotation']
    pair('10_annotation_conflict.jpg',[(labelled(a,conflict[0]['source_annotation'], '#F08035'),'EXCLUDED | '+a,'Same RGB pixels; annotation A (orange)'),(labelled(b,conflict[1]['source_annotation']),'EXCLUDED | '+b,'Same RGB pixels; annotation B (green)')])
    evidence.append({'issue':'conflicting_annotations','images':[a,b],'rgb_identical':True,'labels':[r['source_annotation'] for r in conflict]})

    # 4. The actual zero-height annotation, with a zoomed marker and valid box.
    bad='val/00991.jpg'
    label=(src/'labels/val/00991.txt').read_text()
    badim=load(bad)
    badbox=box(badim,label.splitlines()[0])
    d=ImageDraw.Draw(badim)
    for j,line in enumerate(label.splitlines()):
        bb=box(badim,line)
        d.rectangle(bb,outline='#FF553F' if j==0 else '#00FF77',width=3)
    cx,cy=(badbox[0]+badbox[2])/2,(badbox[1]+badbox[3])/2
    zoom=badim.crop((int(cx-65),int(cy-45),int(cx+85),int(cy+55))).resize((750,500),Image.Resampling.NEAREST)
    pair('11_zero_height.jpg',[(badim,'EXCLUDED | val/00991.jpg','Red: zero-height row | Green: valid row'),(zoom,'ENLARGED LABEL REGION','Zero-height box is a line, not a valid area')])
    evidence.append({'issue':'zero_height','image':bad,'label':label})

    # 5. Reviewed scene match and an uncertain source-context pair.
    scene=['train/04923.jpg','val/02462.jpg']
    reviews=json.loads((data/'scene_visual_reviews.json').read_text())
    review=next(r for r in reviews if {r['image_a'],r['image_b']}==set(scene) and r['decision']=='CONFIRMED_SAME_SCENE')
    assert index[scene[0]]['new_split']==index[scene[1]]['new_split']
    pair('12_scene_leakage.jpg',[(labelled(s),'OLD '+s,f'NEW {index[s]["new_split"]} | {index[s]["scene_group_id"]}') for s in scene])
    evidence.append({'issue':'scene_overlap','images':scene,'saved_review':review,'final_scene':index[scene[0]]['scene_group_id'],'final_split':index[scene[0]]['new_split']})
    uncertain=['test/00310.jpg','test/00311.jpg']
    assert index[uncertain[0]]['allocation_block_id']==index[uncertain[1]]['allocation_block_id']
    pair('13_uncertain_context.jpg',[(labelled(s),s,f'NEW {index[s]["new_split"]} | {index[s]["allocation_block_id"]}') for s in uncertain])
    evidence.append({'issue':'uncertain_source_context','images':uncertain,'scene_ids':[index[s]['scene_group_id'] for s in uncertain],'allocation_block':index[uncertain[0]]['allocation_block_id']})

    history=json.loads((data/'release_audit_history.json').read_text())
    audits=history['unpublished_failed_audits']
    confirmed=[sum(a['checks'][k] for k in ['confirmed_same_scene_train_val','confirmed_same_scene_train_test','confirmed_same_scene_val_test']) for a in audits]+[0]
    fig,ax=plt.subplots(figsize=(8.4,3.5))
    bars=ax.bar(['Fail 1','Fail 2','Fail 3','Fail 4','Fail 5','Final PASS'],confirmed,color=['#E4A34E']*5+['#54A89B'])
    ax.set(ylabel='Confirmed cross-split links',title='Saved pre-publication gate reports',ylim=(0,10))
    for bar,n in zip(bars,confirmed):ax.text(bar.get_x()+bar.get_width()/2,n+.15,str(n),ha='center')
    c['save'](fig,'14_gate_history.png')
    (assets/'issue_evidence.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False),encoding='utf-8')
    for original,name in [(norm,'normalization_changes.csv'),(v1/'excluded_samples.csv','excluded_samples.csv'),(v1/'duplicate_decisions.csv','duplicate_decisions.csv'),(v1/'excluded_artifacts.csv','excluded_artifacts.csv')]:
        shutil.copyfile(original,assets/name)

    title(10,'المشكلات التاريخية — ما الذي تغيّر؟')
    p('الفصل التالي يوثّق إجراءات سابقة حسب سجلات المشروع، ولا يعني تنفيذ تنظيف جديد أثناء إعداد هذا التقرير. أُضيفت صور توضيحية حقيقية من المصدر، مع تحقق مباشر لأمثلة التكرار والتعارض.',size=22)
    table(['المشكلة','الحجم المثبت','الإجراء السابق','الحالة'],[
        ['صناديق خارج الحدود','33 صندوقًا / 33 ملفًا','قص الحدود وتسجيل القيم القديمة والجديدة','معالجة قبل Clean_v1'],
        ['تكرار مطابق في Test','539 نسخة زائدة','الاحتفاظ بنسخة واحدة متفقة الليبل','مستبعدة من Clean_v1'],
        ['صور متطابقة بليبلز متعارضة','8 صور / 4 مجموعات','استبعاد جميع أعضاء التعارض','تحتاج تحكيمًا لإعادتها'],
        ['صندوق ارتفاعه صفر','صورة واحدة / بها صفان','استبعاد الصورة والليبل كاملين','لم يُخمن صندوق بديل'],
        ['ملفات كاش وباك أب يتيمة','4 ملفات مسجلة','عدم نسخها للإصدار النظيف','مستبعدة'],
        ['تشابه مشاهد عبر الأقسام','مثبت في المراجعات وبوابات FAIL','تجميع مشاهد وإعادة تقسيم بكتل','PASS وفق البروتوكول'],
        ['خلفيات غامضة أو قليلة السمات','329 صورة بأقل من 16 نقطة','قيود تخصيص تحفظية','استقلال الفيديو غير مثبت']])
    p('مسار الإصدارات: 9,999 صورة بالمصدر ← استبعاد 548 صورة في Clean_v1 ← الاحتفاظ بكل 9,451 صورة دون تغيير بايتات الصور والليبلز عند بناء Clean_v2.',size=23,bold=True,color='176B87')
    source('Clean_v1/cleaning_report.json؛ docs/dataset_clean_v1.md؛ Clean_v2/cleaning_report.json.')
    p('أسباب نشأة الأخطاء في المصدر غير موثقة بالكامل. كل تفسير محتمل يُذكر صراحة كاحتمال؛ لا توجد أدلة تكفي لنسب المشكلة إلى أداة أو شخص أو عملية تصدير محددة.',size=21)

    title(11,'صناديق تجاوزت حدود الصور')
    p('المشكلة والدليل: سجل التطبيع يثبت تعديل 33 صندوقًا في 33 ملفًا. المثال train/05000.jpg يتجاوز حافة الصورة بنحو 1.0013 بكسل؛ التجاوزات صغيرة وليست صناديق بعيدة عن الصورة.',size=22)
    pic('08_clipping_before_after.jpg')
    p('شكل 8 — البرتقالي قبل القص والأخضر بعده، مع تكبير حافة الصورة ×5. الخلفية الرمادية خارج حدود الصورة أضيفت لإظهار الحافة؛ المصدر لم يُعدّل.',size=20,color='647B88')
    p('سبب المخالفة حسابيًا: مركز الصندوق وحجمه ينتجان ركنًا خارج [0، العرض] أو [0، الارتفاع]. قد يرتبط ذلك بتحويل الإحداثيات أو التقريب أو التعليم عند الحافة، لكن السبب الأصلي غير مثبت.',size=22)
    p('الخطوات السابقة: تحويل الصندوق إلى زوايا بالبكسل ← قص الزوايا إلى حدود الصورة ← إعادة حساب المركز والحجم المطبع ← حفظ نسخة احتياطية وسجل قبل/بعد. الهدف الحفاظ على الجزء المرئي من الصندوق وإنتاج قيم صالحة دون تخمين هدف جديد.',size=22)
    p('تم ذلك قبل تجميد Clean_v1؛ لا توجد إعادة تعديل لليبلز خلال إعادة تقسيم Clean_v2.',bold=True,size=22)
    source('Clean_v1/provenance/annotation_normalization_report.json وannotation_normalization_changes.csv؛ 06_results/dataset_normalization/backups/train/05000.txt.')

    title(12,'نسخ متكررة مطابقة داخل الاختبار')
    p('المشكلة: 539 نسخة Test زائدة تكرر صورًا بنفس البايتات والبكسلات وبنفس multiset رقمي للـlabels. تكرار العينة يزيد وزنها في التقييم وقد يغيّر النتائج، حتى عندما يكون التكرار داخل القسم نفسه.',size=22)
    pic('09_exact_duplicate.jpg')
    p(f'شكل 9 — {keep} احتُفظ بها و{drop} استُبعدت. أُعيد التحقق من تطابق SHA-256 وبكسلات RGB والليبلز الرقمية لهذا المثال عند إعداد الملحق.',size=20,color='647B88')
    p('السبب المثبت هو وجود المحتوى نفسه تحت اسمين. سبب إنشاء النسخة في المصدر غير معلوم؛ لا يثبت السجل خطأ نسخ أو دمج بعينه.',size=22)
    p('المعالجة السابقة: تجميع الصور المتطابقة، ثم مقارنة الليبلز كقيم رقمية مع الاحتفاظ بتكرار الصفوف؛ عند الاتفاق الاحتفاظ بأول اسم معجميًا واستبعاد النسخ الزائدة. اختلاف المسافات أو ترتيب الصفوف وحده لا يُعد تعارضًا.',size=22)
    p('لماذا؟ منع وزن إضافي للعينة مع الاحتفاظ بمحتواها المعلّم مرة واحدة. هذه الحالة لا تثبت تسريبًا تامًا بين الأقسام؛ سجل Clean_v1 يقول إنه لم توجد مجموعات تطابق تام عابرة للأقسام في تلك المرحلة.',size=21)
    source('Clean_v1/duplicate_decisions.csv وexcluded_samples.csv؛ نسخة من السجلين مرفقة داخل word_assets.')

    title(13,'نفس الصورة ولكن الصناديق مختلفة')
    p('المشكلة: أربع مجموعات صور متطابقة البكسلات تحمل annotations مختلفة: Train 00004/00014 و00020/00158 و00248/00700؛ وValidation 00002/00007. الإجمالي 8 صور.',size=22)
    pic('10_annotation_conflict.jpg')
    p('شكل 10 — نفس بكسلات RGB، مع صندوق برتقالي من الليبل الأول وأخضر من الثاني. اختلاف الصندوق حقيقي في القيم المسجلة، وقد يكون صغيرًا بصريًا.',size=20,color='647B88')
    for r in conflict:
        p(r['image']+' : '+r['source_annotation'],size=18,align='left',rtl=False)
    p('السبب المثبت: تعدد نسخ التعليق على الصورة نفسها. لا نعرف أي النسختين أدق، ولا يثبت السجل من أنشأ كل ليبل أو كيف نشأ الاختلاف.',size=22)
    p('المعالجة السابقة: عزل كل أعضاء المجموعات الأربع، دون دمج الصناديق أو اختيار نسخة بتنبؤ نموذج. لماذا؟ تجنب ترجيح تعليق غير محكّم أو تكرار الصورة بإشراف متناقض. الاسترجاع الآمن يتطلب تحكيمًا بشريًا وإصدارًا جديدًا.',size=22)
    source('Clean_v1/excluded_samples.csv؛ source_annotation وduplicate_group؛ تطابق RGB في المثال أُعيد فحصه حاليًا.')

    title(14,'تعليق غير صالح — ارتفاع الصندوق صفر')
    p('المشكلة: الصورة val/00991.jpg تحتوي صفًا بارتفاع 0.000000 وصفًا آخر صالحًا. الصندوق الأول مساحته صفر ولا يصلح كصندوق تدريب.',size=22)
    pic('11_zero_height.jpg')
    p('شكل 11 — الأحمر يوضح صف الارتفاع الصفري كخط، والأخضر يوضح الصف الصالح. يمينًا تكبير المنطقة؛ هذه علامات من الليبل وليست صناديق مصححة.',size=20,color='647B88')
    for line in label.splitlines():p(line,size=20,align='left',rtl=False)
    p('السبب المباشر: h = 0 في الليبل. أصل الخطأ، سواء تعليم أو تحويل أو تقريب، غير موثق. لا تكفي الصورة وحدها لتحديد صندوق بديل موثوق.',size=22)
    p('المعالجة السابقة: استبعاد الصورة والليبل كاملين انتظارًا للتحكيم. لماذا لم نحذف الصف فقط؟ قد يترك حذف الصف هدفًا ظاهرًا دون تعليم، فيتحول إلى إشراف سلبي خاطئ. لم نخمن صندوقًا ولم نغير الهدف الصالح.',size=22)
    source('Clean_v1/excluded_samples.csv؛ provenance/annotation_normalization_errors.csv؛ المصدر labels/val/00991.txt.')

    title(15,'تشابه المشاهد بين أقسام قديمة مختلفة')
    p('المشكلة: إزالة النسخ المطابقة لا تمنع وجود صور مختلفة للمشهد نفسه في Train وValidation أوTest؛ قد يتعلم النموذج سياق الخلفية فيؤثر ذلك على تقدير التعميم.',size=22)
    pic('12_scene_leakage.jpg')
    p('شكل 12 — Train/04923 وValidation/02462 وفق الأسماء القديمة. سجل المراجعة يصف تطابق الأفق المنخفض المميز وهياكل الأسطح وخط الأشجار. أصبحا في Train داخل SCENE_01254 وBLOCK_00002.',size=20,color='647B88')
    p('السبب المدعوم بالسجلات: التقسيم القديم لم يفرض فصل مجموعات المشاهد؛ Clean_v1 حافظ على الأقسام القديمة. لا يثبت ذلك أن الصورتين من فيديو أصلي محدد، ولا توجد هويات فيديو مستعادة.',size=22)
    p('المعالجة السابقة: استرجاع مرشحين على كامل الصور بسمات DINOv2 وpHash ← إخفاء مناطق الدرون للتحقق من الخلفية ← مطابقة RootSIFT وهندسة RANSAC ← مراجعات بصرية ← تجميع الروابط المؤكدة ← تخصيص المشاهد والكتل كاملة لقسم واحد.',size=22)
    source('Clean_v2/scene_visual_reviews.json: هذا الزوج CONFIRMED_SAME_SCENE؛ dataset_manifest.csv: التخصيص النهائي؛ METHODOLOGY.md.')

    title(16,'الحالات الغامضة والخلفيات قليلة السمات')
    pic('13_uncertain_context.jpg')
    p('شكل 13 — test/00310 وtest/00311 في المصدر. سجل المراجعة يسجل عرض Failtest مشتركًا، لكن الإطار الثاني لا يقدم معالم كافية لتأكيد المشهد الفيزيائي. احتُفظ بمجموعتي مشاهد مختلفتين SCENE_00240 وSCENE_00241، مع جمعهما داخل BLOCK_00002 في Train.',size=21)
    p('سبب صعوبة التحقق: السماء أو الخلفية ضعيفة النسيج تقلل نقاط المطابقة؛ وقد تعكس الشعارات المتكررة مصدرًا مشتركًا دون إثبات مشهد واحد. 329 صورة أقل من 16 نقطة خلفية، بينها 172 صورة بلا نقاط.',size=22)
    p('المعالجة السابقة: جمع الحالات غير المحسومة عالية الخطورة في كتلة تخصيص تحفظية؛ الصور الـ329 تُجمع ضمن عائلة تخصيص تحفظية. تشابه DINO ≥ 0.85 يفرض co-allocation وفق السياسة الخامسة، ولا يؤكد وحده المشهد نفسه.',size=22)
    p('لماذا؟ تقليل احتمال عبور علاقة غير محسومة بين الأقسام. المقابل هو تجميع واسع وتوازن أقل؛ أكبر كتلة 6,299 صورة. تشابه الشعار أو تجاور الأسماء لا يُعد هوية فيديو.',size=22)
    source('targeted_visual_extension.json؛ visual_families6_review.json؛ background_feature_summary.json؛ scene_grouping_policy.json؛ dataset_manifest.csv.')

    title(17,'لماذا لم نكتفِ بأول PASS آلي؟')
    pic('14_gate_history.png')
    p('شكل 14 — مجموع الروابط المؤكدة العابرة للأقسام في تقارير الفشل المحفوظة، ثم التقرير النهائي. ازدياد الرقم في بعض الجولات يرتبط بتوسع البحث والمراجعة؛ القوائم والسياسات تغيرت، فلا يمثل الرسم تجربة مقارنة على مجموعة مرشحين ثابتة.',size=20)
    table(['التقرير','الروابط المؤكدة العابرة','مخاطر غير محسومة عابرة'],[[f'FAIL {i+1}',confirmed[i],a['checks']['unresolved_high_risk_cross_split']] for i,a in enumerate(audits)]+[['Final PASS',0,0]])
    p('المشكلة: أول PASS آلي استُبدل بفشل بصري موثق. لم تكن إزالة التكرار أو حدود القرار الأولى كافية لاكتشاف كل الروابط التي ظهرت بالمراجعة اللاحقة.',size=22)
    p('المعالجة: توسيع k من 60 إلى120 لكل واصف، وفحص pHash، وإكمال تغطية المرشحين، وإضافة زوج مستهدف خارج الاسترجاع العام، ثم إصلاح التخصيص بكتل كاملة وإعادة Gate قبل النشر. الإجمالي النهائي 1,106,071 زوجًا، وحدود التقسيم النهائي فُحص منها 282,293 زوجًا مباشرة.',size=21)
    source('release_audit_history.json؛ تقارير *_FAILED.json؛ fresh_retrieval_report.json؛ targeted_visual_extension.json؛ gate2_report.json.')

    title(18,'تنظيف الملفات وحماية الإصدار وما بقي مفتوحًا')
    p('ملفات غير صالحة كعينات',size=26,bold=True)
    table(['المسار القديم','القرار وسببه'],[['labels/train/00579_backup_old.txt','ملف ليبل احتياطي/يتيم؛ لم يُنسخ كعينة'],['labels/train.cache','كاش مشتق؛ لم يُنسخ'],['labels/val.cache','كاش مشتق؛ لم يُنسخ'],['labels/test.cache','كاش مشتق؛ لم يُنسخ']])
    p('المعالجة السابقة: نسخ أزواج الصور والليبلز المؤهلة فقط، وإبقاء runs وvisual_check خارج الإصدار. السبب: فصل مصدر التدريب عن بقايا التشغيل والنسخ الاحتياطية والكاش؛ هذه الملفات لا تزيد عدد العينات.',size=21)
    p('حماية النتيجة: أسماء فريدة ببادئة القسم القديم، وخرائط نقل، وSHA-256 وMANIFEST وFREEZE، ووسم الملفات للقراءة فقط. الأسماء المتكررة بين الأقسام ليست دليل فساد، لكن البادئات تمنع تعارضها عند إعادة التخصيص. ملفات المصدر والتجارب التاريخية لم تُحذف.',size=21)
    p('قيود لم تُحل بمجرد التنظيف',size=26,bold=True,color='176B87')
    table(['القيد','ما فعله العمل السابق وما يلزم لاحقًا'],[['صورة سلبية واحدة فقط','أُبقيت؛ لم تُضف سلبيات مصطنعة. يلزم اختبار خلفيات خالية مستقل'],['هويات الفيديوهات الأصلية مفقودة','لم تُستعد؛ لا ندّعي sequence-disjoint'],['دقة الصناديق بصريًا','موروثة من Clean_v1؛ لم يُحكّم كل صندوق بشريًا'],['احتمال رؤية checkpoints قديمة لصور Test','التنظيف لا يزيل تاريخ التعلم؛ تبدأ التجارب النهائية من أوزان عامة'],['انحياز الأحجام والكتلة الكبيرة','وُثّق بدل حذف الصور لتحسين النسب؛ تقرير أداء حسب الحجم']])
    source('Clean_v1/excluded_artifacts.csv؛ DATASET_CARD.md للإصدارين؛ Clean_v2/FREEZE.json؛ configs/datasets/DUT_Anti_UAV_Clean_v2.release.json.')
    p('تتبّع الأدلة: word_assets/issue_evidence.json يحفظ هويات الأمثلة وقيم الليبلز وقراراتها. نُسخت سجلات الاستبعاد والتكرار والتطبيع بجانب الصور لتسهيل الرجوع إليها.',size=19)
