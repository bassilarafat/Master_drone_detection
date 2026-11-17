"""Create an Arabic DOCX with real annotated samples, using standard OOXML."""
import csv
import json
import random
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '02_datasets/DUT_Anti_UAV_Clean_v2'
OUT = ROOT / '06_results/dataset_review_clean_v2_20261114'
ASSETS = OUT / 'word_assets'
ASSETS.mkdir(parents=True, exist_ok=True)
STATS = json.loads((OUT / 'review_statistics.json').read_text(encoding='utf-8'))
ROWS = list(csv.DictReader((DATA / 'dataset_manifest.csv').open(encoding='utf-8-sig')))
SPLITS = ['train', 'val', 'test']
COLORS = ['#176B87', '#54A89B', '#E4A34E', '#8A78AD']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white', 'axes.titleweight': 'bold'})

def save(fig, name):
    fig.savefig(ASSETS / name, dpi=190, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return name

fig, ax = plt.subplots(figsize=(8.4, 3.5))
vals = [STATS['splits'][s]['images'] for s in SPLITS]
bars = ax.bar(['Train', 'Validation', 'Test'], vals, color=COLORS[:3], width=.55)
ax.set(ylabel='Images', title='Dataset split | 9,451 images', ylim=(0, 7600))
for b, n in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, n+130, f'{n:,}\n{n/9451:.2%}', ha='center')
save(fig, '01_splits.png')

sizes = ['tiny', 'small', 'medium', 'large']
totals = [sum(STATS['splits'][s][k] for s in SPLITS) for k in sizes]
fig, ax = plt.subplots(figsize=(8.4, 3.6))
bars = ax.bar([s.title() for s in sizes], totals, color=COLORS, width=.6)
ax.set(ylabel='Annotated objects', title='Object size distribution | 9,561 objects', ylim=(0, 5700))
for b, n in zip(bars, totals):
    ax.text(b.get_x()+b.get_width()/2, n+90, f'{n:,}\n{n/9561:.2%}', ha='center')
save(fig, '02_sizes.png')

fig, ax = plt.subplots(figsize=(8.4, 3.8))
left = np.zeros(3)
for k, color in zip(sizes, COLORS):
    v = np.array([STATS['splits'][s][k]/STATS['splits'][s]['objects']*100 for s in SPLITS])
    ax.barh(['Train', 'Validation', 'Test'], v, left=left, color=color, label=k.title(), height=.5)
    for i, (a, b) in enumerate(zip(left, v)):
        ax.text(a+b/2, i, f'{b:.1f}%', va='center', ha='center', fontsize=10, color='white' if k in ['tiny','large'] else '#162D3E')
    left += v
ax.set(xlim=(0,100), xlabel='Share of objects within each split (%)', title='Size mix differs across splits')
ax.invert_yaxis()
ax.legend(loc='upper center', bbox_to_anchor=(.5,-.20), ncol=4, frameon=False)
save(fig, '03_size_mix.png')

fig, ax = plt.subplots(figsize=(8.4, 3.3))
v = [7397,1782,272]
bars = ax.barh(['1920 x 1080','1280 x 720','Other resolutions'],v,color=COLORS[:3])
ax.set(xlim=(0,9000), xlabel='Images', title='Image resolutions | manifest metadata')
ax.invert_yaxis()
for b,n in zip(bars,v):
    ax.text(n+100,b.get_y()+b.get_height()/2,f'{n:,}  ({n/9451:.2%})',va='center')
save(fig,'04_resolutions.png')

matrix = np.array([[STATS['old_to_new_splits'][a+' -> '+b] for b in SPLITS] for a in SPLITS])
fig, ax = plt.subplots(figsize=(7,4.6))
ax.imshow(matrix,cmap='Blues')
ax.set(xticks=range(3),yticks=range(3),xticklabels=['Train','Validation','Test'],yticklabels=['Train','Validation','Test'],xlabel='New split',ylabel='Old split',title='Image reassignment | 5,205 moved (55.07%)')
for i in range(3):
    for j in range(3):
        ax.text(j,i,f'{matrix[i,j]:,}',ha='center',va='center',fontsize=16,color='white' if matrix[i,j]>2000 else '#162D3E')
save(fig,'05_reassignment.png')

font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 23)
smallfont = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 19)
selected = []

def annotated(r):
    im = Image.open(DATA/r['new_relative_image']).convert('RGB')
    boxes = []
    for line in (DATA/r['new_relative_label']).read_text().splitlines():
        if not line.strip(): continue
        _,x,y,w,h = map(float,line.split())
        boxes.append(((x-w/2)*im.width,(y-h/2)*im.height,(x+w/2)*im.width,(y+h/2)*im.height))
    d = ImageDraw.Draw(im)
    for box in boxes: d.rectangle(box,outline='#00FF77',width=max(3,im.width//450))
    return im,boxes

def fit(canvas, im, rect):
    x,y,w,h = rect
    im = im.copy()
    im.thumbnail((w,h),Image.Resampling.LANCZOS)
    canvas.paste(im,(x+(w-im.width)//2,y+(h-im.height)//2))

rng = random.Random(17)
sheet = Image.new('RGB',(1500,1490),'#EFF4F8')
d = ImageDraw.Draw(sheet)
for i,s in enumerate(SPLITS):
    pool = sorted([r for r in ROWS if r['new_split']==s and int(r['objects'])>0], key=lambda r:r['new_filename'])
    for j,r in enumerate(rng.sample(pool,2)):
        selected.append({'purpose':'split_gallery', 'image':r['new_relative_image']})
        im,_ = annotated(r)
        x,y = j*750,i*490
        d.text((x+20,y+15),f'{s.upper()} | {r["new_filename"]}',font=font,fill='#16364A')
        fit(sheet,im,(x+15,y+55,720,410))
sheet.save(ASSETS/'06_samples.jpg',quality=93)

sheet = Image.new('RGB',(1500,1660),'#EFF4F8')
d = ImageDraw.Draw(sheet)
used = set()
for i,k in enumerate(sizes):
    pool = sorted([r for r in ROWS if int(r[k])>0 and int(r['objects'])==1 and r['scene_group_id'] not in used], key=lambda r:r['new_filename'])
    r = pool[len(pool)//2]
    used.add(r['scene_group_id'])
    selected.append({'purpose':k, 'image':r['new_relative_image']})
    im,boxes = annotated(r)
    x1,y1,x2,y2 = boxes[0]
    cx,cy = (x1+x2)/2,(y1+y2)/2
    extent = max(80, max(x2-x1,y2-y1)*2.8)
    crop = im.crop((max(0,int(cx-extent/2)),max(0,int(cy-extent/2)),min(im.width,int(cx+extent/2)),min(im.height,int(cy+extent/2))))
    y=i*410
    d.text((20,y+12),f'{k.upper()} | {r["new_split"]} | {r["new_filename"]}',font=font,fill='#16364A')
    fit(sheet,im,(15,y+55,1030,340))
    # Enlarge only the illustrative target crop; counts always use original pixels.
    crop.thumbnail((400,320))
    scale=min(400/crop.width,320/crop.height)
    crop=crop.resize((int(crop.width*scale),int(crop.height*scale)),Image.Resampling.NEAREST)
    fit(sheet,crop,(1070,y+75,400,320))
    d.text((1100,y+48),'Enlarged target crop',font=smallfont,fill='#16364A')
sheet.save(ASSETS/'07_size_examples.jpg',quality=94)
(ASSETS/'selected_samples.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf-8')

# Minimal standards-based Word package. Arabic text is stored in logical order;
# Word performs shaping and right-to-left layout through bidi and rtl properties.
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
parts=[]
relationships=[]
media=[]

def para(text='',size=23,bold=False,color='243C4B',align='right',rtl=True):
    ppr=f'<w:pPr><w:jc w:val="{align}"/>'+('<w:bidi/>' if rtl else '')+'<w:spacing w:after="110" w:line="300" w:lineRule="auto"/></w:pPr>'
    rpr=f'<w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/><w:sz w:val="{size}"/><w:szCs w:val="{size}"/><w:color w:val="{color}"/>'+('<w:b/><w:bCs/>' if bold else '')+('<w:rtl/>' if rtl else '')+'<w:lang w:val="ar-EG" w:bidi="ar-EG"/></w:rPr>'
    return '<w:p>'+ppr+'<w:r>'+rpr+'<w:t xml:space="preserve">'+escape(text)+'</w:t></w:r></w:p>'

def p(text='',**kwargs): parts.append(para(text,**kwargs))
def title(n,text):
    if parts: parts.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
    p(f'{n:02d}  /  تقرير مراجعة الداتاسيت',size=20,color='176B87')
    p(text,size=34,bold=True,color='16364A')
def table(headers,rows):
    content='<w:tbl><w:tblPr><w:tblW w:w="9900" w:type="dxa"/><w:bidiVisual/><w:tblBorders>'+''.join(f'<w:{k} w:val="single" w:sz="4" w:color="D8E3E9"/>' for k in ['top','left','bottom','right','insideH','insideV'])+'</w:tblBorders><w:tblCellMar><w:top w:w="65" w:type="dxa"/><w:bottom w:w="65" w:type="dxa"/><w:left w:w="90" w:type="dxa"/><w:right w:w="90" w:type="dxa"/></w:tblCellMar></w:tblPr>'
    for i,row in enumerate([headers]+rows):
        content+='<w:tr><w:trPr><w:cantSplit/>'+('<w:tblHeader/>' if i==0 else '')+'</w:trPr>'
        for value in row:
            shade='176B87' if i==0 else ('EFF4F8' if i%2 else 'FFFFFF')
            content+=f'<w:tc><w:tcPr><w:tcW w:w="{9900//len(headers)}" w:type="dxa"/><w:shd w:fill="{shade}"/></w:tcPr>'+para(str(value),size=20,bold=i==0,color='FFFFFF' if i==0 else '243C4B')+'</w:tc>'
        content+='</w:tr>'
    parts.append(content+'</w:tbl>')
    p('',size=6)

def pic(name,width=6.5):
    idx=len(media)+1
    media.append((name,ASSETS/name))
    rid=f'rIdImage{idx}'
    relationships.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{name}"/>')
    with Image.open(ASSETS/name) as im: ratio=im.height/im.width
    cx=int(width*914400);cy=int(cx*ratio)
    parts.append(f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{idx}" name="{escape(name)}" descr="Dataset review figure"/><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="{idx}" name="{escape(name)}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')

title(1,'توزيع البيانات — نظرة عامة')
p('DUT_Anti_UAV_Clean_v2',size=30,bold=True,align='center',rtl=False)
p('14 نوفمبر 2026  |  مراجعة النسخة المحلية',color='647B88',align='center')
p('9,451 صورة   •   9,561 جسمًا معلّمًا   •   فئة واحدة: UAV',size=27,bold=True)
p('داتاسيت لاكتشاف الدرون بصناديق إحاطة بصيغة YOLO. أُعد هذا التقرير من نتائج المراجعة المحفوظة وملفات الإصدار، مع صور فعلية ورسوم بيانية مشتقة من الأعداد المسجلة.')
table(['القسم','الصور','نسبة الصور','الأجسام'],[[s,str(STATS['splits'][s]['images']),f'{STATS["splits"][s]["images"]/9451:.2%}',str(STATS['splits'][s]['objects'])] for s in SPLITS])
pic('01_splits.png')
p('شكل 1 — النسب محسوبة من إجمالي 9,451 صورة.',size=19,color='647B88')
p('الملفات المفحوصة متطابقة مع بصمات الإصدار ضمن نطاق المراجعة. لم تُعدّل الصور أو الليبلز أو التقسيم.',bold=True,color='176B87')

title(2,'أحجام الأجسام وترتيب انتشارها')
p('83.96% من الأجسام متناهية الصغر أو صغيرة؛ لذلك يُستحسن عرض أداء النموذج حسب الحجم إلى جانب الأداء الإجمالي.',size=25,bold=True)
pic('02_sizes.png')
table(['الحجم','العدد','نسبة الأجسام','مساحة الصندوق بالبكسل²'],[['Tiny',4636,'48.49%','أقل من 1,024'],['Small',3391,'35.47%','1,024 إلى أقل من 9,216'],['Medium',1152,'12.05%','9,216 إلى أقل من 65,536'],['Large',382,'4.00%','65,536 فأكثر']])
p('شكل 2 — مقام النسب هو 9,561 جسمًا. المساحة تُحسب في الصورة الأصلية قبل أي تغيير لحجم الإدخال. هذه حدود المشروع وليست تعريفات COCO AP-small؛ قد يصبح مجموع النسب 100.01% بسبب التقريب.',size=21)

title(3,'اختلاف توزيع الأحجام واستقلال المجموعات')
pic('03_size_mix.png')
p('شكل 3 — كل شريط يساوي 100% من أجسام القسم نفسه. نسبة Tiny في التدريب 52.94% مقابل 41.90% في الاختبار، بفارق 11.04 نقطة مئوية.')
table(['القسم','مجموعات المشاهد','كتل التخصيص'],[['train',1064,1],['val',321,192],['test',186,120],['الإجمالي',1571,313]])
p('التدريب كله داخل كتلة تخصيص تحفظية واحدة تضم 6,299 صورة و1,064 مجموعة مشاهد. هذا لا يعني أن التدريب من مشهد واحد أو فيديو واحد؛ قد تربط الكتلة مشاهد مختلفة لتقليل مخاطر التسريب.')
p('الأهداف الأصلية كانت 54.95% تدريب / 27.48% تحقق / 17.57% اختبار. الحفاظ على الكتل كاملة أخذ أولوية على تحقيق هذه النسب، فنتج التقسيم الفعلي 66.65% / 19.33% / 14.02%.')

title(4,'تنظيم الملفات ودقات الصور')
pic('04_resolutions.png')
p('شكل 4 — الدقات من ملف manifest. إجمالي حجم الصور والليبلز نحو 1.34 GB، بخلاف أدلة التدقيق.',size=21)
table(['الملف أو المسار','الدور'],[['images/{train,val,test}','الصور حسب القسم'],['labels/{train,val,test}','ملفات YOLO المقابلة للصور'],['data.yaml','تعريف المسارات والفئة 0: UAV'],['dataset_manifest.csv','الأبعاد والأعداد والبصمات وربط الأسماء'],['dataset_changes.csv','سجل إعادة التسمية والنقل'],['scene_groups.csv / split_assignment.csv','مجموعات المشاهد والتقسيم'],['gate2_report.json','نتيجة التدقيق التاريخية'],['MANIFEST.csv / FREEZE.json','هوية الإصدار المجمد']])
p('مسار الجذر داخل data.yaml مطلق ويخص هذا الجهاز. عند النقل لجهاز آخر يُستخدم إعداد تشغيل خارجي مناسب للمسار.',size=21)

title(5,'صور فعلية من الأقسام الثلاثة')
p('صورتان من كل قسم، مختارتان عشوائيًا ببذرة 17 من الصور ذات الأجسام المعلّمة. الصناديق الخضراء من الليبلز الأصلية، وليست تنبؤات نموذج.',size=21)
pic('06_samples.jpg',6.35)
p('شكل 5 — أعلى: التدريب؛ الوسط: التحقق؛ أسفل: الاختبار. اسم كل ملف مكتوب أعلى صورته. العينات توضيحية ولا تمثل تحكيمًا بصريًا شاملًا.',size=20,color='647B88')

title(6,'أمثلة أحجام الدرون وصور مكبرة')
p('مثال واحد لكل فئة حجم. الصورة الكاملة يسارًا والجزء المحيط بالهدف مكبر يمينًا. التكبير للتوضيح فقط ولا يدخل في حساب فئة الحجم.',size=21)
pic('07_size_examples.jpg',6.15)
p('شكل 6 — الترتيب من أعلى: Tiny ثم Small ثم Medium ثم Large. الاختيار حتمي من صور ذات جسم واحد ومجموعات مشاهد مختلفة؛ الأسماء والأقسام موضحة داخل اللوحة.',size=20,color='647B88')

title(7,'نتائج مراجعة السلامة وحدودها')
table(['الفحص الحالي','النتيجة'],[['الصور وملفات الليبلز','9,451 صورة + 9,451 ليبل'],['الملفات المفقودة أو الزائدة في المسارات المفحوصة','0'],['اختلاف SHA-256 للصور والليبلز','0'],['أخطاء صيغة الليبل والفئة وحدود الصناديق','0'],['اختلاف أعداد الأجسام والأحجام عن manifest','0'],['مجموعات صور متطابقة بايتًا','0'],['مجموعات مشاهد أو كتل تعبر بين الأقسام','0'],['هويات manifest والتقسيم وملف التجميد','متطابقة مع المراجع المحفوظة']])
p('توزيع عدد الأجسام في الصورة',size=26,bold=True)
table(['الأجسام في الصورة','الصور','النسبة'],[['صفر',1,'0.01%'],['جسم واحد',9376,'99.21%'],['أكثر من جسم',74,'0.78%']])
p('مراجعة السلامة قرأت كل الليبلز وحسبت بصمات كل الصور والليبلز. أبعاد جميع الصور مأخوذة من manifest؛ فُكّت الصور المختارة فقط لإنشاء اللوحات. لم تُعَد عمليات استرجاع المشاهد أو التحقق الهندسي، ولم تُعَد بصمات جميع ملفات الأدلة المساعدة.',size=21)
p('سلامة الصيغة لا تثبت دقة موضع كل صندوق. لا يتضمن هذا العمل تحكيمًا بصريًا لكل صورة.',size=23,bold=True,color='176B87')

title(8,'إعادة التخصيص وتفسير نتائج التدقيق')
pic('05_reassignment.png',5.8)
p('شكل 7 — الصف يمثل القسم القديم والعمود يمثل القسم الجديد. الأرقام أعداد صور. انتقلت 5,205 صور، أي 55.07%، إلى قسم مختلف. أضيفت بادئة القسم القديم للأسماء لمنع التعارض.',size=21)
p('Gate 2 التاريخية: PASS',size=26,bold=True,color='176B87')
p('تسجل الأدلة فحص 282,293 زوجًا مباشرة على حدود الأقسام، مع صفر للروابط المؤكدة العابرة للمشهد نفسه والحالات عالية الخطورة غير المحسومة العابرة. المراجعة الحالية تحققت من ارتباط manifest والتقسيم بتقرير Gate 2، ولم تُعِد حساب هذه الأزواج.')
p('الاستنتاج المسموح: مجموعات المشاهد منفصلة وفق البروتوكول المعلن. استقلال الفيديوهات والتسلسلات الأصلية غير مثبت؛ هويات التسجيلات والتوقيتات وأرقام الإطارات الأصلية غير معروفة.',bold=True)

title(9,'الأولويات العملية والمصادر')
for heading,body in [
    ('1. استقلال التجربة النهائية','بدء التدريب من أوزان عامة مسبقة التدريب؛ checkpoints المشروع التاريخية قد تكون شاهدت صورًا انتقلت إلى Test.'),
    ('2. تقييم الإنذارات الكاذبة','صورة سلبية واحدة فقط في التدريب، وصفر في التحقق والاختبار. يلزم تقييم إضافي على مشاهد خالية من الدرون لقياس الإنذارات الكاذبة بصورة أشمل.'),
    ('3. الأداء حسب الحجم','إظهار مقاييس Tiny وSmall وMedium وLarge، مع توضيح حدود المساحة واختلاف توزيع الأحجام بين الأقسام.'),
    ('4. الحفاظ على الإصدار المجمد','أي تعديل في الليبلز أو إعادة تقسيم لاحقة يُوثّق في إصدار جديد وفق سياسة الداتاسيت الحالية.')]:
    p(heading,size=25,bold=True,color='176B87');p(body,size=22)
p('المصادر وإمكانية إعادة الإنتاج',size=27,bold=True)
for text in ['docs/dataset_clean_v2_review_ar_20261114.md','docs/dataset_clean_v2.md','configs/datasets/DUT_Anti_UAV_Clean_v2.release.json','06_results/dataset_review_clean_v2_20261114/review_statistics.json','scripts/review_clean_v2_dataset.py','scripts/build_clean_v2_word_report.py']:
    p(text,size=18,align='left',rtl=False)
p('مصادر داخل مجلد الداتاسيت: dataset_manifest.csv، allocation_report.json، gate2_report.json، DATASET_CARD.md، FREEZE.json. مصدر التقرير هو النسخة المحلية؛ لم يُثبت التطابق مع الفرع البعيد الحالي.',size=20)
p('الرسوم صور عالية الدقة مدمجة داخل Word، والجداول والنصوص قابلة للتحرير. تفاصيل الصور المختارة محفوظة في word_assets/selected_samples.json.',size=20)

from build_clean_v2_issue_appendix import append as append_issue_history
append_issue_history(globals())

sect='<w:sectPr><w:footerReference w:type="default" r:id="rIdFooter"/><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="780" w:right="900" w:bottom="780" w:left="900" w:header="300" w:footer="300"/></w:sectPr>'
document=f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{W}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:body>'+''.join(parts)+sect+'</w:body></w:document>'
relationships.append('<Relationship Id="rIdFooter" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>')
footer=f'<w:ftr xmlns:w="{W}">'+para('DUT_Anti_UAV_Clean_v2  |  14 November 2026',size=16,align='center',rtl=False)+'<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:fldSimple w:instr="PAGE"/></w:p></w:ftr>'
target=OUT/'DUT_Anti_UAV_Clean_v2_Report_AR.docx'
previous=OUT/'DUT_Anti_UAV_Clean_v2_Report_AR.before_issue_history.docx'
if target.exists() and not previous.exists():
    import shutil
    shutil.copyfile(target,previous)
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Default Extension="jpg" ContentType="image/jpeg"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>')
    z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
    z.writestr('word/document.xml',document)
    z.writestr('word/_rels/document.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(relationships)+'</Relationships>')
    z.writestr('word/footer1.xml',footer)
    for name,path in media:z.write(path,'word/media/'+name)
with zipfile.ZipFile(target) as z:
    import xml.etree.ElementTree as ET
    assert z.testzip() is None
    for name in z.namelist():
        if name.endswith(('.xml','.rels')):ET.fromstring(z.read(name))
print(target)
print(f'Embedded figures: {len(media)}; selected source images: {len(selected)}')
