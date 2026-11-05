# Ai_photo_editor

محرر صور ذكي لملفات **RAW** (خصوصًا **Sony ARW**) وصور JPEG/PNG/TIFF.
يكتشف الأشخاص ويفصلهم عن الخلفية، يصحح **توازن اللون الأبيض** ليكون طبيعيًا،
يضبط الإضاءة والألوان، ويطبق **إعدادات جاهزة (Presets)** يغيّر كل منها جميع
إعدادات الصورة دفعة واحدة.

An AI photo editor for RAW files (Sony ARW and every format LibRaw reads) and
regular images: people detection & subject/background separation, natural auto
white balance, exposure, colour, and one-click presets that set every parameter.

## تثبيت المهارة في Claude / Install the skill

**في claude.ai أو Claude Desktop (المحادثة العادية):**

1. حمّل الملف [`dist/raw-photo-editor.zip`](dist/raw-photo-editor.zip) من هذا الفرع.
2. في Claude: **Settings ← Capabilities**، فعّل **Code execution and file creation**.
3. في نفس الصفحة، قسم **Skills** ← **Upload skill** ← اختر الملف المضغوط.
4. افتح محادثة جديدة وارفع صورة ARW واكتب مثلًا: "عدّل الصورة تجميل فقط وأزل الحبوب".

**في Claude Code:** المهارة موجودة في `.claude/skills/raw-photo-editor/SKILL.md`
(مجلد مخفي يبدأ بنقطة) وتعمل تلقائيًا عند فتح هذا المستودع.

لإعادة بناء الملف المضغوط بعد أي تعديل: `python tools/build_skill.py`

## الصيغ المدعومة / Supported formats

Sony `ARW` `SR2` `SRF` · Canon `CR2` `CR3` `CRW` · Nikon `NEF` `NRW` · Fujifilm `RAF` ·
Olympus/OM `ORF` · Panasonic `RW2` · Pentax `PEF` · Adobe `DNG` (iPhone ProRAW, Pixel, DJI, Leica) ·
Hasselblad `3FR` `FFF` · Phase One `IIQ` · Samsung `SRW` · Sigma `X3F` … + `JPG` `PNG` `TIFF` `WEBP`.

## التثبيت / Install

```bash
pip install -r requirements.txt
python -m photo_editor models      # تحميل نماذج كشف الأشخاص (~20MB) مرة واحدة
# Linux بدون واجهة: apt-get install -y libegl1 libgles2
```

## الاستخدام / Usage

```bash
# 1) تحليل الصورة: الأشخاص، لون الإضاءة، أخطاء التعريض، الإعداد المقترح
python -m photo_editor analyze DSC01234.ARW --preview-dir output/

# 2) معاينة كل الإعدادات الجاهزة في صورة واحدة للاختيار
python -m photo_editor grid DSC01234.ARW -o output/presets.jpg

# 0) تحويل RAW إلى JPEG للعرض
python -m photo_editor preview DSC01234.ARW --out-dir output/

# تجميل فقط: الصورة كما هي (ألوان وإضاءة وخلفية) مع تعديل الوجه فقط
python -m photo_editor edit DSC01234.ARW --preset beauty_only --set retouch=0.8 --compare

# 3) التعديل بإعداد جاهز (يغير كل الإعدادات) + تعديلات إضافية اختيارية
python -m photo_editor edit DSC01234.ARW --preset natural --out-dir output/ --compare --masks
python -m photo_editor edit DSC01234.ARW --preset portrait_pop --set warmth=8 --set bg_blur=0.6

# 16-bit TIFF، أو دفعة كاملة
python -m photo_editor edit *.ARW --preset natural --tiff --out-dir output/
```

From Python:

```python
from photo_editor import edit
report = edit("DSC01234.ARW", preset="natural", output="out.jpg", bg_blur=0.3, warmth=5)
```

## الإعدادات الجاهزة / Presets

| preset | الاسم | الوصف |
|---|---|---|
| `standard` | تعديل عادي | توازن أبيض، تعريض، تباين، ألوان، حدة - بدون تغيير الخلفية أو الوجه |
| `beauty_only` | تجميل فقط | الصورة كما هي تمامًا، ويُعدَّل الوجه فقط |
| `natural` | طبيعي | ألوان حقيقية، توازن أبيض تلقائي كامل، فصل خفيف للأشخاص عن الخلفية |
| `portrait_pop` | بورتريه بارز | خلفية مموهة وأغمق، وضوح أعلى للشخص، بشرة صحية |
| `soft_portrait` | بورتريه ناعم | تباين منخفض، سواد مرفوع، بشرة دافئة وناعمة |
| `cinematic` | سينمائي | تدرج أزرق مخضر/برتقالي، تباين أعمق، حبيبات |
| `vivid` | حيوي | ألوان غنية للطبيعة والسفر مع بشرة طبيعية |
| `warm_golden` | ذهبي دافئ | دفء الساعة الذهبية |
| `cool_clean` | بارد ونظيف | نقي وبارد قليلًا |
| `moody` | درامي داكن | داكن وهادئ الألوان، الشخص مضاء والمحيط يغمق |
| `matte_film` | فيلم مطفي | طابع فيلم قديم مع حبيبات |
| `bw_classic` | أبيض وأسود | أبيض وأسود غني |

كل إعداد يمكن تعديله بـ `--set key=value`؛ القائمة الكاملة في `photo_editor/presets.py`.

## التعديل العادي / Regular editing

تعديل كلاسيكي بأشرطة مثل Lightroom، بدون فصل الخلفية وبدون تجميل:

```bash
python -m photo_editor adjust DSC01234.ARW --contrast 25 --temperature 5200 --highlights -40 \
    --shadows 30 --vibrance 20 --sharpness 70 --clarity 15 --compare
python -m photo_editor sliders     # قائمة كل الأشرطة ونطاقاتها
```

| الشريط | النطاق |
|---|---|
| `--wb` توازن الأبيض | `auto` / `camera` / `daylight` |
| `--temperature` درجة الحرارة | 2000..12000 كلفن |
| `--warmth` الدفء، `--tint` الصبغة | -100..100 |
| `--exposure` التعريض | -5..5 EV |
| `--contrast` التباين، `--highlights` الإضاءات، `--shadows` الظلال، `--whites` الأبيض، `--blacks` الأسود | -100..100 |
| `--vibrance` الحيوية، `--saturation` التشبع | -100..100 |
| `--sharpness` الحدة | 0..150 |
| `--clarity` الوضوح، `--texture` الملمس | -100..100 |
| `--noise-reduction` إزالة التشويش | 0..100 |
| `--vignette` التظليل، `--grain` الحبيبات | -100..100 / 0..100 |
| `--hsl` تدرج كل لون | `--hsl orange:hue=5,sat=-10,lum=10` |

الأشرطة تعمل أيضًا مع أي إعداد جاهز: `edit --preset portrait_pop --contrast 10 --sharpness 80`.

## تجميل الوجه / Face retouch

`retouch` = القوة العامة (0 إلى 1)، ويمكن ضبط كل خطوة وحدها:

| المعامل | ماذا يفعل |
|---|---|
| `retouch_blemish` | إزالة الحبوب والبثور والبقع الصغيرة مع الحفاظ على ملمس البشرة |
| `retouch_redness` | تخفيف الاحمرار والتهيج وتوحيده مع لون الوجه |
| `retouch_smooth` | توحيد البشرة (فصل الترددات) مع إبقاء المسام |
| `retouch_under_eye` | تفتيح الهالات السوداء وإزالة لونها المزرق |
| `retouch_shine` | تخفيف لمعان البشرة الدهنية |
| `retouch_teeth` | تبييض الأسنان |
| `retouch_eyes` | إزالة احمرار بياض العين |

يعتمد على 478 نقطة لملامح الوجه (MediaPipe) لتحديد العينين والحاجبين والشفاه
والأنف واستثنائها، أو على نموذج YuNet المرفق إن لم تتوفر MediaPipe. ما خارج
بشرة الوجه لا يتغير أبدًا.

## كيف يعمل / How it works

1. **فك RAW** خطيًا بـ LibRaw (16-bit، بدون تصحيح غاما) بتوازن ضوء النهار، فيظهر اللون الحقيقي للإضاءة.
2. **كشف الأشخاص** بنماذج MediaPipe: كاشف الوجوه + DeepLabV3 (شخص) + نموذج متعدد الفئات (شعر/بشرة/ملابس)
   يُشغَّل أيضًا على قص حول كل وجه، ثم تنعيم الحواف بـ guided filter.
3. **توازن اللون الأبيض**: البحث بين مصادر الإضاءة الحقيقية (منحنى بلانك) عن الإضاءة التي تجعل أكبر عدد من
   أسطح **الخلفية** رمادية محايدة، مع grey-edge وshades-of-grey، ثم التحقق من لون البشرة.
   الأشخاص مستبعدون من التقدير حتى لا تؤثر البشرة والملابس عليه.
4. **التعريض**: قياس المشهد والأشخاص مع حماية الإضاءات العالية، واستعادة الإضاءات/الظلال محليًا.
5. **الفصل عن الخلفية**: تمويه عدسي للخلفية في الضوء الخطي بدون هالات، وخفض إضاءتها وتشبعها مقابل إبراز الشخص.
6. **النغمة واللون** في فضاء OKLab: منحنى فيلمي، تباين، حيوية مع حماية البشرة، HSL، تدرج لوني، أبيض وأسود.
7. **التفاصيل**: إزالة تشويش حسب ISO، وضوح، حدة، حبيبات.

## Claude skill

`.claude/skills/raw-photo-editor/SKILL.md` teaches Claude Code the workflow:
analyse → show findings → offer presets → edit → visually verify → deliver.

## Tests

```bash
python -m pytest -q tests
```
