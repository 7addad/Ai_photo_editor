# Ai_photo_editor

محرر صور ذكي لملفات **RAW** (خصوصًا **Sony ARW**) وصور JPEG/PNG/TIFF.
يكتشف الأشخاص ويفصلهم عن الخلفية، يصحح **توازن اللون الأبيض** ليكون طبيعيًا،
يضبط الإضاءة والألوان، ويطبق **إعدادات جاهزة (Presets)** يغيّر كل منها جميع
إعدادات الصورة دفعة واحدة.

An AI photo editor for RAW files (Sony ARW and every format LibRaw reads) and
regular images: people detection & subject/background separation, natural auto
white balance, exposure, colour, and one-click presets that set every parameter.

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
