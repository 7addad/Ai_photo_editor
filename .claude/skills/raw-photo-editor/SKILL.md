---
name: raw-photo-editor
description: Read and edit RAW photos (Sony ARW, CR2/CR3, NEF, RAF, DNG, ORF, RW2 and all LibRaw formats) plus JPEG/PNG/TIFF with every Lightroom setting (basic, tone curve, HSL, colour grading, detail, lens, transform, crop, effects, calibration, sky/subject/background masks) and face retouching. The user picks a category (wedding, portrait, landscape, sunset, sea, night, street, food, product, cinematic, moody, vintage film, B&W, HDR ...) and the adaptive engine tunes all settings to that specific photo's light, colours, sky, people and content; exports Lightroom XMP presets. Also beauty-only and regular slider editing. Use whenever the user uploads or names a photo and wants it viewed, edited, styled, retouched or converted to a look (Arabic too - تعديل صورة، فئة، ستايل، لايت روم، تعديل عادي، صيغة ARW، توازن الأبيض، عزل الشخص، إزالة الحبوب، تجميل الوجه).
---

# RAW photo editor - محرر صور RAW وتجميل الوجه

The engine is the Python package `photo_editor`. It is located either
**next to this SKILL.md** (skill uploaded to claude.ai as a zip) or at the
**repository root** (Claude Code in the Ai_photo_editor repo). Set `PE` to the
folder that contains the `photo_editor/` directory and run everything as:

```bash
PE="<folder containing photo_editor/>"     # e.g. the folder of this SKILL.md, or the repo root
cd "$PE" && python -m photo_editor <command> ...
```

Use absolute paths for input and output files. On claude.ai, uploaded files are
usually in `/mnt/user-data/uploads/`; save results in `/mnt/user-data/outputs/`
so the user can download them.

## 0. Setup (once per session)

```bash
pip install rawpy numpy opencv-python-headless pillow exifread   # required
pip install mediapipe                                              # recommended (better people/face masks)
```

Add `--break-system-packages` if pip refuses. If MediaPipe fails to load
(`libEGL.so.1`), try `apt-get install -y libegl1 libgles2`; if that is not
possible just continue - the editor falls back to OpenCV automatically (face
retouch still works through the bundled YuNet face model, people masks are
coarser). The report field `backend` shows which was used.

## 1. You cannot see RAW files directly - convert first

The Read/image tools cannot display `.ARW/.NEF/.CR3/...`. Always make a JPEG
and look at that:

```bash
python -m photo_editor preview /path/DSC01234.ARW --out-dir /tmp/pe
# -> /tmp/pe/DSC01234_view.jpg  (camera's embedded preview, or a quick render)
```

## 2. Main workflow - categories (الفئات) + adaptive settings

The user chooses **where the photo should go** (a category); the engine
measures **this** photo (light colour, exposure, contrast, palette, sky,
haze, people/faces, lines/tilt, ISO noise) and computes every Lightroom
setting for it, then renders previews and self-corrects (up to 3 passes)
until brightness, contrast, black/white points and colour reach the
category's look - with face highlights protected.

1. Convert & look at the photo (step 1), and decide yourself **what it shows**
   (portrait, group, landscape, sea, sunset, architecture, interior, night,
   street, food, product, animal, event). This is your `--subject` hint.
2. If the user has not named a category, show the options. List them
   (`python -m photo_editor categories`) grouped, and/or render a visual sheet:
   `python -m photo_editor grid PHOTO --categories natural,portrait,wedding,cinematic,moody,bw -o /tmp/pe/cats.jpg`
   (`--categories` alone renders all 22 - slow, ~2 min). Ask with
   AskUserQuestion offering the 3-4 best fits for this photo + "Other".
3. Run the adaptive edit:

```bash
python -m photo_editor auto /path/PHOTO.ARW --category wedding --subject portrait \
       -o /out/PHOTO_wedding.jpg --compare --xmp
```

   Output JSON: `settings` (every Lightroom value used), `why` (Arabic
   reasons, one line per decision - relay them to the user), `check`
   (measured vs target per pass), `applied` (final EV, white balance K,
   geometry, retouch), `lightroom` (XMP file + what Lightroom can't do).
   `--plan-only` computes settings without rendering; `--fast` for quick looks.
4. Show `_compare.jpg`, explain the main decisions from `why` in the user's
   language, then refine with any setting on top (they win over the engine):
   `... --category wedding --exposure 0.3 --sat-orange -10 --crop-aspect 4:5`

| category | العربية | look |
|---|---|---|
| natural | طبيعي | true colours, neutral |
| portrait | بورتريه | warm skin, soft separated background |
| wedding | أعراس ومناسبات | bright & airy, clean whites |
| fashion | أزياء وتحريري | crisp, contrasty, cool shadows |
| family | عائلي وأطفال | cheerful, warm, bright |
| landscape | طبيعة ومناظر | deep skies, rich greens, haze removed |
| sunset | غروب وشروق | warm light kept, rich sky |
| sea | بحر وشاطئ | turquoise water, bright sand |
| travel | سفر | punchy postcard colours |
| night | ليل ومدينة | deep blacks, clean lights, low noise |
| street | شارع ووثائقي | gritty contrast, muted, grain |
| architecture | عمارة | straight lines, neutral, crisp |
| interior | ديكور وداخلي | bright neutral interiors |
| food | طعام | warm, appetising, textured |
| product | منتجات | pure whites, sharp, neutral |
| wildlife | حيوانات | fur/feather detail, earthy |
| cinematic | سينمائي | teal & orange film grade |
| moody | داكن درامي | dark, muted, dramatic |
| bright_airy | مشرق وناعم | light, soft, gentle colour |
| vintage_film | فيلم قديم | faded film, warm highlights, grain |
| pastel | باستيل | dreamy soft candy colours |
| hdr | تفاصيل عالية | everything visible, strong local detail |
| bw | أبيض وأسود | B&W mixed from the photo's colours |

Special modes: `edit --preset beauty_only` (photo as shot, only face
retouch) and `adjust` (fixed classic edit you drive with sliders).

**Lightroom export**: `--xmp` writes a Lightroom preset next to the output
(import: Develop > Presets > + > Import Presets); `--xmp sidecar` writes
`PHOTO.xmp` next to the RAW so Lightroom/Camera Raw opens it already edited.
Masks (sky/person/background), background blur and face retouch exist only in
this editor and are listed under `not_in_lightroom_file`.

## 3. Analyse only (optional)

```bash
python -m photo_editor analyze /path/DSC01234.ARW --preview-dir /tmp/pe
```

JSON: camera/lens/ISO/shutter, **people** (faces, % of frame), **estimated
light** (K) vs the camera's WB, exposure error (EV), clipping, noise, `issues`,
`recommended_presets`; plus `<name>_asshot.jpg` and `<name>_people_mask.jpg`.
Look at the images, then summarise for the user in their language (Arabic if
they wrote Arabic).

## 4. Fixed presets (older, non-adaptive)

Each preset sets **every** parameter (white balance, exposure, tone, colour,
separation, face retouch, detail, effects).

| preset | العربية | what it does |
|---|---|---|
| `standard` | تعديل عادي | classic edit only: WB, exposure, contrast, colour, sharpness, noise - no background change, no face retouch |
| `beauty_only` | تجميل فقط (الصورة كما هي) | photo stays exactly as shot - only faces are retouched |
| `natural` | طبيعي | true colours, natural WB, light separation, light retouch (default) |
| `portrait_pop` | بورتريه بارز | blurred darker background, crisp subject, retouch |
| `soft_portrait` | بورتريه ناعم | soft contrast, warm glowing skin, stronger retouch |
| `cinematic` | سينمائي | teal & orange grade, darker surroundings |
| `vivid` | حيوي | rich colours for travel/landscape |
| `warm_golden` | ذهبي دافئ | golden-hour warmth |
| `cool_clean` | بارد ونظيف | crisp, slightly cool |
| `moody` | درامي داكن | dark, muted, subject lit |
| `matte_film` | فيلم مطفي | faded film with grain |
| `bw_classic` | أبيض وأسود | rich black & white |

Visual choice - all presets on one sheet:
`python -m photo_editor grid /path/photo.ARW -o /tmp/pe/presets.jpg`

If the user already said what they want, pick it yourself. Otherwise ask
(AskUserQuestion when available), offering `beauty_only` + 2-3 recommended presets.

## 5. Edit with a fixed preset

```bash
python -m photo_editor edit /path/DSC01234.ARW --preset natural --out-dir /out --compare --masks
python -m photo_editor edit /path/DSC01234.ARW --preset beauty_only --set retouch=0.8 -o /out/DSC01234_beauty.jpg --compare
```

`--fast` = half-resolution decode for quick previews; `--tiff` = 16-bit TIFF;
several input files = batch. Outputs: the image (full resolution, EXIF kept),
`_compare.jpg` (as shot | edited), `_masks.jpg` (people/skin), `_faces.jpg`
(retouch regions per face).

## 6. Every Lightroom setting - التعديل اليدوي

For a normal photo edit (no AI separation, no retouch) use `adjust` (= `edit`
with the `standard` preset). Every slider also works on `edit` with any preset
and always wins over the preset.

```bash
python -m photo_editor adjust /path/DSC01234.ARW --contrast 25 --temperature 5200 --tint 5 \
    --highlights -40 --shadows 30 --vibrance 20 --sharpness 70 --clarity 15 -o /out/DSC01234_edit.jpg --compare
python -m photo_editor sliders        # list all sliders with Arabic names and ranges
```

| slider | العربية | range |
|---|---|---|
| `--wb` | توازن الأبيض | `auto` (natural, default) / `camera` / `daylight` |
| `--temperature` | درجة حرارة الإضاءة | Kelvin 2000..12000 — the light in the scene; lower = picture becomes cooler (like Lightroom) |
| `--warmth` | الدفء | -100..100 on top of auto WB (+ warmer) |
| `--tint` | الصبغة | -100..100 (+ magenta, - green) |
| `--exposure` | التعريض | EV -5..5 (added to auto exposure; `--auto-exposure off` to disable) |
| `--contrast` | التباين | -100..100 |
| `--highlights` / `--shadows` | الإضاءات / الظلال | -100..100 |
| `--whites` / `--blacks` | الأبيض / الأسود | -100..100 |
| `--vibrance` / `--saturation` | الحيوية / التشبع | -100..100 (`--saturation -100` = black & white) |
| `--sharpness` | الحدة | 0..150 |
| `--clarity` / `--texture` | الوضوح / الملمس | -100..100 |
| `--noise-reduction` | إزالة التشويش | 0..100 (default: automatic from ISO) |
| `--vignette` / `--grain` | التظليل / الحبيبات | -100..100 / 0..100 |
| `--hsl` | تدرج الألوان | `--hsl orange:hue=5,sat=-10,lum=10` (red orange yellow green aqua blue purple magenta) |

All other Lightroom panels are available the same way (`python -m photo_editor sliders --panel <panel>`):

| panel | settings |
|---|---|
| Basic / Presence | `--profile` (color vivid landscape portrait neutral standard monochrome), `--dehaze`, `--texture`, `--clarity` |
| Tone curve | `--curve-highlights --curve-lights --curve-darks --curve-shadows` (-100..100), `--point-curve "0,20; 128,128; 255,240"`, `--point-curve-red/-green/-blue` |
| HSL | `--hue-<c> --sat-<c> --lum-<c>` for red orange yellow green aqua blue purple magenta (-100..100) |
| Colour grading | `--cg-shadows-hue 200 --cg-shadows-sat 20 --cg-shadows-lum -5`, same for `midtones`, `highlights`, `global`; `--cg-blending`, `--cg-balance` |
| B&W | `--monochrome on --bw-red 20 --bw-blue -30 ...` |
| Detail | `--sharpening --sharpen-radius --sharpen-detail --sharpen-masking --nr-luminance --nr-color` |
| Lens | `--remove-ca on --defringe 8 --lens-distortion 20 --lens-vignetting 30` |
| Transform / crop | `--auto-level on --rotate -1.5 --vertical 25 --horizontal -10 --scale 105 --crop-aspect 4:5 --crop-zoom 1.2 --crop-focus 0.4,0.3` |
| Effects | `--vignette -25 --vignette-midpoint --vignette-feather --vignette-roundness --grain 20 --grain-size --grain-roughness` |
| Calibration | `--shadow-tint --red-hue --red-sat --green-hue --green-sat --blue-hue --blue-sat` |
| Masks | `--sky-exposure -0.4 --sky-saturation 20 --sky-warmth 15`, `--subject-exposure 0.2 --subject-clarity --subject-saturation`, `--background-blur 40 --background-exposure -0.3 --background-saturation -15` |
| Retouch | `--retouch 60`, `--retouch-blemish --retouch-redness --retouch-smooth --retouch-under-eye --retouch-shine --retouch-teeth --retouch-eyes` (0..100) |

Typical requests: "more contrast" → `--contrast 25`; "sharper" → `--sharpness 80`;
"colours stronger" → `--vibrance 30`; "white balance natural" → `--wb auto`;
"photo too yellow" → `--warmth -30` (or `--temperature 3500`); "darker sky" →
`--hsl blue:lum=-30 --highlights -50`.

## 7. Overrides - map the user's words to parameters (`--set key=value`)

**Face beauty (تجميل الوجه)** - `retouch` is the master strength 0..1
(0.3 light, 0.6 normal, 0.9 strong); each step can be set on its own:

| user asks | parameter |
|---|---|
| remove pimples / spots / marks — إزالة الحبوب والبثور | `retouch_blemish=0.8` |
| reduce redness / irritation — تخفيف الاحمرار | `retouch_redness=0.7` |
| even / smoother skin, keep pores — توحيد البشرة | `retouch_smooth=0.5` (keep ≤ 0.7) |
| dark circles — الهالات السوداء | `retouch_under_eye=0.6` |
| oily shine — لمعان البشرة | `retouch_shine=0.6` |
| whiter teeth — تبييض الأسنان | `retouch_teeth=0.5` |
| clear whites of eyes — بياض العين | `retouch_eyes=0.5` |
| no face retouch at all | `retouch=0` |
| keep the photo as is, beauty only — الصورة كما هي | `--preset beauty_only` (or `keep_original=true`) |

**Light & colour**

| user asks | parameters |
|---|---|
| natural white balance — وايت بالنس طبيعي | `wb_mode=auto wb_strength=1 warmth=0 tint=0` |
| warmer / cooler — أدفأ / أبرد | `warmth=10` / `warmth=-10` |
| too green / too magenta | `tint=10` / `tint=-10` |
| camera colours / exact light | `wb_mode=camera` / `wb_mode=kelvin wb_kelvin=3200` |
| brighter / darker — أفتح / أغمق | `exposure=0.3` / `exposure=-0.3` |
| recover sky / open shadows | `highlights=-0.6` / `shadows=0.5` |
| contrast, colour | `contrast=±0.1`, `vibrance=0.2`, `saturation=-0.15` |
| isolate person more / slightly — عزل الشخص | `bg_blur=0.5 bg_exposure=-0.4 bg_saturation=-0.2 subject_exposure=0.15` / `bg_blur=0.2 bg_exposure=-0.15` |
| no background changes | `separate_subject=false` |
| sharper, less noise | `sharpen=0.7 clarity=0.15`, `denoise=0.6` |

Full list: `photo_editor/presets.py` (`DEFAULTS`).

## 8. Verify before delivering

Look at `_compare.jpg`, and for retouching crop/zoom the face area of the
result (e.g. with PIL) to check at 100%:

- spots gone, pores and skin texture still visible (not plastic); if too
  smooth lower `retouch_smooth`; if spots remain raise `retouch_blemish`;
- eyes, brows, lips and beard untouched (see `_faces.jpg`); moles the person
  may want to keep: lower `retouch_blemish` or ask;
- whites neutral, skin natural, no halo around hair if the background was blurred.

Then tell the user what was done (preset, number of blemishes removed, light
~K corrected, EV) and give the output path(s).
