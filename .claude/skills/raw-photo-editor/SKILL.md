---
name: raw-photo-editor
description: Read and edit RAW photos (Sony ARW/SR2/SRF, Canon CR2/CR3, Nikon NEF, Fuji RAF, DNG, ORF, RW2 and every LibRaw format) and JPEG/PNG/TIFF. Detects people and separates them from the background, fixes white balance to natural, corrects exposure and colour, and applies presets that change every setting at once. Use whenever the user uploads or names a photo/RAW file and wants it analysed, enhanced, white-balanced, colour-graded, or wants the person isolated/emphasised (Arabic requests too - تعديل صورة، توازن اللون الأبيض، عزل الشخص عن الخلفية، صورة RAW / ARW).
---

# RAW photo editor - مهارة تعديل صور RAW

The editing engine lives in this repository as the Python package `photo_editor`
(run every command from the repository root). It decodes RAW files linearly with
LibRaw, so white balance and exposure are corrected on real sensor data, not on a
baked JPEG.

## 0. One-time setup

```bash
pip install -r requirements.txt
python -m photo_editor models        # downloads the people/face/skin models (~20 MB)
```

If MediaPipe fails with `libEGL.so.1` / `libGLES` missing (headless Linux):
`apt-get install -y libegl1 libgles2`. Without MediaPipe the editor still works
but falls back to weaker OpenCV people detection - tell the user if that happens
(the report shows `"backend": "opencv:..."`).

## 1. Analyse first

```bash
python -m photo_editor analyze PHOTO.ARW --preview-dir output/
```

Returns JSON with camera/lens/ISO/shutter, **people** (faces, % of frame), the
**estimated light colour** (CCT in K + light type) vs the camera's as-shot WB,
exposure error in EV, clipping, noise level, a list of `issues` and
`recommended_presets`. It also writes `output/<name>_asshot.jpg` and, if people
were found, `output/<name>_people_mask.jpg` (people tinted green, skin orange,
face boxes yellow).

**Always look at the preview images with the Read tool** before talking to the
user, and check the people mask actually covers the people.

Summarise to the user in their language (Arabic if they wrote Arabic): how many
people, what light it was shot in, what is wrong (e.g. "الكاميرا وازنت اللون على
3200K لكن الإضاءة الفعلية ~5600K فالصورة مائلة للأزرق"), what you suggest.

## 2. Offer the options (presets)

Choosing a preset sets **every** parameter: white balance, exposure, tone,
colour, subject/background separation, detail and effects.

| preset | العربية | what it does |
|---|---|---|
| `natural` | طبيعي | true colours, full auto WB, people slightly separated (default) |
| `portrait_pop` | بورتريه بارز | blurred & darker background, crisp subject, healthy skin |
| `soft_portrait` | بورتريه ناعم | low contrast, lifted blacks, warm skin, skin smoothing |
| `cinematic` | سينمائي | teal & orange grade, deeper contrast, darker surroundings, grain |
| `vivid` | حيوي | rich colour for travel/landscape, skin kept natural |
| `warm_golden` | ذهبي دافئ | golden-hour warmth |
| `cool_clean` | بارد ونظيف | crisp, slightly cool, clean whites |
| `moody` | درامي داكن | dark, muted, subject lit, surroundings fall away |
| `matte_film` | فيلم مطفي | faded film, lifted blacks, grain |
| `bw_classic` | أبيض وأسود | rich black & white |

To let the user choose visually, render all presets on one sheet and show it:

```bash
python -m photo_editor grid PHOTO.ARW -o output/presets.jpg
```

If the user already said what they want ("طبيعي", "عزل الشخص", "صحح الوايت بالنس"),
don't ask - pick the preset and overrides yourself (table in step 4). Otherwise
ask with AskUserQuestion, offering 3-4 of the `recommended_presets`.

## 3. Edit

```bash
python -m photo_editor edit PHOTO.ARW --preset natural --out-dir output/ --compare --masks --report
# quick look first (half-resolution decode, ~4x faster):
python -m photo_editor edit PHOTO.ARW --preset natural --out-dir output/ --compare --fast
```

Outputs: `<name>_<preset>.jpg` (full resolution, sRGB, EXIF kept), plus
`_compare.jpg` (as shot | edited), `_masks.jpg`, `_report.json`.
Use `--tiff` for a 16-bit TIFF, `-o path.jpg` for an explicit name, several
files at once for a batch.

## 4. Adjust with overrides

Any parameter can be overridden on top of a preset: `--set key=value` (repeat)
or `--params file.json`. Map the user's words to parameters:

| user asks | parameters |
|---|---|
| natural WB / remove the yellow, blue, green cast — وايت بالنس طبيعي | `wb_mode=auto wb_strength=1 warmth=0 tint=0` |
| a bit warmer / cooler — أدفأ / أبرد | `warmth=+10` / `warmth=-10` (mireds; ±5 subtle, ±25 strong) |
| too green / too magenta | `tint=+10` (adds magenta) / `tint=-10` |
| keep the camera's colours | `wb_mode=camera` |
| set light to e.g. 3200K | `wb_mode=kelvin wb_kelvin=3200` |
| brighter / darker — أفتح / أغمق | `exposure=+0.3` / `exposure=-0.3` (EV, on top of auto) |
| recover sky/highlights | `highlights=-0.6` |
| open up shadows/faces in shade | `shadows=+0.5` |
| more / less contrast | `contrast=+0.1` / `contrast=-0.1` |
| more colour / calmer colour | `vibrance=+0.2` / `saturation=-0.15` |
| isolate person more — عزل الشخص أكثر | `bg_blur=0.5 bg_exposure=-0.4 bg_saturation=-0.2 subject_exposure=0.15` |
| isolate only slightly — عزل خفيف | `bg_blur=0.2 bg_exposure=-0.15 bg_saturation=-0.05` |
| no background blur | `bg_blur=0` (or `separate_subject=false` to disable all separation) |
| smoother skin | `skin_smooth=0.3` (0.2 subtle, 0.5 strong; never above 0.6) |
| more natural skin tone | `skin_tone=0.6` |
| sharper / more detail | `sharpen=0.7 clarity=0.15 texture=0.1` |
| noise (high ISO) | `denoise=0.6` (default `auto` derives it from ISO) |
| per-colour tweaks | `hue_<band>`, `sat_<band>`, `lum_<band>`, bands: red orange yellow green aqua blue purple magenta |
| vignette / grain | `vignette=0.3`, `grain=0.3` |

Full list with defaults: `photo_editor/presets.py` (`DEFAULTS`).

## 5. Verify before delivering

Look at `_compare.jpg` (and `_masks.jpg` when people were separated) with Read:

- whites/greys neutral, skin natural (not orange, grey or magenta);
- no halo or blurred patch around hair/shoulders - if the mask missed a body
  part, lower `bg_blur`, or disable separation for that image;
- highlights not blown, shadows not crushed (`output_stats` in the JSON:
  `clipped_highlights_pct` should normally stay < 1%).

Fix with overrides and re-run (use `--fast` while iterating, full resolution for
the final). Then tell the user which preset and settings were applied, what was
corrected (e.g. "light ~3100K tungsten → neutralised", "+1.2 EV"), and give the
output path(s).

## How it works (for explaining to the user)

1. **RAW decode** (rawpy/LibRaw, AHD demosaic, 16-bit linear, daylight WB) → linear sRGB.
2. **People** - MediaPipe face detector + DeepLabV3 person + selfie multiclass
   (hair/skin/clothes), also run on a crop around each face, refined with a
   guided filter for soft hair edges.
3. **White balance** - searches real light sources (Planckian locus) for the one
   that turns the most background surfaces neutral, combined with grey-edge /
   shades-of-grey / bright-patch, then checks skin hue; people are excluded from
   the estimate so skin and clothes do not bias it.
4. **Exposure** - meters the scene and the people, protects highlights; local
   highlight/shadow recovery with an edge-aware base layer.
5. **Separation** - halo-free lens-like background blur in linear light,
   background exposure/saturation/clarity vs subject.
6. **Tone & colour** - filmic highlight roll-off, contrast/levels, vibrance with
   skin protection, HSL, colour grading, B&W - in OKLab.
7. **Detail** - ISO-aware noise reduction, clarity, texture, sharpening, grain.
