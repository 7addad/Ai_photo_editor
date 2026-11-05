"""Every Lightroom Classic / Camera Raw setting the editor supports, on
Lightroom's own scales - التعديل العادي بكل إعدادات لايت روم.

Settings are a flat dict ``{"contrast": 20, "temperature": 5200, ...}``.
``sliders_to_params`` converts them to the pipeline's internal parameters;
``xmp.py`` writes the same dict as a Lightroom/Camera Raw XMP sidecar.
"""
from __future__ import annotations

HSL_BANDS = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")
BANDS_AR = {"red": "الأحمر", "orange": "البرتقالي", "yellow": "الأصفر", "green": "الأخضر",
            "aqua": "السماوي", "blue": "الأزرق", "purple": "البنفسجي", "magenta": "الأرجواني"}
WHEELS_AR = {"shadows": "الظلال", "midtones": "الدرجات المتوسطة", "highlights": "الإضاءات", "global": "عام"}


def _f(v):
    return float(v)


def _b(v):
    return str(v).lower() in ("1", "true", "on", "yes")


def _same(key, conv=_f):
    return lambda v: {key: conv(v)}


# key: (panel, arabic, range text, converter)
SLIDERS: dict = {}


def _add(key, panel, ar, rng, conv):
    SLIDERS[key] = (panel, ar, rng, conv)


# ---- Profile & white balance -------------------------------------------------
_PROFILES = {
    "color": {}, "adobe_color": {},
    "vivid": {"base_contrast": 0.38, "saturation_profile": 0.12},
    "landscape": {"base_contrast": 0.36, "saturation_profile": 0.08},
    "portrait": {"base_contrast": 0.25, "saturation_profile": -0.04},
    "neutral": {"base_contrast": 0.16, "saturation_profile": -0.08},
    "standard": {"base_contrast": 0.32, "saturation_profile": 0.04},
    "monochrome": {"monochrome": True},
}
_add("profile", "basic", "الملف اللوني", "color | vivid | landscape | portrait | neutral | standard | monochrome",
     lambda v: dict(_PROFILES.get(str(v).lower(), {})))
_add("wb", "basic", "توازن الأبيض", "auto | camera | daylight", lambda v: {"wb_mode": str(v)})
_add("temperature", "basic", "درجة الحرارة", "Kelvin 2000..12000 (the light; lower = cooler picture)",
     lambda v: {"wb_mode": "kelvin", "wb_kelvin": _f(v)})
_add("tint", "basic", "الصبغة", "-150..150 (+ magenta, - green)", _same("tint"))
_add("warmth", "basic", "الدفء", "-100..100 relative to auto/camera WB (+ warmer)",
     lambda v: {"warmth": _f(v) * 0.4})
# ---- Basic tone ------------------------------------------------------------
_add("exposure", "basic", "التعريض", "EV -5..5 (added to auto exposure)", _same("exposure"))
_add("auto_exposure", "basic", "تعريض تلقائي", "on | off", lambda v: {"auto_exposure": _b(v)})
_add("contrast", "basic", "التباين", "-100..100", lambda v: {"contrast": _f(v) / 100 * 0.6})
_add("highlights", "basic", "الإضاءات العالية", "-100..100", lambda v: {"highlights": _f(v) / 100})
_add("shadows", "basic", "الظلال", "-100..100", lambda v: {"shadows": _f(v) / 100})
_add("whites", "basic", "الأبيض", "-100..100", lambda v: {"whites": _f(v) / 100})
_add("blacks", "basic", "الأسود", "-100..100", lambda v: {"blacks": _f(v) / 100})
# ---- Presence --------------------------------------------------------------
_add("texture", "presence", "الملمس", "-100..100", lambda v: {"texture": _f(v) / 100 * 0.5})
_add("clarity", "presence", "الوضوح", "-100..100", lambda v: {"clarity": _f(v) / 100 * 0.6})
_add("dehaze", "presence", "إزالة الضباب", "-100..100", _same("dehaze"))
_add("vibrance", "presence", "الحيوية", "-100..100", lambda v: {"vibrance": _f(v) / 100})
_add("saturation", "presence", "التشبع", "-100..100 (-100 = black & white)",
     lambda v: {"saturation": _f(v) / 100} if _f(v) > -100 else {"monochrome": True})
# ---- Tone curve ------------------------------------------------------------
for k, ar in (("highlights", "الإضاءات"), ("lights", "الفاتحة"), ("darks", "الداكنة"), ("shadows", "الظلال")):
    _add(f"curve_{k}", "tone_curve", f"منحنى: {ar}", "-100..100", _same(f"curve_{k}"))
for k, ar in (("", "المنحنى النقطي"), ("_red", "منحنى الأحمر"), ("_green", "منحنى الأخضر"), ("_blue", "منحنى الأزرق")):
    _add(f"point_curve{k}", "tone_curve", ar, "points 'x,y; x,y; ...' in 0..255, e.g. '0,20; 128,128; 255,240'",
         _same(f"point_curve{k}", str))
# ---- HSL / colour mixer ------------------------------------------------------
for band in HSL_BANDS:
    _add(f"hue_{band}", "hsl", f"تدرج {BANDS_AR[band]}", "-100..100", lambda v, b=band: {f"hue_{b}": _f(v) * 0.3})
    _add(f"sat_{band}", "hsl", f"تشبع {BANDS_AR[band]}", "-100..100", lambda v, b=band: {f"sat_{b}": _f(v) / 100})
    _add(f"lum_{band}", "hsl", f"إضاءة {BANDS_AR[band]}", "-100..100", lambda v, b=band: {f"lum_{b}": _f(v) / 100})
# ---- Colour grading --------------------------------------------------------
for w, ar in WHEELS_AR.items():
    _add(f"cg_{w}_hue", "color_grading", f"تلوين {ar}: اللون", "0..360 (0 red, 40 orange, 60 yellow, 120 green, 200 teal, 240 blue)",
         _same(f"cg_{w}_hue"))
    _add(f"cg_{w}_sat", "color_grading", f"تلوين {ar}: الشدة", "0..100", _same(f"cg_{w}_sat"))
    _add(f"cg_{w}_lum", "color_grading", f"تلوين {ar}: الإضاءة", "-100..100", _same(f"cg_{w}_lum"))
_add("cg_blending", "color_grading", "مزج التلوين", "0..100 (default 50)", _same("cg_blending"))
_add("cg_balance", "color_grading", "توازن التلوين", "-100..100 (+ toward highlights)", _same("cg_balance"))
# ---- Black & white mix -----------------------------------------------------
_add("monochrome", "bw", "أبيض وأسود", "on | off", lambda v: {"monochrome": _b(v)})
for band in HSL_BANDS:
    _add(f"bw_{band}", "bw", f"مزج الرمادي: {BANDS_AR[band]}", "-100..100", lambda v, b=band: {f"_bw_{b}": _f(v) / 100})
# ---- Detail ----------------------------------------------------------------
_add("sharpening", "detail", "الحدة", "0..150", lambda v: {"sharpen": _f(v) / 100})
_add("sharpness", "detail", "الحدة (اسم بديل)", "0..150", lambda v: {"sharpen": _f(v) / 100})
_add("sharpen_radius", "detail", "نصف قطر الحدة", "0.5..3.0", _same("sharpen_radius"))
_add("sharpen_detail", "detail", "تفاصيل الحدة", "0..100", _same("sharpen_detail"))
_add("sharpen_masking", "detail", "قناع الحدة", "0..100 (higher = edges only)", _same("sharpen_masking"))
_add("nr_luminance", "detail", "إزالة تشويش الإضاءة", "0..100", _same("nr_luminance"))
_add("nr_color", "detail", "إزالة تشويش اللون", "0..100", _same("nr_color"))
_add("noise_reduction", "detail", "إزالة التشويش", "0..100 (both)", lambda v: {"denoise": _f(v) / 100})
# ---- Lens --------------------------------------------------------------------
_add("remove_ca", "lens", "إزالة الانحراف اللوني", "on | off", lambda v: {"remove_ca": _b(v)})
_add("defringe", "lens", "إزالة الحواف البنفسجية", "0..20", _same("defringe"))
_add("lens_distortion", "lens", "تشوه العدسة", "-100..100 (+ removes barrel)", _same("lens_distortion"))
_add("lens_vignetting", "lens", "تصحيح تعتيم العدسة", "-100..100 (+ brightens corners)", _same("lens_vignetting"))
# ---- Transform & crop ----------------------------------------------------------
_add("auto_level", "transform", "تسوية تلقائية للأفق", "on | off", lambda v: {"auto_level": _b(v)})
_add("rotate", "transform", "تدوير", "degrees -45..45 (+ counter-clockwise)", _same("rotate"))
_add("vertical", "transform", "منظور عمودي", "-100..100 (+ fixes buildings leaning back)", _same("vertical"))
_add("horizontal", "transform", "منظور أفقي", "-100..100", _same("horizontal"))
_add("scale", "transform", "تكبير", "100..150", _same("scale"))
_add("crop_aspect", "crop", "نسبة القص", "original | 1:1 | 4:5 | 5:4 | 3:2 | 2:3 | 4:3 | 3:4 | 16:9 | 9:16",
     _same("crop_aspect", str))
_add("crop_zoom", "crop", "تكبير القص", "1.0..3.0", _same("crop_zoom"))
_add("crop_focus", "crop", "مركز القص", "auto | x,y (0..1)", _same("crop_focus", str))
# ---- Effects -----------------------------------------------------------------
_add("vignette", "effects", "التظليل الجانبي", "-100..100 (- darker corners)",
     lambda v: {"vignette": -_f(v) / 100 * 0.6})
_add("vignette_midpoint", "effects", "منتصف التظليل", "0..100", _same("vignette_midpoint"))
_add("vignette_feather", "effects", "نعومة التظليل", "0..100", _same("vignette_feather"))
_add("vignette_roundness", "effects", "استدارة التظليل", "-100..100", _same("vignette_roundness"))
_add("vignette_highlights", "effects", "حماية الإضاءات في التظليل", "0..100", _same("vignette_highlights"))
_add("grain", "effects", "الحبيبات", "0..100", lambda v: {"grain": _f(v) / 100})
_add("grain_size", "effects", "حجم الحبيبات", "0..100", _same("grain_size"))
_add("grain_roughness", "effects", "خشونة الحبيبات", "0..100", _same("grain_roughness"))
# ---- Calibration -------------------------------------------------------------
_add("shadow_tint", "calibration", "صبغة الظلال", "-100..100", _same("shadow_tint"))
for c, ar in (("red", "الأحمر"), ("green", "الأخضر"), ("blue", "الأزرق")):
    _add(f"{c}_hue", "calibration", f"الأساسي {ar}: التدرج", "-100..100", _same(f"{c}_hue"))
    _add(f"{c}_sat", "calibration", f"الأساسي {ar}: التشبع", "-100..100", _same(f"{c}_sat"))
# ---- Masks (people / background / sky) -------------------------------------------
_add("subject_exposure", "masks", "إضاءة الشخص", "EV -2..2", _same("subject_exposure"))
_add("subject_clarity", "masks", "وضوح الشخص", "-100..100", lambda v: {"subject_clarity": _f(v) / 100 * 0.6})
_add("subject_saturation", "masks", "تشبع الشخص", "-100..100", lambda v: {"subject_saturation": _f(v) / 100})
_add("background_blur", "masks", "تمويه الخلفية", "0..100", lambda v: {"bg_blur": _f(v) / 100})
_add("background_exposure", "masks", "إضاءة الخلفية", "EV -2..2", _same("bg_exposure"))
_add("background_saturation", "masks", "تشبع الخلفية", "-100..100", lambda v: {"bg_saturation": _f(v) / 100})
_add("background_clarity", "masks", "وضوح الخلفية", "-100..100", lambda v: {"bg_clarity": _f(v) / 100 * 0.6})
_add("separate_subject", "masks", "فصل الشخص عن الخلفية", "on | off", lambda v: {"separate_subject": _b(v)})
_add("sky_exposure", "masks", "إضاءة السماء", "EV -2..2", _same("sky_exposure"))
_add("sky_saturation", "masks", "تشبع السماء", "-100..100", _same("sky_saturation"))
_add("sky_warmth", "masks", "دفء السماء", "-100..100", _same("sky_warmth"))
# ---- Face retouch ----------------------------------------------------------------
_add("retouch", "retouch", "تجميل الوجه", "0..100", lambda v: {"retouch": _f(v) / 100})
for k, ar in (("blemish", "إزالة الحبوب"), ("redness", "الاحمرار"), ("smooth", "توحيد البشرة"),
              ("under_eye", "الهالات"), ("shine", "اللمعان"), ("teeth", "تبييض الأسنان"), ("eyes", "بياض العين")):
    _add(f"retouch_{k}", "retouch", ar, "0..100", lambda v, k=k: {f"retouch_{k}": _f(v) / 100})
_add("skin_smooth", "retouch", "نعومة بشرة الجسم", "0..100", lambda v: {"skin_smooth": _f(v) / 100})
_add("keep_original", "retouch", "الصورة كما هي", "on | off", lambda v: {"keep_original": _b(v)})

PANELS_AR = {"basic": "الأساسيات", "presence": "الحضور", "tone_curve": "منحنى النغمة", "hsl": "مزج الألوان HSL",
             "color_grading": "تدرج الألوان", "bw": "أبيض وأسود", "detail": "التفاصيل", "lens": "العدسة",
             "transform": "التحويل", "crop": "القص", "effects": "المؤثرات", "calibration": "المعايرة",
             "masks": "الأقنعة (شخص/خلفية/سماء)", "retouch": "تجميل الوجه"}


def sliders_to_params(values: dict) -> dict:
    """Lightroom-scale settings -> internal pipeline parameters."""
    out: dict = {}
    bw = {}
    for k, v in values.items():
        if v is None:
            continue
        if k not in SLIDERS:
            raise KeyError(f"unknown setting '{k}'. Run `python -m photo_editor sliders` for the list")
        for ik, iv in SLIDERS[k][3](v).items():
            if ik.startswith("_bw_"):
                bw[ik[4:]] = iv
            else:
                out[ik] = iv
    if bw:
        out["bw_mix"] = bw
    if "saturation_profile" in out:
        out["saturation"] = out.get("saturation", 0.0) + out.pop("saturation_profile")
    return out


def parse_hsl(specs: list[str] | None) -> dict:
    """['orange:sat=-10,lum=5', 'blue:hue=-8'] (Lightroom -100..100) -> internal params."""
    vals: dict = {}
    for spec in specs or []:
        band, _, rest = spec.partition(":")
        band = band.strip().lower()
        if band not in HSL_BANDS:
            raise KeyError(f"unknown colour '{band}'. Use one of: {', '.join(HSL_BANDS)}")
        for item in rest.split(","):
            if not item.strip():
                continue
            key, _, val = item.partition("=")
            key = {"saturation": "sat", "luminance": "lum"}.get(key.strip().lower(), key.strip().lower())
            if key not in ("hue", "sat", "lum"):
                raise KeyError(f"HSL key must be hue, sat or lum, got '{key}'")
            vals[f"{key}_{band}"] = float(val)
    return sliders_to_params(vals)


def describe(panel: str | None = None) -> list[dict]:
    return [{"setting": k, "panel": v[0], "panel_ar": PANELS_AR.get(v[0], v[0]), "label_ar": v[1], "range": v[2]}
            for k, v in SLIDERS.items() if panel is None or v[0] == panel]
