"""Lightroom-style sliders for regular (manual) editing - التعديل العادي.

Sliders use familiar scales (mostly -100..+100) and are translated to the
pipeline's internal parameters.  They can be combined with any preset; a
slider always wins over the preset value.
"""
from __future__ import annotations

# name: (arabic label, scale help, converter -> dict of internal params)
SLIDERS: dict = {
    # white balance
    "wb": ("توازن الأبيض", "auto | camera | daylight",
           lambda v: {"wb_mode": str(v)}),
    "temperature": ("درجة حرارة الإضاءة", "Kelvin 2000..12000 (sets the light; lower = image gets cooler)",
                    lambda v: {"wb_mode": "kelvin", "wb_kelvin": float(v)}),
    "warmth": ("الدفء", "-100..100 relative to the auto/camera WB (+ warmer)",
               lambda v: {"warmth": float(v) * 0.4}),
    "tint": ("الصبغة", "-100..100 (+ magenta, - green)",
             lambda v: {"tint": float(v)}),
    # light
    "exposure": ("التعريض", "EV -5..+5",
                 lambda v: {"exposure": float(v)}),
    "auto_exposure": ("تعريض تلقائي", "on | off",
                      lambda v: {"auto_exposure": str(v).lower() in ("1", "true", "on", "yes")}),
    "contrast": ("التباين", "-100..100",
                 lambda v: {"contrast": float(v) / 100 * 0.6}),
    "highlights": ("الإضاءات العالية", "-100..100 (- recovers bright areas)",
                   lambda v: {"highlights": float(v) / 100}),
    "shadows": ("الظلال", "-100..100 (+ opens dark areas)",
                lambda v: {"shadows": float(v) / 100}),
    "whites": ("الأبيض", "-100..100",
               lambda v: {"whites": float(v) / 100}),
    "blacks": ("الأسود", "-100..100 (- deeper blacks, + faded)",
               lambda v: {"blacks": float(v) / 100}),
    # colour
    "vibrance": ("الحيوية", "-100..100 (boosts muted colours, protects skin)",
                 lambda v: {"vibrance": float(v) / 100}),
    "saturation": ("التشبع", "-100..100 (-100 = black & white)",
                   lambda v: {"saturation": float(v) / 100} if float(v) > -100 else {"monochrome": True}),
    # detail
    "sharpness": ("الحدة", "0..150",
                  lambda v: {"sharpen": float(v) / 100}),
    "clarity": ("الوضوح", "-100..100 (mid-tone local contrast)",
                lambda v: {"clarity": float(v) / 100 * 0.6}),
    "texture": ("الملمس", "-100..100 (fine detail)",
                lambda v: {"texture": float(v) / 100 * 0.5}),
    "noise_reduction": ("إزالة التشويش", "0..100 (default: automatic from ISO)",
                        lambda v: {"denoise": float(v) / 100}),
    # effects
    "vignette": ("التظليل الجانبي", "-100..100 (- darker corners)",
                 lambda v: {"vignette": -float(v) / 100 * 0.6}),
    "grain": ("الحبيبات", "0..100",
              lambda v: {"grain": float(v) / 100}),
}

# HSL per colour: --hsl orange:hue=5,sat=-10,lum=10
HSL_BANDS = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")


def sliders_to_params(values: dict) -> dict:
    """{'contrast': 20, 'temperature': 5200, ...} -> internal parameter dict."""
    out: dict = {}
    for k, v in values.items():
        if v is None:
            continue
        if k not in SLIDERS:
            raise KeyError(f"unknown slider '{k}'. Available: {', '.join(SLIDERS)}")
        out.update(SLIDERS[k][2](v))
    return out


def parse_hsl(specs: list[str] | None) -> dict:
    """['orange:sat=-10,lum=5', 'blue:hue=-8'] -> {'sat_orange': -0.1, ...}.
    hue in degrees (-30..30), sat and lum -100..100."""
    out: dict = {}
    for spec in specs or []:
        band, _, rest = spec.partition(":")
        band = band.strip().lower()
        if band not in HSL_BANDS:
            raise KeyError(f"unknown colour '{band}'. Use one of: {', '.join(HSL_BANDS)}")
        for item in rest.split(","):
            if not item.strip():
                continue
            key, _, val = item.partition("=")
            key, val = key.strip().lower(), float(val)
            if key == "hue":
                out[f"hue_{band}"] = val
            elif key in ("sat", "saturation"):
                out[f"sat_{band}"] = val / 100
            elif key in ("lum", "luminance"):
                out[f"lum_{band}"] = val / 100
            else:
                raise KeyError(f"HSL key must be hue, sat or lum, got '{key}'")
    return out


def describe() -> list[dict]:
    return [{"slider": k, "label_ar": v[0], "range": v[1]} for k, v in SLIDERS.items()] + \
        [{"slider": "hsl", "label_ar": "تدرج الألوان (HSL)",
          "range": "--hsl <colour>:hue=deg,sat=-100..100,lum=-100..100; colours: " + ", ".join(HSL_BANDS)}]
