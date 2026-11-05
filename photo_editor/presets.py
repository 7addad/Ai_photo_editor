"""Editing presets.  Choosing a preset sets *every* parameter of the pipeline
(white balance, exposure, tone, colour, subject separation, detail, effects).
Any parameter can still be overridden individually.
"""
from __future__ import annotations

import copy

# Every parameter the pipeline understands, with neutral defaults.
DEFAULTS: dict = {
    # white balance
    "wb_mode": "auto",          # auto | camera | daylight | kelvin
    "wb_kelvin": 5500,          # used when wb_mode == kelvin
    "wb_tint_duv": 0.0,
    "wb_strength": 1.0,         # 0..1 share of the auto correction
    "warmth": 0.0,              # creative shift in mireds (+ warmer)
    "tint": 0.0,                # + magenta / - green
    # exposure & tone
    "auto_exposure": True,
    "exposure": 0.0,            # EV added on top of auto exposure
    "highlights": 0.0,          # -1..1 (negative recovers)
    "shadows": 0.0,             # -1..1 (positive lifts)
    "contrast": 0.0,            # -0.5..0.6
    "whites": 0.0,
    "blacks": 0.0,              # + matte lift, - deeper blacks
    "tone_white": 8.0,          # tone-curve white point (scene-linear)
    "base_contrast": 0.3,       # RAW only: camera-like base contrast before 'contrast'
    "auto_black": True,         # RAW only: set black point from the darkest pixels
    # colour
    "vibrance": 0.0,
    "saturation": 0.0,
    "skin_tone": 0.0,           # 0..1 nudges skin to a natural hue
    "monochrome": False,
    # HSL: hue_<band> (deg), sat_<band>, lum_<band>, bands: red orange yellow green aqua blue purple magenta
    # grading
    "grade_shadows_hue": 200, "grade_shadows_amount": 0.0,
    "grade_highlights_hue": 60, "grade_highlights_amount": 0.0,
    # subject / background separation (needs people)
    "separate_subject": True,
    "bg_blur": 0.0,             # 0..1 lens-like blur of the background
    "bg_exposure": 0.0,         # EV
    "bg_saturation": 0.0,       # added to colour scale
    "bg_clarity": 0.0,
    "subject_exposure": 0.0,    # EV
    "subject_saturation": 0.0,
    "subject_clarity": 0.0,
    "skin_smooth": 0.0,         # 0..1 (body skin; faces use retouch_smooth)
    # face beauty retouch (0..1). 'retouch' is the master strength; each step can be
    # set on its own: retouch_blemish, retouch_redness, retouch_smooth,
    # retouch_under_eye, retouch_shine, retouch_teeth, retouch_eyes
    "retouch": 0.0,
    "keep_original": False,     # True = keep colours/light exactly as shot, only retouch faces
    # detail
    "clarity": 0.0,
    "texture": 0.0,
    "sharpen": 0.4,
    "denoise": "auto",          # auto (from ISO) or number 0..1
    # effects
    "vignette": 0.0,
    "grain": 0.0,
}

PRESETS: dict[str, dict] = {
    "beauty_only": {
        "label_ar": "تجميل فقط (الصورة كما هي)",
        "description": "Keeps the photo exactly as shot (colours, light, background untouched) and only retouches faces: blemishes, redness, skin, under-eyes, shine, teeth, eyes.",
        "description_ar": "تبقى الصورة كما هي تمامًا (الألوان والإضاءة والخلفية دون تغيير) ويُعدَّل الوجه فقط: إزالة البثور والحبوب، تخفيف الاحمرار، توحيد البشرة، الهالات، اللمعان، تبييض الأسنان، بياض العين.",
        "params": {"keep_original": True, "retouch": 0.7},
    },
    "standard": {
        "label_ar": "تعديل عادي",
        "description": "Classic photo edit only: white balance, exposure, contrast, highlights/shadows, colour, sharpness and noise. No background changes, no face retouch. Fine-tune with the sliders.",
        "description_ar": "تعديل كلاسيكي فقط: توازن الأبيض، التعريض، التباين، الإضاءات والظلال، الألوان، الحدة وإزالة التشويش. بدون تغيير الخلفية وبدون تجميل الوجه، ويمكن ضبط كل شريط يدويًا.",
        "params": {
            "separate_subject": False, "retouch": 0.0,
            "wb_strength": 1.0, "contrast": 0.15, "highlights": -0.35, "shadows": 0.3,
            "vibrance": 0.18, "saturation": 0.03, "clarity": 0.08, "texture": 0.04,
            "sharpen": 0.55, "vignette": 0.0, "skin_tone": 0.2,
        },
    },
    "natural": {
        "label_ar": "طبيعي",
        "description": "Neutral, true-to-life colours. Auto white balance fully corrected, people slightly separated from the background.",
        "description_ar": "ألوان حقيقية ومحايدة، توازن أبيض تلقائي كامل، وفصل خفيف للأشخاص عن الخلفية.",
        "params": {
            "retouch": 0.35,
            "wb_strength": 1.0, "warmth": 3, "contrast": 0.12, "highlights": -0.35, "shadows": 0.25,
            "vibrance": 0.12, "saturation": 0.0, "skin_tone": 0.35,
            "bg_blur": 0.18, "bg_exposure": -0.15, "bg_saturation": -0.06, "subject_exposure": 0.08,
            "subject_clarity": 0.08, "clarity": 0.05, "sharpen": 0.45, "vignette": 0.08,
        },
    },
    "portrait_pop": {
        "label_ar": "بورتريه بارز",
        "description": "The person stands out: blurred and darker background, crisp subject, healthy skin.",
        "description_ar": "إبراز الشخص: خلفية مموهة وأغمق قليلًا، وضوح أعلى للشخص، وبشرة صحية.",
        "params": {
            "retouch": 0.55,
            "wb_strength": 1.0, "warmth": 6, "contrast": 0.16, "highlights": -0.4, "shadows": 0.3,
            "vibrance": 0.18, "skin_tone": 0.5, "skin_smooth": 0.1,
            "bg_blur": 0.45, "bg_exposure": -0.35, "bg_saturation": -0.18, "bg_clarity": -0.15,
            "subject_exposure": 0.15, "subject_clarity": 0.15, "subject_saturation": 0.04,
            "sharpen": 0.5, "vignette": 0.18,
        },
    },
    "soft_portrait": {
        "label_ar": "بورتريه ناعم",
        "description": "Airy and gentle: softer contrast, lifted blacks, warm skin, smooth background.",
        "description_ar": "ناعم ومضيء: تباين منخفض، سواد مرفوع قليلًا، بشرة دافئة وخلفية ناعمة.",
        "params": {
            "retouch": 0.7,
            "warmth": 10, "exposure": 0.15, "contrast": -0.05, "highlights": -0.45, "shadows": 0.35,
            "blacks": 0.25, "vibrance": 0.08, "saturation": -0.05, "skin_tone": 0.5, "skin_smooth": 0.15,
            "bg_blur": 0.4, "bg_exposure": 0.05, "bg_saturation": -0.12, "subject_exposure": 0.1,
            "clarity": -0.08, "sharpen": 0.3, "grade_highlights_hue": 70, "grade_highlights_amount": 0.25,
        },
    },
    "cinematic": {
        "label_ar": "سينمائي",
        "description": "Teal & orange grade, deeper contrast, darker surroundings, slight grain.",
        "description_ar": "تدرج سينمائي (أزرق مخضر وبرتقالي)، تباين أعمق، محيط أغمق وحبيبات خفيفة.",
        "params": {
            "retouch": 0.4,
            "warmth": 2, "contrast": 0.25, "highlights": -0.5, "shadows": 0.15, "blacks": -0.1,
            "vibrance": 0.05, "saturation": -0.08, "skin_tone": 0.4,
            "grade_shadows_hue": 215, "grade_shadows_amount": 0.55,
            "grade_highlights_hue": 65, "grade_highlights_amount": 0.4,
            "hue_green": -12, "sat_green": -0.25, "hue_blue": -8,
            "bg_blur": 0.3, "bg_exposure": -0.4, "bg_saturation": -0.1, "subject_exposure": 0.1,
            "clarity": 0.1, "vignette": 0.3, "grain": 0.25, "sharpen": 0.4,
        },
    },
    "vivid": {
        "label_ar": "حيوي",
        "description": "Rich, punchy colours for travel and landscapes, skin kept natural.",
        "description_ar": "ألوان غنية وقوية للسفر والطبيعة مع الحفاظ على طبيعية البشرة.",
        "params": {
            "retouch": 0.3,
            "contrast": 0.2, "highlights": -0.45, "shadows": 0.35, "vibrance": 0.35, "saturation": 0.08,
            "skin_tone": 0.4, "sat_blue": 0.1, "lum_blue": -0.2, "sat_green": 0.08,
            "clarity": 0.18, "texture": 0.1, "bg_blur": 0.0, "bg_exposure": 0.0,
            "subject_exposure": 0.05, "sharpen": 0.55, "vignette": 0.1,
        },
    },
    "warm_golden": {
        "label_ar": "ذهبي دافئ",
        "description": "Golden-hour warmth, glowing highlights, soft background.",
        "description_ar": "دفء الساعة الذهبية، إضاءات متوهجة وخلفية ناعمة.",
        "params": {
            "retouch": 0.45,
            "wb_strength": 0.85, "warmth": 28, "tint": 6, "contrast": 0.12, "highlights": -0.4, "shadows": 0.25,
            "vibrance": 0.15, "skin_tone": 0.3, "grade_highlights_hue": 70, "grade_highlights_amount": 0.5,
            "grade_shadows_hue": 30, "grade_shadows_amount": 0.15,
            "bg_blur": 0.3, "bg_exposure": -0.15, "subject_exposure": 0.1, "vignette": 0.15, "sharpen": 0.4,
        },
    },
    "cool_clean": {
        "label_ar": "بارد ونظيف",
        "description": "Crisp, slightly cool, clean whites - modern / architectural feel.",
        "description_ar": "نقي وبارد قليلًا مع أبيض نظيف - طابع عصري ومعماري.",
        "params": {
            "retouch": 0.4,
            "warmth": -10, "contrast": 0.15, "highlights": -0.3, "shadows": 0.2, "whites": 0.3,
            "vibrance": 0.08, "saturation": -0.05, "skin_tone": 0.4, "sat_orange": -0.05,
            "clarity": 0.15, "bg_blur": 0.12, "bg_exposure": -0.1, "sharpen": 0.55,
        },
    },
    "moody": {
        "label_ar": "درامي داكن",
        "description": "Dark, dramatic and muted; the subject is lit, the surroundings fall away.",
        "description_ar": "داكن ودرامي بألوان هادئة؛ الشخص مضاء والمحيط يغمق.",
        "params": {
            "retouch": 0.4,
            "exposure": -0.3, "contrast": 0.3, "highlights": -0.55, "shadows": 0.05, "blacks": -0.15,
            "vibrance": -0.05, "saturation": -0.18, "skin_tone": 0.3,
            "grade_shadows_hue": 230, "grade_shadows_amount": 0.3,
            "bg_blur": 0.3, "bg_exposure": -0.7, "bg_saturation": -0.2, "subject_exposure": 0.2,
            "subject_clarity": 0.15, "clarity": 0.12, "vignette": 0.45, "sharpen": 0.45,
        },
    },
    "matte_film": {
        "label_ar": "فيلم مطفي",
        "description": "Faded film look: lifted blacks, softer colours, warm highlights and grain.",
        "description_ar": "طابع فيلم قديم: سواد مرفوع، ألوان هادئة، إضاءات دافئة وحبيبات.",
        "params": {
            "retouch": 0.4,
            "warmth": 8, "contrast": 0.08, "highlights": -0.35, "shadows": 0.2, "blacks": 0.45,
            "saturation": -0.15, "vibrance": 0.05, "skin_tone": 0.3,
            "grade_shadows_hue": 190, "grade_shadows_amount": 0.3,
            "grade_highlights_hue": 75, "grade_highlights_amount": 0.3,
            "hue_green": -15, "sat_green": -0.2, "bg_blur": 0.15, "bg_exposure": -0.1,
            "grain": 0.5, "sharpen": 0.25, "vignette": 0.15,
        },
    },
    "bw_classic": {
        "label_ar": "أبيض وأسود",
        "description": "Rich black & white with deep contrast; skin rendered bright, skies darker.",
        "description_ar": "أبيض وأسود غني بتباين عميق؛ البشرة مضيئة والسماء أغمق.",
        "params": {
            "retouch": 0.4,
            "monochrome": True, "contrast": 0.3, "highlights": -0.35, "shadows": 0.2, "blacks": -0.1,
            "bw_mix": {"red": 0.25, "orange": 0.35, "yellow": 0.1, "green": -0.1, "blue": -0.35},
            "bg_blur": 0.25, "bg_exposure": -0.3, "subject_exposure": 0.12, "subject_clarity": 0.2,
            "clarity": 0.15, "grain": 0.25, "vignette": 0.25, "sharpen": 0.5,
        },
    },
}


def list_presets() -> list[dict]:
    return [{"name": k, "label_ar": v["label_ar"], "description": v["description"],
             "description_ar": v["description_ar"]} for k, v in PRESETS.items()]


def resolve(preset: str | None = "natural", overrides: dict | None = None) -> dict:
    p = copy.deepcopy(DEFAULTS)
    if preset:
        if preset not in PRESETS:
            raise KeyError(f"unknown preset '{preset}'. Available: {', '.join(PRESETS)}")
        p.update(copy.deepcopy(PRESETS[preset]["params"]))
    if overrides:
        p.update(overrides)
    return p


def parse_value(v: str):
    s = v.strip()
    if s.lower() in ("true", "yes", "on"):
        return True
    if s.lower() in ("false", "no", "off"):
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if s.startswith("{"):
        import json
        return json.loads(s)
    return s
