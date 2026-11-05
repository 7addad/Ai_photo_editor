"""Categories (الفئات) + the adaptive engine.

A category describes *where the photo should end up* (wedding, landscape,
cinematic, food ...).  ``adapt()`` looks at the actual photo - its light,
exposure, contrast, colour palette, sky, haze, people, lines, noise - and
computes every Lightroom setting so that *this* photo reaches the category's
look, then renders small previews and corrects itself (2 passes) until
brightness, contrast, black/white points and colourfulness hit the targets.

Settings are Lightroom-scale (see sliders.py), so they can be shown to the
user, tweaked, and exported to Lightroom as XMP.
"""
from __future__ import annotations

import copy
import dataclasses

import numpy as np

# targets are OKLab lightness / chroma of the rendered photo:
#   mid = median L, black = 0.5th pct, white = 99.5th pct, std = global contrast,
#   chroma = mean chroma of coloured pixels.  None = keep what the photo has.
CATEGORIES: dict[str, dict] = {
    # ---------------------------------------------------------------- people
    "natural": dict(
        ar="طبيعي", group="عام", en="True-to-life colours and light; clean, neutral, nothing stylised.",
        targets=dict(mid=None, black=0.035, white=0.965, std=None, chroma=None),
        rules=dict(wb_keep=0.0, warmth=4, dehaze=0.6, sky=0.6, people=0.35, retouch=30, level=True, clarity=6,
                   sharpen=50, greens="natural", vignette=-6),
        base=dict(profile="color", vibrance=12, texture=4)),
    "portrait": dict(
        ar="بورتريه", group="أشخاص", en="Flattering portrait: natural warm skin, subject separated, soft background.",
        targets=dict(mid=None, black=0.04, white=0.96, std=0.2, chroma=0.075),
        rules=dict(wb_keep=0.1, warmth=8, dehaze=0.2, sky=0.4, people=1.0, retouch=50, level=True, clarity=0,
                   sharpen=45, masking=60, greens="muted", skin="warm", vignette=-15),
        base=dict(profile="portrait", vibrance=12, texture=-5, subject_clarity=12, lum_orange=6, sat_orange=-4,
                  cg_highlights_hue=40, cg_highlights_sat=6)),
    "wedding": dict(
        ar="أعراس ومناسبات", group="أشخاص", en="Bright & airy, soft contrast, clean whites, glowing skin.",
        targets=dict(mid=0.64, black=0.07, white=0.975, std=0.17, chroma=0.065),
        rules=dict(wb_keep=0.1, warmth=6, dehaze=0.0, sky=0.3, people=0.8, retouch=55, level=True, clarity=-4,
                   sharpen=40, masking=60, greens="fresh", skin="bright", vignette=-6),
        base=dict(profile="portrait", vibrance=8, saturation=-6, texture=-8, lum_orange=10, sat_orange=-8,
                  sat_yellow=-15, lum_yellow=8, curve_shadows=12, cg_highlights_hue=45, cg_highlights_sat=8,
                  cg_shadows_hue=210, cg_shadows_sat=4)),
    "fashion": dict(
        ar="أزياء وتحريري", group="أشخاص", en="Editorial: crisp, contrasty, slightly desaturated, cool shadows.",
        targets=dict(mid=0.55, black=0.02, white=0.97, std=0.23, chroma=0.065),
        rules=dict(wb_keep=0.0, warmth=0, dehaze=0.3, sky=0.5, people=0.7, retouch=55, level=True, clarity=8,
                   sharpen=60, masking=45, greens="muted", skin="natural", vignette=-12),
        base=dict(profile="standard", vibrance=-5, saturation=-8, texture=8, subject_clarity=10,
                  cg_shadows_hue=215, cg_shadows_sat=12, cg_highlights_hue=35, cg_highlights_sat=6)),
    "family": dict(
        ar="عائلي وأطفال", group="أشخاص", en="Cheerful, bright and warm with gentle skin.",
        targets=dict(mid=0.62, black=0.05, white=0.97, std=0.19, chroma=0.09),
        rules=dict(wb_keep=0.15, warmth=10, dehaze=0.2, sky=0.5, people=0.6, retouch=35, level=True, clarity=0,
                   sharpen=45, masking=50, greens="fresh", skin="warm", vignette=-8),
        base=dict(profile="color", vibrance=18, lum_orange=6, sat_orange=-3)),
    # ---------------------------------------------------------------- outdoors
    "landscape": dict(
        ar="طبيعة ومناظر", group="أماكن", en="Deep skies, rich greens, detail and depth, haze removed.",
        targets=dict(mid=None, black=0.025, white=0.97, std=0.22, chroma=0.1),
        rules=dict(wb_keep=0.1, warmth=3, dehaze=1.2, sky=1.0, people=0.1, retouch=0, level=True, clarity=14,
                   sharpen=65, masking=20, greens="vivid", vignette=-10, ca=True),
        base=dict(profile="landscape", vibrance=22, texture=12, lum_blue=-12, sat_blue=10)),
    "sunset": dict(
        ar="غروب وشروق", group="أماكن", en="Golden-hour glow kept and enriched: warm light, rich sky, deep silhouettes.",
        targets=dict(mid=0.5, black=0.02, white=0.97, std=0.22, chroma=0.12),
        rules=dict(wb_keep=0.65, warmth=14, dehaze=0.8, sky=1.0, sky_warm=25, people=0.3, retouch=25, level=True,
                   clarity=8, sharpen=55, greens="olive", vignette=-15),
        base=dict(profile="landscape", vibrance=22, sat_orange=10, sat_red=6, hue_red=6,
                  cg_highlights_hue=38, cg_highlights_sat=18, cg_shadows_hue=260, cg_shadows_sat=8)),
    "sea": dict(
        ar="بحر وشاطئ", group="أماكن", en="Clear turquoise water, bright sand, clean blue sky.",
        targets=dict(mid=0.6, black=0.04, white=0.97, std=0.2, chroma=0.11),
        rules=dict(wb_keep=0.0, warmth=2, dehaze=1.0, sky=1.0, people=0.3, retouch=25, level=True, clarity=8,
                   sharpen=55, greens="fresh", vignette=-5),
        base=dict(profile="landscape", vibrance=25, sat_aqua=20, hue_aqua=-6, lum_aqua=6, sat_blue=10,
                  lum_blue=-8, lum_yellow=8, sat_yellow=-6)),
    "travel": dict(
        ar="سفر", group="أماكن", en="Punchy, vivid and sharp - postcard colours with natural skin.",
        targets=dict(mid=None, black=0.03, white=0.97, std=0.22, chroma=0.11),
        rules=dict(wb_keep=0.15, warmth=5, dehaze=0.8, sky=0.8, people=0.3, retouch=25, level=True, clarity=12,
                   sharpen=60, greens="vivid", vignette=-10, ca=True),
        base=dict(profile="vivid", vibrance=25, texture=8, lum_blue=-10)),
    "night": dict(
        ar="ليل ومدينة", group="أماكن", en="Night city: deep blacks, clean lights, cool shadows, low noise.",
        targets=dict(mid=0.36, black=0.015, white=0.97, std=0.23, chroma=0.09),
        rules=dict(wb_keep=0.3, warmth=-4, dehaze=0.2, sky=0.2, people=0.3, retouch=20, level=True, clarity=6,
                   sharpen=40, masking=40, greens="muted", vignette=-12, night=True),
        base=dict(profile="standard", vibrance=18, highlights=-55, cg_shadows_hue=225, cg_shadows_sat=12,
                  sat_orange=6, sat_purple=10, sat_magenta=10)),
    "street": dict(
        ar="شارع ووثائقي", group="أماكن", en="Gritty documentary: strong contrast, muted colour, grain.",
        targets=dict(mid=0.5, black=0.02, white=0.95, std=0.24, chroma=0.055),
        rules=dict(wb_keep=0.25, warmth=0, dehaze=0.4, sky=0.4, people=0.0, retouch=0, level=True, clarity=18,
                   sharpen=55, greens="olive", vignette=-20),
        base=dict(profile="standard", vibrance=-10, saturation=-12, texture=10, grain=22, grain_size=25)),
    "architecture": dict(
        ar="عمارة", group="أماكن", en="Straight lines, neutral whites, crisp detail and clean skies.",
        targets=dict(mid=None, black=0.025, white=0.975, std=0.21, chroma=0.07),
        rules=dict(wb_keep=0.0, warmth=0, dehaze=0.8, sky=0.8, people=0.0, retouch=0, level=True, clarity=15,
                   sharpen=70, masking=15, greens="natural", vignette=0, ca=True),
        base=dict(profile="standard", vibrance=8, texture=12, lum_blue=-10)),
    "interior": dict(
        ar="ديكور وداخلي", group="أماكن", en="Bright, neutral interiors with open shadows and straight verticals.",
        targets=dict(mid=0.62, black=0.03, white=0.975, std=0.19, chroma=0.07),
        rules=dict(wb_keep=0.1, warmth=2, dehaze=0.3, sky=0.0, people=0.0, retouch=0, level=True, clarity=6,
                   sharpen=55, greens="natural", vignette=0, ca=True),
        base=dict(profile="color", vibrance=10, shadows=40, sat_yellow=-8, sat_orange=-4)),
    # ---------------------------------------------------------------- things
    "food": dict(
        ar="طعام", group="أشياء", en="Appetising: warm, rich reds/oranges, crisp texture, fresh greens.",
        targets=dict(mid=0.6, black=0.03, white=0.97, std=0.2, chroma=0.12),
        rules=dict(wb_keep=0.15, warmth=10, dehaze=0.2, sky=0.0, people=0.0, retouch=0, level=False, clarity=8,
                   sharpen=60, greens="fresh", vignette=-10),
        base=dict(profile="vivid", vibrance=20, texture=18, sat_red=8, sat_orange=6, lum_orange=4,
                  sat_green=6, lum_green=6)),
    "product": dict(
        ar="منتجات", group="أشياء", en="Clean commercial: pure whites, neutral colour, sharp edges.",
        targets=dict(mid=0.66, black=0.03, white=0.985, std=0.2, chroma=0.08),
        rules=dict(wb_keep=0.0, warmth=0, dehaze=0.0, sky=0.0, people=0.0, retouch=0, level=True, clarity=10,
                   sharpen=70, masking=20, greens="natural", vignette=0),
        base=dict(profile="standard", vibrance=10, texture=12, whites=10)),
    "wildlife": dict(
        ar="حيوانات وحياة برية", group="أشياء", en="Fur and feather detail, natural earthy colours, subject pops.",
        targets=dict(mid=None, black=0.03, white=0.97, std=0.21, chroma=0.09),
        rules=dict(wb_keep=0.1, warmth=4, dehaze=0.6, sky=0.6, people=0.0, retouch=0, level=False, clarity=10,
                   sharpen=70, masking=30, greens="natural", vignette=-15),
        base=dict(profile="landscape", vibrance=15, texture=22)),
    # ---------------------------------------------------------------- looks
    "cinematic": dict(
        ar="سينمائي", group="أنماط", en="Teal & orange film grade, soft highlights, deep but lifted blacks.",
        targets=dict(mid=0.48, black=0.05, white=0.93, std=0.2, chroma=0.07),
        rules=dict(wb_keep=0.2, warmth=2, dehaze=0.3, sky=0.5, people=0.5, retouch=35, level=True, clarity=6,
                   sharpen=45, greens="olive", vignette=-22),
        base=dict(profile="standard", vibrance=6, saturation=-8, hue_green=-25, sat_green=-25, hue_blue=-10,
                  sat_aqua=10, cg_shadows_hue=195, cg_shadows_sat=28, cg_highlights_hue=35, cg_highlights_sat=20,
                  cg_balance=-10, curve_highlights=-12, grain=12)),
    "moody": dict(
        ar="داكن درامي", group="أنماط", en="Dark, muted and dramatic; light falls on the subject.",
        targets=dict(mid=0.42, black=0.03, white=0.9, std=0.21, chroma=0.05),
        rules=dict(wb_keep=0.25, warmth=-2, dehaze=0.6, sky=0.8, people=0.7, retouch=35, level=True, clarity=14,
                   sharpen=50, greens="olive", vignette=-30),
        base=dict(profile="standard", vibrance=-8, saturation=-15, hue_green=-20, sat_green=-30, lum_green=-15,
                  sat_blue=-15, lum_blue=-15, cg_shadows_hue=220, cg_shadows_sat=14, curve_highlights=-10)),
    "bright_airy": dict(
        ar="مشرق وناعم", group="أنماط", en="Light, soft and pastel-clean; lifted shadows, gentle colour.",
        targets=dict(mid=0.7, black=0.1, white=0.975, std=0.15, chroma=0.055),
        rules=dict(wb_keep=0.1, warmth=5, dehaze=-0.0, sky=0.3, people=0.6, retouch=40, level=True, clarity=-6,
                   sharpen=40, masking=50, greens="fresh", vignette=0),
        base=dict(profile="portrait", vibrance=5, saturation=-10, texture=-5, lum_orange=8, sat_yellow=-12,
                  lum_green=12, sat_green=-15, curve_shadows=15, cg_highlights_hue=50, cg_highlights_sat=6)),
    "vintage_film": dict(
        ar="فيلم قديم", group="أنماط", en="Analogue film: faded blacks, warm highlights, shifted greens, grain.",
        targets=dict(mid=0.55, black=0.11, white=0.92, std=0.17, chroma=0.06),
        rules=dict(wb_keep=0.3, warmth=10, dehaze=-0.2, sky=0.3, people=0.3, retouch=25, level=True, clarity=-4,
                   sharpen=30, greens="olive", vignette=-15),
        base=dict(profile="standard", saturation=-12, vibrance=4, hue_green=-30, sat_green=-20, hue_blue=-15,
                  sat_blue=-15, hue_red=4, point_curve="0,28; 70,70; 190,196; 255,238",
                  cg_shadows_hue=180, cg_shadows_sat=14, cg_highlights_hue=45, cg_highlights_sat=16,
                  grain=35, grain_size=35, grain_roughness=60, red_hue=4, blue_sat=-10)),
    "pastel": dict(
        ar="باستيل", group="أنماط", en="Dreamy pastel: very soft contrast, lifted shadows, candy colours.",
        targets=dict(mid=0.68, black=0.13, white=0.96, std=0.14, chroma=0.05),
        rules=dict(wb_keep=0.1, warmth=4, dehaze=-0.1, sky=0.3, people=0.4, retouch=35, level=True, clarity=-10,
                   sharpen=35, greens="fresh", vignette=0),
        base=dict(profile="portrait", saturation=-15, vibrance=10, hue_green=15, lum_green=15, sat_blue=-10,
                  lum_blue=15, hue_blue=-15, lum_red=10, cg_shadows_hue=290, cg_shadows_sat=10,
                  cg_highlights_hue=50, cg_highlights_sat=10, curve_shadows=25)),
    "hdr": dict(
        ar="تفاصيل عالية HDR", group="أنماط", en="Everything visible: recovered skies, open shadows, strong local detail.",
        targets=dict(mid=0.55, black=0.03, white=0.95, std=0.17, chroma=0.1),
        rules=dict(wb_keep=0.1, warmth=2, dehaze=1.0, sky=1.0, people=0.0, retouch=0, level=True, clarity=35,
                   sharpen=65, masking=20, greens="vivid", vignette=-8),
        base=dict(profile="landscape", highlights=-90, shadows=75, texture=25, vibrance=22)),
    "bw": dict(
        ar="أبيض وأسود", group="أنماط", en="Rich black & white tuned to the photo's colours, deep contrast, fine grain.",
        targets=dict(mid=0.52, black=0.02, white=0.97, std=0.24, chroma=None),
        rules=dict(wb_keep=0.0, warmth=0, dehaze=0.6, sky=0.8, people=0.5, retouch=35, level=True, clarity=18,
                   sharpen=55, greens="natural", vignette=-18),
        base=dict(monochrome=True, texture=10, grain=15)),
}

# special modes that are not adapted but are offered alongside the categories
SPECIAL = {
    "beauty_only": "تجميل فقط - الصورة كما هي ويُعدَّل الوجه فقط",
    "standard": "تعديل عادي - إعدادات كلاسيكية ثابتة تضبطها أنت بالأشرطة",
}

LIMITS = {"exposure": (-3, 3), "contrast": (-60, 60), "highlights": (-100, 100), "shadows": (-100, 100),
          "whites": (-60, 60), "blacks": (-60, 60), "vibrance": (-60, 60), "saturation": (-60, 40),
          "dehaze": (-40, 50), "clarity": (-40, 50), "texture": (-40, 50)}


# category chroma targets are written as "mean chroma of coloured pixels";
# the measured metric is the 90th percentile, ~1.7x larger on real photos
CHROMA_SCALE = 1.7
_AR_KEYS = {"exposure": "تعريض", "contrast": "تباين", "blacks": "أسود", "whites": "أبيض", "vibrance": "حيوية"}


def list_categories() -> list[dict]:
    out = [{"category": k, "ar": v["ar"], "group": v["group"], "description": v["en"]} for k, v in CATEGORIES.items()]
    out += [{"category": k, "ar": v.split(" - ")[0], "group": "خاص", "description": v} for k, v in SPECIAL.items()]
    return out


def _clamp(s: dict):
    for k, (lo, hi) in LIMITS.items():
        if k in s:
            s[k] = float(np.clip(s[k], lo, hi))
    for k, v in list(s.items()):
        if isinstance(v, float):
            s[k] = round(v, 2) if k.endswith("exposure") else round(v, 1)


def _add(s, k, v):
    s[k] = s.get(k, 0) + v


# ================================================================ rules

def initial_settings(stats: dict, category: str, subject: str | None = None) -> tuple[dict, list[str]]:
    """First guess of every setting from the scene measurements."""
    if category not in CATEGORIES:
        raise KeyError(f"unknown category '{category}'. Available: {', '.join(CATEGORIES)}")
    cat = CATEGORIES[category]
    r = cat["rules"]
    s = copy.deepcopy(cat["base"])
    why: list[str] = []
    subject = subject or stats["subject_guess"]
    tone, pal, sky, ppl = stats["tone"], stats["palette_pct"], stats["sky"], stats["people"]
    T = stats["light"]["cct"]

    # ---- white balance: neutralise, then give back part of the ambience if the category wants it
    s["wb"] = "auto"
    keep = r.get("wb_keep", 0.0)
    warmth = r.get("warmth", 0)
    amb = (1e6 / T - 1e6 / 5500) / 0.4           # warmth-slider units of the light's colour vs daylight
    if keep and abs(amb) > 5:
        warmth += keep * amb
        why.append(f"توازن الأبيض: الإضاءة ~{T}K، صُححت إلى محايد مع إبقاء {int(keep * 100)}% من جوّها "
                   f"({'دافئ' if amb > 0 else 'بارد'}) لأن فئة {cat['ar']} تحتاجه")
    else:
        why.append(f"توازن الأبيض: الإضاءة ~{T}K ({stats['light']['type']}) صُححت إلى أبيض طبيعي")
    if warmth:
        s["warmth"] = round(warmth, 1)

    # ---- exposure & tone
    if stats["night"] or r.get("night"):
        cut = -min(1.6, 0.45 * max(0.0, stats["auto_ev"] - 1.0))
        if cut < -0.05:
            _add(s, "exposure", cut)
            why.append(f"تعريض: صورة ليلية - خُفض {abs(cut):.1f} EV عن القياس التلقائي ليبقى الليل ليلًا")
    hl = -(15 + 45 * min(1.0, tone["clip_pct"] / 2.0) + (30 if (sky["L"] or 0) > 0.85 else 0))
    if "highlights" not in s:
        s["highlights"] = round(hl * (1.15 if r.get("sky", 0) > 0.7 else 0.9))
        why.append(f"الإضاءات العالية {s['highlights']}: مناطق قريبة من الاحتراق {tone['clip_pct']:.1f}%"
                   + (" وسماء ساطعة" if (sky["L"] or 0) > 0.85 else ""))
    if "shadows" not in s:
        dark = max(0.0, 0.5 - tone["L_median"]) * 2 + min(1.0, tone["crush_pct"] / 3)
        s["shadows"] = round(10 + 45 * min(1.0, dark))
        why.append(f"الظلال +{s['shadows']}: لفتح التفاصيل في المناطق الداكنة")

    # ---- presence
    s["clarity"] = s.get("clarity", 0) + r.get("clarity", 0)
    if stats["haze"] > 0.07 and r.get("dehaze", 0) > 0:
        dh = float(np.clip((stats["haze"] - 0.05) * 260, 0, 40)) * r["dehaze"]
        if dh >= 3:
            s["dehaze"] = round(dh)
            why.append(f"إزالة الضباب +{s['dehaze']}: الصورة فيها ضباب/غبش (مؤشر {stats['haze']:.2f})")
    elif r.get("dehaze", 0) < 0:
        s["dehaze"] = round(r["dehaze"] * 30)

    # ---- palette-aware colour mixing
    if pal["green"] + pal["yellow"] > 25:
        g = r.get("greens", "natural")
        mix = {"vivid": dict(hue_green=-8, sat_green=10, lum_green=4, sat_yellow=6),
               "fresh": dict(hue_green=6, sat_green=-6, lum_green=10, hue_yellow=-5),
               "muted": dict(sat_green=-15, lum_green=-4, sat_yellow=-8),
               "olive": dict(hue_green=-18, sat_green=-20, lum_green=-8, hue_yellow=-8),
               "natural": dict(hue_green=-4, sat_green=2)}[g]
        for k, v in mix.items():
            if k not in cat["base"]:
                _add(s, k, v)
        why.append(f"الأخضر يشكل {pal['green'] + pal['yellow']:.0f}% من ألوان الصورة - ضُبط بأسلوب '{g}' المناسب للفئة")
    if sky["pct"] > 4 and r.get("sky", 0) > 0:
        k = r["sky"]
        bright = max(0.0, (sky["L"] or 0) - 0.72)
        s["sky_exposure"] = round(-(0.1 + bright * 2.0) * k, 2)
        if (sky["blue"] or 0) > 0.015:
            s["sky_saturation"] = round(18 * k)
        if r.get("sky_warm"):
            s["sky_warmth"] = r["sky_warm"]
        if "lum_blue" not in cat["base"]:
            _add(s, "lum_blue", -8 * k)
        why.append(f"السماء تغطي {sky['pct']:.0f}% - خُفضت إضاءتها {s['sky_exposure']} EV"
                   + (f" وزيد تشبعها +{s.get('sky_saturation')}" if s.get("sky_saturation") else "")
                   + " لإظهار السحب واللون")
    if pal["aqua"] + pal["blue"] > 35 and subject in ("sea", "landscape", "travel"):
        _add(s, "sat_aqua", 6)

    # ---- people
    has_people = ppl["count"] > 0 or ppl["area_pct"] > 1
    pr = r.get("people", 0.0)
    if has_people and pr > 0 and ppl["area_pct"] > 2:
        ratio = ppl.get("background_sharpness_ratio") or 1.0
        blur = round(55 * pr * float(np.clip(ratio / 0.6, 0.25, 1.0)))
        if blur >= 5 and "background_blur" not in s:
            s["background_blur"] = blur
        s.setdefault("background_exposure", round(-0.3 * pr, 2))
        s.setdefault("background_saturation", round(-12 * pr))
        sl, bl = ppl.get("subject_L"), ppl.get("background_L")
        sub_ev = 0.08 * pr
        if sl is not None and bl is not None and bl - sl > 0.04:
            sub_ev += float(np.clip((bl - sl) * 2.5, 0, 0.6))
            why.append("الشخص أغمق من الخلفية (إضاءة خلفية) - رُفعت إضاءته وحده")
        s["subject_exposure"] = round(sub_ev, 2)
        why.append(f"فصل الشخص عن الخلفية: تمويه {s.get('background_blur', 0)}"
                   + (" (الخلفية أصلًا ناعمة فخُفف التمويه)" if ratio < 0.35 else "")
                   + f"، إضاءة الخلفية {s['background_exposure']} EV")
    else:
        s["separate_subject"] = False
    if ppl["count"] > 0 and ppl["largest_face_pct"] > 3 and r.get("retouch", 0) > 0:
        s["retouch"] = r["retouch"]
        why.append(f"تجميل الوجه بقوة {r['retouch']} (إزالة الحبوب والاحمرار وتوحيد البشرة)")
    sc = ppl.get("skin_chroma")
    if sc is not None and not cat["base"].get("monochrome"):
        if sc > 0.105:
            _add(s, "sat_orange", -10)
            _add(s, "sat_red", -5)
            why.append("البشرة مشبعة/محمرة أكثر من اللازم - خُفض تشبع البرتقالي والأحمر")
        elif sc < 0.055:
            _add(s, "sat_orange", 6)
            why.append("البشرة باهتة - زيد تشبع البرتقالي قليلًا")

    # ---- B&W mix from the palette: skin bright, sky dark, foliage according to share
    if cat["base"].get("monochrome"):
        s.update(bw_red=15, bw_orange=25 if has_people else 10, bw_yellow=10,
                 bw_green=-10 if pal["green"] > 20 else 5, bw_aqua=-15, bw_blue=-30 if sky["pct"] > 4 else -10)
        why.append("مزج الأبيض والأسود: البشرة أفتح والسماء أغمق حسب ألوان الصورة")

    # ---- detail
    iso = stats["iso"]
    s["sharpening"] = r.get("sharpen", 50)
    s["sharpen_masking"] = r.get("masking", 25)
    if iso >= 1600 or stats["night"]:
        s["nr_luminance"] = round(float(np.clip(15 + 12 * np.log2(max(iso, 800) / 800), 15, 70)))
        s["nr_color"] = 40
        s["sharpening"] = round(s["sharpening"] * 0.75)
        why.append(f"ISO {iso:g}: إزالة تشويش الإضاءة {s['nr_luminance']} وخُففت الحدة")
    if r.get("ca") and stats["is_raw"]:
        s["remove_ca"] = True

    # ---- geometry
    tilt = stats["tilt"]
    if r.get("level") and tilt["confidence"] >= 0.4 and 0.3 <= abs(tilt["deg"]) <= 6:
        s["auto_level"] = True
        why.append(f"الأفق/الخطوط مائلة {tilt['deg']:+.1f}° - تسوية تلقائية")

    # ---- effects
    if "vignette" not in s and r.get("vignette"):
        s["vignette"] = r["vignette"]

    # ---- subject hint (from Claude's own look at the photo) - small, targeted extras
    extra = {"food": dict(texture=6, warmth=4), "product": dict(whites=5), "animal": dict(texture=8),
             "architecture": dict(clarity=5), "interior": dict(shadows=10), "portrait": dict(clarity=-4),
             "night": dict(), "sea": dict(sat_aqua=6), "landscape": dict(texture=4)}.get(subject, {})
    for k, v in extra.items():
        _add(s, k, v)
    _clamp(s)
    return s, why


# ================================================================ adaptive refinement

def _measure(data, settings, seg):
    from .pipeline import process
    from .presets import resolve
    from .scene import render_stats
    from .sliders import sliders_to_params
    res = process(data, resolve(None, sliders_to_params(settings)), seg=seg)
    return render_stats(res.srgb), res


def _skin_high(res) -> float | None:
    """95th percentile OKLab L of the skin in a render (None without people)."""
    seg = res.seg
    if seg is None or not seg.has_people or seg.skin.max() < 0.5:
        return None
    from .color import linear_to_oklab, srgb_to_linear
    m = seg.skin > 0.5
    if m.shape != res.srgb.shape[:2] or m.sum() < 30:
        return None
    L = linear_to_oklab(srgb_to_linear(res.srgb[m][None]))[0, :, 0]
    return float(np.percentile(L, 95))


def adapt(data, category: str, subject: str | None = None, seg=None, overrides: dict | None = None,
          passes: int = 3, stats: dict | None = None) -> dict:
    """Full adaptive edit plan for one photo.  Returns
    {category, subject, settings, why, scene, check}.  Rendering of the final
    image is left to the caller (``process`` with the settings)."""
    from .pipeline import _downscale, segment
    from .scene import analyze_scene

    preview = dataclasses.replace(data, linear=_downscale(data.linear, 640))
    pseg = segment(preview) if seg is None else seg
    if seg is not None:
        import cv2
        h, w = preview.linear.shape[:2]
        sx = w / seg.person.shape[1]
        pseg = dataclasses.replace(seg, person=cv2.resize(seg.person, (w, h), interpolation=cv2.INTER_AREA),
                                   skin=cv2.resize(seg.skin, (w, h), interpolation=cv2.INTER_AREA),
                                   faces=[(x * sx, y * sx, a * sx, b * sx, c) for x, y, a, b, c in seg.faces])
    stats = stats or analyze_scene(preview, pseg)
    settings, why = initial_settings(stats, category, subject)
    t = CATEGORIES[category]["targets"]
    overrides = overrides or {}
    settings.update(overrides)

    # geometry is evaluated on the full image later; keep preview passes fast
    geo_keys = ("auto_level", "rotate", "vertical", "horizontal", "scale", "crop_aspect", "crop_zoom",
                "remove_ca", "lens_distortion")
    check = {}
    # metric <- setting, prior slope (metric change per slider unit), max step per pass
    # metric <- setting, response (metric change per slider unit, calibrated on real
    # renders; chroma responds proportionally to the colour already present), max step
    controls = [("L_median", "exposure", 0.15, 0.6), ("L_std", "contrast", 0.0010, 30),
                ("L_p01", "blacks", 0.0016, 30), ("L_p99", "whites", 0.0019, 30),
                ("chroma", "vibrance", None, 25)]
    for i in range(passes):
        trial = {k: v for k, v in settings.items() if k not in geo_keys}
        m, res = _measure(preview, trial, pseg)
        skin_hi = _skin_high(res)
        if skin_hi is not None:
            m["skin_p95"] = skin_hi
        check[f"pass{i + 1}"] = {k: round(m[k], 3) for k in ("L_median", "L_p01", "L_p99", "L_std", "chroma", "clip_pct", "skin_p95") if k in m}
        goal = {"L_median": t["mid"], "L_std": t["std"], "L_p01": t["black"], "L_p99": t["white"], "chroma": t["chroma"]}
        if goal["L_std"] is None and not (0.15 <= m["L_std"] <= 0.27):
            goal["L_std"] = 0.21
        if goal["chroma"] is not None:
            goal["chroma"] = goal["chroma"] * CHROMA_SCALE
            # never push colour more than +60% / -40% of what the photo has: no garish results
            goal["chroma"] = float(np.clip(goal["chroma"], m["chroma"] * 0.6, m["chroma"] * 1.6))
        if m["clip_pct"] > 1.0 and goal["L_p99"] is not None:
            goal["L_p99"] = min(goal["L_p99"], m["L_p99"] - 0.01)
        notes = []
        for metric, key, prior, cap in controls:
            g = goal[metric]
            if g is None or key in overrides:
                continue
            err = g - m[metric]
            tol = 0.012 if metric != "chroma" else 0.004
            if abs(err) < tol:
                continue
            slope = prior if prior is not None else 0.008 * max(m["chroma"], 0.02)
            step = float(np.clip(err / slope * 0.75, -cap, cap))
            if abs(step) < (0.04 if key == "exposure" else 1.5):
                continue
            _add(settings, key, step)
            notes.append(f"{_AR_KEYS[key]} {step:+.2f}" if key == "exposure" else f"{_AR_KEYS[key]} {step:+.0f}")
        # faces must never burn out: pull exposure/whites back if skin highlights clip
        if m.get("skin_p95", 0) > 0.9 and "exposure" not in overrides:
            d = -float(np.clip((m["skin_p95"] - 0.87) / 0.12, 0.05, 0.6))
            _add(settings, "exposure", d)
            _add(settings, "whites", -8)
            if settings.get("subject_exposure", 0) > 0.05:
                settings["subject_exposure"] = round(settings["subject_exposure"] * 0.5, 2)
            notes.append(f"حماية بشرة الوجه من الاحتراق: تعريض {d:+.2f}")
        _clamp(settings)
        if notes:
            why.append(f"معايرة تلقائية (جولة {i + 1}) لتصل الصورة لشكل الفئة: " + "، ".join(notes))
        else:
            break
    m, _ = _measure(preview, {k: v for k, v in settings.items() if k not in geo_keys}, pseg)
    check["final"] = {k: round(m[k], 3) for k in ("L_median", "L_p01", "L_p99", "L_std", "chroma", "clip_pct")}
    check["targets"] = t
    return {"category": category, "category_ar": CATEGORIES[category]["ar"],
            "subject": subject or stats["subject_guess"], "settings": settings, "why": why,
            "scene": stats, "check": check}
