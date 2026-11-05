"""Scene understanding for adaptive editing: measures the photo's light,
colours, palette, sky, haze, people, lines and noise, and guesses the subject
(portrait / group / landscape / architecture / night / food ...).

Claude also looks at the photo itself; its own description can be passed as a
subject hint and overrides the guess.
"""
from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from . import adjustments as adj
from .color import linear_to_oklab, linear_to_srgb, srgb_to_linear
from .raw_io import ImageData, iso_value

BANDS = {"red": 20, "orange": 55, "yellow": 95, "green": 140, "aqua": 195, "blue": 255, "purple": 300, "magenta": 340}
SUBJECTS = ("portrait", "group", "landscape", "sea", "sunset", "architecture", "interior", "night", "street",
            "food", "product", "animal", "event", "general")


def render_stats(srgb: np.ndarray) -> dict:
    """Tone & colour measurements of a display image (used for every pass)."""
    small = cv2.resize(srgb, (min(512, srgb.shape[1]), max(1, round(min(512, srgb.shape[1]) * srgb.shape[0] / srgb.shape[1]))),
                       interpolation=cv2.INTER_AREA)
    lab = linear_to_oklab(srgb_to_linear(small))
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    col = C > 0.025
    return {
        "L_median": float(np.median(L)), "L_mean": float(L.mean()),
        "L_p01": float(np.percentile(L, 0.5)), "L_p99": float(np.percentile(L, 99.5)),
        "L_std": float(L.std()),
        # colour strength: 90th percentile of chroma (stable when pixels gain colour)
        "chroma": float(np.percentile(C, 90)),
        "colourful_pct": float(col.mean() * 100),
        "clip_pct": float((small.max(-1) > 0.995).mean() * 100),
        "crush_pct": float((small.max(-1) < 0.01).mean() * 100),
        "warm_cool": float(np.mean(b[C < 0.06])) if (C < 0.06).any() else 0.0,   # + warm cast, - cool
    }


def _palette(lab: np.ndarray) -> dict:
    a, b = lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a)) % 360
    w = np.clip((C - 0.02) / 0.1, 0, 1)
    tot = w.sum() + 1e-6
    out = {}
    for name, c in BANDS.items():
        d = np.abs((h - c + 180) % 360 - 180)
        out[name] = round(float((w * np.exp(-(d / 25) ** 2)).sum() / tot * 100), 1)
    return out


def _sharpness(gray: np.ndarray, mask: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(np.sqrt(np.mean(lap[mask] ** 2))) if mask.sum() > 50 else 0.0


def analyze_scene(data: ImageData, seg=None) -> dict:
    from . import lr_tools as lrt
    from .pipeline import _downscale, auto_black_point, segment, suggested_ev
    from .white_balance import estimate_illuminant, white_balance_gains

    small = dataclasses.replace(data, linear=_downscale(data.linear, 900))
    h, w = small.linear.shape[:2]
    if seg is None:
        seg = segment(small)
        person, skin, faces = seg.person, seg.skin, seg.faces
    else:
        person = cv2.resize(seg.person, (w, h), interpolation=cv2.INTER_AREA)
        skin = cv2.resize(seg.skin, (w, h), interpolation=cv2.INTER_AREA)
        sx = w / seg.person.shape[1]
        faces = [(x * sx, y * sx, fw * sx, fh * sx, c) for x, y, fw, fh, c in seg.faces]
    has_people = bool(seg.has_people)
    pm = person if has_people else None
    est = estimate_illuminant(small.linear, pm, skin if has_people else None)
    gains, _ = white_balance_gains({"wb_mode": "auto"}, small, est)
    lin = small.linear * gains
    ev = suggested_ev(small, lin, pm)
    lin = lin * np.float32(2 ** ev)
    disp = adj.tone_map(lin, scene_referred=data.is_raw)
    if data.is_raw:
        disp = adj.apply_look(disp, {"_base_contrast": 0.3, "_black_point": auto_black_point(disp)})
    srgb = linear_to_srgb(disp)
    lab = linear_to_oklab(disp)
    L = lab[..., 0]
    st = render_stats(srgb)

    # sky, haze, lines
    sky = lrt.sky_mask(srgb, pm)
    sky_frac = float(sky.mean())
    sky_m = sky > 0.5
    dark = cv2.erode(srgb.min(-1), np.ones((9, 9), np.uint8))
    haze = float(np.median(dark[~sky_m])) if (~sky_m).sum() > 100 else float(np.median(dark))
    tilt, tilt_conf, n_lines = lrt.detect_tilt(srgb)
    gray = cv2.cvtColor(srgb.astype(np.float32), cv2.COLOR_RGB2GRAY)

    # people
    face_w = max((f[2] for f in faces), default=0) / w
    people = {"count": len(faces), "area_pct": round(100 * float(person.mean()), 1) if has_people else 0.0,
              "largest_face_pct": round(100 * face_w, 1)}
    if has_people:
        pmask, bmask = person > 0.5, person < 0.2
        people["subject_L"] = round(float(L[pmask].mean()), 3) if pmask.any() else None
        people["background_L"] = round(float(L[bmask].mean()), 3) if bmask.any() else None
        people["background_sharpness_ratio"] = round(_sharpness(gray, bmask) / max(_sharpness(gray, pmask), 1e-4), 2)
        sk = skin > 0.5
        if sk.sum() > 30:
            people["skin_chroma"] = round(float(np.hypot(lab[..., 1], lab[..., 2])[sk].mean()), 3)
            people["skin_L"] = round(float(L[sk].mean()), 3)

    iso = iso_value(data.meta)
    bright_points = float((L > 0.92).mean() * 100)
    night = (st["L_median"] < 0.38 and ev > 1.0) or (iso >= 3200 and ev > 2.0) or \
            (data.is_raw and ev > 3.0 and bright_points > 0.05)
    pal = _palette(lab)
    warm_light = est["cct"] < 4200

    # subject guess
    if people["count"] >= 3:
        guess = "group"
    elif has_people and (face_w > 0.07 or people["area_pct"] > 12):
        guess = "portrait"
    elif night:
        guess = "night"
    elif sky_frac > 0.12 and (pal["green"] + pal["yellow"] + pal["blue"] + pal["aqua"]) > 45:
        guess = "sunset" if (pal["orange"] + pal["red"] + pal["yellow"] > 45 and warm_light) else \
            ("sea" if pal["aqua"] + pal["blue"] > 45 and sky_frac > 0.2 else "landscape")
    elif n_lines >= 10 and sky_frac < 0.4:
        guess = "architecture" if sky_frac > 0.03 else "interior"
    elif has_people:
        guess = "street"
    elif pal["orange"] + pal["red"] + pal["yellow"] > 60 and st["chroma"] > 0.08 and sky_frac < 0.02:
        guess = "food"
    else:
        guess = "general"

    return {
        "is_raw": data.is_raw, "iso": iso,
        "light": {"cct": est["cct"], "duv": est["duv"], "type": est["light"], "confidence": est["confidence"]},
        "auto_ev": round(ev, 2),
        "tone": {k: round(v, 3) for k, v in st.items()},
        "palette_pct": pal,
        "sky": {"pct": round(100 * sky_frac, 1),
                "L": round(float(L[sky_m].mean()), 3) if sky_m.any() else None,
                "blue": round(float(-lab[..., 2][sky_m].mean()), 3) if sky_m.any() else None},
        "haze": round(haze, 3),
        "tilt": {"deg": round(tilt, 2), "confidence": round(tilt_conf, 2), "lines": n_lines},
        "people": people, "night": bool(night),
        "orientation": "portrait" if h > w * 1.05 else ("square" if abs(h - w) <= w * 0.05 else "landscape"),
        "subject_guess": guess,
    }
