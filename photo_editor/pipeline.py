"""The full editing pipeline: RAW/JPEG in -> analysed, segmented, white
balanced, toned, graded and sharpened sRGB out."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import adjustments as adj
from .color import linear_to_srgb, luminance, srgb_to_linear
from .raw_io import ImageData, iso_value
from .segmentation import Segmentation, segment_people
from .white_balance import estimate_illuminant, white_balance_gains


@dataclass
class Result:
    srgb: np.ndarray                       # final display-referred image 0..1
    params: dict
    report: dict = field(default_factory=dict)
    seg: Segmentation | None = None


def _quick_render(lin: np.ndarray, gains, ev: float) -> np.ndarray:
    disp = adj.tone_map(lin * np.asarray(gains, np.float32) * np.float32(2.0 ** ev))
    return linear_to_srgb(adj.apply_look(disp, {"_base_contrast": 0.3, "_black_point": auto_black_point(disp)}))


def render_before(data: ImageData) -> np.ndarray:
    """What the file looks like 'as shot': camera WB, auto exposure only."""
    if not data.is_raw:
        return linear_to_srgb(np.clip(data.linear, 0, 1))
    ev = adj.auto_exposure_ev(_downscale(data.linear, 1024) * data.as_shot_gains)
    return _quick_render(data.linear, data.as_shot_gains, ev)


def suggested_ev(data: ImageData, lin: np.ndarray, person=None) -> float:
    """Auto exposure.  RAW: full metering.  JPEG/PNG were already exposed by
    the camera, so only clearly wrong exposure is corrected, gently."""
    ev = adj.auto_exposure_ev(lin, person)
    if data.is_raw:
        return ev
    # display-referred files: the log-average of a good JPEG sits lower than
    # scene mid-grey, so allow a wide dead-band and correct only half of it
    ev = ev - 1.0
    if abs(ev) < 0.8:
        return 0.0
    return float(np.clip((ev - np.sign(ev) * 0.8) * 0.5, -1.0, 1.0))


def auto_black_point(disp: np.ndarray) -> float:
    """OKLab-L black point that clips the darkest ~0.05% (removes flat, hazy
    blacks of linear RAW renders) - capped so real shadow detail survives."""
    from .color import linear_to_oklab
    L = linear_to_oklab(_downscale(disp, 800))[..., 0]
    return float(np.clip(np.percentile(L, 0.05) * 0.85, 0.0, 0.12))


def auto_denoise(data: ImageData) -> tuple[float, float]:
    if not data.is_raw:
        return 0.0, 0.0
    iso = iso_value(data.meta)
    pts = [(100, 0.0, 0.15), (400, 0.0, 0.25), (800, 0.12, 0.4), (1600, 0.25, 0.6),
           (3200, 0.4, 0.8), (6400, 0.55, 1.0), (12800, 0.7, 1.0), (51200, 0.85, 1.0)]
    isos = [p[0] for p in pts]
    l = float(np.interp(np.log2(max(iso, 50)), np.log2(isos), [p[1] for p in pts]))
    c = float(np.interp(np.log2(max(iso, 50)), np.log2(isos), [p[2] for p in pts]))
    return l, c


def _downscale(a: np.ndarray, side: int) -> np.ndarray:
    h, w = a.shape[:2]
    s = side / max(h, w)
    if s >= 1:
        return a
    return cv2.resize(a, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def segment(data: ImageData) -> Segmentation:
    """Run people segmentation on a quick, roughly balanced rendering.  Models
    run at ~1280px; the guided-filter refinement uses a 2048px guide and the
    final mask is upsampled to full resolution."""
    H, W = data.linear.shape[:2]
    small = _downscale(data.linear, 2048)
    if data.is_raw:
        ev = adj.auto_exposure_ev(small * data.as_shot_gains)
        disp = _quick_render(small, data.as_shot_gains, ev)
    else:
        disp = linear_to_srgb(np.clip(small, 0, 1))
    seg = segment_people(disp)
    if (H, W) != seg.person.shape:
        sx, sy = W / seg.person.shape[1], H / seg.person.shape[0]
        seg.person = np.clip(cv2.resize(seg.person, (W, H), interpolation=cv2.INTER_LINEAR), 0, 1)
        seg.skin = np.clip(cv2.resize(seg.skin, (W, H), interpolation=cv2.INTER_LINEAR), 0, 1)
        seg.faces = [(round(x * sx), round(y * sy), round(w * sx), round(h * sy), c) for x, y, w, h, c in seg.faces]
    return seg


def process(data: ImageData, params: dict, seg: Segmentation | None = None,
            need_seg: bool = True) -> Result:
    t0 = time.time()
    p = dict(params)
    rep: dict = {"input": data.meta}

    # 1. people / skin
    use_people = need_seg and (p.get("separate_subject", True) or p.get("wb_mode") == "auto")
    if seg is None and use_people:
        seg = segment(data)
    person = seg.person if seg is not None and seg.has_people else None
    skin = seg.skin if seg is not None and seg.has_people else None
    if seg is not None:
        rep["people"] = {"detected": bool(seg.has_people), "faces": seg.people_count,
                         "person_area_pct": round(100 * seg.person_fraction, 2), "backend": seg.backend}
    t_seg = time.time()

    # 2. white balance (estimated on the daylight rendering, people excluded)
    est = estimate_illuminant(data.linear, person, skin) if p.get("wb_mode", "auto") == "auto" else None
    gains, wb_rep = white_balance_gains(p, data, est)
    rep["white_balance"] = {"estimate": {k: v for k, v in (est or {}).items() if k != "e"}, **wb_rep}
    if data.is_raw:
        from .color import illuminant_to_cct
        T, duv = illuminant_to_cct(1.0 / np.asarray(data.as_shot_gains, np.float64))
        rep["white_balance"]["as_shot_cct"] = round(T)
    lin = data.linear * gains

    # 3. noise reduction (linear)
    dn = p.get("denoise", "auto")
    if dn == "auto":
        dl, dc = auto_denoise(data)
    else:
        dl = dc = float(dn)
    lin = adj.denoise(lin, dl, dc)
    rep["denoise"] = {"luma": round(dl, 2), "chroma": round(dc, 2)}

    # 4. exposure
    ev = suggested_ev(data, lin, person) if p.get("auto_exposure", True) else 0.0
    ev += float(p.get("exposure", 0.0))
    lin = lin * np.float32(2.0 ** ev)
    rep["exposure_ev"] = round(ev, 2)

    # 5. local tone
    lin = adj.local_tone(lin, float(p.get("highlights", 0)), float(p.get("shadows", 0)))

    # 6. subject / background separation
    if person is not None and p.get("separate_subject", True):
        lin = adj.separate_subject(lin, person, p)
        rep["separation"] = {k: p.get(k) for k in ("bg_blur", "bg_exposure", "bg_saturation",
                                                   "subject_exposure", "subject_clarity", "skin_smooth")}

    # 7. vignette + tone map
    lin = adj.vignette(lin, float(p.get("vignette", 0)))
    disp = adj.tone_map(lin, white=float(p.get("tone_white", 8.0)), scene_referred=data.is_raw)
    if data.is_raw:
        p["_base_contrast"] = float(p.get("base_contrast", 0.3))
        p["_black_point"] = auto_black_point(disp) if p.get("auto_black", True) else 0.0
        rep["base_tone"] = {"contrast": p["_base_contrast"], "black_point": round(p["_black_point"], 4)}

    # 8. look (OKLab)
    sep_person = person if (person is not None and p.get("separate_subject", True)) else None
    disp = adj.apply_look(disp, p, skin=skin, person=sep_person)
    srgb = linear_to_srgb(disp)

    # 9. detail
    if skin is not None:
        srgb = adj.skin_smooth(srgb, skin, float(p.get("skin_smooth", 0)))
    H, W = srgb.shape[:2]
    amt = np.full((H, W), float(p.get("clarity", 0)), np.float32)
    if sep_person is not None:
        amt = amt + float(p.get("subject_clarity", 0)) * sep_person + float(p.get("bg_clarity", 0)) * (1 - sep_person)
    if np.abs(amt).max() > 1e-3:
        srgb = adj.local_contrast(srgb, 1.0, max(H, W) * 0.012, mask=amt)
    if abs(float(p.get("texture", 0))) > 1e-3:
        srgb = adj.local_contrast(srgb, float(p["texture"]), max(H, W) * 0.002)
    srgb = adj.sharpen(srgb, float(p.get("sharpen", 0)), radius=max(0.7, max(H, W) / 5000))
    srgb = adj.film_grain(srgb, float(p.get("grain", 0)))

    rep["timing_s"] = {"segmentation": round(t_seg - t0, 2), "total": round(time.time() - t0, 2)}
    rep["output_stats"] = image_stats(srgb)
    return Result(srgb=np.clip(srgb, 0, 1).astype(np.float32), params=p, report=rep, seg=seg)


def image_stats(srgb: np.ndarray) -> dict:
    small = cv2.resize(srgb, (400, max(1, round(400 * srgb.shape[0] / srgb.shape[1]))), interpolation=cv2.INTER_AREA)
    lin = srgb_to_linear(small)
    Y = luminance(lin)
    return {
        "mean_brightness": round(float(small.mean()), 3),
        "clipped_highlights_pct": round(100 * float((small.max(-1) > 0.995).mean()), 2),
        "crushed_shadows_pct": round(100 * float((small.max(-1) < 0.01).mean()), 2),
        "median_luminance": round(float(np.median(Y)), 3),
    }


def side_by_side(before: np.ndarray, after: np.ndarray, max_side: int = 1800, gap: int = 12) -> np.ndarray:
    """Before | after, horizontally, each scaled to fit max_side/2."""
    h, w = after.shape[:2]
    s = min(1.0, (max_side / 2) / max(h, w))
    size = (max(1, round(w * s)), max(1, round(h * s)))
    b = cv2.resize(before, size, interpolation=cv2.INTER_AREA)
    a = cv2.resize(after, size, interpolation=cv2.INTER_AREA)
    return np.concatenate([b, np.ones((size[1], gap, 3), np.float32), a], axis=1)
