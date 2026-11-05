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

    # 0. geometry: lens corrections, straighten, perspective, crop
    data, seg, geo = apply_geometry(data, p, seg)
    if geo:
        rep["geometry"] = geo

    if p.get("keep_original"):
        return _beauty_only(data, p, seg, rep, t0)

    # 1. people / skin
    use_people = need_seg and (p.get("separate_subject", True) or p.get("wb_mode") == "auto"
                               or float(p.get("retouch", 0) or 0) > 0)
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
    from . import lr_tools as lrt
    M = lrt.calibration_matrix(p)
    if M is not None:
        lin = np.maximum(lin @ M.T, 0).astype(np.float32)

    # 3. noise reduction (linear)
    dn = p.get("denoise", "auto")
    if dn == "auto":
        dl, dc = auto_denoise(data)
    else:
        dl = dc = float(dn)
    if p.get("nr_luminance") is not None:
        dl = float(p["nr_luminance"]) / 100
    if p.get("nr_color") is not None:
        dc = float(p["nr_color"]) / 100
    lin = adj.denoise(lin, dl, dc)
    rep["denoise"] = {"luma": round(dl, 2), "chroma": round(dc, 2)}

    # 4. exposure
    ev = suggested_ev(data, lin, person) if p.get("auto_exposure", True) else 0.0
    ev += float(p.get("exposure", 0.0))
    lin = lin * np.float32(2.0 ** ev)
    rep["exposure_ev"] = round(ev, 2)

    # 5. dehaze + local tone
    lin = lrt.dehaze(lin, float(p.get("dehaze", 0) or 0))
    lin = adj.local_tone(lin, float(p.get("highlights", 0)), float(p.get("shadows", 0)))

    # 5b. sky mask (only when a sky adjustment is requested)
    sky = None
    if any(abs(float(p.get(k, 0) or 0)) > 1e-3 for k in ("sky_exposure", "sky_saturation", "sky_warmth")):
        sky = lrt.sky_mask(linear_to_srgb(adj.tone_map(lin)), person)
        rep["sky_pct"] = round(100 * float(sky.mean()), 1)
        if sky.max() > 0.05 and abs(float(p.get("sky_exposure", 0) or 0)) > 1e-3:
            lin = lin * np.exp2(float(p["sky_exposure"]) * sky)[..., None].astype(np.float32)
        if sky.max() <= 0.05:
            sky = None

    # 6. subject / background separation
    if person is not None and p.get("separate_subject", True):
        lin = adj.separate_subject(lin, person, p)
        rep["separation"] = {k: p.get(k) for k in ("bg_blur", "bg_exposure", "bg_saturation",
                                                   "subject_exposure", "subject_clarity", "skin_smooth")}

    # 7. vignette + tone map
    lin = lrt.vignette_lr(lin, -float(p.get("vignette", 0) or 0) / 0.6 * 100,
                          float(p.get("vignette_midpoint", 50)), float(p.get("vignette_feather", 50)),
                          float(p.get("vignette_roundness", 0)), float(p.get("vignette_highlights", 0)))
    disp = adj.tone_map(lin, white=float(p.get("tone_white", 8.0)), scene_referred=data.is_raw)
    if data.is_raw:
        p["_base_contrast"] = float(p.get("base_contrast", 0.3))
        p["_black_point"] = auto_black_point(disp) if p.get("auto_black", True) else 0.0
        rep["base_tone"] = {"contrast": p["_base_contrast"], "black_point": round(p["_black_point"], 4)}

    # 8. look (OKLab)
    sep_person = person if (person is not None and p.get("separate_subject", True)) else None
    disp = adj.apply_look(disp, p, skin=skin, person=sep_person, sky=sky)
    srgb = lrt.tone_curve(linear_to_srgb(disp), p)
    srgb = lrt.defringe(srgb, float(p.get("defringe", 0) or 0))

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
    srgb, rep["retouch"] = apply_retouch(srgb, p, seg)
    srgb = lrt.sharpen_lr(srgb, float(p.get("sharpen", 0)) * 100,
                          radius=float(p.get("sharpen_radius", 1.0)) * max(0.7, max(H, W) / 5000),
                          detail=float(p.get("sharpen_detail", 25)), masking=float(p.get("sharpen_masking", 0)))
    srgb = lrt.grain_lr(srgb, float(p.get("grain", 0)) * 100, float(p.get("grain_size", 25)),
                        float(p.get("grain_roughness", 50)))

    rep["timing_s"] = {"segmentation": round(t_seg - t0, 2), "total": round(time.time() - t0, 2)}
    rep["output_stats"] = image_stats(srgb)
    return Result(srgb=np.clip(srgb, 0, 1).astype(np.float32), params=p, report=rep, seg=seg)


def apply_geometry(data: ImageData, p: dict, seg: Segmentation | None):
    """Lens corrections, straighten (rotate / auto-level), perspective, scale
    and aspect crop, applied to the linear image before anything else."""
    import dataclasses
    from . import lr_tools as lrt
    lin = data.linear
    rep: dict = {}
    changed = False
    if p.get("remove_ca"):
        lin, rep["chromatic_aberration_scale"] = lrt.remove_chromatic_aberration(lin)
        changed = True
    if abs(float(p.get("lens_distortion", 0) or 0)) >= 0.5:
        lin = lrt.lens_distortion(lin, float(p["lens_distortion"])); changed = True
    if abs(float(p.get("lens_vignetting", 0) or 0)) >= 0.5:
        lin = lrt.lens_vignetting(lin, float(p["lens_vignetting"]), float(p.get("lens_vignetting_midpoint", 50)))
    angle = float(p.get("rotate", 0) or 0)
    if p.get("auto_level"):
        a, conf, n = lrt.detect_tilt(_downscale(lin, 1000))
        rep["auto_level"] = {"detected_deg": round(a, 2), "confidence": round(conf, 2), "lines": n}
        if conf >= 0.3 and 0.15 <= abs(a) <= 8:
            angle += a
    if abs(angle) >= 0.02:
        lin = lrt.rotate(lin, angle); changed = True
        rep["rotate_deg"] = round(angle, 2)
    v, h, sc = (float(p.get(k, d) or d) for k, d in (("vertical", 0), ("horizontal", 0), ("scale", 100)))
    if abs(v) >= 0.5 or abs(h) >= 0.5 or abs(sc - 100) >= 0.5:
        lin = lrt.perspective(lin, v, h, sc); changed = True
        rep["perspective"] = {"vertical": v, "horizontal": h, "scale": sc}
    if changed:
        seg = None                       # masks must be recomputed on the new geometry
    aspect = p.get("crop_aspect")
    zoom = float(p.get("crop_zoom", 1.0) or 1.0)
    if (aspect and aspect != "original") or zoom > 1.001:
        H, W = lin.shape[:2]
        tmp = dataclasses.replace(data, linear=lin)
        if seg is None:
            seg = segment(tmp)
        focus = p.get("crop_focus", "auto")
        if isinstance(focus, str) and "," in focus:
            fx, fy = (float(v) for v in focus.split(","))
        elif seg.faces:
            xs = [f[0] + f[2] / 2 for f in seg.faces]; ys = [f[1] + f[3] / 2 for f in seg.faces]
            fx, fy = float(np.mean(xs)) / W, min(0.9, float(np.mean(ys)) / H + 0.12)
        elif seg.has_people:
            yy, xx = np.nonzero(seg.person > 0.5)
            fx, fy = (xx.mean() / W, yy.mean() / H) if len(xx) else (0.5, 0.5)
        else:
            fx, fy = 0.5, 0.5
        x0, y0, x1, y1 = lrt.crop_window(H, W, aspect, (fx, fy), zoom)
        lin = lin[y0:y1, x0:x1]
        seg = Segmentation(person=np.ascontiguousarray(seg.person[y0:y1, x0:x1]),
                           skin=np.ascontiguousarray(seg.skin[y0:y1, x0:x1]),
                           faces=[(fx_ - x0, fy_ - y0, fw_, fh_, c) for fx_, fy_, fw_, fh_, c in seg.faces
                                  if x0 <= fx_ + fw_ / 2 <= x1 and y0 <= fy_ + fh_ / 2 <= y1],
                           backend=seg.backend)
        rep["crop"] = {"aspect": aspect or "original", "box": [x0, y0, x1, y1],
                       "normalised": [round(y0 / H, 4), round(x0 / W, 4), round(y1 / H, 4), round(x1 / W, 4)]}
        changed = True
    if lin is data.linear:
        return data, seg, rep
    return dataclasses.replace(data, linear=np.ascontiguousarray(lin, dtype=np.float32)), seg, rep


def apply_retouch(srgb: np.ndarray, p: dict, seg: Segmentation | None):
    """Face beauty retouch (blemishes, redness, smoothing, under-eye, shine,
    teeth, eyes) on the display image."""
    from .retouch import retouch, strengths
    if max(strengths(p).values()) <= 0:
        return srgb, {"enabled": False}
    from .face_landmarks import find_faces
    boxes = seg.faces if seg is not None and seg.faces else None
    faces = find_faces(srgb, boxes)
    skin = seg.skin if seg is not None and seg.has_people else None
    out, rep = retouch(srgb, faces, p, skin=skin)
    rep["enabled"] = True
    return out, rep


def _beauty_only(data: ImageData, p: dict, seg: Segmentation | None, rep: dict, t0: float) -> Result:
    """'Keep the photo as it is' mode: no white balance, exposure, colour,
    background or detail changes - only the face retouch.  JPEG/PNG/TIFF
    pixels outside the retouched skin stay identical; RAW files are rendered
    with the camera's own as-shot white balance."""
    before = render_before(data)
    if seg is None:
        seg = segment(data)
    srgb, rep["retouch"] = apply_retouch(before, p, seg)
    if p.get("skin_smooth") and seg.has_people:
        srgb = adj.skin_smooth(srgb, seg.skin, float(p["skin_smooth"]))
    rep["mode"] = "keep_original (beauty only)"
    rep["people"] = {"detected": bool(seg.has_people), "faces": seg.people_count, "backend": seg.backend}
    rep["timing_s"] = {"total": round(time.time() - t0, 2)}
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
