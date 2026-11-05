"""Automatic white balance.

The image arrives in linear sRGB rendered with the camera's *daylight* white
balance (for RAW).  A grey surface therefore shows the colour of the light that
illuminated it, and correcting is a matter of dividing by that colour.

Estimation combines several classic illuminant estimators computed on the
*background* (people are excluded because skin and clothing bias them):

* grey search                 - the real light under which most surfaces turn neutral
* shades-of-grey (Minkowski p=6)
* grey-edge (1st-order, p=6)  - colour of edges, robust to big coloured areas
* bright-patch (white patch on the brightest, unclipped pixels)

The combined estimate is pulled toward the Planckian locus (real light sources
lie close to it), and - when people are present - checked against skin tone:
skin under neutral light has an OKLab hue of roughly 45-70 degrees.  If the
background is not trustworthy (few grey pixels), skin gets more weight.
"""
from __future__ import annotations

import cv2
import numpy as np

from .color import describe_cct, illuminant_rgb, illuminant_to_cct, linear_to_oklab, luminance

SKIN_HUE_TARGET = 58.0     # degrees in OKLab, centre of natural skin tones
SKIN_HUE_TOL = 13.0


def _small(a: np.ndarray, side: int = 640) -> np.ndarray:
    h, w = a.shape[:2]
    s = min(1.0, side / max(h, w))
    if s >= 1:
        return a
    interp = cv2.INTER_AREA
    return cv2.resize(a, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)


def _norm(e) -> np.ndarray:
    e = np.maximum(np.asarray(e, dtype=np.float64), 1e-6)
    return e / e[1]


def _geo_mean(vectors, weights) -> np.ndarray:
    v = np.log(np.array([_norm(x) for x in vectors]))
    w = np.asarray(weights, dtype=np.float64)[:, None]
    return np.exp((v * w).sum(0) / w.sum())


def _grey_search(px: np.ndarray, max_px: int = 40000) -> tuple[np.ndarray, float]:
    """Try plausible light sources (Planckian locus 2000-12000K, +-0.012 Duv)
    and keep the one under which the most pixels become neutral grey.
    Constrained to real-world lights, so big coloured areas (grass, walls,
    curtains) cannot drag it to an impossible colour."""
    if len(px) > max_px:
        px = px[np.random.default_rng(0).choice(len(px), max_px, replace=False)]
    chrom = px / (px.sum(1, keepdims=True) + 1e-9)                     # N x 3
    w = np.sqrt(px.sum(1))                                              # favour brighter (less noisy) pixels
    mireds = np.linspace(1e6 / 12000, 1e6 / 2000, 41)
    duvs = np.linspace(-0.012, 0.012, 7)
    cands = np.array([illuminant_rgb(1e6 / m, d) for m in mireds for d in duvs])   # K x 3
    q = chrom[None, :, :] / cands[:, None, :]
    q /= q.sum(2, keepdims=True)
    dev = np.sqrt(((q - 1 / 3) ** 2).sum(2))                            # K x N
    score = (np.exp(-(dev / 0.018) ** 2) * w).sum(1) / w.sum()
    best = int(np.argmax(score))
    # soft-argmax around the best candidates for sub-grid precision
    top = np.argsort(score)[-6:]
    sw = score[top] / score[top].sum()
    e = np.exp((np.log(cands[top]) * sw[:, None]).sum(0))
    frac = float((dev[best] < 0.02).mean())
    return _norm(e), frac


def _pull_to_locus(e: np.ndarray, keep_duv: float = 0.35) -> np.ndarray:
    T, duv = illuminant_to_cct(e)
    duv = float(np.clip(duv * keep_duv, -0.02, 0.02))
    return illuminant_rgb(T, duv)


def _skin_hue(lin_px: np.ndarray, e: np.ndarray) -> tuple[float, float]:
    lab = linear_to_oklab((lin_px / e).astype(np.float32))
    a, b = float(np.median(lab[:, 1])), float(np.median(lab[:, 2]))
    return float(np.degrees(np.arctan2(b, a))), float(np.hypot(a, b))


def estimate_illuminant(lin: np.ndarray, person: np.ndarray | None = None,
                        skin: np.ndarray | None = None) -> dict:
    """Return a dict with the estimated light colour ``e`` (linear RGB, G=1),
    its CCT/Duv, a confidence 0..1 and per-method details."""
    img = _small(lin).astype(np.float32)
    h, w = img.shape[:2]
    pm = _small(person, max(h, w)) if person is not None else np.zeros((h, w), np.float32)
    sm = _small(skin, max(h, w)) if skin is not None else np.zeros((h, w), np.float32)
    if pm.shape != (h, w):
        pm = cv2.resize(pm, (w, h)); sm = cv2.resize(sm, (w, h))

    mx = img.max(-1)
    clip_level = np.percentile(mx, 99.9)
    valid = (mx < min(0.97, clip_level * 0.98 + 0.02)) & (img.min(-1) > 0.004)
    bg = valid & (pm < 0.3)
    if bg.sum() < 500:
        bg = valid
    px = img[bg].astype(np.float64)
    details = {}

    e_grey, grey_frac = _grey_search(px)
    details["grey_search"] = e_grey
    e_sog = _norm((px ** 6).mean(0) ** (1 / 6))
    details["shades_of_grey"] = e_sog
    gx = cv2.Sobel(cv2.GaussianBlur(img, (0, 0), 1.0), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(cv2.GaussianBlur(img, (0, 0), 1.0), cv2.CV_32F, 0, 1)
    gm = np.sqrt(gx ** 2 + gy ** 2)[bg].astype(np.float64)
    e_edge = _norm((gm ** 6).mean(0) ** (1 / 6)) if len(gm) else e_sog
    details["grey_edge"] = e_edge
    lum = luminance(img)[bg]
    top = px[lum >= np.percentile(lum, 97)] if len(lum) > 100 else px
    e_white = _norm(top.mean(0))
    details["bright_patch"] = e_white

    # trust the grey search more when many neutral surfaces exist
    wg = 0.6 + 3.0 * min(1.0, grey_frac / 0.12)
    e = _geo_mean([e_grey, e_edge, e_sog, e_white], [wg, 0.7, 0.3, 0.4])
    e = _pull_to_locus(e)
    confidence = float(np.clip(0.3 + grey_frac * 5.0, 0.0, 1.0))

    # --- skin check -------------------------------------------------------
    skin_info = None
    sk = valid & (sm > 0.5)
    if sk.sum() > 150:
        spx = img[sk].astype(np.float64)
        hue0, chroma0 = _skin_hue(spx, e)
        skin_info = {"hue_before": round(hue0, 1), "chroma": round(chroma0, 4)}
        T0, duv0 = illuminant_to_cct(e)
        if abs(hue0 - SKIN_HUE_TARGET) > SKIN_HUE_TOL * 0.5:
            # search along the locus (and a little across it) for the light that
            # gives the most natural skin hue, staying close to the bg estimate
            best, best_cost = e, np.inf
            for mired in np.linspace(-80, 80, 33):
                T = 1e6 / max(1e6 / T0 + mired, 40.0)
                for d in (-0.006, -0.003, 0.0, 0.003, 0.006):
                    cand = illuminant_rgb(T, duv0 * 0.5 + d)
                    hue, chroma = _skin_hue(spx, cand)
                    cost = (max(0.0, abs(hue - SKIN_HUE_TARGET) - SKIN_HUE_TOL * 0.5) / SKIN_HUE_TOL) ** 2
                    cost += (mired / 60.0) ** 2 * confidence + (d / 0.006) ** 2 * 0.2
                    if chroma < 0.02:
                        cost += 1.0
                    if cost < best_cost:
                        best, best_cost = cand, cost
            skin_weight = float(np.clip(1.0 - confidence, 0.25, 0.75))
            e = _geo_mean([e, best], [1 - skin_weight, skin_weight])
            skin_info["weight"] = round(skin_weight, 2)
        skin_info["hue_after"] = round(_skin_hue(spx, e)[0], 1)

    T, duv = illuminant_to_cct(e)
    return {
        "e": _norm(e).astype(np.float32),
        "cct": round(T), "duv": round(duv, 4),
        "light": describe_cct(T),
        "confidence": round(confidence, 2),
        "grey_pixel_fraction": round(grey_frac, 3),
        "skin": skin_info,
        "methods": {k: [round(float(x), 4) for x in v] for k, v in details.items()},
    }


def white_balance_gains(params: dict, data, estimate: dict | None) -> tuple[np.ndarray, dict]:
    """Translate WB parameters into per-channel linear gains.

    params keys:
      wb_mode     auto | camera | daylight | kelvin
      wb_kelvin   scene light temperature for 'kelvin' mode (e.g. 3200)
      wb_tint_duv light Duv for 'kelvin' mode (+ green light, - magenta light)
      wb_strength 0..1, how much of the auto correction to apply
      warmth      creative shift in mireds after correction (+ warmer, - cooler)
      tint        creative tint (+ magenta, - green), roughly -100..100
    """
    mode = params.get("wb_mode", "auto")
    if mode == "camera":
        g = np.asarray(data.as_shot_gains, np.float64)
        e = 1.0 / g
    elif mode == "daylight":
        e = np.ones(3)
    elif mode == "kelvin":
        e = illuminant_rgb(float(params.get("wb_kelvin", 5500)), float(params.get("wb_tint_duv", 0.0)))
    else:
        # (for JPEG input e is relative to the camera's own balance already)
        e = np.asarray(estimate["e"], np.float64) if estimate else np.ones(3)
        s = float(params.get("wb_strength", 1.0))
        e = np.exp(np.log(_norm(e)) * s)
    gains = 1.0 / _norm(e)

    warmth = float(params.get("warmth", 0.0))
    if warmth:
        T = 1e6 / (1e6 / 6504.0 + warmth)
        gains = gains * illuminant_rgb(T)
    tint = float(params.get("tint", 0.0))
    if tint:
        gains = gains * np.array([1.0, 2.0 ** (-tint / 150.0), 1.0])
    gains = gains / gains[1]
    T, duv = illuminant_to_cct(1.0 / gains)
    return gains.astype(np.float32), {"mode": mode, "gains": [round(float(x), 4) for x in gains],
                                      "neutralised_light_cct": round(T), "neutralised_light_duv": round(duv, 4)}
