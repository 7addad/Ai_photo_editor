"""Lightroom-equivalent tools that complete the editor:

Lens        distortion, lens vignetting, lateral chromatic aberration, defringe
Transform   rotate / auto-level (horizon), vertical & horizontal perspective, scale
Crop        aspect-ratio crop centred on the subject
Basic       dehaze
Tone curve  parametric (shadows/darks/lights/highlights) + point curves (RGB, R, G, B)
Calibration red/green/blue primary hue & saturation, shadow tint
Masking     sky mask (+ sky exposure / saturation / warmth)
Detail      sharpening with radius / detail / masking
Effects     post-crop vignette (midpoint, feather, roundness), grain (size, roughness)

Sliders use Lightroom scales (mostly -100..100).
"""
from __future__ import annotations

import colorsys

import cv2
import numpy as np

from .color import linear_to_oklab, oklab_to_linear, srgb_to_linear
from .segmentation import guided_filter


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _down(a, side):
    h, w = a.shape[:2]
    s = min(1.0, side / max(h, w))
    if s >= 1:
        return a, 1.0
    return cv2.resize(a, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA), s


def lr_hue_to_oklab(h_deg: float) -> float:
    """Lightroom/HSV hue (0 red, 60 yellow, 120 green, 240 blue) -> OKLab hue angle."""
    r, g, b = colorsys.hsv_to_rgb((h_deg % 360) / 360.0, 1.0, 1.0)
    lab = linear_to_oklab(srgb_to_linear(np.array([[r, g, b]], np.float32)))[0]
    return float(np.degrees(np.arctan2(lab[2], lab[1])))


# ================================================================ geometry

def _fill_zoom(H, W, mapping) -> float:
    """Smallest zoom >= 1 so that every border point of the output maps inside
    the source (no empty corners).  mapping(xn, yn, z) -> source (xn, yn)
    in normalised [-1, 1] coordinates."""
    t = np.linspace(-1, 1, 41)
    border = np.concatenate([np.stack([t, -np.ones_like(t)], 1), np.stack([t, np.ones_like(t)], 1),
                             np.stack([-np.ones_like(t), t], 1), np.stack([np.ones_like(t), t], 1)])
    lo, hi = 1.0, 3.0
    for _ in range(30):
        z = (lo + hi) / 2
        sx, sy = mapping(border[:, 0], border[:, 1], z)
        if np.all(np.abs(sx) <= 1.0001) and np.all(np.abs(sy) <= 1.0001):
            hi = z
        else:
            lo = z
    return hi


def lens_distortion(lin: np.ndarray, amount: float) -> np.ndarray:
    """Lightroom 'Distortion' (-100..100): + removes barrel distortion,
    - removes pincushion.  The result is zoomed to avoid empty borders."""
    if abs(amount) < 0.5:
        return lin
    H, W = lin.shape[:2]
    k = -amount / 100.0 * 0.12
    asp = W / H
    rn = np.hypot(asp, 1.0)

    def mapping(xn, yn, z):
        x, y = xn * asp / z, yn / z
        r2 = (x * x + y * y) / (rn * rn)
        f = 1 + k * r2
        return x * f / asp, y * f

    z = _fill_zoom(H, W, mapping) if k != 0 else 1.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn = (xx - (W - 1) / 2) / ((W - 1) / 2)
    yn = (yy - (H - 1) / 2) / ((H - 1) / 2)
    sx, sy = mapping(xn, yn, z)
    mx = (sx * (W - 1) / 2 + (W - 1) / 2).astype(np.float32)
    my = (sy * (H - 1) / 2 + (H - 1) / 2).astype(np.float32)
    return cv2.remap(lin, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def lens_vignetting(lin: np.ndarray, amount: float, midpoint: float = 50) -> np.ndarray:
    """Lens-correction vignetting (-100..100): + brightens dark corners."""
    if abs(amount) < 0.5:
        return lin
    H, W = lin.shape[:2]
    y, x = np.ogrid[-1:1:complex(0, H), -1:1:complex(0, W)]
    r2 = (x * x * (W / max(H, W)) ** 2 + y * y * (H / max(H, W)) ** 2) / 1.0
    r2 = r2 / r2.max()
    m = 0.2 + 0.6 * midpoint / 100
    g = np.exp2(amount / 100.0 * 1.2 * _ss(m * 0.5, 1.0, np.sqrt(r2)) ** 2).astype(np.float32)
    return lin * g[..., None]


def _scale_channel(ch: np.ndarray, s: float) -> np.ndarray:
    H, W = ch.shape
    M = np.array([[s, 0, (1 - s) * (W - 1) / 2], [0, s, (1 - s) * (H - 1) / 2]], np.float32)
    return cv2.warpAffine(ch, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def remove_chromatic_aberration(lin: np.ndarray) -> tuple[np.ndarray, dict]:
    """Automatic lateral CA removal: scales R and B about the centre to best
    align their edges with G (like Lightroom 'Remove Chromatic Aberration')."""
    small, s = _down(lin, 1600)
    g = cv2.Sobel(small[..., 1], cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(small[..., 1], cv2.CV_32F, 0, 1) ** 2
    edges = g > np.percentile(g, 92)
    out = lin.copy()
    rep = {}
    for c, name in ((0, "red"), (2, "blue")):
        best, best_err = 1.0, np.inf
        for sc in np.linspace(0.996, 1.004, 17):
            ch = _scale_channel(small[..., c], sc)
            ratio = ch[edges] / np.maximum(small[..., 1][edges], 1e-4)
            k = np.median(ratio)
            err = np.mean(np.abs(ch[edges] - k * small[..., 1][edges]))
            if err < best_err:
                best, best_err = sc, err
        rep[name] = round(float(best), 5)
        if abs(best - 1.0) > 1e-4:
            out[..., c] = _scale_channel(lin[..., c], best)
    return out, rep


def defringe(srgb: np.ndarray, amount: float) -> np.ndarray:
    """Remove purple / green fringes along high-contrast edges (0..20 like LR)."""
    if amount <= 0:
        return srgb
    lab = linear_to_oklab(srgb_to_linear(srgb))
    L = lab[..., 0]
    gm = np.hypot(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3))
    edge = cv2.dilate(_ss(0.08, 0.25, gm), np.ones((5, 5), np.uint8))
    h = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360
    purple = np.exp(-(((h - 315 + 180) % 360 - 180) / 30) ** 2)
    green = np.exp(-(((h - 150 + 180) % 360 - 180) / 20) ** 2) * 0.6
    k = np.clip(edge * np.maximum(purple, green) * amount / 10.0, 0, 1)
    lab[..., 1] *= 1 - k
    lab[..., 2] *= 1 - k
    from .color import gamut_clip, linear_to_srgb
    return linear_to_srgb(gamut_clip(oklab_to_linear(lab)))


def detect_tilt(srgb_or_lin: np.ndarray) -> tuple[float, float, int]:
    """Estimate the camera roll from long near-horizontal / near-vertical lines.
    Returns (angle_to_rotate_deg, confidence 0..1, number_of_lines)."""
    small, _ = _down(srgb_or_lin, 1000)
    g8 = (np.clip(cv2.cvtColor(small.astype(np.float32), cv2.COLOR_RGB2GRAY), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    edges = cv2.Canny(cv2.GaussianBlur(g8, (3, 3), 0), 50, 140)
    min_len = max(small.shape[:2]) * 0.12
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=80, minLineLength=min_len, maxLineGap=6)
    if lines is None:
        return 0.0, 0.0, 0
    angs, wts = [], []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        a = (a + 90) % 180 - 90                    # -90..90
        length = np.hypot(x2 - x1, y2 - y1)
        if abs(a) < 8:                             # horizontal-ish
            angs.append(a); wts.append(length * 1.5)
        elif abs(abs(a) - 90) < 8:                 # vertical-ish
            angs.append(a - np.sign(a) * 90); wts.append(length)
    if len(angs) < 2:
        return 0.0, 0.0, len(angs)
    angs, wts = np.array(angs), np.array(wts)
    order = np.argsort(angs)
    cw = np.cumsum(wts[order])
    med = float(angs[order][np.searchsorted(cw, cw[-1] / 2)])
    spread = float(np.sqrt(np.average((angs - med) ** 2, weights=wts)))
    conf = float(np.clip(len(angs) / 8, 0, 1) * np.clip(1 - spread / 3, 0, 1))
    return -med, conf, len(angs)


def rotate(lin: np.ndarray, angle: float) -> np.ndarray:
    """Rotate by angle (deg, + = counter-clockwise) and zoom to remove corners."""
    if abs(angle) < 0.02:
        return lin
    H, W = lin.shape[:2]
    t = np.radians(abs(angle))
    z = max((W * np.cos(t) + H * np.sin(t)) / W, (W * np.sin(t) + H * np.cos(t)) / H)
    M = cv2.getRotationMatrix2D(((W - 1) / 2, (H - 1) / 2), angle, z)
    return cv2.warpAffine(lin, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def perspective(lin: np.ndarray, vertical: float, horizontal: float, scale: float = 100) -> np.ndarray:
    """Lightroom Transform: vertical (+ fixes buildings leaning back, top
    narrower), horizontal, and scale (>100 zooms in).  Never leaves borders."""
    if abs(vertical) < 0.5 and abs(horizontal) < 0.5 and abs(scale - 100) < 0.5:
        return lin
    H, W = lin.shape[:2]
    kv, kh = vertical / 100 * 0.3, horizontal / 100 * 0.3
    tl, tr, br, bl = [0.0, 0.0], [W - 1.0, 0.0], [W - 1.0, H - 1.0], [0.0, H - 1.0]
    # source quad (region of the input that becomes the full output)
    if kv > 0:   # top is too narrow in the photo -> take a narrower top region
        tl[0] += kv * W / 2; tr[0] -= kv * W / 2
    elif kv < 0:
        bl[0] -= kv * W / 2; br[0] += kv * W / 2
    if kh > 0:
        tl[1] += kh * H / 2; bl[1] -= kh * H / 2
    elif kh < 0:
        tr[1] -= kh * H / 2; br[1] += kh * H / 2
    src = np.array([tl, tr, br, bl], np.float32)
    s = max(scale, 100) / 100.0
    c = np.array([(W - 1) / 2, (H - 1) / 2], np.float32)
    src = c + (src - c) / s
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(lin, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


ASPECTS = {"1:1": 1.0, "4:5": 4 / 5, "5:4": 5 / 4, "3:2": 3 / 2, "2:3": 2 / 3, "16:9": 16 / 9,
           "9:16": 9 / 16, "4:3": 4 / 3, "3:4": 3 / 4, "2:1": 2.0, "21:9": 21 / 9}


def crop_window(H: int, W: int, aspect: str, focus: tuple[float, float] = (0.5, 0.45),
                zoom: float = 1.0) -> tuple[int, int, int, int]:
    """Largest window with the requested aspect, containing the focus point as
    close to the window centre as the frame allows.  Returns (x0, y0, x1, y1)."""
    if aspect in (None, "", "original"):
        r = W / H
    else:
        r = ASPECTS.get(aspect) or float(aspect)
    if W / H > r:
        ch, cw = H, H * r
    else:
        cw, ch = W, W / r
    cw, ch = cw / max(zoom, 1.0), ch / max(zoom, 1.0)
    fx, fy = focus[0] * W, focus[1] * H
    x0 = int(np.clip(fx - cw / 2, 0, W - cw)); y0 = int(np.clip(fy - ch / 2, 0, H - ch))
    return x0, y0, int(x0 + round(cw)), int(y0 + round(ch))


# ================================================================ tone

def dehaze(lin: np.ndarray, amount: float) -> np.ndarray:
    """Lightroom Dehaze (-100..100) using the dark-channel prior."""
    if abs(amount) < 0.5:
        return lin
    a = amount / 100.0
    small, _ = _down(lin, 700)
    I = np.clip(small, 0, None)
    ref = np.percentile(I, 99.5)
    In = np.clip(I / max(ref, 1e-4), 0, 1)
    patch = max(3, int(max(In.shape[:2]) * 0.02) | 1)
    dark = cv2.erode(In.min(-1), np.ones((patch, patch), np.uint8))
    idx = dark.reshape(-1).argsort()[-max(10, dark.size // 1000):]
    A = np.clip(In.reshape(-1, 3)[idx].mean(0), 0.3, 1.0)
    t = 1 - 0.92 * cv2.erode((In / A).min(-1), np.ones((patch, patch), np.uint8))
    gray = In.mean(-1).astype(np.float32)
    t = np.clip(guided_filter(gray, t.astype(np.float32), patch * 2, 1e-3), 0.05, 1)
    t = cv2.resize(t, (lin.shape[1], lin.shape[0]), interpolation=cv2.INTER_LINEAR)[..., None]
    A_lin = (A * ref).astype(np.float32)
    if a > 0:
        te = 1 - a * (1 - t)
        out = (lin - A_lin) / np.maximum(te, 0.25) + A_lin
        # dehaze darkens; give back part of the mean brightness
        out = out * (np.mean(lin) / max(np.mean(np.maximum(out, 0)), 1e-6)) ** 0.5
    else:
        h = -a * 0.45 * (1 - t * 0.5)
        out = lin * (1 - h) + A_lin * h
    return np.maximum(out, 0).astype(np.float32)


def _pchip(xs, ys, x):
    """Monotone cubic (Fritsch-Carlson) interpolation."""
    xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    h = np.diff(xs); d = np.diff(ys) / h
    m = np.zeros_like(ys)
    m[0], m[-1] = d[0], d[-1]
    for k in range(1, len(xs) - 1):
        if d[k - 1] * d[k] <= 0:
            m[k] = 0
        else:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    i = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
    t = (x - xs[i]) / h[i]
    h00, h10 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t
    h01, h11 = -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    return h00 * ys[i] + h10 * h[i] * m[i] + h01 * ys[i + 1] + h11 * h[i] * m[i + 1]


def parse_points(spec) -> list[tuple[float, float]] | None:
    """'0,0; 64,56; 192,200; 255,255' (0..255) or a list of pairs -> points 0..1."""
    if spec in (None, "", "linear"):
        return None
    if isinstance(spec, str):
        pts = [tuple(float(v) for v in p.split(",")) for p in spec.replace(" ", "").split(";") if p]
    else:
        pts = [tuple(map(float, p)) for p in spec]
    pts = sorted((x / 255.0, y / 255.0) for x, y in pts)
    if pts[0][0] > 0:
        pts.insert(0, (0.0, 0.0))
    if pts[-1][0] < 1:
        pts.append((1.0, 1.0))
    return pts


def _curve_lut(p: dict, key: str) -> np.ndarray | None:
    pts = parse_points(p.get(key))
    if not pts:
        return None
    x = np.linspace(0, 1, 4096)
    return np.clip(_pchip([q[0] for q in pts], [q[1] for q in pts], x), 0, 1).astype(np.float32)


def _parametric_lut(p: dict) -> np.ndarray | None:
    vals = [float(p.get(k, 0) or 0) for k in ("curve_shadows", "curve_darks", "curve_lights", "curve_highlights")]
    if not any(abs(v) > 0.5 for v in vals):
        return None
    x = np.linspace(0, 1, 4096)
    y = x.copy()
    for v, c in zip(vals, (0.125, 0.375, 0.625, 0.875)):
        bump = np.clip(1 - np.abs(x - c) / 0.25, 0, 1)
        bump = np.sin(bump * np.pi / 2) ** 2
        y += v / 100 * 0.12 * bump * np.sin(np.pi * x) ** 0.5     # endpoints stay fixed
    y = np.maximum.accumulate(np.clip(y, 0, 1))
    return y.astype(np.float32)


def tone_curve(srgb: np.ndarray, p: dict) -> np.ndarray:
    """Parametric + point curves on display values (like Lightroom's curve)."""
    master = _parametric_lut(p)
    pc = _curve_lut(p, "point_curve")
    if pc is not None:
        master = pc if master is None else np.interp(master, np.linspace(0, 1, 4096), pc).astype(np.float32)
    chans = [_curve_lut(p, f"point_curve_{c}") for c in ("red", "green", "blue")]
    if master is None and all(c is None for c in chans):
        return srgb
    out = np.empty_like(srgb)
    idx = np.clip((srgb * 4095).astype(np.int32), 0, 4095)
    for c in range(3):
        v = srgb[..., c] if master is None else master[idx[..., c]]
        if chans[c] is not None:
            v = chans[c][np.clip((v * 4095).astype(np.int32), 0, 4095)]
        out[..., c] = v
    return out


# ================================================================ calibration

def calibration_matrix(p: dict) -> np.ndarray | None:
    """3x3 matrix moving the red/green/blue primaries (hue -100..100,
    saturation -100..100), like Lightroom's Calibration panel."""
    keys = [(f"{c}_hue", f"{c}_sat") for c in ("red", "green", "blue")]
    if not any(abs(float(p.get(k, 0) or 0)) > 0.5 for pair in keys for k in pair):
        return None
    cols = []
    for i, (hk, sk) in enumerate(keys):
        e = np.zeros((1, 3), np.float32); e[0, i] = 1.0
        lab = linear_to_oklab(e)[0]
        C, h = np.hypot(lab[1], lab[2]), np.arctan2(lab[2], lab[1])
        h += np.radians(float(p.get(hk, 0) or 0) * 0.3)
        C *= 1 + float(p.get(sk, 0) or 0) / 100 * 0.6
        new = oklab_to_linear(np.array([[lab[0], C * np.cos(h), C * np.sin(h)]], np.float32))[0]
        cols.append(new)
    M = np.stack(cols, 1)
    w = M @ np.ones(3)
    return (M / w[:, None]).astype(np.float32)          # keep white white


# ================================================================ sky

def sky_mask(srgb: np.ndarray, person: np.ndarray | None = None) -> np.ndarray:
    """Heuristic sky mask (bright, smooth, blue/neutral/sunset region connected
    to the top of the frame), refined with a guided filter.  0..1, full size."""
    H, W = srgb.shape[:2]
    small, s = _down(srgb, 640)
    h, w = small.shape[:2]
    lab = linear_to_oklab(srgb_to_linear(small))
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    tex = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 2.0)
    hue = np.degrees(np.arctan2(b, a)) % 360
    C = np.hypot(a, b)
    blue = np.exp(-(((hue - 245 + 180) % 360 - 180) / 45) ** 2) * _ss(0.01, 0.04, C)
    neutral = 1 - _ss(0.02, 0.06, C)
    warm = np.exp(-(((hue - 60 + 180) % 360 - 180) / 40) ** 2) * _ss(0.03, 0.08, C)
    yy = np.linspace(0, 1, h)[:, None] * np.ones((1, w))
    colour = np.maximum.reduce([blue, neutral * _ss(0.6, 0.8, L), warm * (1 - yy) * _ss(0.55, 0.75, L)])
    smooth = 1 - _ss(0.03, 0.12, tex)
    prob = colour * smooth * _ss(0.35, 0.6, L) * (1 - _ss(0.55, 0.95, yy))
    if person is not None:
        pm = cv2.resize(person, (w, h), interpolation=cv2.INTER_AREA)
        prob *= 1 - pm
    cand = (prob > 0.35).astype(np.uint8)
    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    keep = np.zeros_like(cand)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if y <= h * 0.05 and area > 0.01 * h * w:          # touches the top
            keep[lab_cc == i] = 1
    if keep.sum() == 0:
        return np.zeros((H, W), np.float32)
    # close small holes (birds, wires) and refine edges to tree branches/buildings
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    gray = cv2.cvtColor(small.astype(np.float32), cv2.COLOR_RGB2GRAY)
    m = np.clip(guided_filter(gray, keep.astype(np.float32) * np.clip(prob * 1.5, 0, 1), 6, 1e-3), 0, 1)
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
    gfull, _ = _down(srgb, 2000)
    if gfull.shape[:2] != (H, W):
        return np.clip(m, 0, 1).astype(np.float32)
    return np.clip(guided_filter(cv2.cvtColor(srgb.astype(np.float32), cv2.COLOR_RGB2GRAY), m, 4, 2e-3), 0, 1)


# ================================================================ detail & effects

def sharpen_lr(srgb: np.ndarray, amount: float, radius: float = 1.0, detail: float = 25,
               masking: float = 0) -> np.ndarray:
    """Lightroom-style sharpening: amount 0..150, radius 0.5..3, detail 0..100
    (low = only strong edges), masking 0..100 (high = only edges, no flat areas)."""
    if amount <= 0:
        return srgb
    Y = cv2.transform(np.ascontiguousarray(srgb, np.float32), np.array([[0.2126, 0.7152, 0.0722]], np.float32))
    Y = Y.reshape(srgb.shape[:2])
    d = Y - cv2.GaussianBlur(Y, (0, 0), max(0.4, radius))
    thr = 0.012 * (1 - detail / 100) + 0.0015
    m = _ss(thr, thr * 2.5, np.abs(d))
    if masking > 0:
        gm = np.hypot(cv2.Sobel(Y, cv2.CV_32F, 1, 0), cv2.Sobel(Y, cv2.CV_32F, 0, 1))
        gm = cv2.GaussianBlur(gm, (0, 0), 1.5)
        t = masking / 100 * 0.35
        m = m * _ss(t * 0.5, t + 1e-4, gm)
    k = amount / 100.0 * (0.7 + 0.6 * detail / 100)
    return np.clip(srgb + (k * d * m)[..., None], 0.0, 1.0)


def vignette_lr(lin: np.ndarray, amount: float, midpoint: float = 50, feather: float = 50,
                roundness: float = 0, highlights: float = 0) -> np.ndarray:
    """Post-crop vignette: amount -100..100 (- darkens corners), midpoint,
    feather 0..100, roundness -100..100, highlights 0..100 (protect bright)."""
    if abs(amount) < 0.5:
        return lin
    H, W = lin.shape[:2]
    y, x = np.ogrid[-1:1:complex(0, H), -1:1:complex(0, W)]
    rnd = (roundness + 100) / 200          # 0 = follows frame shape, 1 = circle
    ax, ay = 1.0, 1.0
    if rnd > 0:
        ar = W / H
        ax, ay = (ar ** rnd, 1.0) if ar > 1 else (1.0, (1 / ar) ** rnd)
    r = np.sqrt((x * ax) ** 2 + (y * ay) ** 2) / np.sqrt(ax ** 2 + ay ** 2) * np.sqrt(2)
    m = 0.25 + 0.9 * midpoint / 100
    f = 0.1 + 0.9 * feather / 100
    fall = _ss(m - f * 0.5, m + f * 0.6, r)
    g = np.exp2(amount / 100 * 1.5 * fall).astype(np.float32)[..., None]
    out = lin * g
    if highlights > 0 and amount < 0:
        Y = lin.max(-1, keepdims=True)
        prot = _ss(0.6, 1.2, Y) * highlights / 100
        out = out * (1 - prot) + lin * prot
    return out


def grain_lr(srgb: np.ndarray, amount: float, size: float = 25, roughness: float = 50, seed: int = 7) -> np.ndarray:
    if amount <= 0:
        return srgb
    H, W = srgb.shape[:2]
    rng = np.random.default_rng(seed)
    cell = max(1.0, max(H, W) / 2500) * (0.5 + size / 25)
    n1 = rng.standard_normal((max(1, round(H / cell)), max(1, round(W / cell)))).astype(np.float32)
    n1 = cv2.resize(cv2.GaussianBlur(n1, (0, 0), 0.6), (W, H), interpolation=cv2.INTER_LINEAR)
    n2 = rng.standard_normal((max(1, round(H / (cell * 3))), max(1, round(W / (cell * 3))))).astype(np.float32)
    n2 = cv2.resize(n2, (W, H), interpolation=cv2.INTER_CUBIC)
    n = n1 * (1 - roughness / 200) + n2 * roughness / 200
    Y = srgb.mean(-1)
    w = 4 * Y * (1 - Y)
    return np.clip(srgb + (amount / 100 * 0.06 * n * w)[..., None], 0, 1)
