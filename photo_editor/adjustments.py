"""Image adjustments.  Scene-referred ones work on linear light; the look
(contrast, colour, grading) works in OKLab on display-referred values."""
from __future__ import annotations

import cv2
import numpy as np

from .color import gamut_clip, linear_to_oklab, linear_to_srgb, luminance, oklab_to_linear, srgb_to_linear  # noqa: F401
from .segmentation import guided_filter


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.3:
        return a
    # blur large radii on a downscaled copy for speed
    h, w = a.shape[:2]
    if sigma > 12:
        f = sigma / 6.0
        sm = cv2.resize(a, (max(1, round(w / f)), max(1, round(h / f))), interpolation=cv2.INTER_AREA)
        sm = cv2.GaussianBlur(sm, (0, 0), 6.0, borderType=cv2.BORDER_REFLECT)
        return cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(a, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------- exposure

def auto_exposure_ev(lin: np.ndarray, person: np.ndarray | None = None, target: float = 0.18,
                     person_weight: float = 0.6) -> float:
    """EV that brings the log-average luminance to mid-grey, metering people
    more when present, while protecting highlights."""
    small = cv2.resize(lin, (512, max(1, round(512 * lin.shape[0] / lin.shape[1]))), interpolation=cv2.INTER_AREA)
    Y = np.maximum(luminance(small), 1e-5)
    logavg = float(np.exp(np.mean(np.log(Y))))
    meter = logavg
    if person is not None and person.mean() > 0.01:
        pm = cv2.resize(person, Y.shape[::-1], interpolation=cv2.INTER_AREA)
        w = pm > 0.5
        if w.sum() > 50:
            p_avg = float(np.exp(np.mean(np.log(Y[w]))))
            # people should sit a little brighter than mid grey
            meter = np.exp((1 - person_weight) * np.log(logavg) + person_weight * np.log(p_avg / 1.25))
    ev = float(np.log2(target / max(meter, 1e-5)))
    # highlight guard: don't push the 99.7th percentile far past the shoulder
    p_hi = float(np.percentile(Y, 99.7))
    ev_hi = float(np.log2(6.0 / max(p_hi, 1e-5)))
    return float(np.clip(min(ev, ev_hi), -3.0, 3.5))


# ---------------------------------------------------------------- local tone (highlights / shadows)

def local_tone(lin: np.ndarray, highlights: float, shadows: float) -> np.ndarray:
    """Edge-aware highlight recovery / shadow lift.  The gain is computed from a
    guided-filtered base layer, so local contrast and texture are preserved."""
    if abs(highlights) < 1e-3 and abs(shadows) < 1e-3:
        return lin
    H, W = lin.shape[:2]
    s = min(1.0, 800 / max(H, W))
    Y = cv2.resize(luminance(lin), (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_AREA)
    L = np.log2(np.maximum(Y, 1e-5) / 0.18)                # stops from mid grey
    base = guided_filter(L, L, max(4, round(max(Y.shape) / 40)), 0.5)
    base = cv2.resize(base, (W, H), interpolation=cv2.INTER_LINEAR)
    stops = np.zeros_like(base)
    # highlights: act on everything brighter than ~1 stop above mid grey
    hw = smoothstep(0.5, 3.0, base)
    stops += highlights * 1.6 * hw
    # shadows: lift what is darker than ~1.5 stops below mid grey
    sw = smoothstep(-0.5, -4.0, base)
    stops += shadows * 1.8 * sw
    return (lin * np.exp2(stops)[..., None]).astype(np.float32)


# ---------------------------------------------------------------- tone mapping

def tone_map(lin: np.ndarray, white: float = 8.0, highlight_desat: float = 0.6,
             scene_referred: bool = True) -> np.ndarray:
    """Scene linear -> display linear with a soft highlight shoulder.
    Hue-preserving (applied to a norm, not per channel) with gentle
    desaturation of the brightest highlights like film.

    ``scene_referred=False`` is for JPEG/PNG input that already went through a
    camera tone curve: values are left alone up to 0.8 and only the part
    pushed above that by our edits is rolled off softly."""
    lin = np.maximum(lin, 0.0, dtype=np.float32)
    Y = luminance(lin)
    n = np.maximum(0.6 * lin.max(-1) + 0.4 * Y, 1e-6)
    if scene_referred:
        x = n * np.float32(1.25)
        t = x * (1 + x / np.float32(white * white)) / (1 + x)
    else:
        over = np.maximum(n - 0.8, 0)
        t = np.minimum(n, 0.8) + 0.2 * (1 - np.exp(-over / 0.2))
        highlight_desat *= 0.5
    ratio = t / n
    out = lin * ratio[..., None]
    if highlight_desat > 0:
        k = smoothstep(0.6, 1.0, t) * np.float32(highlight_desat)
        yo = Y * ratio
        out += (yo[..., None] - out) * k[..., None]
    return gamut_clip(out)


# ---------------------------------------------------------------- display-referred look (OKLab)

def contrast_curve(L: np.ndarray, contrast: float, pivot: float = 0.56) -> np.ndarray:
    """Sigmoid contrast on OKLab L around ``pivot`` (0.56 ≈ 18% grey)."""
    if abs(contrast) < 1e-3:
        return L
    k = 1.0 + contrast
    Lc = np.clip(L, 1e-4, 1 - 1e-4)
    r = ((1 - Lc) / Lc) ** k * ((1 - pivot) / pivot) ** (1 - k)
    return (1.0 / (1.0 + r)).astype(np.float32)


def levels(L: np.ndarray, blacks: float, whites: float) -> np.ndarray:
    """blacks > 0 lifts (matte), < 0 crushes; whites > 0 brightens top end."""
    out = L
    if whites:
        out = out * (1 + 0.25 * whites) if whites > 0 else out * (1 + 0.2 * whites)
        out = np.where(out > 0.85, 0.85 + (out - 0.85) / (1 + 2 * np.maximum(out - 0.85, 0)), out) if whites > 0 else out
    if blacks > 0:
        out = blacks * 0.12 + out * (1 - blacks * 0.12)
    elif blacks < 0:
        b = -blacks * 0.06
        out = np.clip((out - b) / (1 - b), 0, None)
    return out.astype(np.float32)


HUE_BANDS = {  # OKLab hue centres in degrees
    "red": 20, "orange": 55, "yellow": 95, "green": 140,
    "aqua": 195, "blue": 255, "purple": 300, "magenta": 340,
}


def _hue_weight(h_deg: np.ndarray, centre: float, width: float = 32.0) -> np.ndarray:
    d = np.abs((h_deg - centre + 180.0) % 360.0 - 180.0)
    return np.exp(-(d / width) ** 2).astype(np.float32)


def apply_look(srgb_lin: np.ndarray, p: dict, skin: np.ndarray | None = None,
               person: np.ndarray | None = None) -> np.ndarray:
    """Contrast, levels, vibrance/saturation, HSL, split toning, B&W - all in
    OKLab on display-linear input.  Returns display-linear."""
    lab = linear_to_oklab(np.maximum(srgb_lin, 0))
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a))

    # --- tone
    # RAW data gets a base "camera-like" contrast and black point; JPEG/PNG
    # already have them baked in (pipeline sets _base_contrast/_black_point)
    bp = float(p.get("_black_point", 0.0))
    if bp > 0:
        L = np.clip((L - bp) / (1 - bp), 0, None)
    L = contrast_curve(L, float(p.get("_base_contrast", 0.0)) + float(p.get("contrast", 0.0)))
    L = levels(L, float(p.get("blacks", 0.0)), float(p.get("whites", 0.0)))

    # --- colour
    skin_w = np.zeros_like(L) if skin is None else skin
    # skin-like colours (hue ~58deg, moderate chroma) are protected even outside the skin mask
    skin_hue_w = _hue_weight(h, 58, 22) * smoothstep(0.02, 0.05, C) * (1 - smoothstep(0.10, 0.16, C))
    protect = np.maximum(skin_w, 0.5 * skin_hue_w)
    sat = float(p.get("saturation", 0.0))
    vib = float(p.get("vibrance", 0.0))
    scale = 1.0 + sat * (1 - 0.5 * protect)
    if vib:
        scale = scale + vib * (1 - smoothstep(0.0, 0.2, C)) * (1 - 0.7 * protect)
    if person is not None:
        ps = float(p.get("subject_saturation", 0.0))
        bs = float(p.get("bg_saturation", 0.0))
        scale = scale + ps * person + bs * (1 - person)

    hue_shift = np.zeros_like(L)
    for name, centre in HUE_BANDS.items():
        hs = float(p.get(f"hue_{name}", 0.0))
        ss = float(p.get(f"sat_{name}", 0.0))
        ls = float(p.get(f"lum_{name}", 0.0))
        if not (hs or ss or ls):
            continue
        wgt = _hue_weight(h, centre) * smoothstep(0.0, 0.04, C)
        hue_shift += hs * wgt
        scale = scale + ss * wgt
        L = L + ls * 0.15 * wgt
    C = C * np.maximum(scale, 0.0)
    h = h + hue_shift

    # skin: nudge hue toward a healthy range and calm over-saturated/red skin
    skin_fix = float(p.get("skin_tone", 0.0))
    if skin_fix and skin is not None:
        d = ((58.0 - h + 180) % 360) - 180
        h = h + skin_fix * 0.5 * np.clip(d, -15, 15) * skin
        C = C * (1 - skin_fix * 0.25 * skin * smoothstep(0.08, 0.16, C))

    hr = np.radians(h)
    a, b = C * np.cos(hr), C * np.sin(hr)

    # --- split toning / colour grading: (hue degrees, amount) for shadows & highlights
    for key, wfun in (("shadows", lambda x: 1 - smoothstep(0.15, 0.6, x)),
                      ("highlights", lambda x: smoothstep(0.45, 0.95, x))):
        amt = float(p.get(f"grade_{key}_amount", 0.0))
        if amt:
            hh = np.radians(float(p.get(f"grade_{key}_hue", 0.0)))
            w = wfun(L) * amt * 0.06
            a = a + w * np.cos(hh)
            b = b + w * np.sin(hh)

    # --- black & white
    if p.get("monochrome"):
        mix = p.get("bw_mix", {"red": 0.1, "orange": 0.15, "yellow": 0.05, "blue": -0.15, "green": -0.05})
        for name, v in mix.items():
            L = L + float(v) * 0.12 * _hue_weight(np.degrees(np.arctan2(lab[..., 2], lab[..., 1])), HUE_BANDS[name]) \
                * smoothstep(0.0, 0.1, np.hypot(lab[..., 1], lab[..., 2]))
        a = np.zeros_like(a)
        b = np.zeros_like(b)
        tone_amt = float(p.get("bw_tone_amount", 0.0))
        if tone_amt:
            hh = np.radians(float(p.get("bw_tone_hue", 60)))
            a = a + tone_amt * 0.03 * np.cos(hh)
            b = b + tone_amt * 0.03 * np.sin(hh)

    out = oklab_to_linear(np.stack([np.clip(L, 0, 1), a, b], -1).astype(np.float32))
    return gamut_clip(out)


# ---------------------------------------------------------------- detail

def _luma(srgb: np.ndarray) -> np.ndarray:
    y = cv2.transform(np.ascontiguousarray(srgb, dtype=np.float32),
                      np.array([[0.2126, 0.7152, 0.0722]], np.float32))
    return y.reshape(srgb.shape[:2])


def local_contrast(srgb: np.ndarray, amount: float, sigma: float, mask: np.ndarray | None = None) -> np.ndarray:
    """Clarity / texture: boosts luma detail at scale ``sigma`` (display sRGB
    in/out).  The same delta is added to R, G and B so hue is preserved."""
    if abs(amount) < 1e-3:
        return srgb
    Y = _luma(srgb)
    detail = Y - _blur(Y, sigma)
    mid = 1.0 - np.square(2.0 * Y - 1.0)                # protect deep shadows / highlights
    k = amount * mid if mask is None else amount * mid * mask
    delta = (k * detail * 1.5).astype(np.float32)
    return np.clip(srgb + delta[..., None], 0.0, 1.0)


def sharpen(srgb: np.ndarray, amount: float, radius: float = 1.0, threshold: float = 0.01) -> np.ndarray:
    """Unsharp mask on luma with a noise threshold."""
    if amount <= 0:
        return srgb
    Y = _luma(srgb)
    d = Y - cv2.GaussianBlur(Y, (0, 0), radius)
    m = smoothstep(threshold, threshold * 3, np.abs(d))
    return np.clip(srgb + (amount * d * m)[..., None], 0.0, 1.0)


def denoise(lin: np.ndarray, luma: float, chroma: float) -> np.ndarray:
    """Noise reduction on linear data: chroma noise is smoothed with a guided
    filter steered by luminance, luminance noise with a bilateral filter."""
    if luma <= 0 and chroma <= 0:
        return lin
    lab = linear_to_oklab(np.maximum(lin, 0))
    L = lab[..., 0]
    H, W = L.shape
    scale = max(H, W) / 3000.0
    if chroma > 0:
        r = max(2, round(6 * chroma * max(scale, 0.5)))
        for c in (1, 2):
            lab[..., c] = guided_filter(L, lab[..., c], r, 4e-4)
    if luma > 0:
        d = max(3, int(round(5 * max(scale, 0.6))) | 1)
        Ld = cv2.bilateralFilter(L, d, sigmaColor=0.02 + 0.06 * luma, sigmaSpace=d)
        lab[..., 0] = L * (1 - min(luma, 1.0) * 0.85) + Ld * min(luma, 1.0) * 0.85
    return np.maximum(oklab_to_linear(lab), 0).astype(np.float32)


def skin_smooth(srgb: np.ndarray, skin: np.ndarray, amount: float) -> np.ndarray:
    """Gentle frequency-separation style smoothing restricted to skin."""
    if amount <= 0 or skin is None or skin.max() < 0.05:
        return srgb
    H, W = skin.shape
    sigma = max(1.5, max(H, W) / 900)
    lab = linear_to_oklab(srgb_to_linear(srgb))
    L = lab[..., 0]
    low = guided_filter(L, L, max(2, round(sigma * 2)), 1e-3)
    fine = L - cv2.GaussianBlur(L, (0, 0), sigma * 0.35)     # pores/texture to keep
    m = cv2.GaussianBlur(skin, (0, 0), sigma) * min(amount, 1.0)
    lab[..., 0] = L * (1 - m) + (low + fine * 0.6) * m
    return linear_to_srgb(gamut_clip(oklab_to_linear(lab)))


def vignette(lin: np.ndarray, amount: float, midpoint: float = 0.6) -> np.ndarray:
    if abs(amount) < 1e-3:
        return lin
    H, W = lin.shape[:2]
    y, x = np.ogrid[-1:1:complex(0, H), -1:1:complex(0, W)]
    ar = W / H
    r = np.sqrt((x * min(ar, 1.6) / 1.6) ** 2 * 1.0 + y ** 2 * 1.0) / np.sqrt(2) * 1.4
    fall = smoothstep(midpoint, 1.35, r)
    g = np.exp2(-amount * 1.4 * fall).astype(np.float32)
    return lin * g[..., None]


def film_grain(srgb: np.ndarray, amount: float, seed: int = 7) -> np.ndarray:
    if amount <= 0:
        return srgb
    H, W = srgb.shape[:2]
    rng = np.random.default_rng(seed)
    size = max(1.0, max(H, W) / 2500)
    n = rng.standard_normal((max(1, round(H / size)), max(1, round(W / size)))).astype(np.float32)
    n = cv2.resize(cv2.GaussianBlur(n, (0, 0), 0.6), (W, H), interpolation=cv2.INTER_LINEAR)
    Y = srgb.mean(-1)
    w = 4 * Y * (1 - Y)                                      # strongest in midtones
    return np.clip(srgb + (amount * 0.05 * n * w)[..., None], 0, 1)


# ---------------------------------------------------------------- subject / background separation

def separate_subject(lin: np.ndarray, person: np.ndarray, p: dict) -> np.ndarray:
    """Subtle 'pop': lens-like background blur (in linear light, halo-free),
    background exposure drop and subject lift.  All controlled by params:
    bg_blur (0..1), bg_exposure (EV), subject_exposure (EV)."""
    if person is None or person.max() < 0.05:
        return lin
    m = np.clip(person, 0, 1).astype(np.float32)
    out = lin
    blur = float(p.get("bg_blur", 0.0))
    if blur > 0:
        H, W = m.shape
        sigma = blur * 0.010 * max(H, W)
        # exclude the subject (slightly dilated) from the blur so it can't bleed
        k = max(3, int(sigma * 0.6) | 1)
        md = cv2.dilate(m, np.ones((k, k), np.uint8))
        wbg = (1.0 - md)[..., None]
        num = _blur(lin * wbg, sigma)
        den = _blur(wbg, sigma)
        den = den if den.ndim == 3 else den[..., None]
        bgb = num / np.maximum(den, 1e-4)
        # where the background is fully hidden (den≈0) fall back to a plain blur
        plain = _blur(lin, sigma)
        trust = np.clip(den * 4, 0, 1)
        bgb = bgb * trust + plain * (1 - trust)
        m3 = m[..., None]
        out = out * m3 + bgb * (1 - m3)
    ev_bg = float(p.get("bg_exposure", 0.0))
    ev_sub = float(p.get("subject_exposure", 0.0))
    if ev_bg or ev_sub:
        soft = cv2.GaussianBlur(m, (0, 0), max(1.0, max(m.shape) / 300))
        gain = np.exp2(ev_sub * soft + ev_bg * (1 - soft))
        out = out * gain[..., None]
    return out.astype(np.float32)
