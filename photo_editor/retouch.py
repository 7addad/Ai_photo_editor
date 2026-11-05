"""Natural face retouching (beauty) - تجميل الوجه.

Works on a display-referred sRGB image, face by face, inside a crop around
each face so it is fast on 24-60 MP files.  Every step is restricted to real
skin (face oval minus eyes, brows, lips, nostrils, and minus hair/beard by a
colour check) and keeps pore texture, so the result does not look plastic.

Steps (each with its own strength 0..1):
  blemish    - pimples, spots, small marks: detected as small dark and/or red
               blobs at several scales, healed by inpainting + transplanted
               neighbouring skin texture.
  redness    - blotchy redness / irritation evened toward the face's own tone.
  smooth     - frequency separation: removes mid-frequency unevenness, keeps pores.
  under_eye  - lifts dark circles and removes their blue/purple tint.
  shine      - tames oily specular highlights.
  teeth      - whitens teeth (removes yellow) inside the mouth opening.
  eyes       - clears redness from the whites of the eyes.
"""
from __future__ import annotations

import cv2
import numpy as np

from .color import gamut_clip, linear_to_oklab, linear_to_srgb, oklab_to_linear, srgb_to_linear
from .face_landmarks import Face
from .segmentation import guided_filter

STEPS = ("blemish", "redness", "smooth", "under_eye", "shine", "teeth", "eyes")
# share of the master 'retouch' strength each step gets by default
DEFAULT_MIX = {"blemish": 1.0, "redness": 0.8, "smooth": 0.55, "under_eye": 0.6,
               "shine": 0.6, "teeth": 0.5, "eyes": 0.5}


def strengths(p: dict) -> dict:
    master = float(p.get("retouch", 0.0) or 0.0)
    out = {}
    for k in STEPS:
        v = p.get(f"retouch_{k}")
        out[k] = float(np.clip(master * DEFAULT_MIX[k] if v is None else float(v), 0.0, 1.0))
    return out


def _g(a, sigma):
    return cv2.GaussianBlur(a, (0, 0), max(0.3, float(sigma)), borderType=cv2.BORDER_REFLECT)


def _poly_mask(shape, poly, offset) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    if poly is not None and len(poly) >= 3:
        cv2.fillPoly(m, [np.round(poly - offset).astype(np.int32)], 1)
    return m


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class _FaceCtx:
    """Crop + masks for one face."""

    def __init__(self, img: np.ndarray, face: Face, extra_skin: np.ndarray | None):
        H, W = img.shape[:2]
        fw = max(face.width, 20.0)
        x0, y0 = face.oval.min(0) - fw * 0.25
        x1, y1 = face.oval.max(0) + fw * 0.25
        self.x0, self.y0 = int(max(0, x0)), int(max(0, y0))
        self.x1, self.y1 = int(min(W, x1)), int(min(H, y1))
        self.fw = fw
        off = np.array([self.x0, self.y0], np.float32)
        crop = img[self.y0:self.y1, self.x0:self.x1]
        shp = crop.shape[:2]
        self.lab = linear_to_oklab(srgb_to_linear(crop))
        self.orig = linear_to_srgb(gamut_clip(oklab_to_linear(self.lab)))   # round-trip reference

        oval = _poly_mask(shp, face.oval, off)
        excl = np.zeros(shp, np.uint8)
        for poly in face.eyes + face.brows + [face.lips, face.nostrils]:
            excl |= _poly_mask(shp, poly, off)
        k = max(3, int(fw * 0.035) | 1)
        excl_d = cv2.dilate(excl, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        base = (oval > 0) & (excl_d == 0)
        L, a, b = self.lab[..., 0], self.lab[..., 1], self.lab[..., 2]
        if base.sum() < 50:
            base = oval > 0
        # face's own skin tone (robust) and a colour-likelihood to drop hair/beard/glasses
        self.L0 = float(np.median(L[base])); self.a0 = float(np.median(a[base])); self.b0 = float(np.median(b[base]))
        # likelihood is computed on smoothed colour so that small spots (the
        # things we want to fix) stay inside the skin mask instead of being holes
        sg = max(1.0, fw * 0.02)
        ab, bb, Lb = _g(a, sg), _g(b, sg), _g(L, sg)
        d = np.sqrt((ab - self.a0) ** 2 + (bb - self.b0) ** 2)
        like = np.exp(-(d / 0.045) ** 2) * _smoothstep(self.L0 - 0.32, self.L0 - 0.18, Lb)
        skin = base * like
        if extra_skin is not None:
            es = extra_skin[self.y0:self.y1, self.x0:self.x1]
            skin = np.maximum(skin, es * (excl_d == 0) * like * 0.9)
        self.skin = cv2.GaussianBlur(skin.astype(np.float32), (0, 0), max(1.0, fw * 0.006))
        er = max(3, int(fw * 0.02) | 1)
        self.skin_core = cv2.erode((self.skin > 0.5).astype(np.uint8),
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er, er)))
        self.face, self.off, self.shape = face, off, shp

    def write(self, img: np.ndarray):
        """Write back only pixels that were actually changed, so everything
        outside the retouched skin stays bit-identical."""
        new = linear_to_srgb(gamut_clip(oklab_to_linear(self.lab)))
        old = img[self.y0:self.y1, self.x0:self.x1]
        changed = np.abs(new - self.orig).max(-1, keepdims=True) > 2e-4
        img[self.y0:self.y1, self.x0:self.x1] = np.where(changed, new, old)


# ---------------------------------------------------------------- steps

def _masked_blur(x, m, sigma):
    """Gaussian blur that only averages pixels where m > 0 (normalized
    convolution) - neighbouring hair, eyes or background can't leak in."""
    return _g(x * m, sigma) / np.maximum(_g(m, sigma), 1e-4)


def _blemishes(ctx: _FaceCtx, s: float, srgb_crop: np.ndarray) -> tuple[np.ndarray, int]:
    """Find and heal pimples / spots.  Returns healed sRGB crop and count."""
    L, a = ctx.lab[..., 0], ctx.lab[..., 1]
    fw = ctx.fw
    er = max(3, int(fw * 0.03) | 1)
    core = cv2.erode(ctx.skin_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er, er))) > 0
    if core.sum() < 100:
        return srgb_crop, 0
    m = (ctx.skin > 0.35).astype(np.float32)
    score = np.zeros_like(L)
    for frac in (0.005, 0.008, 0.013):
        sg = fw * frac
        dark = _masked_blur(L, m, sg * 3) - _masked_blur(L, m, sg)
        red = _masked_blur(a, m, sg) - _masked_blur(a, m, sg * 3)
        sd = 1.4826 * np.median(np.abs(dark[core] - np.median(dark[core]))) + 1e-4
        sr = 1.4826 * np.median(np.abs(red[core] - np.median(red[core]))) + 1e-4
        dn, rn = dark / sd, red / sr
        sc = np.maximum.reduce([dn, rn, (np.maximum(dn, 0) + np.maximum(rn, 0)) * 0.6])
        score = np.maximum(score, sc)
    thr = 5.2 - 1.8 * s
    cand = ((score > thr) & core).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    rmax = fw * 0.035
    blem = np.zeros_like(cand)
    count = 0
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        sel = lab[y:y + h, x:x + w] == i
        sc = score[y:y + h, x:x + w]
        peak = float(sc[sel].max())
        if peak < thr + 0.8:
            continue
        # measure the blob at half its peak response: independent of how noisy
        # (and therefore how sensitive) the skin is
        sel = sel & (sc > max(thr, 0.4 * peak))
        area = int(sel.sum())
        if area == 0 or area > np.pi * rmax ** 2:
            continue                                   # too big: shadow / feature
        if area >= 6:
            ys, xs = np.nonzero(sel)
            ev = np.linalg.eigvalsh(np.cov(np.stack([xs, ys]).astype(np.float64)) + np.eye(2) * 0.25)
            if ev[1] / max(ev[0], 1e-6) > 4.0:
                continue                               # a line: wrinkle, crease, hair - not a spot
        blem[y:y + h, x:x + w][sel] = 1
        count += 1
    if count == 0:
        return srgb_crop, 0
    blem = cv2.dilate(blem, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(fw * 0.012) | 1),) * 2))
    keep = ((1 - blem) * (ctx.skin > 0.2)).astype(np.float32)
    # smooth fill from the surrounding clean skin
    sig = max(1.5, fw * 0.01)
    base = _g(srgb_crop * keep[..., None], sig) / np.maximum(_g(keep, sig), 1e-4)[..., None]
    # transplant fine skin texture from a nearby clean patch so the spot isn't flat
    fine = srgb_crop - _g(srgb_crop, max(0.6, fw * 0.003))
    tex = np.zeros_like(srgb_crop)
    n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(blem, connectivity=8)
    H, W = blem.shape
    for i in range(1, n2):
        x, y, w, h, _ = stats2[i]
        r = max(w, h)
        best, best_cov = None, -1e9
        for ang in range(0, 360, 45):
            dx = int(round(np.cos(np.radians(ang)) * r * 1.5))
            dy = int(round(np.sin(np.radians(ang)) * r * 1.5))
            if not (0 <= x + dx and x + w + dx <= W and 0 <= y + dy and y + h + dy <= H):
                continue
            cov = float(ctx.skin_core[y + dy:y + h + dy, x + dx:x + w + dx].mean()) \
                - 2.0 * float(blem[y + dy:y + h + dy, x + dx:x + w + dx].mean())
            if cov > best_cov:
                best, best_cov = (dx, dy), cov
        if best is None:
            continue
        dx, dy = best
        sel = lab2[y:y + h, x:x + w] == i
        tex[y:y + h, x:x + w][sel] = fine[y + dy:y + h + dy, x + dx:x + w + dx][sel]
    healed = base + tex
    wgt = cv2.GaussianBlur(blem.astype(np.float32), (0, 0), max(0.8, fw * 0.004))[..., None]
    wgt = np.clip(wgt * 1.6, 0, 1) * min(1.0, 0.4 + s)
    return np.clip(srgb_crop * (1 - wgt) + healed * wgt, 0, 1), count


def _redness(ctx: _FaceCtx, s: float):
    a = ctx.lab[..., 1]
    a_mid = _g(a, ctx.fw * 0.02)
    excess = np.maximum(a_mid - ctx.a0 - 0.004, 0)
    ctx.lab[..., 1] = a - s * 0.9 * excess * ctx.skin
    # slight lift of the darkening that comes with redness
    ctx.lab[..., 0] += s * 0.6 * excess * ctx.skin


def _smooth(ctx: _FaceCtx, s: float):
    fw = ctx.fw
    L = ctx.lab[..., 0]
    r = max(2, int(fw * 0.03))
    low = guided_filter(L, L, r, 4e-4)                  # edge-aware: keeps nose/jaw edges
    fine_base = _g(L, max(0.5, fw * 0.0035))
    fine = L - fine_base                                 # pores / micro texture - kept
    mid = fine_base - low                                # blotches, bumps, uneven tone - reduced
    m = ctx.skin * s
    ctx.lab[..., 0] = L - m * (0.8 * mid + 0.12 * fine)
    for c in (1, 2):
        ch = ctx.lab[..., c]
        lowc = _g(ch, fw * 0.025)
        ctx.lab[..., c] = ch - m * 0.65 * (ch - lowc)


def _under_eye(ctx: _FaceCtx, s: float):
    fw, off, shp = ctx.fw, ctx.off, ctx.shape
    L, b = ctx.lab[..., 0], ctx.lab[..., 2]
    Llow, blow = _g(L, fw * 0.02), _g(b, fw * 0.02)
    region = np.zeros(shp, np.float32)
    eyes_m = np.zeros(shp, np.uint8)
    for e in ctx.face.eyes:
        eyes_m |= _poly_mask(shp, e, off)
        ew = float(e[:, 0].max() - e[:, 0].min())
        cx = float(e[:, 0].mean()) - off[0]
        bottom = float(e[:, 1].max()) - off[1]
        cv2.ellipse(region, (int(cx), int(bottom + ew * 0.22)), (max(1, int(ew * 0.55)), max(1, int(ew * 0.25))),
                    0, 0, 360, 1.0, -1)
    eyes_d = cv2.dilate(eyes_m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(fw * 0.03) | 1),) * 2))
    region *= (eyes_d == 0)
    region = cv2.GaussianBlur(region, (0, 0), fw * 0.02) * np.clip(ctx.skin * 1.5, 0, 1)
    if region.max() < 0.05:
        return
    cheek = (ctx.skin > 0.6) & (region < 0.05)
    Lref = float(np.percentile(L[cheek], 60)) if cheek.sum() > 50 else ctx.L0
    ctx.lab[..., 0] = L + s * 0.75 * np.maximum(Lref - Llow, 0) * region
    ctx.lab[..., 2] = b + s * 0.6 * np.maximum(ctx.b0 - blow, 0) * region   # remove blue/purple


def _shine(ctx: _FaceCtx, s: float):
    L, a, b = ctx.lab[..., 0], ctx.lab[..., 1], ctx.lab[..., 2]
    core = ctx.skin > 0.5
    if core.sum() < 50:
        return
    Ls = _g(L, ctx.fw * 0.008)
    p85, p99 = np.percentile(Ls[core], [85, 99.5])
    if p99 - p85 < 0.03:
        return
    C = np.hypot(a, b)
    C0 = float(np.hypot(ctx.a0, ctx.b0))
    w = _smoothstep(p85, p99, Ls) * _smoothstep(C0 * 1.0, C0 * 0.55, C) * ctx.skin * s
    ctx.lab[..., 0] = L - w * 0.55 * (Ls - p85)
    ctx.lab[..., 1] = a + w * 0.5 * (ctx.a0 - a)
    ctx.lab[..., 2] = b + w * 0.5 * (ctx.b0 - b)


def _teeth(ctx: _FaceCtx, s: float):
    if ctx.face.mouth is None:
        return
    m = _poly_mask(ctx.shape, ctx.face.mouth, ctx.off).astype(np.float32)
    if m.sum() < 20:
        return
    L, a, b = ctx.lab[..., 0], ctx.lab[..., 1], ctx.lab[..., 2]
    inside = m > 0
    Lmed = float(np.median(L[inside]))
    teeth = m * _smoothstep(Lmed - 0.05, Lmed + 0.08, L) * _smoothstep(0.04, -0.01, a - 0.0)  # bright, not red (gums/lips)
    teeth = cv2.GaussianBlur(teeth, (0, 0), max(0.7, ctx.fw * 0.003)) * s
    ctx.lab[..., 2] = b - teeth * 0.75 * np.maximum(b - 0.0, 0)
    ctx.lab[..., 1] = a - teeth * 0.4 * np.maximum(a, 0)
    ctx.lab[..., 0] = L + teeth * 0.05 * (1 - L)


def _eyes(ctx: _FaceCtx, s: float):
    if not ctx.face.eyes:
        return
    shp, off = ctx.shape, ctx.off
    m = np.zeros(shp, np.float32)
    for e in ctx.face.eyes:
        m = np.maximum(m, _poly_mask(shp, e, off).astype(np.float32))
    for cx, cy, r in ctx.face.irises:
        cv2.circle(m, (int(cx - off[0]), int(cy - off[1])), int(r * 1.1), 0.0, -1)
    if not ctx.face.irises:          # without iris landmarks only touch bright, reddish pixels
        m *= _smoothstep(ctx.L0, ctx.L0 + 0.1, ctx.lab[..., 0])
    m = cv2.GaussianBlur(m, (0, 0), max(0.6, ctx.fw * 0.002)) * s
    a, L = ctx.lab[..., 1], ctx.lab[..., 0]
    sclera = m * _smoothstep(ctx.L0 - 0.1, ctx.L0 + 0.05, L)
    ctx.lab[..., 1] = a - sclera * 0.7 * np.maximum(a, 0)
    ctx.lab[..., 0] = L + sclera * 0.04 * (1 - L)


# ---------------------------------------------------------------- public

def retouch(srgb: np.ndarray, faces: list[Face], params: dict, skin: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Apply beauty retouching to every face.  Returns (image, report)."""
    st = strengths(params)
    rep = {"faces": len(faces), "strengths": {k: round(v, 2) for k, v in st.items()}, "blemishes_removed": 0}
    if not faces or max(st.values()) <= 0:
        return srgb, rep
    out = np.ascontiguousarray(srgb, dtype=np.float32).copy()
    for f in faces:
        if f.width < 40:
            continue
        ctx = _FaceCtx(out, f, skin)
        if st["blemish"] > 0:
            crop = out[ctx.y0:ctx.y1, ctx.x0:ctx.x1]
            healed, n = _blemishes(ctx, st["blemish"], crop)
            rep["blemishes_removed"] += n
            if n:
                ctx.lab = linear_to_oklab(srgb_to_linear(healed))
        if st["redness"] > 0:
            _redness(ctx, st["redness"])
        if st["smooth"] > 0:
            _smooth(ctx, st["smooth"])
        if st["under_eye"] > 0:
            _under_eye(ctx, st["under_eye"])
        if st["shine"] > 0:
            _shine(ctx, st["shine"])
        if st["teeth"] > 0:
            _teeth(ctx, st["teeth"])
        if st["eyes"] > 0:
            _eyes(ctx, st["eyes"])
        ctx.write(out)
    rep["landmarks"] = sorted({f.source for f in faces})
    return np.clip(out, 0, 1), rep
