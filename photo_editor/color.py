"""Colour-science helpers: sRGB transfer, OKLab/OKLCh, CCT <-> chromaticity.

All "linear" images in this package are float32 arrays in linear Rec.709/sRGB
primaries with D65 white, shape (H, W, 3).
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- sRGB transfer

def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return np.where(v <= 0.04045, v / 12.92, ((np.maximum(v, 0) + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(v: np.ndarray) -> np.ndarray:
    v = np.clip(np.asarray(v, dtype=np.float32), 0.0, None)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1 / 2.4) - 0.055).astype(np.float32)


def luminance(lin: np.ndarray) -> np.ndarray:
    """Relative luminance (Y) of linear sRGB."""
    return (lin[..., 0] * 0.2126 + lin[..., 1] * 0.7152 + lin[..., 2] * 0.0722).astype(np.float32)


# ---------------------------------------------------------------- OKLab

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]], dtype=np.float32)
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]], dtype=np.float32)
_M1_INV = np.linalg.inv(_M1.astype(np.float64)).astype(np.float32)
_M2_INV = np.linalg.inv(_M2.astype(np.float64)).astype(np.float32)


def linear_to_oklab(lin: np.ndarray) -> np.ndarray:
    lms = lin @ _M1.T
    lms = np.cbrt(lms)
    return (lms @ _M2.T).astype(np.float32)


def oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    lms = lab @ _M2_INV.T
    lms = lms * lms * lms
    return (lms @ _M1_INV.T).astype(np.float32)


def gamut_clip(lin: np.ndarray) -> np.ndarray:
    """Bring out-of-range linear RGB back in [0,1] by desaturating toward its own
    luminance (keeps hue and brightness instead of hard per-channel clipping).
    Only the out-of-gamut pixels are touched, so this is cheap on big images."""
    lin = np.asarray(lin, dtype=np.float32)
    bad = (lin.min(-1) < 0.0) | (lin.max(-1) > 1.0)
    if not bad.any():
        return lin
    out = lin.copy()
    px = lin[bad]
    y = np.clip(px @ np.array([0.2126, 0.7152, 0.0722], np.float32), 0.0, 1.0)[:, None]
    d = px - y
    with np.errstate(divide="ignore", invalid="ignore"):
        t_hi = np.where(d > 1e-6, (1.0 - y) / d, np.inf)
        t_lo = np.where(d < -1e-6, (0.0 - y) / d, np.inf)
    t = np.clip(np.minimum(t_hi.min(-1), t_lo.min(-1)), 0.0, 1.0)[:, None]
    out[bad] = np.clip(y + d * t, 0.0, 1.0)
    return out


# ---------------------------------------------------------------- CCT / chromaticity

_XYZ_TO_SRGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                         [-0.9692660, 1.8760108, 0.0415560],
                         [0.0556434, -0.2040259, 1.0572252]])
_SRGB_TO_XYZ = np.linalg.inv(_XYZ_TO_SRGB)


def cct_to_xy(T: float) -> tuple[float, float]:
    """Planckian-locus chromaticity (Kim et al. 2002 cubic spline), 1667K-25000K."""
    T = float(np.clip(T, 1667.0, 25000.0))
    if T <= 4000:
        x = -0.2661239e9 / T**3 - 0.2343589e6 / T**2 + 0.8776956e3 / T + 0.179910
    else:
        x = -3.0258469e9 / T**3 + 2.1070379e6 / T**2 + 0.2226347e3 / T + 0.240390
    if T <= 2222:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif T <= 4000:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    return x, y


def xy_to_rgb(x: float, y: float) -> np.ndarray:
    """Linear sRGB colour of a white (Y=1) with chromaticity xy, normalised so G=1."""
    X, Y, Z = x / y, 1.0, (1 - x - y) / y
    rgb = _XYZ_TO_SRGB @ np.array([X, Y, Z])
    rgb = np.maximum(rgb, 1e-4)
    return rgb / rgb[1]


def rgb_to_xy(rgb) -> tuple[float, float]:
    X, Y, Z = _SRGB_TO_XYZ @ np.asarray(rgb, dtype=np.float64)
    s = X + Y + Z
    return X / s, Y / s


def _xy_to_uv(x, y):
    d = -2 * x + 12 * y + 3
    return 4 * x / d, 6 * y / d


def _uv_to_xy(u, v):
    d = 2 * u - 8 * v + 4
    return 3 * u / d, 2 * v / d


_LOCUS_T = np.concatenate([np.arange(1667, 4000, 10), np.arange(4000, 25001, 50)]).astype(np.float64)
_LOCUS_UV = np.array([_xy_to_uv(*cct_to_xy(t)) for t in _LOCUS_T])


def xy_to_cct_duv(x: float, y: float) -> tuple[float, float]:
    """Nearest point on the Planckian locus: returns (CCT in K, signed Duv)."""
    u, v = _xy_to_uv(x, y)
    d = np.hypot(_LOCUS_UV[:, 0] - u, _LOCUS_UV[:, 1] - v)
    i = int(np.argmin(d))
    # sign: positive above locus (greenish), negative below (magenta)
    j = min(i + 1, len(_LOCUS_T) - 1)
    k = max(i - 1, 0)
    tu, tv = _LOCUS_UV[j] - _LOCUS_UV[k]
    cross = tu * (v - _LOCUS_UV[i, 1]) - tv * (u - _LOCUS_UV[i, 0])
    return float(_LOCUS_T[i]), float(np.sign(cross) * d[i])


def cct_duv_to_xy(T: float, duv: float) -> tuple[float, float]:
    x, y = cct_to_xy(T)
    u, v = _xy_to_uv(x, y)
    x2, y2 = cct_to_xy(T + 1.0 if T < 24999 else T - 1.0)
    u2, v2 = _xy_to_uv(x2, y2)
    tu, tv = u2 - u, v2 - v
    n = np.hypot(tu, tv)
    if T >= 24999:
        tu, tv = -tu, -tv
    nu, nv = -tv / n, tu / n  # normal pointing "above" the locus
    return _uv_to_xy(u + duv * nu, v + duv * nv)


# Reference white of the working space: the Planckian locus at 6504K (within
# 0.003 Duv of D65), so "6500K, tint 0" renders exactly neutral.
_REF_WHITE = xy_to_rgb(*cct_to_xy(6504.0))


def illuminant_rgb(T: float, duv: float = 0.0) -> np.ndarray:
    """Linear-sRGB colour (G=1) of a white surface lit by a light of CCT T,
    as rendered by a D65-balanced pipeline."""
    rgb = xy_to_rgb(*cct_duv_to_xy(T, duv)) / _REF_WHITE
    return rgb / rgb[1]


def illuminant_to_cct(e) -> tuple[float, float]:
    """Inverse of :func:`illuminant_rgb` for an estimated illuminant colour."""
    return xy_to_cct_duv(*rgb_to_xy(np.asarray(e, dtype=np.float64) * _REF_WHITE))


def describe_cct(T: float) -> str:
    if T < 2700:
        return "candle / very warm tungsten"
    if T < 3500:
        return "tungsten / warm indoor light"
    if T < 4500:
        return "warm white LED / fluorescent / golden hour"
    if T < 5600:
        return "neutral / direct sun"
    if T < 7000:
        return "daylight / overcast"
    return "open shade / blue hour (cool)"
