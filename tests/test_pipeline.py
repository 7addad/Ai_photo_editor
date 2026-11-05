"""Fast tests on synthetic images (no models or RAW samples needed).

Run:  python -m pytest -q tests
"""
import numpy as np
import pytest

from photo_editor import presets
from photo_editor.color import (gamut_clip, illuminant_rgb, illuminant_to_cct, linear_to_oklab,
                                linear_to_srgb, oklab_to_linear, srgb_to_linear)
from photo_editor.pipeline import process
from photo_editor.raw_io import ImageData
from photo_editor.segmentation import Segmentation
from photo_editor.white_balance import estimate_illuminant


def _scene(h=240, w=320, seed=0):
    """Grey cards + coloured patches + gradient, in linear light."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    img[:] = np.linspace(0.05, 0.6, w, dtype=np.float32)[None, :, None]       # neutral ramp
    colours = [(0.4, 0.1, 0.05), (0.05, 0.3, 0.08), (0.05, 0.1, 0.45), (0.5, 0.4, 0.05)]
    for i, c in enumerate(colours):
        img[20 + i * 50:60 + i * 50, 30:110] = c
    img *= rng.uniform(0.95, 1.05, (h, w, 1)).astype(np.float32)
    return img


def test_color_roundtrips():
    x = np.random.default_rng(1).uniform(0, 1, (50, 3)).astype(np.float32)
    assert np.allclose(srgb_to_linear(linear_to_srgb(x)), x, atol=1e-4)
    assert np.allclose(oklab_to_linear(linear_to_oklab(x)), x, atol=1e-4)
    assert np.allclose(illuminant_rgb(6504), 1, atol=1e-3)
    T, duv = illuminant_to_cct(illuminant_rgb(3200, 0.004))
    assert abs(T - 3200) < 30 and abs(duv - 0.004) < 5e-4


def test_gamut_clip_preserves_in_gamut():
    x = np.array([[[0.2, 0.5, 0.7], [1.5, 0.2, 0.1], [-0.1, 0.3, 0.2]]], np.float32)
    y = gamut_clip(x)
    assert np.allclose(y[0, 0], x[0, 0])
    assert y.min() >= 0 and y.max() <= 1


@pytest.mark.parametrize("T", [2800, 3500, 5000, 8000, 11000])
def test_auto_white_balance_recovers_cast(T):
    scene = _scene()
    est = estimate_illuminant(scene * illuminant_rgb(T).astype(np.float32))
    mired_err = abs(1e6 / est["cct"] - 1e6 / T)
    assert mired_err < 15, est


def _data(lin):
    return ImageData(linear=lin, path="synthetic.arw", is_raw=True, meta={"iso": "400"})


def test_all_presets_run_and_neutralise():
    lin = _scene() * illuminant_rgb(3000).astype(np.float32)
    h, w = lin.shape[:2]
    person = np.zeros((h, w), np.float32)
    person[60:200, 150:260] = 1.0
    seg = Segmentation(person=person, skin=person * 0.5, faces=[(180, 60, 50, 50, 0.9)], backend="test")
    for name in presets.PRESETS:
        res = process(_data(lin.copy()), presets.resolve(name), seg=seg)
        assert res.srgb.shape == (h, w, 3)
        assert np.isfinite(res.srgb).all()
        assert 0.0 <= res.srgb.min() and res.srgb.max() <= 1.0
    # natural: the neutral ramp must come out neutral (tungsten cast removed)
    res = process(_data(lin.copy()), presets.resolve("natural", {"bg_blur": 0, "vignette": 0}), seg=seg)
    ramp = res.srgb[5:15, 140:300].reshape(-1, 3).mean(0)
    assert np.ptp(ramp) < 0.04, ramp


def test_overrides_and_parse():
    p = presets.resolve("cinematic", {"warmth": 12})
    assert p["warmth"] == 12 and p["grade_shadows_amount"] > 0
    assert presets.parse_value("0.5") == 0.5 and presets.parse_value("true") is True
    with pytest.raises(KeyError):
        presets.resolve("does_not_exist")


# ---------------------------------------------------------------- face retouch

def _synthetic_face(spot=True):
    """Skin-coloured face ellipse on a grey background with a dark red spot."""
    from photo_editor.face_landmarks import _ellipse, Face
    h, w = 400, 400
    img = np.full((h, w, 3), 0.45, np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    face_m = ((xx - 200) / 110.0) ** 2 + ((yy - 200) / 150.0) ** 2 < 1
    rng = np.random.default_rng(0)
    img[face_m] = np.array([0.80, 0.60, 0.48], np.float32)
    img += rng.normal(0, 0.006, img.shape).astype(np.float32)        # skin texture
    if spot:
        g = np.exp(-((xx - 150) ** 2 + (yy - 240) ** 2) / (2 * 3.5 ** 2))[..., None]
        img = img * (1 - 0.55 * g) + np.array([0.55, 0.25, 0.22], np.float32) * 0.55 * g
    face = Face(box=(90, 50, 220, 300), oval=_ellipse(200, 200, 110, 150),
                eyes=[_ellipse(160, 160, 22, 10), _ellipse(240, 160, 22, 10)],
                brows=[_ellipse(160, 135, 26, 7), _ellipse(240, 135, 26, 7)],
                lips=_ellipse(200, 290, 40, 14), nostrils=_ellipse(200, 245, 18, 7), source="test")
    return np.clip(img, 0, 1), face


def test_retouch_removes_spot_and_keeps_rest():
    from photo_editor.retouch import retouch
    img, face = _synthetic_face()
    out, rep = retouch(img, [face], {"retouch": 0.8})
    assert rep["blemishes_removed"] >= 1
    spot_before = np.abs(img[236:245, 146:155] - img[236:245, 170:179].mean((0, 1))).mean()
    spot_after = np.abs(out[236:245, 146:155] - out[236:245, 170:179].mean((0, 1))).mean()
    assert spot_after < spot_before * 0.4
    # background (outside the face) is bit-identical
    assert np.array_equal(out[:30], img[:30])


def test_beauty_only_keeps_colours():
    img, face = _synthetic_face(spot=False)
    data = ImageData(linear=srgb_to_linear(img), path="x.jpg", is_raw=False)
    seg = Segmentation(person=np.zeros(img.shape[:2], np.float32), skin=np.zeros(img.shape[:2], np.float32),
                       faces=[], backend="test")
    res = process(data, presets.resolve("beauty_only", {"retouch": 0.0}), seg=seg)
    assert np.abs(res.srgb - img).max() < 1e-3          # nothing but retouch may change
