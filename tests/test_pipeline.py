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


# ---------------------------------------------------------------- regular editing sliders

def test_sliders_mapping_and_direction():
    from photo_editor.sliders import parse_hsl, sliders_to_params
    p = sliders_to_params({"contrast": 50, "temperature": 3200, "sharpness": 80, "saturation": -100, "vignette": -50})
    assert p["contrast"] == pytest.approx(0.3) and p["wb_mode"] == "kelvin" and p["wb_kelvin"] == 3200
    assert p["sharpen"] == pytest.approx(0.8) and p["monochrome"] is True and p["vignette"] > 0
    assert parse_hsl(["orange:hue=10,sat=-20"]) == {"hue_orange": pytest.approx(3.0), "sat_orange": -0.2}
    with pytest.raises(KeyError):
        sliders_to_params({"nope": 1})

    lin = _scene()
    seg = Segmentation(person=np.zeros(lin.shape[:2], np.float32), skin=np.zeros(lin.shape[:2], np.float32),
                       faces=[], backend="test")
    run = lambda s: process(_data(lin.copy()), presets.resolve("standard", sliders_to_params(s)), seg=seg).srgb
    base = run({})
    assert run({"exposure": 1}).mean() > base.mean() + 0.03
    warm = run({"warmth": 60}).reshape(-1, 3).mean(0)
    assert warm[0] - warm[2] > base.reshape(-1, 3).mean(0)[0] - base.reshape(-1, 3).mean(0)[2]
    assert run({"contrast": 60}).std() > base.std()


# ---------------------------------------------------------------- Lightroom tools, categories, XMP

def test_lightroom_tools():
    from photo_editor import lr_tools as lrt
    lin = _scene()
    H, W = lin.shape[:2]
    for out in (lrt.rotate(lin, 3), lrt.perspective(lin, 30, -20, 110), lrt.lens_distortion(lin, 40),
                lrt.dehaze(lin, 40), lrt.dehaze(lin, -30), lrt.vignette_lr(lin, -50, 40, 60, 20)):
        assert out.shape == lin.shape and np.isfinite(out).all() and out.min() >= 0
    x0, y0, x1, y1 = lrt.crop_window(H, W, "1:1", (0.9, 0.5))
    assert x1 - x0 == y1 - y0 == H and x1 == W
    srgb = linear_to_srgb(np.clip(lin, 0, 1))
    curved = lrt.tone_curve(srgb, {"point_curve": "0,30; 128,128; 255,230", "curve_shadows": 30})
    assert curved.min() >= 30 / 255 - 1e-3 and curved.max() <= 230 / 255 + 1e-3
    M = lrt.calibration_matrix({"red_hue": 30, "blue_sat": -40})
    assert np.allclose(M @ np.ones(3), 1, atol=1e-4)
    assert lrt.sharpen_lr(srgb, 80, 1.2, 40, 50).shape == srgb.shape


def test_categories_and_xmp():
    import xml.etree.ElementTree as ET
    from photo_editor.categories import CATEGORIES, initial_settings
    from photo_editor.xmp import build_xmp
    stats = {"light": {"cct": 3200, "duv": 0, "type": "tungsten", "confidence": 0.8}, "auto_ev": 1.0,
             "tone": {"L_median": 0.4, "L_p01": 0.05, "L_p99": 0.95, "L_std": 0.2, "chroma": 0.08,
                      "colourful_pct": 50, "clip_pct": 0.5, "crush_pct": 1.0, "warm_cool": 0.01},
             "palette_pct": {"red": 10, "orange": 20, "yellow": 10, "green": 30, "aqua": 5, "blue": 20,
                             "purple": 3, "magenta": 2},
             "sky": {"pct": 20, "L": 0.9, "blue": 0.03}, "haze": 0.12,
             "tilt": {"deg": 2.0, "confidence": 0.8, "lines": 12},
             "people": {"count": 1, "area_pct": 20, "largest_face_pct": 12, "subject_L": 0.4,
                        "background_L": 0.6, "background_sharpness_ratio": 0.8, "skin_chroma": 0.12},
             "night": False, "iso": 3200, "is_raw": True, "subject_guess": "portrait"}
    for name in CATEGORIES:
        s, why = initial_settings(stats, name)
        assert why and isinstance(s, dict)
        sliders = __import__("photo_editor.sliders", fromlist=["sliders_to_params"])
        sliders.sliders_to_params(s)                        # every key must be a valid setting
    s, _ = initial_settings(stats, "sunset")
    assert s["warmth"] > 10                                  # warm light partly kept for sunsets
    s, _ = initial_settings(stats, "landscape")
    assert s["auto_level"] and s["sky_exposure"] < 0 and s["dehaze"] > 0
    xml, skipped = build_xmp(s, {"exposure_ev": 0.7, "white_balance": {"neutralised_light_cct": 3400}})
    root = ET.fromstring(xml)
    desc = root.find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
    crs = "{http://ns.adobe.com/camera-raw-settings/1.0/}"
    assert desc.get(crs + "Temperature") == "3400" and desc.get(crs + "Exposure2012") == "+0.70"
    assert desc.get(crs + "Dehaze") is not None and "sky_exposure" in skipped
