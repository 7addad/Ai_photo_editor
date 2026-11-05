"""Image analysis: what is in the photo, what is wrong with it, which preset fits."""
from __future__ import annotations

import numpy as np

from .color import illuminant_to_cct, luminance
from .pipeline import auto_denoise, segment, suggested_ev
from .raw_io import ImageData, iso_value
from .white_balance import estimate_illuminant


def analyze(data: ImageData, seg=None) -> dict:
    seg = seg or segment(data)
    person = seg.person if seg.has_people else None
    skin = seg.skin if seg.has_people else None
    est = estimate_illuminant(data.linear, person, skin)

    lin_cam = data.linear * data.as_shot_gains
    ev = suggested_ev(data, lin_cam, person)
    small = lin_cam[::8, ::8]
    clipped = float((data.linear[::8, ::8].max(-1) > 0.98).mean()) * 100
    Y = luminance(small)
    dark = float((Y < 0.002).mean()) * 100

    rep: dict = {
        "file": data.meta.get("file"),
        "camera": " ".join(x for x in (data.meta.get("make"), data.meta.get("model")) if x) or None,
        "lens": data.meta.get("lens"),
        "settings": {k: data.meta.get(k) for k in ("iso", "shutter", "aperture", "focal_length") if data.meta.get(k)},
        "is_raw": data.is_raw,
        "people": {
            "detected": bool(seg.has_people),
            "faces": seg.people_count,
            "person_area_pct": round(100 * seg.person_fraction, 2),
            "face_boxes": [list(map(int, f[:4])) for f in seg.faces],
            "backend": seg.backend,
        },
        "white_balance": {
            "estimated_light_cct": est["cct"], "estimated_light_duv": est["duv"],
            "light_type": est["light"], "confidence": est["confidence"], "skin_check": est["skin"],
        },
        "exposure": {
            "suggested_ev": round(ev, 2),
            "raw_clipped_pct": round(clipped, 2),
            "deep_shadow_pct": round(dark, 2),
        },
        "noise": dict(zip(("luma", "chroma"), (round(v, 2) for v in auto_denoise(data)))),
    }
    if data.is_raw:
        T, duv = illuminant_to_cct(1.0 / np.asarray(data.as_shot_gains, np.float64))
        rep["white_balance"]["camera_as_shot_cct"] = round(T)

    issues, recs = [], []
    T = est["cct"]
    if data.is_raw:
        shot_T = rep["white_balance"]["camera_as_shot_cct"]
        mired_err = abs(1e6 / T - 1e6 / shot_T)
        if mired_err > 20:
            issues.append(f"Camera white balance (~{shot_T}K) does not match the light (~{T}K): "
                          f"{'too warm/orange' if shot_T < T else 'too cool/blue'} - auto WB will neutralise it.")
    elif T < 4800 or T > 7500 or abs(est["duv"]) > 0.008:
        issues.append(f"Colour cast detected (light ~{T}K, duv {est['duv']}): auto WB will neutralise it.")
    if ev > 0.8:
        issues.append(f"Under-exposed by about {ev:.1f} EV.")
    elif ev < -0.8:
        issues.append(f"Over-exposed by about {-ev:.1f} EV.")
    if clipped > 2:
        issues.append(f"{clipped:.1f}% of pixels are clipped - use negative highlights.")
    if iso_value(data.meta) >= 3200:
        issues.append(f"High ISO ({iso_value(data.meta):g}) - noise reduction will be stronger.")

    area = seg.person_fraction
    if seg.has_people and area > 0.05:
        recs = [("natural", "people present - true colours, light separation"),
                ("portrait_pop", "make the person stand out more"),
                ("soft_portrait", "gentle, airy portrait"),
                ("cinematic", "stylised film grade")]
    elif seg.has_people:
        recs = [("natural", "small people in a scene - keep it realistic"),
                ("vivid", "travel / landscape with people"),
                ("cinematic", "stylised")]
    else:
        recs = [("natural", "true-to-life"), ("vivid", "landscape / travel colours"), ("cool_clean", "architecture")]
    if T < 3600:
        recs.append(("warm_golden", "keep some of the warm indoor/golden mood instead of full neutral"))
    rep["issues"] = issues
    rep["recommended_presets"] = [{"preset": n, "why": w} for n, w in recs]
    return rep
