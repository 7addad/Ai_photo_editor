"""Command line interface.

  python -m photo_editor presets
  python -m photo_editor preview  PHOTO.ARW            (RAW -> JPEG you can look at)
  python -m photo_editor info     PHOTO.ARW
  python -m photo_editor analyze  PHOTO.ARW [--preview-dir out/]
  python -m photo_editor edit     PHOTO.ARW --preset beauty_only     (as shot, faces retouched only)
  python -m photo_editor edit     PHOTO.ARW --preset natural [--set warmth=10 ...] [-o out.jpg]
                                  [--compare] [--masks] [--tiff] [--fast]
  python -m photo_editor grid     PHOTO.ARW [-o presets.jpg]   (all presets on one sheet)
  python -m photo_editor models   (download the segmentation models)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _print(d):
    print(json.dumps(d, ensure_ascii=False, indent=2, default=_json))


def cmd_presets(_):
    from .presets import list_presets
    _print(list_presets())


def cmd_info(a):
    from .raw_io import read_metadata
    for f in a.files:
        _print(read_metadata(f))


def cmd_models(_):
    from .segmentation import download_all_models
    _print(download_all_models())


def cmd_analyze(a):
    from .analyze import analyze
    from .pipeline import render_before, segment
    from .raw_io import load_image, save_image, to_preview
    from .segmentation import mask_overlay
    out = []
    for f in a.files:
        data = load_image(f, half_size=True)
        seg = segment(data)
        rep = analyze(data, seg)
        if a.preview_dir:
            base = os.path.join(a.preview_dir, os.path.splitext(os.path.basename(f))[0])
            before = to_preview(render_before(data))
            save_image(before, base + "_asshot.jpg", quality=88)
            if seg.has_people:
                save_image(to_preview(mask_overlay(render_before(data), seg)), base + "_people_mask.jpg", quality=88)
            rep["previews"] = [p for p in (base + "_asshot.jpg", base + "_people_mask.jpg") if os.path.exists(p)]
        out.append(rep)
    _print(out if len(out) > 1 else out[0])


def _parse_sets(sets):
    from .presets import parse_value
    o = {}
    for s in sets or []:
        if "=" not in s:
            raise SystemExit(f"--set expects key=value, got '{s}'")
        k, v = s.split("=", 1)
        o[k.strip()] = parse_value(v)
    return o


def cmd_edit(a):
    from .pipeline import process, render_before, side_by_side
    from .presets import resolve
    from .raw_io import load_image, save_image, to_preview
    from .segmentation import mask_overlay

    overrides = _parse_sets(a.set)
    if a.params:
        with open(a.params) as fh:
            overrides = {**json.load(fh), **overrides}
    params = resolve(a.preset, overrides)
    reports = []
    for f in a.files:
        data = load_image(f, half_size=a.fast)
        res = process(data, params)
        stem = os.path.splitext(os.path.basename(f))[0]
        ext = ".tif" if a.tiff else ".jpg"
        if a.output and len(a.files) == 1:
            out = a.output
        else:
            out = os.path.join(a.out_dir or os.path.dirname(os.path.abspath(f)), f"{stem}_{a.preset or 'custom'}{ext}")
        save_image(res.srgb, out, data, quality=a.quality)
        res.report["output"] = out
        base = os.path.splitext(out)[0]
        if a.compare:
            cmp_path = base + "_compare.jpg"
            save_image(side_by_side(render_before(data), res.srgb), cmp_path, quality=88)
            res.report["compare"] = cmp_path
        if a.masks and res.seg is not None and res.seg.has_people:
            m_path = base + "_masks.jpg"
            save_image(to_preview(mask_overlay(res.srgb, res.seg)), m_path, quality=88)
            res.report["masks"] = m_path
            if (res.report.get("retouch") or {}).get("enabled"):
                from .face_landmarks import draw_faces, find_faces
                f_path = base + "_faces.jpg"
                save_image(to_preview(draw_faces(res.srgb, find_faces(res.srgb, res.seg.faces))), f_path, quality=88)
                res.report["faces_overlay"] = f_path
        if a.preview:
            pv = base + "_preview.jpg"
            save_image(to_preview(res.srgb, 1600), pv, quality=88)
            res.report["preview"] = pv
        res.report["preset"] = a.preset
        res.report["params"] = {k: v for k, v in res.params.items()}
        if a.report:
            with open(base + "_report.json", "w") as fh:
                json.dump(res.report, fh, ensure_ascii=False, indent=2, default=_json)
        reports.append({k: res.report.get(k) for k in ("output", "compare", "masks", "faces_overlay", "preview", "preset",
                                                        "mode", "people", "retouch", "white_balance", "exposure_ev", "denoise",
                                                        "output_stats", "timing_s")})
    _print(reports if len(reports) > 1 else reports[0])


def cmd_grid(a):
    """Render every preset (or a chosen list) on a downscaled copy and tile them
    into one labelled image so the user can pick visually."""
    import cv2
    from .pipeline import _downscale, process, render_before, segment
    from .presets import PRESETS, resolve
    from .raw_io import load_image, save_image

    data = load_image(a.file, half_size=True)
    data.linear = _downscale(data.linear, a.size)
    seg = segment(data)
    names = a.presets.split(",") if a.presets else list(PRESETS)
    tiles = [("as shot", render_before(data))]
    for n in names:
        tiles.append((n, process(data, resolve(n), seg=seg).srgb))
    h, w = tiles[0][1].shape[:2]
    cols = a.cols or (4 if len(tiles) > 6 else 3)
    rows = -(-len(tiles) // cols)
    pad, bar = 8, max(24, h // 14)
    sheet = np.full((rows * (h + bar + pad) + pad, cols * (w + pad) + pad, 3), 255, np.uint8)
    for i, (n, img) in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = pad + r * (h + bar + pad), pad + c * (w + pad)
        sheet[y:y + h, x:x + w] = (np.clip(cv2.resize(img, (w, h)), 0, 1) * 255 + 0.5).astype(np.uint8)
        cv2.putText(sheet, n, (x + 4, y + h + int(bar * 0.75)), cv2.FONT_HERSHEY_SIMPLEX,
                    bar / 40, (25, 25, 25), max(1, bar // 18), cv2.LINE_AA)
    out = a.output or os.path.splitext(a.file)[0] + "_presets.jpg"
    save_image(sheet.astype(np.float32) / 255.0, out, quality=88)
    _print({"grid": out, "order": [t[0] for t in tiles], "people": seg.people_count})


def cmd_preview(a):
    """Convert RAW (ARW...) to a viewable JPEG: the camera's embedded preview
    when it is large enough, else a quick half-size render.  Claude cannot
    look at RAW files directly - view this JPEG instead."""
    from .pipeline import render_before
    from .raw_io import embedded_preview, is_raw, load_image, save_image, to_preview
    out = []
    for f in a.files:
        stem = os.path.splitext(os.path.basename(f))[0]
        dst = os.path.join(a.out_dir or os.path.dirname(os.path.abspath(f)), stem + "_view.jpg")
        how = None
        if is_raw(f) and not a.render:
            im = embedded_preview(f)
            if im is not None and max(im.size) >= min(a.size, 1000):
                im = im.convert("RGB")
                im.thumbnail((a.size, a.size))
                os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
                im.save(dst, quality=90)
                how = "embedded camera preview"
        if how is None:
            data = load_image(f, half_size=True)
            save_image(to_preview(render_before(data), a.size), dst, quality=90)
            how = "rendered from RAW (camera white balance)" if data.is_raw else "resized"
        out.append({"file": f, "view": dst, "source": how})
    _print(out if len(out) > 1 else out[0])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="photo_editor", description="RAW (Sony ARW & co.) AI photo editor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("presets", help="list presets").set_defaults(fn=cmd_presets)
    sub.add_parser("models", help="download segmentation models").set_defaults(fn=cmd_models)
    p = sub.add_parser("info", help="camera / exposure metadata")
    p.add_argument("files", nargs="+"); p.set_defaults(fn=cmd_info)
    p = sub.add_parser("analyze", help="detect people, light colour, exposure problems; recommend presets")
    p.add_argument("files", nargs="+"); p.add_argument("--preview-dir")
    p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("edit", help="apply a preset (all settings) and save")
    p.add_argument("files", nargs="+")
    p.add_argument("--preset", default="natural")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override any parameter (repeatable)")
    p.add_argument("--params", help="JSON file of parameter overrides")
    p.add_argument("-o", "--output"); p.add_argument("--out-dir")
    p.add_argument("--tiff", action="store_true", help="16-bit TIFF instead of JPEG")
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--compare", action="store_true", help="also save before|after image")
    p.add_argument("--masks", action="store_true", help="also save the people/skin mask overlay")
    p.add_argument("--preview", action="store_true", help="also save a 1600px preview")
    p.add_argument("--report", action="store_true", help="also save a JSON report")
    p.add_argument("--fast", action="store_true", help="half-resolution RAW decode (quick previews)")
    p.set_defaults(fn=cmd_edit)
    p = sub.add_parser("preview", help="RAW -> viewable JPEG (to look at the photo)")
    p.add_argument("files", nargs="+"); p.add_argument("--out-dir")
    p.add_argument("--size", type=int, default=1600)
    p.add_argument("--render", action="store_true", help="render from RAW data instead of the embedded preview")
    p.set_defaults(fn=cmd_preview)
    p = sub.add_parser("grid", help="preview all presets side by side on one image")
    p.add_argument("file"); p.add_argument("-o", "--output")
    p.add_argument("--presets", help="comma-separated subset")
    p.add_argument("--size", type=int, default=640, help="long side of each tile")
    p.add_argument("--cols", type=int)
    p.set_defaults(fn=cmd_grid)
    a = ap.parse_args(argv)
    if a.cmd == "edit" and a.preset in ("none", "custom"):
        a.preset = None
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
