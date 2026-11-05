"""Loading RAW (Sony ARW and every format LibRaw supports) and regular images,
metadata extraction and saving.

RAW files are decoded *linearly* (no gamma, no auto-brightening) into linear
sRGB with a fixed daylight white balance, so later white-balance estimation
sees the real colour of the light.  The camera's "as shot" white balance is
kept as a set of gains so it can be reproduced on request.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from .color import linear_to_srgb, srgb_to_linear

RAW_EXTENSIONS = {
    ".arw", ".srf", ".sr2",            # Sony
    ".cr2", ".cr3", ".crw",            # Canon
    ".nef", ".nrw",                    # Nikon
    ".raf",                            # Fujifilm
    ".orf",                            # Olympus / OM System
    ".rw2", ".rwl",                    # Panasonic / Leica
    ".pef", ".ptx",                    # Pentax
    ".dng",                            # Adobe DNG (Leica, Pixel, iPhone ProRAW, DJI...)
    ".3fr", ".fff",                    # Hasselblad
    ".iiq",                            # Phase One
    ".x3f",                            # Sigma
    ".srw",                            # Samsung
    ".kdc", ".dcr", ".k25",            # Kodak
    ".mrw",                            # Minolta
    ".erf", ".mef", ".mos", ".raw", ".rwz", ".gpr",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


@dataclass
class ImageData:
    linear: np.ndarray                       # HxWx3 float32 linear sRGB, daylight-balanced for RAW
    path: str
    is_raw: bool
    as_shot_gains: np.ndarray = field(default_factory=lambda: np.ones(3, np.float32))
    meta: dict = field(default_factory=dict)
    exif_bytes: bytes | None = None          # original EXIF block (non-RAW inputs)


def is_raw(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in RAW_EXTENSIONS


# ---------------------------------------------------------------- metadata

def _fmt_shutter(s: float | None) -> str | None:
    if not s:
        return None
    if s >= 0.5:
        return f"{s:g}s"
    return f"1/{round(1 / s)}s"


def _read_exif_tags(path: str) -> dict:
    out: dict = {}
    try:
        import exifread  # optional, handles ARW/NEF/CR2/DNG... and JPEG
    except ImportError:
        return out
    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)
    except Exception:
        return out
    wanted = {
        "Image Make": "make", "Image Model": "model", "EXIF LensModel": "lens",
        "EXIF ISOSpeedRatings": "iso", "EXIF ExposureTime": "shutter",
        "EXIF FNumber": "aperture", "EXIF FocalLength": "focal_length",
        "EXIF DateTimeOriginal": "datetime", "EXIF ExposureBiasValue": "exposure_bias",
        "EXIF WhiteBalance": "white_balance_mode", "EXIF Flash": "flash",
        "Image Orientation": "orientation",
    }
    for k, name in wanted.items():
        if k in tags:
            out[name] = str(tags[k]).strip()
    return out


def read_metadata(path: str) -> dict:
    meta = {"file": os.path.basename(path), "format": os.path.splitext(path)[1].lstrip(".").upper()}
    meta.update(_read_exif_tags(path))
    if is_raw(path):
        import rawpy
        with rawpy.imread(path) as raw:
            o = raw.other
            meta.setdefault("iso", f"{o.iso_speed:g}")
            meta.setdefault("shutter", _fmt_shutter(o.shutter_speed))
            meta.setdefault("aperture", f"f/{o.aperture:g}")
            meta.setdefault("focal_length", f"{o.focal_length:g}mm")
            if o.timestamp:
                meta.setdefault("datetime", str(o.timestamp))
            try:
                if raw.lens.model:
                    meta.setdefault("lens", raw.lens.model)
            except Exception:
                pass
            s = raw.sizes
            meta["raw_size"] = f"{s.width}x{s.height}"
            meta["bayer"] = raw.color_desc.decode(errors="ignore")
            meta["camera_wb"] = [round(float(v), 4) for v in raw.camera_whitebalance]
            meta["daylight_wb"] = [round(float(v), 4) for v in raw.daylight_whitebalance]
            meta["white_level"] = int(raw.white_level)
    return meta


def iso_value(meta: dict) -> float:
    try:
        return float(str(meta.get("iso", "100")).split()[0].strip("[]").split(",")[0])
    except ValueError:
        return 100.0


# ---------------------------------------------------------------- loading

def _decode(raw, wb, half_size: bool) -> np.ndarray:
    import rawpy
    rgb16 = raw.postprocess(
        gamma=(1, 1), no_auto_bright=True, output_bps=16, user_wb=list(wb),
        output_color=rawpy.ColorSpace.sRGB, half_size=half_size,
        highlight_mode=rawpy.HighlightMode.Blend,
        demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        use_camera_wb=False, use_auto_wb=False,
    )
    return rgb16.astype(np.float32) / 65535.0


def _normalise_wb(wb) -> list[float]:
    wb = [float(v) for v in wb]
    if len(wb) == 4 and wb[3] == 0:
        wb[3] = wb[1]
    g = wb[1] or 1.0
    return [v / g for v in wb]


def load_raw(path: str, half_size: bool = False) -> ImageData:
    import rawpy
    meta = read_metadata(path)
    with rawpy.imread(path) as raw:
        day = _normalise_wb(raw.daylight_whitebalance)
        cam = _normalise_wb(raw.camera_whitebalance)
        if not any(day[:3]) or day[0] <= 0:
            day = cam
        lin = _decode(raw, day, half_size)
        # Express the camera's as-shot WB as diagonal gains in our working space
        # by comparing a small daylight render with a small as-shot render.
        try:
            small_day = _decode(raw, day, True) if not half_size else lin
            small_cam = _decode(raw, cam, True)
            ok = (small_day.max(-1) < 0.9) & (small_day.min(-1) > 0.01)
            if ok.sum() > 1000:
                gains = np.median(small_cam[ok], axis=0) / np.median(small_day[ok], axis=0)
            else:
                gains = small_cam.reshape(-1, 3).mean(0) / np.maximum(small_day.reshape(-1, 3).mean(0), 1e-6)
            gains = (gains / gains[1]).astype(np.float32)
        except Exception:
            gains = np.ones(3, np.float32)
    return ImageData(linear=np.ascontiguousarray(lin), path=path, is_raw=True,
                     as_shot_gains=gains, meta=meta)


def load_image(path: str, half_size: bool = False) -> ImageData:
    """Load any supported file into linear float32 sRGB."""
    if is_raw(path):
        return load_raw(path, half_size)
    from PIL import Image, ImageOps
    meta = read_metadata(path)
    with Image.open(path) as im:
        exif = im.info.get("exif")
        im = ImageOps.exif_transpose(im)
        if im.mode in ("I;16", "I;16B", "I"):
            arr = np.asarray(im, dtype=np.float32) / 65535.0
            arr = np.repeat(arr[..., None], 3, -1)
        else:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.float32) / 255.0
    if path.lower().endswith((".tif", ".tiff")):
        import cv2  # 16-bit RGB TIFFs
        t = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if t is not None and t.dtype == np.uint16 and t.ndim == 3:
            arr = cv2.cvtColor(t[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
    meta["size"] = f"{arr.shape[1]}x{arr.shape[0]}"
    if half_size:
        import cv2
        arr = cv2.resize(arr, (arr.shape[1] // 2, arr.shape[0] // 2), interpolation=cv2.INTER_AREA)
    return ImageData(linear=srgb_to_linear(arr), path=path, is_raw=False, meta=meta, exif_bytes=exif)


def embedded_preview(path: str):
    """Return the camera JPEG preview embedded in a RAW (PIL.Image) or None."""
    import rawpy
    from PIL import Image
    try:
        with rawpy.imread(path) as raw:
            th = raw.extract_thumb()
        if th.format == rawpy.ThumbFormat.JPEG:
            return Image.open(io.BytesIO(th.data))
        return Image.fromarray(th.data)
    except Exception:
        return None


# ---------------------------------------------------------------- saving

def _rational(v: float) -> tuple[int, int]:
    f = Fraction(v).limit_denominator(10000)
    return f.numerator, f.denominator


def _build_exif(meta: dict):
    from PIL import Image
    ex = Image.Exif()
    if meta.get("make"):
        ex[0x010F] = str(meta["make"])
    if meta.get("model"):
        ex[0x0110] = str(meta["model"])
    ex[0x0131] = "Ai_photo_editor"
    try:
        sub = ex.get_ifd(0x8769)
        iso = iso_value(meta)
        if iso:
            sub[0x8827] = int(iso)
        if meta.get("lens"):
            sub[0xA434] = str(meta["lens"])
        if meta.get("datetime"):
            sub[0x9003] = str(meta["datetime"]).replace("-", ":")
    except Exception:
        pass
    return ex


def save_image(srgb: np.ndarray, out_path: str, data: ImageData | None = None, quality: int = 95) -> str:
    """Save a display-referred sRGB float image (0..1).  Format from extension:
    .jpg/.jpeg (8-bit, sRGB ICC + EXIF), .png (8-bit), .tif/.tiff (16-bit)."""
    from PIL import Image, ImageCms
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()
    srgb = np.clip(srgb, 0.0, 1.0)
    if ext in (".tif", ".tiff"):
        import cv2
        a16 = (srgb * 65535.0 + 0.5).astype(np.uint16)
        cv2.imwrite(out_path, cv2.cvtColor(a16, cv2.COLOR_RGB2BGR))
        return out_path
    # light ordered dither before 8-bit quantisation to avoid banding in skies
    rng = np.random.default_rng(0)
    a8 = np.clip(srgb * 255.0 + rng.uniform(-0.5, 0.5, srgb.shape[:2])[..., None], 0, 255).astype(np.uint8)
    im = Image.fromarray(a8, "RGB")
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    kw = {"icc_profile": icc}
    if data is not None:
        if data.exif_bytes:
            try:
                ex = Image.Exif()
                ex.load(data.exif_bytes)
                ex[0x0112] = 1  # pixels are already upright
                kw["exif"] = ex.tobytes()
            except Exception:
                pass
        else:
            kw["exif"] = _build_exif(data.meta).tobytes()
    if ext in (".jpg", ".jpeg"):
        im.save(out_path, quality=quality, subsampling=0, optimize=True, **kw)
    else:
        im.save(out_path, **kw)
    return out_path


def to_preview(srgb: np.ndarray, max_side: int = 1600) -> np.ndarray:
    import cv2
    h, w = srgb.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        srgb = cv2.resize(srgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return srgb


def linear_preview(lin: np.ndarray) -> np.ndarray:
    """Quick display of a linear image (simple clip + sRGB gamma)."""
    return linear_to_srgb(np.clip(lin, 0, 1))
