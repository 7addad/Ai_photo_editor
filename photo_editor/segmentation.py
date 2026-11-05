"""People detection and person / skin / background masks.

Primary backend: MediaPipe Tasks (all models run on CPU, downloaded once to
``~/.cache/ai_photo_editor/models``):

* ``blaze_face_full_range``      - face boxes (counts people, small faces too)
* ``deeplab_v3``                 - generic "person" class for the whole scene
* ``selfie_multiclass_256x256``  - hair / face-skin / body-skin / clothes, run on
                                   the whole frame *and* on a crop around every
                                   face for fine detail on people far away.

Fallback (no MediaPipe / offline): OpenCV Haar faces + GrabCut.

Masks are refined with an edge-aware guided filter so hair and shoulders get
soft, accurate edges at full resolution.
"""
from __future__ import annotations

import contextlib
import os
import ssl
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

from .color import linear_to_srgb

MODEL_DIR = os.environ.get("AI_PHOTO_EDITOR_MODELS",
                           os.path.join(os.path.expanduser("~"), ".cache", "ai_photo_editor", "models"))
MODELS = {
    "face": ("blaze_face_full_range.tflite",
             "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite"),
    "face_short": ("blaze_face_short_range.tflite",
                   "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"),
    "deeplab": ("deeplab_v3.tflite",
                "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite"),
    "multiclass": ("selfie_multiclass_256x256.tflite",
                   "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"),
}
DEEPLAB_PERSON = 15
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


@contextlib.contextmanager
def _quiet_native_stderr():
    """MediaPipe/LiteRT print INFO lines from C++ straight to fd 2; hide them
    unless AI_PHOTO_EDITOR_DEBUG is set."""
    if os.environ.get("AI_PHOTO_EDITOR_DEBUG"):
        yield
        return
    saved = None
    try:
        saved = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
    except OSError:
        saved = None
    try:
        yield
    finally:
        if saved is not None:
            os.dup2(saved, 2)
            os.close(saved)
MC_BG, MC_HAIR, MC_BODY_SKIN, MC_FACE_SKIN, MC_CLOTHES, MC_OTHER = range(6)


@dataclass
class Segmentation:
    person: np.ndarray                 # HxW float32 0..1 (soft, refined)
    skin: np.ndarray                   # HxW float32 0..1
    faces: list = field(default_factory=list)   # [(x, y, w, h, score)] in full-res pixels
    backend: str = "none"

    @property
    def people_count(self) -> int:
        return len(self.faces)

    @property
    def person_fraction(self) -> float:
        return float(self.person.mean())

    @property
    def has_people(self) -> bool:
        return self.person_fraction > 0.004 or bool(self.faces)


# ---------------------------------------------------------------- models

def model_path(key: str, download: bool = True) -> str | None:
    name, url = MODELS[key]
    path = os.path.join(MODEL_DIR, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    if not download:
        return None
    os.makedirs(MODEL_DIR, exist_ok=True)
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    ctx = ssl.create_default_context(cafile=cafile) if cafile and os.path.exists(cafile) else None
    tmp = path + ".part"
    try:
        with urllib.request.urlopen(url, timeout=60, context=ctx) as r, open(tmp, "wb") as f:
            f.write(r.read())
        os.replace(tmp, path)
        return path
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        return None


def download_all_models() -> dict:
    return {k: model_path(k) for k in MODELS}


# ---------------------------------------------------------------- guided filter

def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He et al. guided filter with a single-channel guide (fast, box filters)."""
    k = (2 * radius + 1, 2 * radius + 1)
    mean = lambda x: cv2.boxFilter(x, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)
    I = guide.astype(np.float32)
    p = src.astype(np.float32)
    mI, mp = mean(I), mean(p)
    cov = mean(I * p) - mI * mp
    var = mean(I * I) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return (mean(a) * I + mean(b)).astype(np.float32)


def refine_mask(mask: np.ndarray, srgb: np.ndarray) -> np.ndarray:
    """Upscale a coarse soft mask to the image size and snap its edges to
    image edges (hair, shoulders) with a two-scale guided filter."""
    H, W = srgb.shape[:2]
    gray = cv2.cvtColor(np.ascontiguousarray(srgb, dtype=np.float32), cv2.COLOR_RGB2GRAY)
    # mid scale
    s = min(1.0, 1280 / max(H, W))
    gs = cv2.resize(gray, (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_AREA)
    ms = cv2.resize(mask.astype(np.float32), gs.shape[::-1], interpolation=cv2.INTER_LINEAR)
    r = max(2, round(max(gs.shape) / 160))
    ms = np.clip(guided_filter(gs, ms, r, 1e-3), 0, 1)
    # full scale, small radius
    m = cv2.resize(ms, (W, H), interpolation=cv2.INTER_LINEAR)
    if s < 1:
        r2 = max(2, round(max(H, W) / 1200))
        m = guided_filter(gray, m, r2, 2e-3)
    # sharpen the soft ramp a bit, keep it soft enough for natural compositing
    m = np.clip((m - 0.5) * 1.6 + 0.5, 0, 1)
    return m.astype(np.float32)


# ---------------------------------------------------------------- MediaPipe backend

def _mp_image(u8: np.ndarray):
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(u8))


def _detect_faces(u8: np.ndarray) -> list:
    from mediapipe.tasks.python import BaseOptions, vision
    faces = []
    for key in ("face", "face_short"):
        path = model_path(key)
        if not path:
            continue
        opts = vision.FaceDetectorOptions(base_options=BaseOptions(model_asset_path=path),
                                          min_detection_confidence=0.5)
        try:
            with vision.FaceDetector.create_from_options(opts) as det:
                res = det.detect(_mp_image(u8))
        except Exception:
            continue
        for d in res.detections:
            b = d.bounding_box
            faces.append((b.origin_x, b.origin_y, b.width, b.height, float(d.categories[0].score)))
        if faces:
            break
    return _nms(faces)


def _nms(boxes, thr=0.3):
    boxes = sorted(boxes, key=lambda b: -b[4])
    keep = []
    for b in boxes:
        if all(_iou(b, k) < thr for k in keep):
            keep.append(b)
    return keep


def _iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(1e-6, a[2] * a[3] + b[2] * b[3] - inter)


def _segment(u8: np.ndarray, key: str) -> list[np.ndarray] | None:
    from mediapipe.tasks.python import BaseOptions, vision
    path = model_path(key)
    if not path:
        return None
    opts = vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=path),
                                        output_confidence_masks=True, output_category_mask=False)
    with vision.ImageSegmenter.create_from_options(opts) as seg:
        res = seg.segment(_mp_image(u8))
    out = []
    for m in res.confidence_masks:
        a = m.numpy_view().astype(np.float32)
        if a.ndim == 3:
            a = a[..., 0]
        if a.shape != u8.shape[:2]:
            a = cv2.resize(a, (u8.shape[1], u8.shape[0]), interpolation=cv2.INTER_LINEAR)
        out.append(a)
    return out


def _body_box(face, W, H):
    """Rough region a person occupies given the face box."""
    x, y, w, h = face[:4]
    cx = x + w / 2
    x0 = int(max(0, cx - 2.6 * w)); x1 = int(min(W, cx + 2.6 * w))
    y0 = int(max(0, y - 0.9 * h)); y1 = int(min(H, y + 7.5 * h))
    return x0, y0, x1, y1


def _mediapipe_masks(u8: np.ndarray):
    H, W = u8.shape[:2]
    faces = _detect_faces(u8)
    person = np.zeros((H, W), np.float32)
    skin = np.zeros((H, W), np.float32)
    region = np.zeros((H, W), np.float32)
    used = []

    dl = _segment(u8, "deeplab")
    if dl is not None and len(dl) > DEEPLAB_PERSON:
        person = np.maximum(person, dl[DEEPLAB_PERSON])
        used.append("deeplab_v3")
        k = max(3, int(0.04 * max(H, W)) | 1)
        region = np.maximum(region, cv2.dilate((person > 0.3).astype(np.float32), np.ones((k, k), np.uint8)))

    for f in faces:
        x0, y0, x1, y1 = _body_box(f, W, H)
        region[y0:y1, x0:x1] = 1.0

    mc = _segment(u8, "multiclass")
    if mc is not None:
        used.append("selfie_multiclass")
        p_mc = 1.0 - mc[MC_BG]
        sk = np.maximum(mc[MC_BODY_SKIN], mc[MC_FACE_SKIN])
        # whole-frame multiclass is only trusted where a person is plausible
        person = np.maximum(person, p_mc * region)
        skin = np.maximum(skin, sk * region)
        # per-person crops: much better detail for people that are small in frame
        for f in faces:
            if f[2] > 0.35 * W:  # already a close-up, whole frame was enough
                continue
            x0, y0, x1, y1 = _body_box(f, W, H)
            if (x1 - x0) < 16 or (y1 - y0) < 16:
                continue
            crop = _segment(u8[y0:y1, x0:x1], "multiclass")
            if crop is None:
                continue
            person[y0:y1, x0:x1] = np.maximum(person[y0:y1, x0:x1], 1.0 - crop[MC_BG])
            skin[y0:y1, x0:x1] = np.maximum(skin[y0:y1, x0:x1],
                                            np.maximum(crop[MC_BODY_SKIN], crop[MC_FACE_SKIN]))
    if not used:
        raise RuntimeError("no segmentation model available")
    return person, skin * (person > 0.3), faces, "mediapipe:" + "+".join(used)


# ---------------------------------------------------------------- OpenCV fallback

def _opencv_masks(u8: np.ndarray):
    H, W = u8.shape[:2]
    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    faces = []
    cascade = os.path.join(getattr(getattr(cv2, "data", None), "haarcascades", ""),
                           "haarcascade_frontalface_default.xml")
    if os.path.exists(cascade):
        casc = cv2.CascadeClassifier(cascade)
        det = casc.detectMultiScale(gray, 1.1, 5, minSize=(max(16, W // 60), max(16, W // 60)))
        faces = [(int(x), int(y), int(w), int(h), 0.8) for (x, y, w, h) in det]
    person = np.zeros((H, W), np.float32)
    for f in faces:
        x0, y0, x1, y1 = _body_box(f, W, H)
        gc = np.full((H, W), cv2.GC_BGD, np.uint8)
        gc[y0:y1, x0:x1] = cv2.GC_PR_BGD
        x, y, w, h = f[:4]
        gc[y0:y1, max(0, x - w // 2):min(W, x + w + w // 2)] = cv2.GC_PR_FGD
        gc[y:y + h, x:x + w] = cv2.GC_FGD
        try:
            bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            cv2.grabCut(cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
            person = np.maximum(person, ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.float32))
        except cv2.error:
            person[y0:y1, x0:x1] = np.maximum(person[y0:y1, x0:x1], 0.5)
    ycc = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)
    skin = ((ycc[..., 1] > 135) & (ycc[..., 1] < 180) & (ycc[..., 2] > 85) & (ycc[..., 2] < 135)).astype(np.float32)
    skin = cv2.GaussianBlur(skin, (0, 0), 2) * person
    return person, skin, faces, "opencv:haar+grabcut"


# ---------------------------------------------------------------- public API

def segment_people(srgb: np.ndarray, work_size: int = 1280) -> Segmentation:
    """Detect people in a display-referred sRGB float image (0..1) and return
    full-resolution soft person and skin masks."""
    H, W = srgb.shape[:2]
    s = min(1.0, work_size / max(H, W))
    small = cv2.resize(srgb, (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_AREA) if s < 1 else srgb
    u8 = (np.clip(small, 0, 1) * 255 + 0.5).astype(np.uint8)
    try:
        with _quiet_native_stderr():
            person, skin, faces, backend = _mediapipe_masks(u8)
    except Exception:
        person, skin, faces, backend = _opencv_masks(u8)
    if person.max() < 0.05:
        z = np.zeros((H, W), np.float32)
        return Segmentation(person=z, skin=z.copy(), faces=[], backend=backend)
    # remove specks: drop components that are tiny relative to the frame
    hard = (person > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    min_area = 0.0015 * hard.size
    face_pts = [(int(f[0] + f[2] / 2), int(f[1] + f[3] / 2)) for f in faces]
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area and not any(lab[min(y, lab.shape[0]-1), min(x, lab.shape[1]-1)] == i for x, y in face_pts):
            person[lab == i] = 0
    person = refine_mask(person, srgb)
    skin = np.clip(cv2.resize(skin, (W, H), interpolation=cv2.INTER_LINEAR), 0, 1) * person
    faces_full = [(round(x / s), round(y / s), round(w / s), round(h / s), sc) for x, y, w, h, sc in faces]
    return Segmentation(person=person, skin=skin.astype(np.float32), faces=faces_full, backend=backend)


def segment_linear(lin: np.ndarray, exposure_ev: float = 0.0) -> Segmentation:
    """Convenience wrapper for linear input (applies a quick display transform)."""
    disp = linear_to_srgb(np.clip(lin * (2.0 ** exposure_ev), 0, 1))
    return segment_people(disp)


def mask_overlay(srgb: np.ndarray, seg: Segmentation) -> np.ndarray:
    """Visualisation: people tinted, skin outlined, background dimmed, face boxes."""
    out = srgb.copy()
    m = seg.person[..., None]
    out = out * (0.35 + 0.65 * m)
    out = out * (1 - 0.35 * m) + np.array([0.15, 0.85, 0.3], np.float32) * 0.35 * m
    sk = seg.skin[..., None]
    out = out * (1 - 0.3 * sk) + np.array([1.0, 0.55, 0.2], np.float32) * 0.3 * sk
    t = max(2, round(max(out.shape[:2]) / 400))
    for x, y, w, h, _ in seg.faces:
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), (1.0, 1.0, 0.0), t)
    return np.clip(out, 0, 1)
