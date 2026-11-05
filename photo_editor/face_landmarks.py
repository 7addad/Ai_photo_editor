"""Face geometry for retouching: face oval, eyes, brows, lips, mouth, nostrils,
irises - as polygons in full-image pixel coordinates.

Backends
* MediaPipe FaceLandmarker (478 points, bundled model) run on a crop around
  every detected face, so small faces in group shots get precise landmarks.
* Fallback: OpenCV YuNet (bundled) - 5 points per face; regions are
  approximated with ellipses. Works with only OpenCV installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .segmentation import _quiet_native_stderr, mediapipe_enabled, model_path, yunet_detect

# inner lip contour (mouth opening) of the MediaPipe 468/478 mesh
INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]


@dataclass
class Face:
    box: tuple                                  # x, y, w, h
    oval: np.ndarray                            # N x 2 polygon
    eyes: list = field(default_factory=list)    # [poly, poly]
    brows: list = field(default_factory=list)
    lips: np.ndarray | None = None              # outer lip polygon
    mouth: np.ndarray | None = None             # inner mouth opening (teeth live here)
    nostrils: np.ndarray | None = None
    irises: list = field(default_factory=list)  # [(cx, cy, r)]
    source: str = "mediapipe"

    @property
    def width(self) -> float:
        x0, x1 = self.oval[:, 0].min(), self.oval[:, 0].max()
        return float(x1 - x0)


def _hull(pts: np.ndarray) -> np.ndarray:
    return cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)


def _indices(connections) -> list[int]:
    s = set()
    for c in connections:
        s.add(c.start); s.add(c.end)
    return sorted(s)


def _ellipse(cx, cy, ax, ay, angle=0.0, n=36) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ca, sa = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    x, y = ax * np.cos(t), ay * np.sin(t)
    return np.stack([cx + x * ca - y * sa, cy + x * sa + y * ca], -1).astype(np.float32)


# ---------------------------------------------------------------- MediaPipe

_LM_CACHE: dict = {}


def _landmark_sets():
    if "sets" not in _LM_CACHE:
        from mediapipe.tasks.python.vision import FaceLandmarksConnections as C
        _LM_CACHE["sets"] = {
            "oval": _indices(C.FACE_LANDMARKS_FACE_OVAL),
            "leye": _indices(C.FACE_LANDMARKS_LEFT_EYE), "reye": _indices(C.FACE_LANDMARKS_RIGHT_EYE),
            "lbrow": _indices(C.FACE_LANDMARKS_LEFT_EYEBROW), "rbrow": _indices(C.FACE_LANDMARKS_RIGHT_EYEBROW),
            "lips": _indices(C.FACE_LANDMARKS_LIPS), "nose": _indices(C.FACE_LANDMARKS_NOSE),
            "liris": _indices(C.FACE_LANDMARKS_LEFT_IRIS), "riris": _indices(C.FACE_LANDMARKS_RIGHT_IRIS),
        }
    return _LM_CACHE["sets"]


def _mediapipe_face(u8_crop: np.ndarray, ox: int, oy: int, scale: float) -> Face | None:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    path = model_path("landmarker")
    if not path:
        return None
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=path), num_faces=3,
                                        min_face_detection_confidence=0.4, min_face_presence_confidence=0.4)
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(u8_crop)))
    if not res.face_landmarks:
        return None
    h, w = u8_crop.shape[:2]
    # the face closest to the crop centre is the one we were asked about
    best = min(res.face_landmarks, key=lambda f: (f[1].x - 0.5) ** 2 + (f[1].y - 0.5) ** 2)
    P = np.array([[p.x * w, p.y * h] for p in best], np.float32) / scale + np.array([ox, oy], np.float32)
    S = _landmark_sets()
    nose = P[S["nose"]]
    ny0, ny1 = nose[:, 1].min(), nose[:, 1].max()
    nostrils = nose[nose[:, 1] > ny0 + 0.62 * (ny1 - ny0)]
    face = Face(box=(0, 0, 0, 0), oval=_hull(P[S["oval"]]),
                eyes=[_hull(P[S["leye"]]), _hull(P[S["reye"]])],
                brows=[_hull(P[S["lbrow"]]), _hull(P[S["rbrow"]])],
                lips=_hull(P[S["lips"]]),
                mouth=P[INNER_LIPS] if P.shape[0] > max(INNER_LIPS) else None,
                nostrils=_hull(nostrils) if len(nostrils) >= 3 else None)
    if P.shape[0] >= 478:
        for k in ("liris", "riris"):
            ip = P[S[k]]
            c = ip.mean(0)
            face.irises.append((float(c[0]), float(c[1]), float(np.linalg.norm(ip - c, axis=1).mean())))
    x0, y0 = face.oval.min(0); x1, y1 = face.oval.max(0)
    face.box = (float(x0), float(y0), float(x1 - x0), float(y1 - y0))
    return face


# ---------------------------------------------------------------- YuNet approximation

def _face_from_5pts(x, y, w, h, pts) -> Face:
    re, le, nose, rm, lm = [np.asarray(p, np.float32) for p in pts]
    eye_d = float(np.linalg.norm(le - re))
    ang = float(np.degrees(np.arctan2(le[1] - re[1], le[0] - re[0])))
    mid_eye = (re + le) / 2
    mouth_c = (rm + lm) / 2
    up = mid_eye - mouth_c
    up = up / (np.linalg.norm(up) + 1e-6)
    oc = (mid_eye + mouth_c) / 2 + up * eye_d * 0.15
    oval = _ellipse(oc[0], oc[1], eye_d * 1.05, eye_d * 1.45, ang)
    eyes = [_ellipse(e[0], e[1], eye_d * 0.26, eye_d * 0.13, ang) for e in (re, le)]
    brows = [_ellipse(*(e + up * eye_d * 0.26), eye_d * 0.32, eye_d * 0.12, ang) for e in (re, le)]
    mw = float(np.linalg.norm(lm - rm))
    lips = _ellipse(mouth_c[0], mouth_c[1], mw * 0.62, eye_d * 0.2, ang)
    nos = nose - up * eye_d * 0.08
    nostrils = _ellipse(nos[0], nos[1], eye_d * 0.3, eye_d * 0.12, ang)
    return Face(box=(x, y, w, h), oval=oval, eyes=eyes, brows=brows, lips=lips, mouth=None,
                nostrils=nostrils, source="yunet")


# ---------------------------------------------------------------- public

def find_faces(srgb: np.ndarray, boxes: list | None = None, min_size: int = 40) -> list[Face]:
    """Landmarks for every face in a display-referred sRGB image (0..1).
    ``boxes`` (x, y, w, h, score) from the people segmentation are reused when
    given; otherwise faces are detected here."""
    H, W = srgb.shape[:2]
    u8_full = None

    def u8_region(x0, y0, x1, y1, target):
        crop = srgb[y0:y1, x0:x1]
        s = min(1.0, target / max(crop.shape[:2]))
        if s < 1:
            crop = cv2.resize(crop, (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))),
                              interpolation=cv2.INTER_AREA)
        return (np.clip(crop, 0, 1) * 255 + 0.5).astype(np.uint8), s

    yn = None
    if not boxes:
        u8_full, s = u8_region(0, 0, W, H, 1600)
        yn = [(x / s, y / s, w / s, h / s, sc, p / s) for x, y, w, h, sc, p in yunet_detect(u8_full)]
        boxes = [b[:5] for b in yn]
    use_mp = mediapipe_enabled()
    faces = []
    for i, b in enumerate(boxes):
        x, y, w, h = b[:4]
        if max(w, h) < min_size:
            continue
        f = None
        if use_mp:
            m = 0.9 * max(w, h)
            x0, y0 = int(max(0, x - m)), int(max(0, y - m))
            x1, y1 = int(min(W, x + w + m)), int(min(H, y + h + m))
            crop, s = u8_region(x0, y0, x1, y1, 768)
            try:
                with _quiet_native_stderr():
                    f = _mediapipe_face(crop, x0, y0, s)
            except Exception:
                f = None
        if f is None:
            # fall back to YuNet points for this face
            if yn is None:
                u8_full, s = u8_region(0, 0, W, H, 1600)
                yn = [(xx / s, yy / s, ww / s, hh / s, sc, p / s) for xx, yy, ww, hh, sc, p in yunet_detect(u8_full)]
            cx, cy = x + w / 2, y + h / 2
            cand = [d for d in yn if abs(d[0] + d[2] / 2 - cx) < w and abs(d[1] + d[3] / 2 - cy) < h]
            if cand:
                d = min(cand, key=lambda d: abs(d[0] + d[2] / 2 - cx) + abs(d[1] + d[3] / 2 - cy))
                f = _face_from_5pts(*d[:4], d[5])
        if f is not None:
            faces.append(f)
    return faces


def draw_faces(srgb: np.ndarray, faces: list[Face]) -> np.ndarray:
    """Debug overlay of the retouch regions."""
    out = (np.clip(srgb, 0, 1) * 255).astype(np.uint8).copy()
    t = max(1, round(max(out.shape[:2]) / 700))
    for f in faces:
        cv2.polylines(out, [f.oval.astype(np.int32)], True, (0, 255, 0), t)
        for p in f.eyes + f.brows:
            cv2.polylines(out, [p.astype(np.int32)], True, (255, 255, 0), t)
        for p, c in ((f.lips, (255, 0, 120)), (f.mouth, (255, 255, 255)), (f.nostrils, (0, 160, 255))):
            if p is not None:
                cv2.polylines(out, [p.astype(np.int32)], True, c, t)
        for cx, cy, r in f.irises:
            cv2.circle(out, (int(cx), int(cy)), int(r), (0, 120, 255), t)
    return out.astype(np.float32) / 255.0
