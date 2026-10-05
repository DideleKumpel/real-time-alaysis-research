import cv2
import numpy as np

PANEL_H = 480
CAM_W = 640
FACE_W = 240
VERDICT_W = 300


def _black(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _text(img, txt, pos, color=(180, 180, 180), scale=0.65, thick=1):
    cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def build_face_panel(face_bgr) -> np.ndarray:
    panel = _black(FACE_W, PANEL_H)
    cv2.rectangle(panel, (0, 0), (FACE_W, 34), (30, 30, 30), -1)
    _text(panel, "CROPPED FACE", (8, 24), (180, 180, 180), 0.6)

    if face_bgr is not None and face_bgr.size > 0:
        max_h, max_w = PANEL_H - 50, FACE_W - 10
        h, w = face_bgr.shape[:2]
        scale = min(max_w / w, max_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(face_bgr, (nw, nh))
        yo = (PANEL_H - nh) // 2 + 10
        xo = (FACE_W - nw) // 2
        panel[yo:yo+nh, xo:xo+nw] = resized
        cv2.rectangle(panel, (xo - 1, yo - 1), (xo + nw, yo + nh), (80, 80, 80), 1)
    else:
        _text(panel, "no face detected", (8, PANEL_H // 2), (60, 60, 60), 0.55)

    return panel


def build_verdict_panel(verdict: str, name: str, dist: float) -> np.ndarray:
    panel = _black(VERDICT_W, PANEL_H)
    cv2.rectangle(panel, (0, 0), (VERDICT_W, 34), (30, 30, 30), -1)
    _text(panel, "DEEPFACE RESULT", (8, 24), (180, 180, 180), 0.6)

    cfg = {
        "known": ((0, 120, 0), "KNOWN", "PERSON"),
        "unknown": ((0, 0, 160), "UNKNOWN", "PERSON"),
        "empty_db": ((60, 60, 0), "NO DB", "register first"),
        "no_face": ((40, 40, 40), "NO FACE", "detected"),
        "no_motion": ((35, 35, 35), "NO MOTION", "AI paused"),
        "motion": ((20, 80, 20), "MOTION", "AI active"),
        "error": ((80, 0, 80), "ERROR", "see console"),
    }

    bg_color, big, sub = cfg.get(verdict, cfg["no_face"])
    cv2.rectangle(panel, (10, 45), (VERDICT_W - 10, PANEL_H - 20), bg_color, -1)
    cv2.rectangle(panel, (10, 45), (VERDICT_W - 10, PANEL_H - 20), (160, 160, 160), 1)

    cx = VERDICT_W // 2
    font = cv2.FONT_HERSHEY_DUPLEX

    (tw, _), _ = cv2.getTextSize(big, font, 1.5, 2)
    cv2.putText(panel, big, (cx - tw // 2, PANEL_H // 2 - 25), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    (tw2, _), _ = cv2.getTextSize(sub, font, 0.9, 1)
    cv2.putText(panel, sub, (cx - tw2 // 2, PANEL_H // 2 + 22), font, 0.9, (210, 210, 210), 1, cv2.LINE_AA)

    if verdict == "known" and name:
        (tw3, _), _ = cv2.getTextSize(name, font, 0.75, 1)
        cv2.putText(panel, name, (cx - tw3 // 2, PANEL_H // 2 + 62), font, 0.75, (160, 255, 160), 1, cv2.LINE_AA)

    if verdict in ("known", "unknown"):
        dist_str = f"dist: {dist:.3f}"
        (tw4, _), _ = cv2.getTextSize(dist_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(panel, dist_str, (cx - tw4 // 2, PANEL_H - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    return panel


def render_monitoring_view(cam_panel: np.ndarray, best_crop, verdict: str, res_name: str, res_dist: float) -> np.ndarray:
    face_panel = build_face_panel(best_crop)
    verdict_panel = build_verdict_panel(verdict, res_name, res_dist)
    composed = np.hstack([cam_panel, face_panel, verdict_panel])
    composed[:, CAM_W - 1:CAM_W + 1] = (60, 60, 60)
    composed[:, CAM_W + FACE_W - 1:CAM_W + FACE_W + 1] = (60, 60, 60)
    return composed
