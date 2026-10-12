from __future__ import annotations

import os
import pickle
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

from ai.motion_detector import MotionDetector
from ai.yolo_detector import YoloDetector
from ai.face_matcher import FaceMatcher
from app.services.video_recording_service import VideoRecordingService

CAMERA_INDEX = 0
YOLO_HEAD_CLASS_ID = 1
SIMILARITY_THRESHOLD = 0.40
MIN_FACE_SIZE = (60, 60)
DEEPFACE_MODEL = "Facenet"
DEEPFACE_BACKEND = "skip"
PANEL_H = 480
CAM_W = 640
FACE_W = 240
VERDICT_W = 300
MOTION_THRESHOLD = 200


class FaceDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.records: dict[str, list[np.ndarray]] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                self.records = pickle.load(f)
            print(f"[INFO] Baza załadowana: {list(self.records.keys())}")
        else:
            print("[INFO] Brak bazy – uruchom tryb rejestracji.")

    def save(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.records, f)
        print("[INFO] Baza zapisana.")

    def add(self, name: str, emb: np.ndarray):
        self.records.setdefault(name, []).append(emb)

    def find(self, emb: np.ndarray) -> tuple[str, float]:
        best_name, best_dist = "UNKNOWN", 1.0
        for name, embs in self.records.items():
            for ref in embs:
                d = cosine(emb, ref)
                if d < best_dist:
                    best_dist, best_name = d, name
        if best_dist > SIMILARITY_THRESHOLD:
            return "UNKNOWN", best_dist
        return best_name, best_dist

    def is_empty(self) -> bool:
        return len(self.records) == 0


class MonitoringPipeline:
    """Główny pipeline: ruch -> YOLO -> face-match -> render 3-paneli."""

    def __init__(self, yolo_model_path: str):
        self.yolo_detector = YoloDetector(yolo_model_path)
        self.face_matcher = FaceMatcher()
        self.db = FaceDatabase(str(Path(__file__).resolve().parents[1] / "data" / "face_database.pkl"))
        self.camera_states: dict[str, dict] = {}
        self.recording_service = VideoRecordingService()

    def initialize(self):
        self.yolo_detector.load()

        _dummy = np.zeros((160, 160, 3), dtype=np.uint8)
        try:
            DeepFace.represent(
                img_path=_dummy,
                model_name=DEEPFACE_MODEL,
                detector_backend=DEEPFACE_BACKEND,
                enforce_detection=False,
            )
        except Exception:
            pass

    def _extract_embedding(self, face_bgr: np.ndarray):
        try:
            results = DeepFace.represent(
                img_path=face_bgr,
                model_name=DEEPFACE_MODEL,
                detector_backend=DEEPFACE_BACKEND,
                enforce_detection=False,
                align=True,
            )
            if results:
                return np.array(results[0]["embedding"], dtype=np.float32)
        except Exception as e:
            print(f"[WARN] DeepFace błąd: {e}")
        return None

    def _detect_heads(self, frame: np.ndarray) -> list:
        results = self.yolo_detector.model(frame, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            if int(box.cls[0]) != YOLO_HEAD_CLASS_ID:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2, float(box.conf[0])))
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        return boxes

    def _get_camera_state(self, camera_id: str | None):
        camera_key = str(camera_id or "default")
        if camera_key not in self.camera_states:
            self.camera_states[camera_key] = {"motion_detector": MotionDetector()}
        return self.camera_states[camera_key]

    def process_frame(self, frame, camera_id: str | None = None):
        camera_key = str(camera_id or "default")
        state = self._get_camera_state(camera_key)
        motion_detector = state["motion_detector"]
        h_orig, w_orig = frame.shape[:2]
        output_frame = frame.copy()
        sx = 1.0
        sy = 1.0

        motion_detected, motion_mask = motion_detector.detect(frame, threshold=MOTION_THRESHOLD)
        print(f"[AI] camera={camera_key} motion_detected={motion_detected} pixels={cv2.countNonZero(motion_mask)} frame={frame.shape[:2]}")

        if not motion_detected:
            boxes = []
            best_crop = None
            verdict = "no_motion"
            res_name = ""
            res_dist = 1.0
            status_text = "NO MOTION"
            person_detected = False
            print("[AI] status=NO MOTION -> skip YOLO")
        else:
            boxes = self._detect_heads(frame)
            best_crop = None
            verdict = "no_face"
            res_name = ""
            res_dist = 1.0
            status_text = "NO FACE DETECTED"
            person_detected = False
            print(f"[AI] YOLO boxes={len(boxes)}")

        for idx, (x1, y1, x2, y2, conf) in enumerate(boxes):
            sx1, sy1 = int(x1 * sx), int(y1 * sy)
            sx2, sy2 = int(x2 * sx), int(y2 * sy)

            marg = 10
            fh, fw = frame.shape[:2]
            crop = frame[max(0, y1 - marg):min(fh, y2 + marg), max(0, x1 - marg):min(fw, x2 + marg)]
            too_small = (crop.size == 0 or crop.shape[0] < MIN_FACE_SIZE[0] or crop.shape[1] < MIN_FACE_SIZE[1])

            if too_small:
                box_color = (0, 165, 255)
                box_label = f"too small ({conf:.2f})"
            else:
                emb = self._extract_embedding(crop)
                if emb is None:
                    box_color = (0, 165, 255)
                    box_label = f"embed error ({conf:.2f})"
                    if idx == 0:
                        best_crop = crop
                        verdict = "error"
                elif self.db.is_empty():
                    box_color = (0, 220, 220)
                    box_label = f"empty db ({conf:.2f})"
                    if idx == 0:
                        best_crop = crop
                        verdict = "empty_db"
                else:
                    name, dist = self.db.find(emb)
                    if name == "UNKNOWN":
                        box_color = (0, 0, 220)
                        box_label = f"UNKNOWN ({conf:.2f})"
                        if idx == 0:
                            best_crop = crop
                            verdict = "unknown"
                            person_detected = True
                            status_text = "UNKNOWN"
                            res_dist = dist
                    else:
                        box_color = (0, 210, 0)
                        box_label = f"{name} ({conf:.2f})"
                        if idx == 0:
                            best_crop = crop
                            verdict = "known"
                            person_detected = True
                            status_text = "PERSON DETECTED"
                            res_name = name
                            res_dist = dist

            if verdict in ("known", "unknown"):
                status_text = "PERSON DETECTED" if verdict == "known" else "UNKNOWN"
            elif motion_detected and verdict == "no_face":
                status_text = "NO FACE DETECTED"

            cv2.rectangle(output_frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(output_frame, box_label, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv2.LINE_AA)

        print(f"[AI] final_status={status_text} boxes={len(boxes)} person_detected={person_detected} verdict={verdict}")

        recording_result = self.recording_service.process_frame(
            camera_id=camera_key,
            frame=output_frame,
            motion_detected=motion_detected,
            person_detected=person_detected,
        )

        return {
            "motion_detected": motion_detected,
            "person_detected": person_detected,
            "status_text": status_text,
            "display": output_frame,
            "boxes": boxes,
            "recording": recording_result.get("recording", False),
            "recording_path": recording_result.get("path"),
        }
