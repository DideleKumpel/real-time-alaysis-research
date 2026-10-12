from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2


class VideoRecordingService:
    """Zapisuje krótkie fragmenty wideo z kamer, gdy wykryto ruch i osobę."""

    def __init__(
        self,
        output_dir: str | Path | None = None,
        fps: float = 15.0,
        grace_period_seconds: float = 1.5,
        max_duration_seconds: float = 12.0,
    ):
        if output_dir is None:
            output_dir = Path(__file__).resolve().parents[2] / "data" / "recordings"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fps = fps
        self.grace_period_seconds = grace_period_seconds
        self.max_duration_seconds = max_duration_seconds

        self.recorders: dict[str, dict] = {}
        self.last_detection_time: dict[str, float] = {}

    def _build_path(self, camera_id: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return self.output_dir / f"{camera_id}_{timestamp}.mp4"

    def _start_recording(self, camera_id: str, frame) -> Path:
        height, width = frame.shape[:2]
        output_path = self._build_path(camera_id)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Nie udało się otworzyć zapisu wideo dla kamery {camera_id}.")

        self.recorders[camera_id] = {
            "writer": writer,
            "path": output_path,
            "started_at": time.monotonic(),
        }
        self.last_detection_time[camera_id] = time.monotonic()
        print(f"[REC] camera={camera_id} start={output_path.name}")
        return output_path

    def process_frame(self, camera_id: str, frame, motion_detected: bool, person_detected: bool):
        camera_key = str(camera_id or "default")

        if motion_detected and person_detected:
            self.last_detection_time[camera_key] = time.monotonic()
            if camera_key not in self.recorders:
                self._start_recording(camera_key, frame)

            writer = self.recorders[camera_key]["writer"]
            writer.write(frame)

            elapsed = time.monotonic() - self.recorders[camera_key]["started_at"]
            if elapsed >= self.max_duration_seconds:
                self.stop_recording(camera_key)
                return {"recording": False, "path": None}

            return {"recording": True, "path": str(self.recorders[camera_key]["path"])}

        if camera_key in self.recorders:
            last_seen = self.last_detection_time.get(camera_key, time.monotonic())
            if (time.monotonic() - last_seen) >= self.grace_period_seconds:
                self.stop_recording(camera_key)
                return {"recording": False, "path": None}

            writer = self.recorders[camera_key]["writer"]
            writer.write(frame)
            return {"recording": True, "path": str(self.recorders[camera_key]["path"])}

        return {"recording": False, "path": None}

    def stop_recording(self, camera_id: str):
        camera_key = str(camera_id or "default")
        recorder = self.recorders.pop(camera_key, None)
        if recorder is None:
            return None

        writer = recorder["writer"]
        writer.release()
        output_path = recorder["path"]
        print(f"[REC] camera={camera_key} saved={output_path.name}")
        return str(output_path)

    def close(self):
        for camera_id in list(self.recorders.keys()):
            self.stop_recording(camera_id)
