import json
from pathlib import Path

import cv2


class CameraService:
    """Mockup serwisu kamer z trwałym zapisem i statusami connected/disconnected."""

    def __init__(self, storage_path: str | None = None):
        project_root = Path(__file__).resolve().parents[2]
        self.storage_path = Path(storage_path) if storage_path else project_root / "data" / "camera_registry.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.cameras = self.load_cameras()

    def load_cameras(self):
        if not self.storage_path.exists():
            default_cameras = [
                {"id": "cam-1", "name": "Kamera wejście", "source": "0", "source_type": "camera_index", "status": "connected"},
                {"id": "cam-2", "name": "Kamera parking", "source": "1", "source_type": "camera_index", "status": "connected"},
            ]
            self._save_camera_store(default_cameras)
            return default_cameras

        try:
            with self.storage_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            pass

        return []

    def _save_camera_store(self, cameras):
        with self.storage_path.open("w", encoding="utf-8") as file:
            json.dump(cameras, file, ensure_ascii=False, indent=2)

    def save(self):
        self._save_camera_store(self.cameras)

    def refresh_statuses(self):
        for camera in self.cameras:
            camera["status"] = self.check_camera_status(camera)
        self.save()
        return self.cameras

    def check_camera_status(self, camera):
        source = str(camera.get("source", "")).strip()
        if not source:
            return "disconnected"

        source_type = camera.get("source_type", "index")
        normalized = "index" if source_type in ("index", "camera_index") else source_type

        try:
            if normalized == "index":
                idx = int(source)
                cap = cv2.VideoCapture(idx)
            else:
                cap = cv2.VideoCapture(source)

            is_open = cap.isOpened()
            cap.release()
            return "connected" if is_open else "disconnected"
        except Exception:
            return "disconnected"

    def add_camera(self, name: str, source: str, source_type: str = "index"):
        name = (name or "Nowa kamera").strip()
        source = (source or "0").strip()

        if not name or not source:
            raise ValueError("Nazwa i źródło kamery są wymagane.")

        normalized_source_type = "index" if source_type in ("index", "camera_index") else source_type
        camera = {
            "id": f"cam-{len(self.cameras) + 1}",
            "name": name,
            "source": source,
            "source_type": normalized_source_type,
            "status": self.check_camera_status({"source": source, "source_type": normalized_source_type}),
        }

        self.cameras.append(camera)
        self.save()
        return camera

    def add_many(self, camera_specs):
        added = []
        for spec in camera_specs:
            added.append(self.add_camera(spec.get("name", "Nowa kamera"), spec.get("source", "0"), spec.get("source_type", "camera_index")))
        return added

    def remove_camera(self, camera_id: str):
        before = len(self.cameras)
        self.cameras = [camera for camera in self.cameras if camera.get("id") != camera_id]
        self.save()
        return len(self.cameras) < before

    def get_cameras(self):
        return self.refresh_statuses()

    def read_camera(self, frame):
        return frame
