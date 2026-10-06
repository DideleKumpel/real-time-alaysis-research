import cv2


class LivePreviewService:
    """Serwis do odczytu wielu streamów z kamer jednocześnie."""

    def __init__(self):
        self.captures = {}

    def _camera_key(self, camera):
        if isinstance(camera, dict):
            return str(camera.get("id", camera.get("name", "camera")))
        return str(camera)

    def open_camera(self, camera, source=None, source_type="index"):
        key = self._camera_key(camera)
        existing = self.captures.get(key)
        if existing is not None and existing.isOpened():
            return True

        if source is None:
            source = camera.get("source") if isinstance(camera, dict) else camera
        if source_type is None:
            source_type = camera.get("source_type", "index") if isinstance(camera, dict) else "index"

        normalized_source_type = "index" if str(source_type).lower() in ("index", "camera_index") else str(source_type).lower()

        try:
            if normalized_source_type == "index":
                camera_source = int(str(source).strip())
            else:
                camera_source = str(source)
        except (TypeError, ValueError):
            camera_source = str(source)

        print(f"[Camera] opening key={key} source={source} source_type={source_type} normalized={normalized_source_type} resolved={camera_source}")

        cap = cv2.VideoCapture(camera_source)
        if not cap or not cap.isOpened():
            print(f"[Camera] FAILED to open: key={key} source={camera_source}")
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.captures[key] = cap
        print(f"[Camera] opened successfully: key={key} source={camera_source}")
        return True

    def read_frame(self, camera):
        key = self._camera_key(camera)
        cap = self.captures.get(key)
        if cap is None or not cap.isOpened():
            return None

        ok, frame = cap.read()
        if not ok:
            return None
        return frame

    def close(self):
        for cap in self.captures.values():
            if cap is not None:
                cap.release()
        self.captures.clear()
