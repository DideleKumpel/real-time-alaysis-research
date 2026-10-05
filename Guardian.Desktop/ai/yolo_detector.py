from __future__ import annotations


class YoloDetector:
    """Detekcja twarzy / obiektów przy pomocy YOLO."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load(self):
        """Załaduj model YOLO."""
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        return self.model

    def detect_heads(self, frame):
        if self.model is None:
            raise RuntimeError("Model YOLO nie został załadowany.")

        results = self.model(frame, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            if int(box.cls[0]) != 1:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2, float(box.conf[0])))
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        return boxes
