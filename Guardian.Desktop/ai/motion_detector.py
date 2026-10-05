import cv2
import numpy as np


class MotionDetector:
    """Detekcja ruchu na klatce kamery."""

    def __init__(self, history: int = 500, var_threshold: int = 25):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=True,
        )

    def detect(self, frame: np.ndarray, threshold: int = 1500) -> tuple[bool, np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        mask = self.background_subtractor.apply(gray)

        _, binary = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        motion_pixels = cv2.countNonZero(binary)
        detected = motion_pixels > threshold
        return detected, binary
