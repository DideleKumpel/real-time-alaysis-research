import cv2
import numpy as np


def draw_status(frame, text: str, color=(0, 255, 0)):
    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame


def resize_frame(frame, width: int, height: int):
    return cv2.resize(frame, (width, height))
