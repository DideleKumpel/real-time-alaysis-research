"""Entry point dla modułowej aplikacji desktopowej Guardian."""

import cv2

from app.bootstrap import bootstrap_application
from core.logger import get_logger


def main() -> None:
    logger = get_logger(__name__)
    logger.info("Starting Guardian Desktop application")

    pipeline = bootstrap_application()
    logger.info("AI pipeline initialized successfully")
    print("Guardian Desktop started.")
    print("Press Q to quit the monitoring window.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Camera could not be opened.")
        print("Camera could not be opened.")
        return None

    cv2.namedWindow("Guardian Desktop", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Guardian Desktop", 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read camera frame.")
            break

        result = pipeline.process_frame(frame)
        output = result.get("display", frame)

        cv2.imshow("Guardian Desktop", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Guardian Desktop closed.")
    return pipeline


if __name__ == "__main__":
    main()
