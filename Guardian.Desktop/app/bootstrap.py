from ai.pipeline import MonitoringPipeline
from app.config.settings import YOLO_MODEL_PATH


def bootstrap_application():
    """Uruchamia pipeline AI dla aplikacji desktopowej."""
    pipeline = MonitoringPipeline(yolo_model_path=YOLO_MODEL_PATH)
    pipeline.initialize()
    return pipeline
