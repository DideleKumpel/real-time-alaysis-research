class CameraController:
    """Kontroler kamery i cyklu analizy obrazu."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def process_frame(self, frame):
        return self.pipeline.process_frame(frame)
