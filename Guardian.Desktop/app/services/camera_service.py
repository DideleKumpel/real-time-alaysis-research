class CameraService:
    """Obsługa kamery i strumienia wideo."""

    def __init__(self, controller):
        self.controller = controller

    def read_camera(self, frame):
        return self.controller.process_frame(frame)
