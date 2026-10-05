class IncidentController:
    """Zarządzanie zdarzeniami alarmowymi."""

    def __init__(self, service):
        self.service = service

    def handle_unknown_face(self, event_data):
        return self.service.record_incident(event_data)
