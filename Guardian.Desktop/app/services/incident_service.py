class IncidentService:
    """Obsługa zapisów zdarzeń alarmowych."""

    def __init__(self, repository=None):
        self.repository = repository

    def record_incident(self, event_data):
        if self.repository is not None:
            return self.repository.save(event_data)
        return {
            "status": "recorded",
            "payload": event_data,
        }
