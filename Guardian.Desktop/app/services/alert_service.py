class AlertService:
    """Wysyłka alertów do serwera / użytkownika."""

    def __init__(self, client=None):
        self.client = client

    def send_alert(self, incident):
        if self.client is not None:
            return self.client.send(incident)
        return {
            "status": "queued",
            "incident": incident,
        }
