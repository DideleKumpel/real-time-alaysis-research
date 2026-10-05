class FaceService:
    """Logika rozpoznawania twarzy i zwracania statusu."""

    def __init__(self, matcher):
        self.matcher = matcher

    def verify_face(self, embedding, known_faces):
        return self.matcher.match(embedding, known_faces)
