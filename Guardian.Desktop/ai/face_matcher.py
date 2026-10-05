from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cosine


class FaceMatcher:
    """Porównywanie twarzy z bazą znanych osó b."""

    def __init__(self, similarity_threshold: float = 0.40):
        self.similarity_threshold = similarity_threshold

    def match(self, embedding: np.ndarray, known_faces: dict[str, list[np.ndarray]]) -> tuple[str, float]:
        best_name = "UNKNOWN"
        best_distance = 1.0

        for name, embeddings in known_faces.items():
            for ref in embeddings:
                distance = cosine(embedding, ref)
                if distance < best_distance:
                    best_distance = float(distance)
                    best_name = name

        if best_distance > self.similarity_threshold:
            return "UNKNOWN", best_distance
        return best_name, best_distance
