from dataclasses import dataclass


@dataclass
class FaceRecord:
    person_name: str
    embedding: list[float]
    source: str = "manual"
