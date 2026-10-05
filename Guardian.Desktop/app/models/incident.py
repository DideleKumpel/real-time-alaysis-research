from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Incident:
    camera_id: str
    event_type: str
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    video_path: str = ""
    person_status: str = "unknown"
