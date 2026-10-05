from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = APP_ROOT.parent
LEGACY_MODEL_PATH = PROJECT_ROOT / "scripts" / "ProofOfCOncepts" / "best.pt"
DEFAULT_MODEL_PATH = APP_ROOT / "models" / "best.pt"


def resolve_model_path(model_name: str = "best.pt") -> str:
    candidates = [
        APP_ROOT / "models" / model_name,
        APP_ROOT / model_name,
        PROJECT_ROOT / "scripts" / "ProofOfCOncepts" / model_name,
        Path.cwd() / model_name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    if LEGACY_MODEL_PATH.exists():
        return str(LEGACY_MODEL_PATH)

    raise FileNotFoundError(
        f"Model AI '{model_name}' not found. Expected in: {[str(p) for p in candidates]}"
    )


CAMERA_INDEX = 0
YOLO_MODEL_PATH = resolve_model_path()
SIMILARITY_THRESHOLD = 0.40
MOTION_THRESHOLD = 1500
FACE_DB_PATH = str(APP_ROOT / "data" / "face_database.pkl")
