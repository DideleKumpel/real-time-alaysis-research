"""
Face Recognition Pipeline  –  3-panelowy podgląd
=================================================
Stos technologiczny:
  • YOLO (własny finetune)  –  detekcja głów, etykieta klasy HEAD = 1
  • DeepFace (model Facenet) –  embeddingi + weryfikacja twarzy

Okno:
  ┌──────────────────────────┬──────────────────┬──────────────────────┐
  │  KAMERA  +  YOLO boxy    │  Przycięta twarz │  KNOWN / UNKNOWN     │
  └──────────────────────────┴──────────────────┴──────────────────────┘

Wymagania:
    pip install opencv-python ultralytics deepface numpy scipy tf-keras
"""

import cv2
import numpy as np
import pickle
import os
from pathlib import Path

from ultralytics import YOLO
from deepface import DeepFace
from scipy.spatial.distance import cosine

# ──────────────────────────────────────────────
# KONFIGURACJA
# ──────────────────────────────────────────────
CAMERA_INDEX        = 0
_DIR                = Path(__file__).parent          # katalog skryptu
YOLO_MODEL          = str(_DIR / "best.pt")          # best.pt obok skryptu
YOLO_HEAD_CLASS_ID  = 1                              # klasa głowy w Twoim modelu = 1
FACE_DB_PATH        = str(_DIR / "face_database.pkl")
SIMILARITY_THRESHOLD= 0.40            # dystans cosinusowy; niżej = bardziej rygorystyczny
MIN_FACE_SIZE       = (60, 60)

DEEPFACE_MODEL      = "Facenet"       # DeepFace: "Facenet" | "Facenet512" | "ArcFace"
DEEPFACE_BACKEND    = "skip"          # "skip" = nie rób ponownej detekcji twarzy w DeepFace
                                      # (YOLO już wyciął twarz); alternatywy: "opencv", "mtcnn"

# Rozmiary paneli
PANEL_H   = 480
CAM_W     = 640
FACE_W    = 240
VERDICT_W = 300
TOTAL_W   = CAM_W + FACE_W + VERDICT_W   # 1180


# ──────────────────────────────────────────────
# BAZA DANYCH TWARZY  (embeddingi DeepFace)
# ──────────────────────────────────────────────
class FaceDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.records: dict[str, list[np.ndarray]] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                self.records = pickle.load(f)
            print(f"[INFO] Baza załadowana: {list(self.records.keys())}")
        else:
            print("[INFO] Brak bazy – uruchom tryb rejestracji.")

    def save(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.records, f)
        print("[INFO] Baza zapisana.")

    def add(self, name: str, emb: np.ndarray):
        self.records.setdefault(name, []).append(emb)

    def find(self, emb: np.ndarray) -> tuple[str, float]:
        """Zwraca (imię, dystans) dla najbliższego wzorca lub ('UNKNOWN', 1.0)."""
        best_name, best_dist = "UNKNOWN", 1.0
        for name, embs in self.records.items():
            for ref in embs:
                d = cosine(emb, ref)
                if d < best_dist:
                    best_dist, best_name = d, name
        if best_dist > SIMILARITY_THRESHOLD:
            return "UNKNOWN", best_dist
        return best_name, best_dist

    def is_empty(self) -> bool:
        return len(self.records) == 0


# ──────────────────────────────────────────────
# POMOCNICZE RYSOWANIE
# ──────────────────────────────────────────────
def _black(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _text(img, txt, pos, color=(180, 180, 180), scale=0.65, thick=1):
    cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def build_face_panel(face_bgr) -> np.ndarray:
    """Panel środkowy – przycięta twarz."""
    panel = _black(FACE_W, PANEL_H)
    cv2.rectangle(panel, (0, 0), (FACE_W, 34), (30, 30, 30), -1)
    _text(panel, "CROPPED FACE", (8, 24), (180, 180, 180), 0.6)

    if face_bgr is not None and face_bgr.size > 0:
        max_h, max_w = PANEL_H - 50, FACE_W - 10
        h, w = face_bgr.shape[:2]
        scale = min(max_w / w, max_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(face_bgr, (nw, nh))
        yo = (PANEL_H - nh) // 2 + 10
        xo = (FACE_W - nw) // 2
        panel[yo:yo+nh, xo:xo+nw] = resized
        cv2.rectangle(panel, (xo-1, yo-1), (xo+nw, yo+nh), (80, 80, 80), 1)
    else:
        _text(panel, "no face detected", (8, PANEL_H // 2), (60, 60, 60), 0.55)

    return panel


def build_verdict_panel(verdict: str, name: str, dist: float) -> np.ndarray:
    """Panel prawy – wynik FaceNet przez DeepFace."""
    panel = _black(VERDICT_W, PANEL_H)
    cv2.rectangle(panel, (0, 0), (VERDICT_W, 34), (30, 30, 30), -1)
    _text(panel, "DEEPFACE  RESULT", (8, 24), (180, 180, 180), 0.6)

    cfg = {
        "known":    ((0, 120, 0),   "KNOWN",    "PERSON"),
        "unknown":  ((0, 0, 160),   "UNKNOWN",  "PERSON"),
        "empty_db": ((60, 60, 0),   "NO DB",    "register first"),
        "no_face":  ((40, 40, 40),  "NO FACE",  "detected"),
        "error":    ((80, 0, 80),   "ERROR",    "see console"),
    }
    bg_color, big, sub = cfg.get(verdict, cfg["no_face"])

    cv2.rectangle(panel, (10, 45), (VERDICT_W-10, PANEL_H-20), bg_color, -1)
    cv2.rectangle(panel, (10, 45), (VERDICT_W-10, PANEL_H-20), (160, 160, 160), 1)

    cx   = VERDICT_W // 2
    font = cv2.FONT_HERSHEY_DUPLEX

    (tw, _), _ = cv2.getTextSize(big, font, 1.5, 2)
    cv2.putText(panel, big, (cx - tw//2, PANEL_H//2 - 25),
                font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    (tw2, _), _ = cv2.getTextSize(sub, font, 0.9, 1)
    cv2.putText(panel, sub, (cx - tw2//2, PANEL_H//2 + 22),
                font, 0.9, (210, 210, 210), 1, cv2.LINE_AA)

    if verdict == "known" and name:
        (tw3, _), _ = cv2.getTextSize(name, font, 0.75, 1)
        cv2.putText(panel, name, (cx - tw3//2, PANEL_H//2 + 62),
                    font, 0.75, (160, 255, 160), 1, cv2.LINE_AA)

    if verdict in ("known", "unknown"):
        dist_str = f"dist: {dist:.3f}"
        (tw4, _), _ = cv2.getTextSize(dist_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(panel, dist_str, (cx - tw4//2, PANEL_H - 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    return panel


# ──────────────────────────────────────────────
# GŁÓWNA KLASA POTOKU
# ──────────────────────────────────────────────
class FaceRecognitionPipeline:
    def __init__(self):
        print("[INFO] Ładowanie YOLO…")
        self.yolo = YOLO(YOLO_MODEL)

        print("[INFO] Rozgrzewanie DeepFace / Facenet…")
        # Pierwsze wywołanie DeepFace pobiera wagi modelu – robimy to z góry
        _dummy = np.zeros((160, 160, 3), dtype=np.uint8)
        try:
            DeepFace.represent(_dummy,
                               model_name=DEEPFACE_MODEL,
                               detector_backend=DEEPFACE_BACKEND,
                               enforce_detection=False)
        except Exception:
            pass  # błąd na pustym obrazie jest normalny przy rozgrzewaniu

        self.db = FaceDatabase(FACE_DB_PATH)
        print("[INFO] Pipeline gotowy.")

    # ── embedding przez DeepFace ─────────────────
    def _extract_embedding(self, face_bgr: np.ndarray) -> np.ndarray | None:
        """
        Przyjmuje wyciętą twarz (BGR numpy), zwraca wektor embeddingu lub None.
        DeepFace.represent() zwraca listę słowników; bierzemy pierwszy wynik.
        enforce_detection=False – twarz już jest wycięta przez YOLO.
        """
        try:
            results = DeepFace.represent(
                img_path      = face_bgr,          # akceptuje np.ndarray BGR
                model_name    = DEEPFACE_MODEL,
                detector_backend = DEEPFACE_BACKEND,
                enforce_detection = False,
                align          = True,
            )
            if results:
                return np.array(results[0]["embedding"], dtype=np.float32)
        except Exception as e:
            print(f"[WARN] DeepFace błąd: {e}")
        return None

    # ── detekcja głów YOLO ───────────────────────
    def _detect_heads(self, frame: np.ndarray) -> list:
        """Zwraca listę (x1, y1, x2, y2, conf) dla klasy HEAD (id=1)."""
        results = self.yolo(frame, verbose=False)[0]
        boxes   = []
        for box in results.boxes:
            if int(box.cls[0]) != YOLO_HEAD_CLASS_ID:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2, float(box.conf[0])))
        # sortuj: największy box (= twarz z przodu) jako pierwszy
        boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        return boxes

    # ── przetwarzanie klatki ─────────────────────
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Zwraca skomponowany obraz 3-panelowy."""
        h_orig, w_orig = frame.shape[:2]

        # Panel 1 – kamera (przeskalowana)
        cam_panel = cv2.resize(frame.copy(), (CAM_W, PANEL_H))
        sx = CAM_W  / w_orig
        sy = PANEL_H / h_orig

        boxes = self._detect_heads(frame)

        best_crop = None
        verdict   = "no_face"
        res_name  = ""
        res_dist  = 1.0

        for idx, (x1, y1, x2, y2, conf) in enumerate(boxes):
            # Skalowane współrzędne na panelu kamery
            sx1, sy1 = int(x1*sx), int(y1*sy)
            sx2, sy2 = int(x2*sx), int(y2*sy)

            # Wytnij twarz z oryginału z małym marginesem
            marg = 10
            fh, fw = frame.shape[:2]
            crop = frame[max(0, y1-marg):min(fh, y2+marg),
                         max(0, x1-marg):min(fw, x2+marg)]

            too_small = (crop.size == 0 or
                         crop.shape[0] < MIN_FACE_SIZE[0] or
                         crop.shape[1] < MIN_FACE_SIZE[1])

            if too_small:
                box_color = (0, 165, 255)
                box_label = f"too small ({conf:.2f})"

            else:
                emb = self._extract_embedding(crop)

                if emb is None:
                    box_color = (0, 165, 255)
                    box_label = f"embed error ({conf:.2f})"
                    if idx == 0:
                        best_crop = crop
                        verdict   = "error"

                elif self.db.is_empty():
                    box_color = (0, 220, 220)
                    box_label = f"empty db ({conf:.2f})"
                    if idx == 0:
                        best_crop = crop
                        verdict   = "empty_db"

                else:
                    name, dist = self.db.find(emb)
                    if name == "UNKNOWN":
                        box_color = (0, 0, 220)
                        box_label = f"UNKNOWN ({conf:.2f})"
                        if idx == 0:
                            best_crop = crop
                            verdict   = "unknown"
                            res_dist  = dist
                    else:
                        box_color = (0, 210, 0)
                        box_label = f"{name} ({conf:.2f})"
                        if idx == 0:
                            best_crop = crop
                            verdict   = "known"
                            res_name  = name
                            res_dist  = dist

            # Rysuj na panelu kamery
            cv2.rectangle(cam_panel, (sx1, sy1), (sx2, sy2), box_color, 2)
            cv2.putText(cam_panel, box_label,
                        (sx1, max(sy1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv2.LINE_AA)

        # Nagłówek panelu kamery
        cv2.rectangle(cam_panel, (0, 0), (CAM_W, 34), (30, 30, 30), -1)
        _text(cam_panel, "YOLO  –  HEAD DETECTION  (class 1)", (8, 24), (200, 200, 200), 0.6)
        _text(cam_panel, "Q = quit", (CAM_W - 80, 24), (120, 120, 120), 0.5)

        # Panele 2 i 3
        face_panel    = build_face_panel(best_crop)
        verdict_panel = build_verdict_panel(verdict, res_name, res_dist)

        # Złóż + separatory
        composed = np.hstack([cam_panel, face_panel, verdict_panel])
        composed[:, CAM_W-1:CAM_W+1]               = (60, 60, 60)
        composed[:, CAM_W+FACE_W-1:CAM_W+FACE_W+1] = (60, 60, 60)
        return composed

    # ── pętla na żywo ────────────────────────────
    def run_live(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("[BŁĄD] Nie można otworzyć kamery.")
            return

        cv2.namedWindow("Face Recognition Pipeline", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face Recognition Pipeline", TOTAL_W, PANEL_H)
        print("[INFO] Rozpoznawanie na żywo.  Q = wyjście.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[BŁĄD] Brak klatki.")
                break

            composed = self.process_frame(frame)
            cv2.imshow("Face Recognition Pipeline", composed)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    # ── rejestracja nowej twarzy ─────────────────
    def register_face(self, name: str, num_samples: int = 5):
        """
        Otwiera kamerę, wykrywa głowę YOLO, pobiera embedding DeepFace
        i zapisuje do bazy.  SPACJA = zapisz próbkę, Q = zakończ.
        """
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("[BŁĄD] Nie można otworzyć kamery.")
            return

        print(f"[REJESTRACJA] '{name}'  –  {num_samples} próbek")
        print("  SPACJA = zapisz  |  Q = zakończ")
        collected = 0

        while collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            display = cv2.resize(frame.copy(), (CAM_W, PANEL_H))
            sx = CAM_W / frame.shape[1]
            sy = PANEL_H / frame.shape[0]

            boxes = self._detect_heads(frame)
            for (x1, y1, x2, y2, conf) in boxes:
                cv2.rectangle(display,
                              (int(x1*sx), int(y1*sy)),
                              (int(x2*sx), int(y2*sy)),
                              (0, 220, 220), 2)

            cv2.rectangle(display, (0, 0), (CAM_W, 34), (30, 30, 30), -1)
            _text(display,
                  f"Próbki: {collected}/{num_samples}   SPACJA=zapisz  Q=wyjście",
                  (8, 24), (0, 220, 220), 0.6)
            cv2.imshow("Rejestracja twarzy", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" ") and boxes:
                x1, y1, x2, y2, _ = boxes[0]
                fh, fw = frame.shape[:2]
                crop = frame[max(0,y1-10):min(fh,y2+10),
                             max(0,x1-10):min(fw,x2+10)]
                emb = self._extract_embedding(crop)
                if emb is not None:
                    self.db.add(name, emb)
                    collected += 1
                    print(f"  ✓ Próbka {collected}/{num_samples}")
                else:
                    print("  ✗ Nie udało się wyekstrahować embeddingu – spróbuj ponownie.")

        cap.release()
        cv2.destroyAllWindows()
        self.db.save()
        print(f"[INFO] Zapisano {collected} próbek dla '{name}'.")


if __name__ == "__main__":
    import sys

    pipeline = FaceRecognitionPipeline()

    if len(sys.argv) > 1 and sys.argv[1] == "register":
        # python face_recognition_pipeline.py register "Jan Kowalski"
        name = sys.argv[2] if len(sys.argv) > 2 else input("Podaj imię osoby: ")
        pipeline.register_face(name)
    else:
        pipeline.run_live()