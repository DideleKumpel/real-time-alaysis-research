# Guardian Desktop

To jest moduł desktopowy systemu monitoringu. Cel: lokalne wykrywanie ruchu, analiza kamer, detekcja twarzy oraz alarmowanie w sytuacji podejrzanego zdarzenia.

## Struktura katalogów

```text
Guardian.Desktop/
├── main.py
├── README.md
├── legacy/
│   └── ProofOfConcept.py
├── ai/
│   ├── __init__.py
│   ├── motion_detector.py
│   ├── yolo_detector.py
│   ├── face_matcher.py
│   └── pipeline.py
├── app/
│   ├── __init__.py
│   ├── bootstrap.py
│   ├── controllers/
│   │   ├── camera_controller.py
│   │   └── incident_controller.py
│   ├── services/
│   │   ├── camera_service.py
│   │   ├── face_service.py
│   │   ├── incident_service.py
│   │   └── alert_service.py
│   ├── models/
│   │   ├── user.py
│   │   ├── incident.py
│   │   └── face_record.py
│   ├── config/
│   │   └── settings.py
│   └── utils/
│       └── image_utils.py
├── core/
│   ├── __init__.py
│   └── logger.py
├── tests/
│   └── test_ai_pipeline.py
└── requirements.txt
```

## Architektura

- `ai/` – warstwa inteligentna: detekcja ruchu, YOLO, dopasowanie twarzy
- `app/controllers/` – kontrolery obsługujące wejście z UI / sterowniki
- `app/services/` – logika biznesowa i operacje na danych
- `app/models/` – modele domenowe
- `core/` – wspólne komponenty: logger, wspólne ustawienia
- `legacy/` – stary prototyp do porównania i dalszego rozwoju

## Przepływ zdarzenia

1. Kamera dostarcza klatkę.
2. `motion_detector` wykrywa ruch.
3. Po wykryciu ruchu uruchamiany jest `yolo_detector`.
4. `face_matcher` porównuje twarze z bazą znanych użytkowników.
5. `incident_service` zapisuje zdarzenie jeśli twarz jest obca lub niewykryta.
6. `alert_service` wysyła alert do serwera i użytkowników.
