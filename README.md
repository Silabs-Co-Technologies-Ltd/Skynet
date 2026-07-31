# SKYNET — University Security Management System (Prototype)

Flask-based prototype implementing the Project15 objectives: AI facial-recognition
access control, unknown-person detection with automated alerts, a Security
Operations Center dashboard, and centralized audit logging.

## Setup

```bash
cd backend
pip install -r requirements.txt
```

The app uses SQLite by default — no separate database server to install or
run. A file named `skynet.db` is created automatically inside `backend/`
the first time the app or `seed.py` runs. To use a different database
instead, set a `DATABASE_URL` environment variable (any SQLAlchemy-
compatible URI) before running the commands below.

Then run:

```bash
python seed.py
```

This creates all tables and two default accounts:

| Username | Password    | Role              |
|----------|-------------|-------------------|
| admin    | Admin@123   | Admin             |
| security | Security@123| Security Officer  |

**Change both passwords after first login.**

Start the app:

```bash
python app.py
```

Then visit `http://localhost:5000/login`.

## Facial recognition engine

Face detection/recognition uses OpenCV Haar cascade detection + LBPH
(Local Binary Pattern Histogram) recognition, ported from a proven
working implementation. There is no TensorFlow, DeepFace, RetinaFace, or
model-download dependency — everything needed ships with this project
(`ai_engine/cascades/haarcascade_frontalface_default.xml`) or comes from
a standard `opencv-python` install.

**Important:** unlike an embeddings-based approach, LBPH requires an
explicit training step. Registering a user or adding/removing photos via
`/register-face` or a user's photo gallery page automatically retrains
the model, so day-to-day use needs no manual step. If you ever add
photos by editing `uploads/known_faces/` directly on disk (not
recommended), or after first-time setup with pre-existing data, use the
**"Train Recognition Model"** button on the Users page (or visit
`/train-model`) to rebuild it manually.

Run `python test_recognition.py` at any time to self-test recognition
against every registered user's own gallery photos without needing the
webcam.

## Notes on scope

- This is a single-camera prototype, per Project15 section 1.4 ("Scope of
  the Study") — the live monitoring page attributes every registered
  camera tile to the one physical capture device available, and says so
  on the page rather than implying full multi-camera ingestion.
- The `routes/` package is legacy scaffolding (empty, never registered as
  blueprints) — all routing lives in `app.py`. It's left in place rather
  than deleted, but is not part of the running application.
- `services/config.py` is an unused duplicate of `config.py`; the app only
  ever imports the top-level `config.py`.
