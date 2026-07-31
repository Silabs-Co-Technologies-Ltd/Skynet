import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "skynet_secret_key")

    # SQLite needs no separate server process, no credentials, and no
    # network port -- the whole database is one file on disk. This removes
    # the "is Postgres running / are the credentials right" failure class
    # entirely for local development and demo/grading. Still overridable
    # via DATABASE_URL for anyone who wants to point at a real Postgres
    # instance later without touching code.
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "sqlite:///" + os.path.join(BASE_DIR, "skynet.db")
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False
