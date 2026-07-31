"""
Database bootstrap script for the NAUB Surveillance System University Security Management
System.

Nothing in the original project created the database schema or a first
login account -- app.py referenced SystemUser for login, but there was no
model of any route that could ever insert one, so the system was
unusable out of the box. This script:

  1. Creates all tables (idempotent - safe to re-run).
  2. Seeds one Admin and one Security Officer account with hashed
     passwords, if they don't already exist.
  3. Seeds a default "Main Gate" camera so the live monitoring page and
     the recognition pipeline have a Camera row to attribute events to.

Usage:
    python seed.py
"""

from app import app
from extensions import db
from models.system_user import SystemUser
from models.camera import Camera


def seed():
    with app.app_context():
        db.create_all()

        if not SystemUser.query.filter_by(username="admin").first():
            admin = SystemUser(username="admin", role="Admin")
            admin.set_password("Admin@123")
            db.session.add(admin)
            print("Created default Admin account -> username: admin / password: Admin@123")
        else:
            print("Admin account already exists, skipping.")

        if not SystemUser.query.filter_by(username="security").first():
            officer = SystemUser(username="security", role="Security Officer")
            officer.set_password("Security@123")
            db.session.add(officer)
            print("Created default Security Officer account -> username: security / password: Security@123")
        else:
            print("Security Officer account already exists, skipping.")

        if not Camera.query.filter_by(camera_name="Main Gate").first():
            db.session.add(Camera(
                camera_name="Main Gate",
                ip_address="0",
                location="Main Gate",
                status="online"
            ))
            print("Created default 'Main Gate' camera.")
        else:
            print("Default camera already exists, skipping.")

        db.session.commit()
        print("Database seeding complete.")


if __name__ == "__main__":
    seed()
