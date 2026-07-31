from datetime import datetime
import os

import cv2
import numpy as np
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    session,
    send_file,
    send_from_directory,
    Response,
    url_for
)
from reportlab.pdfgen import canvas

from config import Config
from extensions import db

from models.user import User
from models.camera import Camera
from models.security_alert import SecurityAlert
from models.incident_report import IncidentReport
from models.unknown_person import UnknownPerson
from models.system_user import SystemUser
from models.logs import AccessLog

from services import face_service, camera_service
from services.alert_service import create_unknown_alert
from utils.decorators import login_required, roles_required

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)

with app.app_context():
    # Safe/idempotent: creates any tables that don't exist yet without
    # touching existing data. Actual seed accounts are created by seed.py
    # (a fresh database has no login-capable SystemUser until that is run).
    try:
        db.create_all()
    except Exception as e:
        print(
            "\n[STARTUP ERROR] Could not open/connect to the database.\n"
            "Check that config.py / DATABASE_URL points at a writable "
            "SQLite file path (or a reachable database if overridden).\n"
            f"Details: {e}\n"
        )

face_service.ensure_upload_dirs()
face_service.warm_up_models()

# ---------------------------------------------------------------------------
# Camera capture handling
#
# The original code opened `cv2.VideoCapture(0)` at import time, which
# meant the entire Flask app would hang or throw on any machine without a
# webcam attached (including this kind of headless/server environment).
# The capture is now opened lazily on first use and every call site checks
# whether it actually succeeded.
# ---------------------------------------------------------------------------
_camera_capture = None


def _configure_capture(cap):
    """
    Optionally force a FOURCC / resolution on the capture immediately
    after opening it, before any frames are read, then print what was
    ACTUALLY negotiated (drivers often silently ignore .set() calls for
    modes they don't support). Both are now opt-in via env vars instead
    of hardcoded, because forcing a resolution/format the camera doesn't
    actually support can itself corrupt frames -- which may be exactly
    what happened with the previous hardcoded 640x480 + MJPG attempt.

    Env vars (all optional):
        CAMERA_FOURCC = MJPG | YUY2 | "none"  (default: YUY2, confirmed
                                                working for this project's
                                                webcam; "none" = don't
                                                force one)
        CAMERA_WIDTH  = e.g. 640              (unset = use camera's default)
        CAMERA_HEIGHT = e.g. 480              (unset = use camera's default)
    """
    # Defaults to YUY2: confirmed via diagnose_camera.py to be this
    # webcam's native format. Explicitly setting it (even though it's
    # already "the default") fixes a DirectShow pipeline/buffer-alignment
    # issue that otherwise produced full-frame static/noise -- leaving
    # FOURCC untouched, or forcing an unsupported format like MJPG, left
    # that buffer misaligned even though isOpened()/read() still "worked".
    fourcc = os.environ.get("CAMERA_FOURCC", "YUY2").strip()
    width = os.environ.get("CAMERA_WIDTH", "").strip()
    height = os.environ.get("CAMERA_HEIGHT", "").strip()

    try:
        if fourcc and fourcc.lower() != "none":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        # A small/1-frame internal buffer means we always read the most
        # recent frame instead of one queued up behind others, which
        # reduces the torn/broken-looking frames that show up when
        # processing (detection/encoding) can't keep up with the camera's
        # own frame rate.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception as e:
        print("[camera] Could not set FOURCC/resolution:", e)

    # Report what the driver is ACTUALLY using, regardless of what we
    # asked for -- this is the key diagnostic for "still static" reports.
    try:
        actual_fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc = "".join(
            chr((actual_fourcc_int >> (8 * i)) & 0xFF) for i in range(4)
        ).strip()
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[camera] Negotiated format: FOURCC='{actual_fourcc}' "
              f"resolution={actual_w}x{actual_h} "
              f"(requested FOURCC='{fourcc or 'default'}', "
              f"{width or 'default'}x{height or 'default'})")
    except Exception as e:
        print("[camera] Could not read back negotiated format:", e)


def _sample_brightness(cap, warmup_frames=5):
    """
    Grab a few frames and return the average pixel brightness of the last
    one. Many webcams (and especially virtual/system cameras) return a
    valid-looking but literally black frame on the very first read() while
    the sensor/exposure is still initializing, and a covered or disabled
    system camera will report frames that are black indefinitely. Reading
    past a short warm-up and checking brightness lets us tell "real video"
    apart from "opened successfully but there's nothing to see" instead of
    accepting the first device that merely responds.
    """
    frame = None
    for _ in range(warmup_frames):
        ok, frame = cap.read()
        if not ok:
            return None, 0.0
    if frame is None:
        return None, 0.0
    return frame, float(frame.mean())


BLACK_FRAME_THRESHOLD = 12.0  # mean pixel value (0-255); below this = "black"


def get_camera_capture():
    """
    Find a working camera. A USB webcam is very often NOT at index 0
    (index 0 is frequently a laptop's built-in/system camera, which may be
    covered, disabled, or a placeholder device that still "opens" but only
    ever returns black frames). So this doesn't just take the first device
    that opens -- it reads a few warm-up frames from each candidate and
    only accepts one that is actually producing non-black video.

    Set the CAMERA_INDEX environment variable to force a specific device
    (skips scanning entirely) if auto-detection ever picks the wrong one
    on a given machine, e.g. on Windows:
        set CAMERA_INDEX=1
    """
    global _camera_capture
    if _camera_capture is not None:
        return _camera_capture or None

    backends = [
        ("CAP_DSHOW", cv2.CAP_DSHOW),  # Windows-preferred, avoids MSMF hangs
        ("DEFAULT", cv2.CAP_ANY),
    ]

    # Defaults to 0: forced to use the first camera device on this
    # machine. Override with CAMERA_INDEX=<n> or CAMERA_INDEX=auto to
    # scan multiple indices/brightness detection instead.
    forced_index = os.environ.get("CAMERA_INDEX", "0")

    if forced_index.lower() != "auto":
        # An explicit index means "trust the user's choice" -- no
        # brightness-based rejection, since a dim room or a webcam that's
        # still adjusting exposure would otherwise get silently skipped.
        index = int(forced_index)
        for backend_name, backend_flag in backends:
            try:
                cap = cv2.VideoCapture(index, backend_flag)
            except Exception as e:
                print(f"[camera] index={index} backend={backend_name} "
                      f"raised on open: {e}")
                continue

            if not cap.isOpened():
                cap.release()
                print(f"[camera] index={index} backend={backend_name} "
                      f"did not open -> trying next backend")
                continue

            _configure_capture(cap)
            _frame, brightness = _sample_brightness(cap)
            print(f"[camera] SUCCESS -> using forced index={index} "
                  f"backend={backend_name} (brightness={brightness:.1f})")
            _camera_capture = cap
            return _camera_capture

        print(
            f"[camera] Forced CAMERA_INDEX={index} could not be opened "
            f"with any backend. Set CAMERA_INDEX=auto to scan instead, "
            f"or run list_cameras.py again to re-check device numbers."
        )
        _camera_capture = False
        return None

    indices = range(4)

    candidates = []

    for index in indices:
        for backend_name, backend_flag in backends:
            try:
                cap = cv2.VideoCapture(index, backend_flag)
            except Exception as e:
                print(f"[camera] index={index} backend={backend_name} "
                      f"raised on open: {e}")
                continue

            if not cap.isOpened():
                cap.release()
                print(f"[camera] index={index} backend={backend_name} "
                      f"did not open -> skipping")
                continue

            _configure_capture(cap)
            _frame, brightness = _sample_brightness(cap)

            if brightness >= BLACK_FRAME_THRESHOLD:
                print(f"[camera] SUCCESS -> using index={index} "
                      f"backend={backend_name} (brightness={brightness:.1f})")
                _camera_capture = cap
                return _camera_capture

            print(f"[camera] index={index} backend={backend_name} opened "
                  f"but frame is black (brightness={brightness:.1f}) "
                  f"-> skipping")
            candidates.append((index, backend_name, cap))

    # Nothing passed the brightness check. Rather than show nothing at
    # all, fall back to the first device that at least opened and
    # produced frames (better than a dead placeholder), but say so
    # loudly -- this usually means the real camera is at an index/backend
    # combo not tried above, or CAMERA_INDEX needs to be set explicitly.
    if candidates:
        index, backend_name, cap = candidates[0]
        for _idx, _name, other_cap in candidates[1:]:
            other_cap.release()
        print(
            f"[camera] WARNING: no candidate produced a non-black frame. "
            f"Falling back to index={index} backend={backend_name} anyway.\n"
            f"[camera] If this is the wrong camera, set the CAMERA_INDEX "
            f"environment variable to the correct device number and "
            f"restart, e.g.:  set CAMERA_INDEX=1"
        )
        _camera_capture = cap
        return _camera_capture

    print(
        "[camera] No working camera found. Checklist:\n"
        "  - Is the USB webcam plugged in and showing up in Windows "
        "'Camera' settings / Device Manager?\n"
        "  - Is another app (Zoom, Teams, Windows Camera app, another "
        "Python process) currently holding the camera open? Close it and "
        "restart this app.\n"
        "  - Windows Settings > Privacy & security > Camera: is 'Let "
        "desktop apps access your camera' turned ON?\n"
        "  - Try setting CAMERA_INDEX explicitly, e.g.: set CAMERA_INDEX=1"
    )
    _camera_capture = False
    return None


@app.route("/")
def home():
    return "system is 2.0 Running Successfully"


@app.route("/test-db")
def test_db():
    return "Database Connected Successfully"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = SystemUser.query.filter_by(
            username=request.form["username"]
        ).first()

        # check_password_hash-backed comparison instead of the previous
        # plaintext `password=request.form["password"]` filter.
        if user and user.check_password(request.form["password"]):
            session["user_id"] = user.id
            session["role"] = user.role
            session["username"] = user.username
            return redirect("/dashboard")

        return render_template("login.html", error="Invalid Credentials")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    total_users = User.query.count()
    student_count = User.query.filter_by(role="Student").count()
    staff_count = User.query.filter_by(role="Staff").count()
    total_cameras = Camera.query.count()
    camera_list = Camera.query.all()
    unknown_persons = UnknownPerson.query.count()
    total_incidents = IncidentReport.query.count()
    total_alerts = SecurityAlert.query.count()
    pending_alerts = SecurityAlert.query.filter_by(
        alert_status="pending").count()
    resolved_alerts = SecurityAlert.query.filter(
        SecurityAlert.alert_status.ilike("resolved")).count()
    recent_unknowns = UnknownPerson.query.order_by(
        UnknownPerson.detection_time.desc()
    ).limit(5).all()
    recent_access = AccessLog.query.order_by(
        AccessLog.id.desc()
    ).limit(5).all()

    return render_template(
        "dashboard.html",
        total_users=total_users,
        student_count=student_count,
        staff_count=staff_count,
        total_cameras=total_cameras,
        unknown_persons=unknown_persons,
        total_incidents=total_incidents,
        total_alerts=total_alerts,
        recent_unknowns=recent_unknowns,
        recent_access=recent_access,
        pending_alerts=pending_alerts,
        resolved_alerts=resolved_alerts,
        cameras=camera_list
    )


# ---------------------------------------------------------------------------
# User management (Use Case: Administrator -> Manage Users add/edit/delete)
# ---------------------------------------------------------------------------
@app.route("/register-user", methods=["GET", "POST"])
@login_required
@roles_required("Admin")
def register_user():
    if request.method == "POST":
        full_name = request.form["full_name"]
        matric_staff_id = request.form["matric_staff_id"]
        role = request.form["role"]
        department = request.form["department"]
        email = request.form.get("email")
        phone = request.form.get("phone")

        existing = User.query.filter_by(
            matric_staff_id=matric_staff_id).first()
        if existing:
            return render_template(
                "register_user.html",
                error="This ID already exists."
            )

        if email:
            existing_email = User.query.filter_by(email=email).first()
            if existing_email:
                return render_template(
                    "register_user.html",
                    error="This email is already registered."
                )

        photos = [f for f in request.files.getlist("photos") if f and f.filename]
        if not photos:
            return render_template(
                "register_user.html",
                error="Please upload at least one face photo."
            )

        user = User(
            full_name=full_name,
            matric_staff_id=matric_staff_id,
            role=role,
            department=department,
            email=email,
            phone=phone
        )
        db.session.add(user)
        db.session.commit()  # need user.id before creating its gallery folder

        # Multiple reference photos (different angles/lighting) directly
        # improve LBPH training quality over a single photo.
        saved_paths = face_service.save_face_images(user.id, photos)

        if saved_paths:
            user.image_path = os.path.relpath(saved_paths[0], face_service.BASE_DIR)
            db.session.commit()

        # LBPH requires an explicit training step -- unlike the previous
        # DeepFace pipeline's on-the-fly comparison, new photos don't
        # take effect until the model is retrained. Doing this
        # synchronously here keeps registration a single, complete step
        # (training a handful of users is fast -- seconds, not minutes).
        face_service.train_model()

        return redirect("/users")

    return render_template("register_user.html")


@app.route("/register-face", methods=["GET", "POST"])
@login_required
@roles_required("Admin")
def register_face():
    """
    Add more reference photos to an existing user's face gallery, without
    re-entering all their registration details. Several templates already
    linked to this URL but the route did not exist (404), and there was
    no way at all to add training photos once a user was registered.

    Recognition with a single reference photo is unreliable -- different
    angles, lighting, and expressions all affect match confidence. This
    lets an admin keep strengthening a user's gallery over time instead
    of being limited to one photo forever.
    """
    users = []
    for u in User.query.order_by(User.full_name).all():
        u.photo_count = face_service.gallery_photo_count(u.id)
        users.append(u)

    if request.method == "POST":
        user_id = request.form["user_id"]
        user = User.query.get_or_404(user_id)
        photos = [f for f in request.files.getlist("photos") if f and f.filename]

        if not photos:
            return render_template(
                "register_face.html",
                users=users,
                error="Please choose at least one photo to upload."
            )

        saved_paths = face_service.save_face_images(user.id, photos)

        if not user.image_path and saved_paths:
            user.image_path = os.path.relpath(saved_paths[0], face_service.BASE_DIR)
            db.session.commit()

        face_service.train_model()

        return redirect("/users")

    return render_template("register_face.html", users=users)


@app.route("/users")
@login_required
@roles_required("Admin")
def users():
    all_users = User.query.order_by(User.full_name).all()
    for u in all_users:
        u.photo_count = face_service.gallery_photo_count(u.id)
    return render_template("users.html", users=all_users)


@app.route("/users/<int:user_id>/edit", methods=["GET", "POST"])
@login_required
@roles_required("Admin")
def edit_user(user_id):
    user = User.query.get_or_404(user_id)

    if request.method == "POST":
        user.full_name = request.form["full_name"]
        user.matric_staff_id = request.form["matric_staff_id"]
        user.role = request.form["role"]
        user.department = request.form["department"]
        user.email = request.form.get("email")
        user.phone = request.form.get("phone")
        user.status = request.form.get("status", user.status)
        db.session.commit()
        return redirect("/users")

    return render_template("edit_user.html", user=user)


@app.route("/users/<int:user_id>/delete", methods=["POST"])
@login_required
@roles_required("Admin")
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    return redirect("/users")


@app.route("/users/<int:user_id>/photos")
@login_required
@roles_required("Admin")
def user_photos(user_id):
    """
    View and manage a single user's face gallery. This is the intended
    way to remove a bad training photo -- deleting through this page
    (rather than by hand in the uploads/known_faces/ folder) always
    triggers a model retrain automatically, so the trained model and
    what's actually in the gallery never fall out of sync.
    """
    user = User.query.get_or_404(user_id)
    photos = face_service.list_gallery_photos(user_id)
    return render_template("user_photos.html", user=user, photos=photos)


@app.route("/users/<int:user_id>/photos/<path:filename>/delete", methods=["POST"])
@login_required
@roles_required("Admin")
def delete_user_photo(user_id, filename):
    face_service.delete_face_image(user_id, filename)
    # LBPH has no implicit cache to invalidate -- the model must be
    # explicitly retrained for a removed photo to actually stop being
    # recognized.
    face_service.train_model()
    return redirect(f"/users/{user_id}/photos")


@app.route("/train-model")
@login_required
@roles_required("Admin")
def train_model_route():
    """
    Manually (re)train the LBPH recognition model from every registered
    user's current photo gallery. Registration and photo management
    routes already call this automatically, so this is mainly useful
    after first-time setup (existing users registered before this was
    wired in) or after any out-of-band change to uploads/known_faces/.
    """
    ok, msg, details = face_service.train_model()
    return render_template(
        "train_model.html",
        success=ok,
        message=msg,
        details=details
    )


# ---------------------------------------------------------------------------
# Camera management (Use Case: Configure Camera)
# ---------------------------------------------------------------------------
@app.route("/add-camera", methods=["GET", "POST"])
@login_required
@roles_required("Admin")
def add_camera():
    if request.method == "POST":
        camera_service.create_camera(
            camera_name=request.form["camera_name"],
            ip_address=request.form["ip_address"],
            location=request.form["location"],
            status="online"
        )
        return redirect("/cameras")
    return render_template("add_camera.html")


@app.route("/cameras")
@login_required
def cameras():
    camera_list = camera_service.get_all_cameras()
    return render_template(
        "cameras.html",
        cameras=camera_list
    )


@app.route("/cameras/<int:camera_id>/edit", methods=["GET", "POST"])
@login_required
@roles_required("Admin")
def edit_camera(camera_id):
    camera = camera_service.get_camera(camera_id)
    if not camera:
        return "Camera Not Found", 404

    if request.method == "POST":
        camera_service.update_camera(
            camera_id,
            camera_name=request.form["camera_name"],
            ip_address=request.form["ip_address"],
            location=request.form["location"],
            status=request.form.get("status", camera.status)
        )
        return redirect("/cameras")

    return render_template("edit_camera.html", camera=camera)


@app.route("/cameras/<int:camera_id>/delete", methods=["POST"])
@login_required
@roles_required("Admin")
def delete_camera(camera_id):
    camera_service.delete_camera(camera_id)
    return redirect("/cameras")


# ---------------------------------------------------------------------------
# Analytics / Alerts / Incidents / Access logs
# ---------------------------------------------------------------------------
@app.route("/analytics")
@login_required
def analytics():
    student_count = User.query.filter_by(role="Student").count()
    staff_count = User.query.filter_by(role="Staff").count()
    return render_template(
        "analytics.html",
        student_count=student_count,
        staff_count=staff_count
    )


@app.route("/alerts")
@login_required
@roles_required("Admin", "Security Officer")
def alerts():
    alert_list = SecurityAlert.query.order_by(
        SecurityAlert.created_at.desc()
    ).all()
    return render_template(
        "alerts.html",
        alerts=alert_list
    )


@app.route("/api/latest-alert-id")
@login_required
@roles_required("Admin", "Security Officer")
def latest_alert_id():
    """
    Polled by the monitoring/dashboard pages every few seconds so the
    browser can detect a brand-new security alert and play an audible
    notification client-side (Web Audio API, no external sound file or
    dependency needed) -- the "audible sound alert" described in the
    system design, alongside the existing on-screen alert entry.
    """
    latest = SecurityAlert.query.order_by(SecurityAlert.id.desc()).first()
    return {"latest_id": latest.id if latest else 0}


@app.route("/resolve-alert/<int:alert_id>")
@login_required
@roles_required("Admin", "Security Officer")
def resolve_alert(alert_id):
    alert = SecurityAlert.query.get_or_404(alert_id)
    alert.alert_status = "Resolved"
    db.session.commit()
    return redirect("/alerts")


@app.route("/unknown-image/<filename>")
@login_required
def unknown_image(filename):
    return send_from_directory("uploads/unknown_faces", filename)


@app.route("/known-face-image/<int:user_id>/<path:filename>")
@login_required
@roles_required("Admin")
def known_face_image(user_id, filename):
    return send_from_directory(
        os.path.join("uploads", "known_faces", str(user_id)),
        filename
    )


@app.route("/access-logs")
@login_required
@roles_required("Admin", "Security Officer")
def access_logs():
    """
    Surfaces the ERD's Access Log entity, which previously had no model,
    no route, and no template anywhere in the project even though the
    functional requirements explicitly call for storing and auditing
    access records ("store access records in a centralized database for
    audit and reporting").
    """
    logs = AccessLog.query.order_by(AccessLog.id.desc()).limit(200).all()
    return render_template("access_logs.html", logs=logs)


@app.route("/reports")
@login_required
@roles_required("Admin", "Supervisor")
def reports():
    return render_template(
        "reports.html",
        total_users=User.query.count(),
        total_cameras=Camera.query.count(),
        unknown_persons=UnknownPerson.query.count(),
        total_alerts=SecurityAlert.query.count(),
        total_incidents=IncidentReport.query.count()
    )


@app.route("/incidents")
@login_required
@roles_required("Admin", "Security Officer")
def incidents():
    reports_list = IncidentReport.query.order_by(
        IncidentReport.created_at.desc()
    ).all()
    return render_template(
        "incidents.html",
        reports=reports_list
    )


@app.route("/create-incident", methods=["GET", "POST"])
@login_required
@roles_required("Admin", "Security Officer")
def create_incident():
    if request.method == "POST":
        alert_id = request.form.get("alert_id") or None

        if alert_id and not SecurityAlert.query.get(alert_id):
            return render_template(
                "create_incident.html",
                error="No alert exists with that ID."
            )

        report = IncidentReport(
            alert_id=alert_id,
            title=request.form["title"],
            description=request.form["description"],
            officer_name=request.form["officer_name"]
        )
        db.session.add(report)
        db.session.commit()
        return redirect("/incidents")

    return render_template("create_incident.html")


@app.route("/test-alert")
@login_required
@roles_required("Admin")
def test_alert():
    create_unknown_alert("uploads/unknown_faces/test.jpg")
    return "Alert Created"


# ---------------------------------------------------------------------------
# Live monitoring / video feed
# ---------------------------------------------------------------------------
@app.route("/monitoring")
@login_required
@roles_required("Admin", "Security Officer")
def monitoring():
    camera_list = camera_service.get_all_cameras()
    return render_template(
        "monitoring.html",
        cameras=camera_list
    )


def generate_frames():
    capture = get_camera_capture()

    if capture is None:
        # No physical webcam available in this environment. Emit a single
        # placeholder frame instead of hanging/crashing the whole app, so
        # the rest of the system (dashboard, reports, user management,
        # alerts) remains fully usable without a camera attached.
        placeholder = np.zeros((480, 640, 3), dtype="uint8")
        cv2.putText(
            placeholder,
            "Camera Unavailable",
            (140, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        ret, buffer = cv2.imencode(".jpg", placeholder)
        frame_bytes = buffer.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )
        return

    frame_counter = 0

    with app.app_context():
        camera_row = camera_service.default_camera()

    while True:
        success, frame = capture.read()
        if not success:
            print("[camera] capture.read() failed mid-stream, stopping feed")
            break

        frame_counter += 1

        try:
            with app.app_context():
                frame = face_service.process_frame(
                    frame,
                    camera=camera_row,
                    frame_counter=frame_counter
                )

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                print("[camera] imencode failed on this frame, skipping")
                continue
            frame_bytes = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                frame_bytes +
                b'\r\n'
            )
        except Exception as e:
            # A single bad frame should never take down the whole stream
            # (or the app). Log it and keep serving subsequent frames.
            print("[camera] Error processing/encoding frame:", e)
            continue


@app.route("/video_feed")
@login_required
@roles_required("Admin", "Security Officer")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------------------------------------------------------------------
# Reports (PDF export + HTML report pages)
# ---------------------------------------------------------------------------
@app.route("/export-report")
@login_required
@roles_required("Admin", "Supervisor")
def export_report():
    pdf_file = "security_report.pdf"
    c = canvas.Canvas(pdf_file)
    c.drawString(100, 800, " Security Analytics Report")
    c.drawString(100, 760, f"Total Users: {User.query.count()}")
    c.drawString(100, 740, f"Total Cameras: {Camera.query.count()}")
    c.drawString(100, 720, f"Unknown Persons: {UnknownPerson.query.count()}")
    c.drawString(100, 700, f"Security Alerts: {SecurityAlert.query.count()}")
    c.drawString(100, 680, f"Incident Reports: {IncidentReport.query.count()}")
    c.save()
    return send_file(pdf_file, as_attachment=True)


@app.route("/student-report")
@login_required
@roles_required("Admin", "Supervisor")
def student_report():
    students = User.query.filter_by(role="Student").all()
    return render_template("student_report.html", students=students)


@app.route("/staff-report")
@login_required
@roles_required("Admin", "Supervisor")
def staff_report():
    staff = User.query.filter_by(role="Staff").all()
    return render_template("staff_report.html", staff=staff)


@app.route("/export-student-report")
@login_required
@roles_required("Admin", "Supervisor")
def export_student_report():
    students = User.query.filter_by(role="Student").all()
    pdf_file = "student_report.pdf"
    c = canvas.Canvas(pdf_file)
    c.drawString(100, 800, "Student Report")
    y = 760
    for student in students:
        c.drawString(
            50, y,
            f"{student.full_name} | "
            f"{student.matric_staff_id} | "
            f"{student.department}"
        )
        y -= 20
    c.save()
    return send_file(pdf_file, as_attachment=True)


@app.route("/export-staff-report")
@login_required
@roles_required("Admin", "Supervisor")
def export_staff_report():
    staff = User.query.filter_by(role="Staff").all()
    pdf_file = "staff_report.pdf"
    c = canvas.Canvas(pdf_file)
    c.drawString(100, 800, "Staff Report")
    y = 760
    for employee in staff:
        c.drawString(
            50, y,
            f"{employee.full_name} | "
            f"{employee.matric_staff_id} | "
            f"{employee.department}"
        )
        y -= 20
    c.save()
    return send_file(pdf_file, as_attachment=True)


@app.route("/unknown-report")
@login_required
@roles_required("Admin", "Security Officer")
def unknown_report():
    unknowns = UnknownPerson.query.order_by(
        UnknownPerson.detection_time.desc()
    ).all()
    return render_template("unknown_report.html", unknowns=unknowns)


@app.route("/export-unknown-report")
@login_required
@roles_required("Admin", "Security Officer")
def export_unknown_report():
    unknowns = UnknownPerson.query.order_by(
        UnknownPerson.detection_time.desc()
    ).all()
    pdf_file = "unknown_persons_report.pdf"
    c = canvas.Canvas(pdf_file)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(180, 800, "Unknown Persons Report")
    y = 730
    for person in unknowns:
        image_path = os.path.join(os.path.dirname(__file__), person.image_path)
        if image_path and os.path.exists(image_path):
            try:
                c.drawImage(image_path, 50, y - 90, width=85, height=113)
            except Exception:
                pass
        c.setFont("Helvetica", 11)
        c.drawString(160, y, f"Unknown Person ID: {person.id}")
        c.drawString(160, y - 20, f"Location: {person.location}")
        c.drawString(160, y - 40, f"Detected: {person.detection_time}")
        c.drawString(160, y - 60, f"Status: {person.status}")
        c.line(40, y - 105, 550, y - 105)
        y -= 140
        if y < 120:
            c.showPage()
            y = 730
    c.save()
    return send_file(pdf_file, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)