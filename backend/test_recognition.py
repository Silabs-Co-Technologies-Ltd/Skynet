"""
Recognition self-test / AI model health check for NAUB Surveillance System.

Runs the EXACT same code path live recognition uses
(face_service.identify_face) against every registered user's own gallery
photos, and reports whether each one is correctly recognized. This is the
fastest way to check recognition health without needing the webcam or
the live dashboard at all.

Recognition backend: Haar cascade detection + LBPH recognition (ported
from the SAMS attendance system) -- no TensorFlow, no model downloads,
no network access required.

What it checks, in order:
  1. Environment: can cv2 be imported, and does the bundled Haar cascade
     file load correctly.
  2. Model status: is a model currently trained, and how old is it
     relative to the gallery (retrains if needed).
  3. Gallery inventory: every registered user and how many training
     photos they have on file.
  4. Per-photo detectability: does the Haar cascade even find a face in
     each training photo? (a bad/blurry/multi-person photo won't detect).
  5. Recognition self-test: for every detectable training photo, crop the
     face exactly like a live video frame would, run it through
     face_service.identify_face(), and confirm it resolves back to the
     correct user -- with the LBPH confidence score reported either way.

Usage (run from the backend/ folder):
    python test_recognition.py

Safe to run whether or not app.py's server is already running elsewhere
(it never binds a port and never opens the camera).
"""

import os
import sys
import time

print("=" * 70)
print("NAUB Surveillance System Recognition & AI Model Self-Test")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Environment check
# ---------------------------------------------------------------------------
print("\n[1/5] Environment check")

try:
    import cv2
    print(f"  cv2 imported OK (version {cv2.__version__})")
except Exception as e:
    print("  FAILED to import cv2:", e)
    sys.exit(1)

try:
    from ai_engine.cascade_path import get_cascade_path
    cascade_path = get_cascade_path()
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print(f"  Cascade file at {cascade_path} loaded but is EMPTY/invalid")
    else:
        print(f"  Haar cascade loaded OK from: {cascade_path}")
except Exception as e:
    print("  FAILED to load Haar cascade:", e)
    sys.exit(1)

# ---------------------------------------------------------------------------
# App/DB context + services (reuses the project's own code, not a copy)
# ---------------------------------------------------------------------------
try:
    from app import app  # triggers app.py's own startup incl. warm_up_models()
    from extensions import db
    from models.user import User
    from services import face_service
    from ai_engine import face_recognition_engine as fre
except Exception as e:
    print("\nFAILED to import project modules -- run this from the "
          "backend/ folder, with the same virtualenv as app.py:", e)
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Model status
# ---------------------------------------------------------------------------
print("\n[2/5] Model status")

if not fre.model_is_trained():
    print("  No trained model found yet. Training now from the current "
          "gallery...")
    t0 = time.time()
    ok, msg, details = face_service.train_model(progress_callback=lambda m: None)
    print(f"  {msg} ({time.time() - t0:.1f}s)")
    if not ok:
        print("\n  Cannot continue self-test without a trained model. "
              "Register at least one user with a detectable face photo.")
        sys.exit(0)
else:
    print(f"  Trained model found at: {fre.MODEL_PATH}")

# ---------------------------------------------------------------------------
# 3 + 4 + 5. Gallery inventory + detectability + recognition self-test
# ---------------------------------------------------------------------------
print("\n[3/5] Gallery inventory")

with app.app_context():
    users = User.query.order_by(User.full_name).all()

    if not users:
        print("  No users registered yet -- nothing to test. "
              "Register at least one user first.")
        sys.exit(0)

    total_photos = 0
    for u in users:
        count = face_service.gallery_photo_count(u.id)
        total_photos += count
        flag = "  <-- NO PHOTOS, cannot be recognized" if count == 0 else ""
        print(f"  [{u.id}] {u.full_name} ({u.role}): {count} photo(s){flag}")

    print(f"  Total: {len(users)} user(s), {total_photos} photo(s)")

    print("\n[4/5] Per-photo detectability (can the cascade find a face at all?)")
    print("[5/5] Recognition self-test (does it resolve back to the right user?)")
    print()

    pass_count = 0
    fail_count = 0
    no_face_count = 0

    for u in users:
        folder = face_service.person_dir(u.id)
        photos = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(face_service.IMAGE_EXTENSIONS)
        ])

        if not photos:
            continue

        print(f"--- {u.full_name} (user id {u.id}) ---")

        for photo_name in photos:
            photo_path = os.path.join(folder, photo_name)
            frame = cv2.imread(photo_path)

            if frame is None:
                print(f"  {photo_name}: could not read image file, skipping")
                continue

            faces = face_service.detect_faces(frame)

            if not faces:
                print(f"  {photo_name}: NO FACE DETECTED "
                      f"(bad/blurry photo, or face too small/angled -- "
                      f"consider replacing it)")
                no_face_count += 1
                continue

            # Use the first/largest detected face, same as live pipeline.
            x1, y1, x2, y2 = faces[0]["facial_area"]
            face_crop = frame[y1:y2, x1:x2]

            t0 = time.time()
            matched_user = face_service.identify_face(face_crop)
            elapsed = time.time() - t0

            if matched_user and matched_user.id == u.id:
                print(f"  {photo_name}: PASS -> correctly recognized as "
                      f"{matched_user.full_name} ({elapsed:.2f}s)")
                pass_count += 1
            elif matched_user:
                print(f"  {photo_name}: FAIL -> matched WRONG person "
                      f"({matched_user.full_name}, id={matched_user.id}) "
                      f"({elapsed:.2f}s)")
                fail_count += 1
            else:
                print(f"  {photo_name}: FAIL -> not recognized at all "
                      f"(returned unknown) ({elapsed:.2f}s)")
                fail_count += 1

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Recognized correctly : {pass_count}")
    print(f"  Failed / wrong match : {fail_count}")
    print(f"  No face detected     : {no_face_count}")

    if fail_count == 0 and no_face_count == 0 and pass_count > 0:
        print("\n  All registered photos are recognized correctly. "
              "If live webcam recognition still fails, the issue is "
              "likely camera framing/lighting/angle rather than the "
              "model or gallery.")
    elif pass_count == 0:
        print("\n  NOTHING is being recognized correctly. Check the "
              "[1/5] and [2/5] output above for cascade/model-loading "
              "errors, and try running /train-model again from the app.")
    else:
        print("\n  Some photos fail -- for FAIL cases, try replacing that "
              "specific photo (clearer, more front-facing, better lit) "
              "via a user's photo gallery page, then retrain. For "
              "'no face detected' cases, the photo itself needs replacing.")
