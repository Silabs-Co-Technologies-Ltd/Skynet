"""
Camera identification utility.

Opens every camera index from 0-5 with DirectShow, saves one snapshot from
each to camera_snapshots/, and reports resolution + brightness. Look at
the saved images to see which index is your USB webcam vs. the system
camera, then set that number as the CAMERA_INDEX environment variable
before running app.py, e.g.:

    set CAMERA_INDEX=1
    python app.py

Usage:
    python list_cameras.py
"""

import os
import cv2

OUT_DIR = "camera_snapshots"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    found_any = False

    for index in range(6):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            cap.release()
            print(f"index={index}: could not open, skipping")
            continue

        frame = None
        for _ in range(5):  # warm-up frames, same as app.py
            ok, frame = cap.read()
            if not ok:
                frame = None
                break

        if frame is None:
            print(f"index={index}: opened but produced no frames, skipping")
            cap.release()
            continue

        found_any = True
        height, width = frame.shape[:2]
        brightness = float(frame.mean())

        out_path = os.path.join(OUT_DIR, f"camera_{index}.jpg")
        cv2.imwrite(out_path, frame)

        print(
            f"index={index}: {width}x{height}, "
            f"brightness={brightness:.1f} -> saved {out_path}"
        )

        cap.release()

    if not found_any:
        print("No cameras responded on indices 0-5.")
    else:
        print(
            f"\nOpen the images in '{OUT_DIR}/' and find the one showing "
            f"your USB webcam's view. Its filename tells you the index "
            f"(e.g. camera_1.jpg -> CAMERA_INDEX=1)."
        )


if __name__ == "__main__":
    main()
