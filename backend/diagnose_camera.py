"""
Camera format diagnostic tool.

Opens CAMERA_INDEX (defaults to 1) with a handful of different FOURCC /
"don't touch it" settings, saves a labeled snapshot for each, and prints
what the driver actually negotiated for every attempt. Static/rainbow
"noise" frames usually mean the pixel format OpenCV assumes doesn't match
what the camera is actually sending -- this makes it possible to SEE
which combination (if any) produces a clean image, instead of guessing.

Usage:
    python diagnose_camera.py            (uses index 1)
    python diagnose_camera.py 0          (uses index 0)
"""

import os
import sys
import cv2

OUT_DIR = "camera_diagnostics"

ATTEMPTS = [
    ("default", None),   # don't touch FOURCC/resolution at all
    ("MJPG", "MJPG"),
    ("YUY2", "YUY2"),
    ("NV12", "NV12"),
]


def describe_fourcc(cap):
    val = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((val >> (8 * i)) & 0xFF) for i in range(4)).strip()


def main():
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Testing camera index={index} with {len(ATTEMPTS)} format attempts...\n")

    for label, fourcc in ATTEMPTS:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print(f"[{label}] could not open index={index}")
            cap.release()
            continue

        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

        frame = None
        for _ in range(5):  # warm-up
            ok, frame = cap.read()
            if not ok:
                frame = None
                break

        if frame is None:
            print(f"[{label}] opened but produced no frames")
            cap.release()
            continue

        negotiated = describe_fourcc(cap)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        brightness = float(frame.mean())
        stddev = float(frame.std())  # noise tends to have very high stddev

        out_path = os.path.join(OUT_DIR, f"{label}_index{index}.jpg")
        cv2.imwrite(out_path, frame)

        print(
            f"[{label}] requested='{fourcc or 'unset'}' "
            f"negotiated='{negotiated}' {w}x{h} "
            f"brightness={brightness:.1f} stddev={stddev:.1f} "
            f"-> saved {out_path}"
        )

        cap.release()

    print(
        f"\nOpen the images in '{OUT_DIR}/' and find the one that looks "
        f"like a real picture (not static). Whichever label produced it "
        f"(e.g. 'MJPG'), set that as CAMERA_FOURCC before running app.py:\n"
        f"    set CAMERA_FOURCC=MJPG\n"
        f"    set CAMERA_INDEX={index}\n"
        f"    python app.py\n"
        f"If NONE of them look clean, the issue is likely elsewhere "
        f"(cable/driver/USB port) rather than pixel format -- try a "
        f"different USB port, or test the camera in the Windows Camera "
        f"app to rule out a hardware/driver problem."
    )


if __name__ == "__main__":
    main()
