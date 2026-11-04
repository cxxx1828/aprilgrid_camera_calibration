"""
to start:
    python3 ir_mono.py calibrate left  -> compute calibration for the left IR sensor
    python3 ir_mono.py calibrate right -> compute calibration for the right IR sensor

Results are saved to ./results/left_ir_intrinsics.yaml and ./results/right_ir_intrinsics.yaml

Note: capture.py disables the IR emitter (laser projector) during capture,
since its dot pattern interferes with tag detection.
"""

import sys
import os
import glob
import cv2
import numpy as np

from common import save_mono_yaml

# ---- AprilGrid parameters (must match the generated PDF board / capture.py) ----
GRID_COLS = 7
GRID_ROWS = 10
TAG_SIZE_MM = 21.0
SPACING_MM = 6.0
TAG_DICT_ID = cv2.aruco.DICT_APRILTAG_36h11
MIN_TAGS = 20  # min. broj detektovanih tagova da se frejm iskoristi

aruco_dict = cv2.aruco.getPredefinedDictionary(TAG_DICT_ID)
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

PITCH_MM = TAG_SIZE_MM + SPACING_MM


def build_apriltag_object_points():
    """id -> 4x3 float32 array of corner coords (TL,TR,BR,BL), matches aruco corner order."""
    obj = {}
    tag_id = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x0 = c * PITCH_MM
            y0 = r * PITCH_MM
            obj[tag_id] = np.array([
                [x0, y0, 0.0],
                [x0 + TAG_SIZE_MM, y0, 0.0],
                [x0 + TAG_SIZE_MM, y0 + TAG_SIZE_MM, 0.0],
                [x0, y0 + TAG_SIZE_MM, 0.0],
            ], dtype=np.float32)
            tag_id += 1
    return obj


TAG_OBJECT_POINTS = build_apriltag_object_points()

LEFT_IMG_DIR = "calib_images/left_ir"
RIGHT_IMG_DIR = "calib_images/right_ir"

RESULTS = {
    "left": ("calib_images/left_ir", "results/left_ir_intrinsics.yaml"),
    "right": ("calib_images/right_ir", "results/right_ir_intrinsics.yaml"),
}

def calibrate(side):
    img_dir, out_file = RESULTS[side]
    images = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if len(images) < 10:
        print(f"Warning: only {len(images)} images, recommended minimum is 15-20 with varied angles/distances.")

    objpoints = []
    imgpoints = []
    img_shape = None

    used = 0
    for fname in images:
        gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        img_shape = gray.shape[::-1]

        corners, ids, _ = aruco_detector.detectMarkers(gray)
        if ids is None or len(ids) < MIN_TAGS:
            n = 0 if ids is None else len(ids)
            print(f"Skipping {fname} ({n} tags found, need >= {MIN_TAGS})")
            continue

        frame_objp = []
        frame_imgp = []
        for tag_corners, tag_id in zip(corners, ids.flatten()):
            tag_id = int(tag_id)
            if tag_id not in TAG_OBJECT_POINTS:
                continue
            frame_objp.append(TAG_OBJECT_POINTS[tag_id])
            frame_imgp.append(tag_corners.reshape(4, 2))

        objpoints.append(np.concatenate(frame_objp, axis=0).astype(np.float32))
        imgpoints.append(np.concatenate(frame_imgp, axis=0).astype(np.float32))
        used += 1

    if used < 5:
        print("Not enough valid images for calibration.")
        return

    print(f"Computing {side} IR calibration from {used} images...")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    print(f"\n=== RESULT ({side} IR) ===")
    print(f"RMS reprojection error: {ret:.4f} px")
    print("Camera matrix K:\n", K)
    print("Distortion coeffs:\n", dist)


    errors = []
    for objp_i, imgp_i, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp_i, rvec, tvec, K, dist)
        err = cv2.norm(imgp_i, proj.reshape(-1, 2), cv2.NORM_L2) / len(proj)
        errors.append(err)

    for i, err in enumerate(errors):
        print(f"  frame {i:03d}: {err:.4f} px")

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    save_mono_yaml(out_file, img_shape, K, dist, ret)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 2 and args[0] == "calibrate" and args[1] in ("left", "right"):
        calibrate(args[1])
    else:
        print(__doc__)
        sys.exit(1)
