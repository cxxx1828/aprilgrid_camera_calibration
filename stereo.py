import sys
import os
import glob
import cv2
import numpy as np

from common import load_mono_yaml, save_stereo_yaml

GRID_COLS = 7
GRID_ROWS = 10
TAG_SIZE_MM = 21.0
SPACING_MM = 6.0
TAG_DICT_ID = cv2.aruco.DICT_APRILTAG_36h11
MIN_TAGS = 20  # min. broj detektovanih tagova (u OBA frejma) da se par iskoristi

aruco_dict = cv2.aruco.getPredefinedDictionary(TAG_DICT_ID)
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

PITCH_MM = TAG_SIZE_MM + SPACING_MM


def build_apriltag_object_points():
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

RGB_DIR = "calib_images/rgb"
LEFT_DIR = "calib_images/left_ir"
RIGHT_DIR = "calib_images/right_ir"

RGB_INTRINSICS = "results/rgb_intrinsics.yaml"
LEFT_INTRINSICS = "results/left_ir_intrinsics.yaml"
RIGHT_INTRINSICS = "results/right_ir_intrinsics.yaml"

PAIRS = {
    "left-right": {
        "dir1": LEFT_DIR,
        "dir2": RIGHT_DIR,
        "intrinsics1": LEFT_INTRINSICS,
        "intrinsics2": RIGHT_INTRINSICS,
        "gray1": lambda f: cv2.imread(f, cv2.IMREAD_GRAYSCALE),
        "gray2": lambda f: cv2.imread(f, cv2.IMREAD_GRAYSCALE),
        "out_file": "results/stereo_left_right.yaml",
        "label": "left IR <-> right IR",
    },
    "rgb-left": {
        "dir1": RGB_DIR,
        "dir2": LEFT_DIR,
        "intrinsics1": RGB_INTRINSICS,
        "intrinsics2": LEFT_INTRINSICS,
        "gray1": lambda f: cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY),
        "gray2": lambda f: cv2.imread(f, cv2.IMREAD_GRAYSCALE),
        "out_file": "results/stereo_rgb_left.yaml",
        "label": "RGB <-> left IR",
    },
}


def calibrate(pair_name):
    cfg = PAIRS[pair_name]

    if not os.path.exists(cfg["intrinsics1"]) or not os.path.exists(cfg["intrinsics2"]):
        print(f"Missing mono intrinsics for {cfg['label']}. Run the relevant *_mono.py calibrate first.")
        return

    K1, dist1, _ = load_mono_yaml(cfg["intrinsics1"])
    K2, dist2, _ = load_mono_yaml(cfg["intrinsics2"])

    images1 = sorted(glob.glob(os.path.join(cfg["dir1"], "*.png")))

    objpoints = []
    imgpoints1 = []
    imgpoints2 = []
    img_shape = None

    used = 0
    for f1 in images1:
        basename = os.path.basename(f1)
        f2 = os.path.join(cfg["dir2"], basename)
        if not os.path.exists(f2):
            print(f"Skipping {basename} (no matching frame in {cfg['dir2']})")
            continue

        gray1 = cfg["gray1"](f1)
        gray2 = cfg["gray2"](f2)
        #img_shape = gray1.shape[::-1]

        if gray1.shape != gray2.shape:
            print(f"Skipping {basename} (resolution mismatch: {gray1.shape[::-1]} vs {gray2.shape[::-1]})")
            continue

        img_shape = gray1.shape[::-1]

        corners1, ids1, _ = aruco_detector.detectMarkers(gray1)
        corners2, ids2, _ = aruco_detector.detectMarkers(gray2)

        if ids1 is None or ids2 is None or len(ids1) < MIN_TAGS or len(ids2) < MIN_TAGS:
            n1 = 0 if ids1 is None else len(ids1)
            n2 = 0 if ids2 is None else len(ids2)
            print(f"Skipping {basename} (tags found: {n1} / {n2}, need >= {MIN_TAGS} in both)")
            continue

        # koristi samo tagove vidljive u OBA frejma, u istom redosledu
        ids1_flat = ids1.flatten()
        ids2_flat = ids2.flatten()
        common_ids = sorted(set(ids1_flat.tolist()) & set(ids2_flat.tolist()))

        if len(common_ids) < MIN_TAGS:
            print(f"Skipping {basename} (only {len(common_ids)} common tags, need >= {MIN_TAGS})")
            continue

        map1 = {int(i): c for i, c in zip(ids1_flat, corners1)}
        map2 = {int(i): c for i, c in zip(ids2_flat, corners2)}

        frame_objp = []
        frame_imgp1 = []
        frame_imgp2 = []
        for tag_id in common_ids:
            if tag_id not in TAG_OBJECT_POINTS:
                continue
            frame_objp.append(TAG_OBJECT_POINTS[tag_id])
            frame_imgp1.append(map1[tag_id].reshape(4, 2))
            frame_imgp2.append(map2[tag_id].reshape(4, 2))

        objpoints.append(np.concatenate(frame_objp, axis=0).astype(np.float32))
        imgpoints1.append(np.concatenate(frame_imgp1, axis=0).astype(np.float32))
        imgpoints2.append(np.concatenate(frame_imgp2, axis=0).astype(np.float32))
        used += 1

    if used < 5:
        print("Not enough valid synchronized pairs for stereo calibration.")
        return

    print(f"Computing stereo calibration ({cfg['label']}) from {used} synchronized pairs...")

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    ret, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints1,
        imgpoints2,
        K1,
        dist1,
        K2,
        dist2,
        img_shape,
        criteria=criteria,
        flags=flags,
    )

    print(f"\n=== RESULT ({cfg['label']}) ===")
    print(f"RMS reprojection error: {ret:.4f} px")
    print("Rotation R:\n", R)
    print("Translation T (mm):\n", T)

    baseline_mm = cv2.norm(T)
    print(f"Baseline: {baseline_mm:.2f} mm")

    out_file = cfg["out_file"]
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    save_stereo_yaml(out_file, K1, dist1, K2, dist2, R, T, E, F, ret)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 2 or args[0] != "calibrate" or args[1] not in PAIRS:
        print(__doc__)
        sys.exit(1)

    calibrate(args[1])


