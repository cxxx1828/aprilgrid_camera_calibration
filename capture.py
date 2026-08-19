"""
Synchronized calibration image capture for RealSense D457.
Captures RGB, left IR, and right IR from the same frameset.

Usage:
    python3 capture.py

SPACE = save synchronized triplet, Q = quit
Images saved to calib_images/{rgb,left_ir,right_ir}/
"""

import os
import glob
import cv2
import numpy as np
import pyrealsense2 as rs

GRID_COLS = 7
GRID_ROWS = 10
TAG_DICT_ID = cv2.aruco.DICT_APRILTAG_36h11
MIN_TAGS = 20  # min. broj detektovanih tagova da se frejm prihvati

aruco_dict = cv2.aruco.getPredefinedDictionary(TAG_DICT_ID)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def find_apriltags(gray):
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    found = ids is not None and len(ids) >= MIN_TAGS
    return found, corners, ids

RGB_DIR = "calib_images/rgb"
LEFT_DIR = "calib_images/left_ir"
RIGHT_DIR = "calib_images/right_ir"

WIDTH, HEIGHT, FPS = 1280, 720, 30
#WIDTH, HEIGHT, FPS = 640, 480, 30

def capture():
    for d in (RGB_DIR, LEFT_DIR, RIGHT_DIR):
        os.makedirs(d, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.infrared, 1, WIDTH, HEIGHT, rs.format.y8, FPS)
    config.enable_stream(rs.stream.infrared, 2, WIDTH, HEIGHT, rs.format.y8, FPS)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 0)

    count = len(glob.glob(os.path.join(RGB_DIR, "*.png")))
    print("SPACE = save synchronized triplet, Q = quit")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            left_frame = frames.get_infrared_frame(1)
            right_frame = frames.get_infrared_frame(2)
            if not (color_frame and left_frame and right_frame):
                continue

            rgb = np.asanyarray(color_frame.get_data())
            left = np.asanyarray(left_frame.get_data())
            right = np.asanyarray(right_frame.get_data())

            found_rgb, corners_rgb, ids_rgb = find_apriltags(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY))
            found_left, corners_left, ids_left = find_apriltags(left)
            found_right, corners_right, ids_right = find_apriltags(right)

            disp_rgb = rgb.copy()
            disp_left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
            disp_right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
            for disp, ids, corners in (
                (disp_rgb, ids_rgb, corners_rgb),
                (disp_left, ids_left, corners_left),
                (disp_right, ids_right, corners_right),
            ):
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(disp, corners, ids)

            small = [cv2.resize(d, (640, 360)) for d in (disp_rgb, disp_left, disp_right)]
            top = cv2.hconcat(small[:2])
            bottom = cv2.hconcat([small[2], np.zeros_like(small[2])])
            combined = cv2.vconcat([top, bottom])

            def n_tags(ids):
                return 0 if ids is None else len(ids)

            labels = [
                ("RGB", found_rgb, n_tags(ids_rgb), (20, 30)),
                ("LEFT", found_left, n_tags(ids_left), (660, 30)),
                ("RIGHT", found_right, n_tags(ids_right), (20, 390)),
            ]
            for text, ok, n, pos in labels:
                color = (0, 255, 0) if ok else (0, 0, 255)
                cv2.putText(combined, f"{text}: {n} tags", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("D457 calibration capture", combined)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if found_rgb and found_left and found_right:
                    basename = f"img_{count:03d}.png"
                    cv2.imwrite(os.path.join(RGB_DIR, basename), rgb)
                    cv2.imwrite(os.path.join(LEFT_DIR, basename), left)
                    cv2.imwrite(os.path.join(RIGHT_DIR, basename), right)
                    print(f"Saved synchronized frame {count:03d}")
                    count += 1
                else:
                    print("Frame rejected: aprilgrid must be detected in RGB + LEFT + RIGHT.")
            elif key == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    capture()
