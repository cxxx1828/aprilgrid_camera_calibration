"""
to start:
    python3 rgb_stream.py

Press Q to quit.
"""

import cv2
import numpy as np
import pyrealsense2 as rs

from common import load_mono_yaml

RGB_INTRINSICS = "results/rgb_intrinsics.yaml"

WIDTH, HEIGHT, FPS = 1280, 720, 30


def main():
    K, dist, _ = load_mono_yaml(RGB_INTRINSICS)

    # optimal new camera matrix - alpha=0 crops to remove all black borders,
    # alpha=1 keeps full FOV but with black borders visible
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (WIDTH, HEIGHT), alpha=0)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (WIDTH, HEIGHT), cv2.CV_16SC2)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipeline.start(config)

    print("Q = quit")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

            side_by_side = cv2.hconcat([img, undistorted])
            cv2.putText(side_by_side, "RAW", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(side_by_side, "RECTIFIED", (WIDTH + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("D457 RGB stream", side_by_side)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
