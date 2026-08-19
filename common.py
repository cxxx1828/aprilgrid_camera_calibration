"""
Shared helper functions used across all calibration modules
(rgb_mono, ir_mono, stereo).
"""

import cv2
import numpy as np


def find_chessboard(gray_img, cols_inner, rows_inner):
    """Find chessboard corners in a grayscale image. Returns (found, corners)."""
    return cv2.findChessboardCorners(gray_img, (cols_inner, rows_inner), None)


def refine_corners(gray_img, corners):
    """Refine detected corners to sub-pixel accuracy."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)


def build_object_points(cols_inner, rows_inner, square_size_mm):
    """Build the 3D object points grid for one chessboard view (Z=0 plane)."""
    objp = np.zeros((rows_inner * cols_inner, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols_inner, 0:rows_inner].T.reshape(-1, 2)
    objp *= square_size_mm
    return objp


def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist):
    """Return per-image reprojection error (list of floats, in pixels)."""
    errors = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        pts1 = imgpoints[i].reshape(-1, 2).astype(np.float64)
        pts2 = imgpoints2.reshape(-1, 2).astype(np.float64)
        err = cv2.norm(pts1, pts2, cv2.NORM_L2) / len(pts2)
        errors.append(err)
    return errors


def save_mono_yaml(path, image_shape, K, dist, rms_error):
    """Save mono calibration result (camera matrix + distortion) to a YAML file."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", image_shape[0])
    fs.write("image_height", image_shape[1])
    fs.write("camera_matrix", K)
    fs.write("distortion_coefficients", dist)
    fs.write("rms_error", rms_error)
    fs.release()


def load_mono_yaml(path):
    """Load mono calibration result from a YAML file. Returns (K, dist, image_shape)."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat()
    width = int(fs.getNode("image_width").real())
    height = int(fs.getNode("image_height").real())
    fs.release()
    return K, dist, (width, height)


def save_stereo_yaml(path, K1, dist1, K2, dist2, R, T, E, F, rms_error):
    """Save stereo calibration result (two intrinsics + extrinsics) to a YAML file."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix_1", K1)
    fs.write("distortion_coefficients_1", dist1)
    fs.write("camera_matrix_2", K2)
    fs.write("distortion_coefficients_2", dist2)
    fs.write("R", R)
    fs.write("T", T)
    fs.write("E", E)
    fs.write("F", F)
    fs.write("rms_error", rms_error)
    fs.release()
