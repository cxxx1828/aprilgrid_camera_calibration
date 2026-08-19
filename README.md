# RealSense D455/D457 AprilGrid Calibration

Calibration of the RGB, left IR, and right IR sensors on an Intel RealSense
D455/D457 camera, using an AprilGrid board (instead of a classic chessboard)
for more precise detection.

## Calibration board

- **7 × 10** AprilTag markers
- Dictionary: **tag36h11**
- Tag size: **21 mm**
- Spacing between tags: **6 mm**
- Custom-generated A4 PDF (183 × 264 mm, centered, no print scaling)

> Print the board on A4 paper at **100% scale** (disable "Fit to page" in
> your printer settings), otherwise the physical dimensions won't match
> `TAG_SIZE_MM`/`SPACING_MM` in the code.

## Project structure

```
aprilgrid_calibration/
├── capture.py          # synchronized RGB + IR1 + IR2 image capture
├── common.py             # shared helper functions (YAML I/O)
├── rgb_mono.py            # mono calibration of the RGB sensor
├── ir_mono.py             # mono calibration of the left/right IR sensor
├── stereo.py              # stereo calibration (left-right, rgb-left)
├── calib_images/
│   ├── rgb/
│   ├── left_ir/
│   └── right_ir/
└── results/
    ├── rgb_intrinsics.yaml
    ├── left_ir_intrinsics.yaml
    ├── right_ir_intrinsics.yaml
    ├── stereo_left_right.yaml
    └── stereo_rgb_left.yaml
```

## Dependencies

```bash
pip3 install pyrealsense2 opencv-contrib-python numpy
```

`opencv-contrib-python` is required (not plain `opencv-python`), since it
includes the `cv2.aruco` module.

## Step-by-step order

### 1. Capturing images

```bash
python3 capture.py
```

- `SPACE` — saves a synchronized triplet (RGB + IR1 + IR2), only if all
  three frames detected **at least `MIN_TAGS`** (default 20) tags
- `Q` — quit
- The IR emitter (laser projector) is automatically disabled during capture,
  since its dot pattern interferes with tag detection
- Recommendation: capture 150–300 images with varied angles, distances, and
  board positions in the frame (including the edges/corners of the image,
  not just the center)

**Resolution/FPS** (`WIDTH, HEIGHT, FPS` at the top of the file) must match
what your USB connection supports:

- USB3 connection: `1280, 720, 30` works for all three streams simultaneously
- USB2 connection: the camera restricts combinations (Color maxes out around
  ~15Hz at 720p, IR around ~5Hz); check `rs-enumerate-devices -c` and use
  `640, 480, 30` if needed

> **Critical**: don't change resolution between captures into the same
> `calib_images/` folder. Mixing resolutions in the same image set silently
> introduces a systematic error into the calibration (RMS jumps 2–4×) with no
> runtime error at all. If you change resolution, delete the old images and
> recapture the whole set from scratch:
> ```bash
> rm -rf calib_images/rgb calib_images/left_ir calib_images/right_ir
> ```

### 2. Mono calibration (must run before the stereo step)

```bash
python3 rgb_mono.py calibrate
python3 ir_mono.py calibrate left
python3 ir_mono.py calibrate right
```

Each command:
- detects the AprilGrid in every image (`MIN_TAGS` threshold)
- computes `cv2.calibrateCamera` (K, distortion coefficients)
- prints the RMS reprojection error for the whole set and per frame
- saves the result to `results/*_intrinsics.yaml`

**`stereo.py` reads these `.yaml` files from disk** — it does not recompute
intrinsics on the fly. If you recapture images (a new `capture.py` run), you
must rerun all three mono commands before the stereo step, otherwise stereo
calibration will use stale intrinsics from the old image set.

### 3. Stereo calibration

```bash
python3 stereo.py calibrate left-right
python3 stereo.py calibrate rgb-left
```

- Uses `cv2.CALIB_FIX_INTRINSIC` — mono intrinsics are held fixed, only the
  relative rotation/translation (R, T) between the two sensors is solved for
- For each frame, uses only the tags visible in **both** sensors
  (`common_ids`), so that `objpoints`/`imgpoints1`/`imgpoints2` have the same
  number of points in the same order — required by `cv2.stereoCalibrate`
- The result (K1, dist1, K2, dist2, R, T, E, F) is saved to
  `results/stereo_*.yaml`

## Expected results (rule of thumb)

| RMS reprojection error | Quality |
|---|---|
| < 0.5 px | excellent, suitable for precise depth/3D |
| 0.5 – 1.5 px | solid, standard for robotics (navigation, AprilTag docking) |
| > 2–3 px | problematic — check for mixed resolutions, poor focus, or the board being too small in frame |

The RGB sensor usually has a higher RMS than the IR sensors (larger factory
distortion), and `stereo_rgb_left` usually comes out somewhat worse than
`stereo_left_right` (two different sensor types instead of an identical
pair) — that's expected, not a bug.

## Parameters that must be changed together

`GRID_COLS`, `GRID_ROWS`, `TAG_SIZE_MM`, `SPACING_MM`, `TAG_DICT_ID`, and
`MIN_TAGS` are defined identically in `capture.py`, `rgb_mono.py`,
`ir_mono.py`, and `stereo.py`. If you change the board (e.g. print a new one
at a different size), these values must be kept in sync across all four
files.

## Troubleshooting

- **`Couldn't resolve requests`** in `capture.py` — the requested
  resolution/FPS combination isn't supported on the current USB connection;
  check `rs-enumerate-devices -c` and adjust `WIDTH, HEIGHT, FPS`
- **`ModuleNotFoundError: pyrealsense2`** — not installed in the active
  virtualenv (`pip3 install pyrealsense2`)
- **Sudden jump in RMS after adding new images** — almost always mixed
  resolutions in the same folder; delete and recapture at a single resolution
- **`stereo.py` reports an unrealistically high RMS even though the mono
  results look good** — check whether you ran the mono calibrations on the
  CURRENT image set before the stereo step (see step 2)
