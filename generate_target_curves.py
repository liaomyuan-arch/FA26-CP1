"""
Regenerate target_curves.npy from the egg illustrations in ../eggs.png.

Requires opencv-python-headless and scipy (not part of requirements_CPU.txt -
this is an authoring-time script, students never need to run it).

Produces a (6, N_POINTS, 2) float32 array where each curve is:
- centered at its own centroid
- resampled to N_POINTS uniformly spaced points by arc length
- smoothed to remove raster staircase noise
- scaled so its RMS radius equals TARGET_SCALE, matching the calibration of
  the original dataset. Curve distance is scale-invariant (each curve is
  re-normalized by its own RMS radius before comparison), so this scale
  choice is now just a convention carried over from the original dataset,
  not something the distance (<=1.5) / complexity (<=15 joints) constraints
  depend on.
"""
import os

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "eggs.png")
OUT = os.path.join(HERE, "target_curves.npy")

N_POINTS = 200
TARGET_SCALE = 0.191250  # RMS radius used by the original target_curves.npy
DARK_STROKE_THRESHOLD = 96
MIN_CONTOUR_AREA = 500
SMOOTH_OVERSAMPLE_POINTS = 2000
SMOOTH_SIGMA = 14
POLYLINE_SIMPLIFY_FRACTION = 0.0015


def equisample(points, n):
    """Resample a closed polyline to n points uniformly spaced by arc length."""
    pts = np.asarray(points, dtype=np.float64)
    pts = np.vstack([pts, pts[0:1]])  # close the loop
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    targets = np.linspace(0, total, n, endpoint=False)
    idx = np.searchsorted(cum, targets, side="right") - 1
    idx = np.clip(idx, 0, len(seg) - 1)
    t = (targets - cum[idx]) / seg[idx]
    return pts[idx] + t[:, None] * (pts[idx + 1] - pts[idx])


def _clean_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _find_external_contours(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_TC89_KCOS,
    )

    streamlined = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= MIN_CONTOUR_AREA:
            continue

        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = POLYLINE_SIMPLIFY_FRACTION * perimeter
        streamlined.append(cv2.approxPolyDP(contour, epsilon, closed=True))

    return streamlined


def extract_contours(src_path):
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(src_path)

    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        stroke = ((alpha > 10) & (gray < DARK_STROKE_THRESHOLD)).astype(np.uint8) * 255
        fallback = ((alpha > 10) & (gray < 250)).astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        stroke = (gray < DARK_STROKE_THRESHOLD).astype(np.uint8) * 255
        fallback = (gray < 250).astype(np.uint8) * 255

    contours = _find_external_contours(_clean_mask(stroke))

    # If the artwork changes to use lighter outlines, keep the old broad
    # foreground extraction as a fallback instead of silently writing bad data.
    if len(contours) != 6:
        contours = _find_external_contours(_clean_mask(fallback))
    return contours


def order_row_major(contours, n_cols=3):
    centers = []
    for c in contours:
        M = cv2.moments(c)
        centers.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
    centers = np.array(centers)
    row = (centers[:, 1] > centers[:, 1].mean()).astype(int)
    return np.lexsort((centers[:, 0], row))


def process_contour(raw_xy):
    c = raw_xy.astype(np.float64)
    c[:, 1] *= -1  # flip so shapes are right-side-up under standard math axes

    oversampled = equisample(c, SMOOTH_OVERSAMPLE_POINTS)
    smoothed = np.stack([
        gaussian_filter1d(oversampled[:, 0], sigma=SMOOTH_SIGMA, mode="wrap"),
        gaussian_filter1d(oversampled[:, 1], sigma=SMOOTH_SIGMA, mode="wrap"),
    ], axis=1)
    resampled = equisample(smoothed, N_POINTS)

    centered = resampled - resampled.mean(0, keepdims=True)
    rms = np.sqrt(np.square(centered).sum() / N_POINTS)
    return centered * (TARGET_SCALE / rms)


def main():
    contours = extract_contours(SRC)
    if len(contours) != 6:
        areas = sorted((cv2.contourArea(c) for c in contours), reverse=True)
        raise RuntimeError(f"Expected 6 egg blobs, found {len(contours)}. Areas: {areas}")

    order = order_row_major(contours)
    curves = np.array([
        process_contour(contours[i].reshape(-1, 2)) for i in order
    ], dtype=np.float32)

    np.save(OUT, curves)
    print(f"Wrote {curves.shape} to {OUT}")


if __name__ == "__main__":
    main()
