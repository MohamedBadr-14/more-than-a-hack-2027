"""Feature extraction for ball candidates: HOG | colour | shape."""
import cv2
import numpy as np
from skimage.feature import hog

CROP_SIZE = 32
FEATURE_SETS = {
    "hog": ["hog"],
    "hog+color": ["hog", "color"],
    "hog+color+shape": ["hog", "color", "shape"],
}


def hog_features(crop):
    gray = cv2.cvtColor(cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")


def color_features(crop):
    """HSV histograms + white/green fractions in the centre vs the surrounding ring."""
    hsv = cv2.cvtColor(cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    h = np.histogram(hsv[..., 0], bins=16, range=(0, 180))[0]
    s = np.histogram(hsv[..., 1], bins=8, range=(0, 256))[0]
    v = np.histogram(hsv[..., 2], bins=8, range=(0, 256))[0]
    hists = np.concatenate([h, s, v]) / hsv[..., 0].size

    white = (hsv[..., 1] <= 90) & (hsv[..., 2] >= 130)
    green = (hsv[..., 0] >= 35) & (hsv[..., 0] <= 85) & (hsv[..., 1] >= 60)
    c0, c1 = CROP_SIZE // 4, 3 * CROP_SIZE // 4  # centre = middle half (the crop is padded 50% each side)
    centre = np.zeros(white.shape, bool)
    centre[c0:c1, c0:c1] = True
    fracs = [white[centre].mean(), green[centre].mean(), white[~centre].mean(), green[~centre].mean()]
    return np.concatenate([hists, fracs])


def shape_features(row, frame_h=1080):
    """Blob geometry from the candidate stage (row from labels.csv or a candidate dict)."""
    x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
    w, h = max(x2 - x1, 1), max(y2 - y1, 1)
    area = float(row["area"])
    return np.array([
        np.log1p(area),
        float(row["circularity"]),
        min(w, h) / max(w, h),  # aspect, 1 = square
        area / (w * h),         # extent, ~0.785 for a filled circle
        (y1 + y2) / 2 / frame_h,
    ])


def extract(crop, row, feature_set="hog+color+shape"):
    parts = {"hog": lambda: hog_features(crop), "color": lambda: color_features(crop), "shape": lambda: shape_features(row)}
    return np.concatenate([parts[p]() for p in FEATURE_SETS[feature_set]])
