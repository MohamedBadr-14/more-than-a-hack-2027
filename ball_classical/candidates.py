"""Ball candidate generator: same steps and values as 01_candidate_pipeline.ipynb."""
import cv2
import numpy as np

CFG = dict(
    # gameplay filter
    green_lower=(35, 60, 50), green_upper=(85, 255, 255),
    min_green_ratio=0.35,

    # pitch mask
    pitch_close_kernel=15,
    pitch_erode=15,
    hud_boxes=[(60, 40, 300, 160), (1700, 40, 1860, 140)],  # x1,y1,x2,y2 at 1920x1080

    # white mask
    white_lower=(0, 0, 130), white_upper=(179, 90, 255),

    # morphology
    open_kernel=5, close_kernel=3,

    # area range, interpolated between pitch top and frame bottom
    area_far=(20, 300),
    area_near=(40, 900),

    min_circularity=0.4,

    crop_pad=0.5,  # fraction of bbox size added on each side of the square crop
)


def green_ratio(frame, cfg=CFG):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return (cv2.inRange(hsv, cfg["green_lower"], cfg["green_upper"]) > 0).mean()


def is_gameplay(frame, cfg=CFG):
    return green_ratio(frame, cfg) >= cfg["min_green_ratio"]


def pitch_mask(frame, cfg=CFG):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, cfg["green_lower"], cfg["green_upper"])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["pitch_close_kernel"],) * 2)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(green)
    if contours:
        cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["pitch_erode"],) * 2))
    for x1, y1, x2, y2 in cfg["hud_boxes"]:
        mask[y1:y2, x1:x2] = 0
    return mask


def white_mask(frame, pm, cfg=CFG):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.bitwise_and(cv2.inRange(hsv, cfg["white_lower"], cfg["white_upper"]), pm)


def clean_mask(mask, cfg=CFG):
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["open_kernel"],) * 2)
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["close_kernel"],) * 2)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko), cv2.MORPH_CLOSE, kc)


def circularity(c):
    p = cv2.arcLength(c, True)
    return 4 * np.pi * cv2.contourArea(c) / (p * p) if p > 0 else 0


def find_candidates(frame, cfg=CFG, pm=None):
    """Return list of dicts: bbox (x1,y1,x2,y2), area, circularity.

    pm: optional precomputed pitch mask (e.g. from a homography); default is the green-based one.
    """
    if pm is None:
        pm = pitch_mask(frame, cfg)
    rows = np.where(pm.any(axis=1))[0]
    if not len(rows):
        return []
    pitch_top, h_img = rows[0], frame.shape[0]
    mask = clean_mask(white_mask(frame, pm, cfg), cfg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        t = np.clip((y + h / 2 - pitch_top) / max(h_img - pitch_top, 1), 0, 1)
        lo = cfg["area_far"][0] + t * (cfg["area_near"][0] - cfg["area_far"][0])
        hi = cfg["area_far"][1] + t * (cfg["area_near"][1] - cfg["area_far"][1])
        area, circ = cv2.contourArea(c), circularity(c)
        if lo < area < hi and circ >= cfg["min_circularity"]:
            out.append(dict(bbox=(x, y, x + w, y + h), area=area, circularity=circ))
    return out


def square_crop(frame, bbox, pad=CFG["crop_pad"]):
    """Square crop centred on bbox at native resolution (resize happens at feature extraction)."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (0.5 + pad)
    H, W = frame.shape[:2]
    a, b = int(max(cx - half, 0)), int(min(cx + half, W))
    c, d = int(max(cy - half, 0)), int(min(cy + half, H))
    return frame[c:d, a:b]
