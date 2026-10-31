"""Zero-shot YOLO ball detections next to the provided tracking ball, drawn onto the video.

A look-only tool: nothing here trains, tunes or labels anything. It runs a COCO-pretrained YOLO
on the full-resolution tactical video, projects the ATD ball into the same frames through the
homography, and writes both onto one video so the two can be compared by eye.

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install ultralytics
    python ball_yolo/overlay.py                       # the public 5-minute window
    python ball_yolo/overlay.py --limit 100           # smoke test on the first 100 frames

Output goes to data/ball_yolo/, which is gitignored -- the video and the per-frame CSV are
derived from the match assets and must not be published (RULES.md).

    blue circle   ATD ball, projected to pixels
    red boxes     every YOLO "sports ball" detection, with its confidence
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, require  # noqa: E402
from load_atd import frame_id_of  # noqa: E402
import load_homography  # noqa: E402

# Params
# The public window, inclusive, as video frames (DATA.md).
FIRST_FRAME, LAST_FRAME = 11185, 18684
OUT_DIR = os.path.join(ROOT, "data", "ball_yolo")
# Kept under data/ because *.pt is not gitignored at the repo root. Ultralytics downloads the
# weights by name the first time; move them here afterwards.
MODEL = os.path.join(OUT_DIR, "yolo11s.pt")
IMGSZ = 1920
CONF = 0.05
SPORTS_BALL = 32  # COCO class id

TRACK_COLOUR, YOLO_COLOUR, TEXT_COLOUR = (255, 128, 0), (0, 0, 255), (255, 255, 255)
TRACK_RADIUS = 12
MISSING = "N"  # raw ATD writes "NaN" for an unseen object


def atd_ball(first, last, game=GAME):
    """{video_frame: (x, y)} in normalised pitch coordinates, NaN where the ball is not tracked.

    Read straight off the raw file rather than through kloppy: only the last ':'-group of each
    line is needed, and only for a few thousand lines.
    """
    raw_path = require(["atd_raw"], game)[0]
    first_id, last_id = frame_id_of(first), frame_id_of(last)
    balls = {}
    with open(raw_path) as handle:
        for line in handle:
            frame_id = int(line[:line.index(":")])
            if frame_id < first_id:
                continue
            if frame_id > last_id:
                break
            ball = line.rstrip().rsplit(":", 1)[1]
            if ball[0] == MISSING:
                balls[frame_id - 1] = (np.nan, np.nan)
            else:
                x, y = ball.split(",")[:2]
                balls[frame_id - 1] = (float(x), float(y))
    return balls


def open_video_at(first, game=GAME):
    """The tactical video, positioned so the next read() returns video frame `first`.

    Frames are skipped with grab() rather than a seek: seeking an mp4 can land a few frames off,
    and a ball drawn on the wrong frame looks exactly like a detector error.
    """
    capture = cv2.VideoCapture(require(["video"], game)[0])
    if not capture.isOpened():
        raise IOError("could not open the tactical video")
    for _ in range(first):
        if not capture.grab():
            raise IOError("video ended before frame {}".format(first))
    return capture


def draw(frame, video_frame, track_px, boxes):
    """Overlay the tracking ball, the YOLO boxes and a one-line status in place."""
    if track_px is not None:
        cv2.circle(frame, (int(round(track_px[0])), int(round(track_px[1]))),
                   TRACK_RADIUS, TRACK_COLOUR, 2)
    for x1, y1, x2, y2, conf in boxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), YOLO_COLOUR, 2)
        cv2.putText(frame, "{:.2f}".format(conf), (int(x1), int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, YOLO_COLOUR, 1, cv2.LINE_AA)

    status = "frame {}   TRACKING: {}   YOLO: {}".format(
        video_frame, "yes" if track_px is not None else "no",
        "yes ({}, best {:.2f})".format(len(boxes), max(b[4] for b in boxes)) if boxes else "no")
    cv2.rectangle(frame, (0, 0), (760, 34), (0, 0, 0), -1)
    cv2.putText(frame, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOUR, 2,
                cv2.LINE_AA)


def run(first, last, model_name, imgsz, conf, out_dir, out_scale):
    from ultralytics import YOLO

    os.makedirs(out_dir, exist_ok=True)
    stem = "yolo_vs_atd_{}_{}".format(first, last)
    video_path = os.path.join(out_dir, stem + ".mp4")
    csv_path = os.path.join(out_dir, stem + ".csv")

    matrices, _, valid = load_homography.load()
    balls = atd_ball(first, last)
    model = YOLO(model_name)

    print("skipping to frame {} ...".format(first))
    capture = open_video_at(first)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_size = (int(width * out_scale), int(height * out_scale))
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             capture.get(cv2.CAP_PROP_FPS), out_size)

    started = time.time()
    with open(csv_path, "w", newline="") as handle:
        rows = csv.writer(handle)
        rows.writerow(["video_frame", "track_x", "track_y", "yolo_n",
                       "yolo_x1", "yolo_y1", "yolo_x2", "yolo_y2", "yolo_conf", "dist_px",
                       "nearest_dist_px"])

        for video_frame in range(first, last + 1):
            ok, frame = capture.read()
            if not ok:
                print("video ended early at frame {}".format(video_frame))
                break

            track_px = None
            ball_x, ball_y = balls.get(video_frame, (np.nan, np.nan))
            if valid[video_frame] and not np.isnan(ball_x):
                px, py = load_homography.project(matrices[video_frame], ball_x, ball_y)
                track_px = (float(px), float(py))

            result = model.predict(frame, imgsz=imgsz, conf=conf, classes=[SPORTS_BALL],
                                   verbose=False)[0]
            boxes = [tuple(xyxy) + (float(c),) for xyxy, c in
                     zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())]

            best = max(boxes, key=lambda b: b[4]) if boxes else None
            # Two distances, because the most confident box is not always the ball: a static
            # object by the touchline can outscore it. nearest says whether YOLO found the
            # tracked ball at all, whatever else it also fired on.
            distance = nearest = ""
            if boxes and track_px is not None:
                gaps = [np.hypot((b[0] + b[2]) / 2 - track_px[0], (b[1] + b[3]) / 2 - track_px[1])
                        for b in boxes]
                distance = "{:.1f}".format(gaps[boxes.index(best)])
                nearest = "{:.1f}".format(min(gaps))
            rows.writerow([video_frame]
                          + (["{:.1f}".format(v) for v in track_px] if track_px else ["", ""])
                          + [len(boxes)]
                          + (["{:.1f}".format(v) for v in best[:4]] + ["{:.3f}".format(best[4])]
                             if best else [""] * 5)
                          + [distance, nearest])

            draw(frame, video_frame, track_px, boxes)
            writer.write(frame if out_scale == 1.0 else cv2.resize(frame, out_size))

            done = video_frame - first + 1
            if done % 250 == 0:
                elapsed = time.time() - started
                print("  {} / {} frames  {:.2f} fps  ~{:.0f} min left".format(
                    done, last - first + 1, done / elapsed,
                    (last - video_frame) / (done / elapsed) / 60))

    capture.release()
    writer.release()
    print("\nwrote {}\n      {}".format(video_path, csv_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--first", type=int, default=FIRST_FRAME)
    parser.add_argument("--last", type=int, default=LAST_FRAME)
    parser.add_argument("--limit", type=int, help="process only this many frames")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--conf", type=float, default=CONF)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--out-scale", type=float, default=1.0,
                        help="0.5 writes a 960x540 video; detection always runs on full frames")
    args = parser.parse_args()

    last = args.last if args.limit is None else min(args.last, args.first + args.limit - 1)
    run(args.first, last, args.model, args.imgsz, args.conf, args.out_dir, args.out_scale)
