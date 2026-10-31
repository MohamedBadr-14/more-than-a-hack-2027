"""Interactive labelling of ball candidates.

Shows sampled frames with every candidate box; you mark which box(es) are the ball.
Unmarked boxes are saved as negatives when you confirm the frame.

Controls
  left-click box        toggle ball (positive) - several allowed, e.g. ball merged with a boot
  right-click box       toggle ignore (unclear crop, left out of training)
  left-click empty spot ball is visible but NOT among the candidates -> records its position (candidate-stage miss)
  right-click empty     clear that miss point
  Enter / Space         confirm frame (unmarked boxes -> negative) and go to the next one
  s                     skip frame (nothing saved)
  u                     undo last confirmed / skipped frame
  q / Esc               quit (progress is saved after every frame; re-run to resume)

Output (in --out)
  crops/{pos,neg,ign}/f<frame>_<k>.png   square crops at native resolution
  labels.csv   one row per crop:  frame,k,x1,y1,x2,y2,area,circularity,label,path
  frames.csv   one row per frame: frame,status,n_candidates,miss_x,miss_y
               status: ball | ball_missed | no_ball | skipped | not_gameplay
  prelabel.csv one row per reviewed frame when pre-labelling: frame,pred_k,pred_score
               (compare with labels.csv to measure the model on new data)

Pre-labelling: if models/ball_svm.pkl exists, each frame opens with the model's top-scoring
candidate already marked as ball (when its score > --prelabel-threshold). Scores are shown on
every box/thumbnail. Just correct it if wrong. Disable with --no-prelabel.
"""
import argparse
import csv
import os
import pickle

import cv2
import numpy as np

from candidates import CFG, find_candidates, is_gameplay, square_crop
from features import extract

LABEL_FIELDS = ["frame", "k", "x1", "y1", "x2", "y2", "area", "circularity", "label", "path"]
FRAME_FIELDS = ["frame", "status", "n_candidates", "miss_x", "miss_y"]
PRELABEL_FIELDS = ["frame", "pred_k", "pred_score"]
COLORS = {"neg": (0, 0, 255), "pos": (0, 255, 0), "ign": (0, 255, 255)}
HIT_MARGIN = 6  # px (frame coords) added around boxes for easier clicking
THUMB, COLS = 80, 8


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows, append=False):
    new = not os.path.exists(path) or not append
    with open(path, "a" if append else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


class Labeler:
    def __init__(self, args):
        self.args = args
        self.out = args.out
        self.cfg = dict(CFG, hud_boxes=CFG["hud_boxes"] + [tuple(map(int, b.split(","))) for b in args.extra_hud])
        for lab in COLORS:
            os.makedirs(os.path.join(self.out, "crops", lab), exist_ok=True)
        self.labels_csv = os.path.join(self.out, "labels.csv")
        self.frames_csv = os.path.join(self.out, "frames.csv")
        self.prelabel_csv = os.path.join(self.out, "prelabel.csv")

        self.model = None
        if args.prelabel and os.path.exists(args.model):
            with open(args.model, "rb") as f:
                self.model = pickle.load(f)
            print(f"pre-labelling with {args.model} ({self.model['feature_set']})")

        self.cap = cv2.VideoCapture(args.video)
        n_total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end = min(args.end, n_total) if args.end else n_total
        self.todo = list(range(args.start, end, args.step))
        self.scale = args.width / 1920

        cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("crops", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("frame", 0, 0)
        cv2.moveWindow("crops", args.width + 10, 0)
        cv2.setMouseCallback("frame", self.on_frame_click)
        cv2.setMouseCallback("crops", self.on_crops_click)

    # ---------- state per frame ----------
    def load(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return False
        self.idx, self.frame = idx, frame
        self.cands = find_candidates(frame, self.cfg)
        self.marks = ["neg"] * len(self.cands)
        self.miss = None
        self.scores, self.pred_k = None, None
        if self.model and self.cands:
            X = [extract(square_crop(frame, c["bbox"]), dict(zip(("x1", "y1", "x2", "y2"), c["bbox"]), **c),
                         self.model["feature_set"]) for c in self.cands]
            self.scores = self.model["model"].decision_function(np.array(X))
            self.pred_k = int(np.argmax(self.scores))
            if self.scores[self.pred_k] > self.args.prelabel_threshold:
                self.marks[self.pred_k] = "pos"
        return True

    def toggle(self, k, lab):
        self.marks[k] = "neg" if self.marks[k] == lab else lab
        if lab == "pos" and self.marks[k] == "pos":
            self.miss = None

    def hit(self, x, y):
        """Index of the smallest candidate box containing (x, y) in frame coords, else None."""
        best, best_area = None, None
        for k, c in enumerate(self.cands):
            x1, y1, x2, y2 = c["bbox"]
            if x1 - HIT_MARGIN <= x <= x2 + HIT_MARGIN and y1 - HIT_MARGIN <= y <= y2 + HIT_MARGIN:
                a = (x2 - x1) * (y2 - y1)
                if best is None or a < best_area:
                    best, best_area = k, a
        return best

    # ---------- mouse ----------
    def on_frame_click(self, event, x, y, *_):
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        x, y = int(x / self.scale), int(y / self.scale)
        k = self.hit(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if k is not None:
                self.toggle(k, "pos")
            elif "pos" not in self.marks:
                self.miss = (x, y)
        else:
            if k is not None:
                self.toggle(k, "ign")
            else:
                self.miss = None

    def on_crops_click(self, event, x, y, *_):
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        k = (y // THUMB) * COLS + x // THUMB
        if k < len(self.cands):
            self.toggle(k, "pos" if event == cv2.EVENT_LBUTTONDOWN else "ign")

    # ---------- drawing ----------
    def render(self, pos_in_todo):
        img = self.frame.copy()
        for k, (c, m) in enumerate(zip(self.cands, self.marks)):
            x1, y1, x2, y2 = c["bbox"]
            cv2.rectangle(img, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), COLORS[m], 3 if m != "neg" else 2)
            txt = str(k) if self.scores is None else f"{k}:{self.scores[k]:.1f}"
            cv2.putText(img, txt, (x1 - 4, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[m], 2)
        if self.miss:
            cv2.drawMarker(img, self.miss, (255, 0, 255), cv2.MARKER_CROSS, 30, 3)
            cv2.circle(img, self.miss, 18, (255, 0, 255), 2)

        n_pos = self.marks.count("pos")
        status = "BALL x%d" % n_pos if n_pos else ("MISSED (point set)" if self.miss else "no ball")
        lines = [
            f"frame {self.idx}  [{pos_in_todo + 1}/{len(self.todo)}]  candidates {len(self.cands)}  -> {status}",
            "L-click box: ball | R-click box: ignore | L-click empty: missed ball | Enter: next | s: skip | u: undo | q: quit",
        ]
        cv2.rectangle(img, (0, 0), (1920, 75), (0, 0, 0), -1)
        for i, t in enumerate(lines):
            cv2.putText(img, t, (10, 30 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("frame", cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA))

        rows = max(1, int(np.ceil(len(self.cands) / COLS)))
        panel = np.full((rows * THUMB, COLS * THUMB, 3), 40, np.uint8)
        for k, (c, m) in enumerate(zip(self.cands, self.marks)):
            r, col = divmod(k, COLS)
            t = cv2.resize(square_crop(self.frame, c["bbox"]), (THUMB, THUMB), interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(t, (0, 0), (THUMB - 1, THUMB - 1), COLORS[m], 4 if m != "neg" else 1)
            txt = str(k) if self.scores is None else f"{k}:{self.scores[k]:.1f}"
            cv2.putText(t, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[m], 1)
            panel[r * THUMB:(r + 1) * THUMB, col * THUMB:(col + 1) * THUMB] = t
        if not self.cands:
            cv2.putText(panel, "no candidates", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("crops", panel)

    # ---------- persistence ----------
    def save(self, status):
        rows = []
        if status not in ("skipped", "not_gameplay"):  # only frames you actually reviewed produce crops
            for k, (c, m) in enumerate(zip(self.cands, self.marks)):
                path = os.path.join(self.out, "crops", m, f"f{self.idx:06d}_{k:02d}.png")
                cv2.imwrite(path, square_crop(self.frame, c["bbox"]))
                rows.append(dict(zip(LABEL_FIELDS, [self.idx, k, *c["bbox"], round(c["area"], 1),
                                                    round(c["circularity"], 3), m, path])))
        write_csv(self.labels_csv, LABEL_FIELDS, rows, append=True)
        miss = self.miss or ("", "")
        write_csv(self.frames_csv, FRAME_FIELDS, [dict(frame=self.idx, status=status,
                  n_candidates=len(self.cands), miss_x=miss[0], miss_y=miss[1])], append=True)
        if self.pred_k is not None and status not in ("skipped", "not_gameplay"):
            write_csv(self.prelabel_csv, PRELABEL_FIELDS, [dict(frame=self.idx, pred_k=self.pred_k,
                      pred_score=round(float(self.scores[self.pred_k]), 3))], append=True)

    def undo(self):
        """Remove the last saved frame's rows and crop files. Returns its frame index or None."""
        frames = read_csv(self.frames_csv)
        if not frames:
            return None
        last = frames[-1]["frame"]
        labels = read_csv(self.labels_csv)
        for r in labels:
            if r["frame"] == last and os.path.exists(r["path"]):
                os.remove(r["path"])
        write_csv(self.labels_csv, LABEL_FIELDS, [r for r in labels if r["frame"] != last])
        write_csv(self.frames_csv, FRAME_FIELDS, frames[:-1])
        pre = read_csv(self.prelabel_csv)
        if pre:
            write_csv(self.prelabel_csv, PRELABEL_FIELDS, [r for r in pre if r["frame"] != last])
        return int(last)

    # ---------- main loop ----------
    def run(self):
        done = {int(r["frame"]) for r in read_csv(self.frames_csv)}
        i = 0
        while i < len(self.todo):
            idx = self.todo[i]
            if idx in done or not self.load(idx):
                i += 1
                continue
            if self.args.gameplay_filter and not is_gameplay(self.frame):
                self.save("not_gameplay")
                done.add(idx)
                i += 1
                continue

            action = None
            while action is None:
                self.render(i)
                key = cv2.waitKey(20) & 0xFF
                if key in (13, 32):
                    action = "confirm"
                elif key == ord("s"):
                    action = "skip"
                elif key == ord("u"):
                    action = "undo"
                elif key in (ord("q"), 27) or cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) < 1:
                    action = "quit"

            if action == "quit":
                break
            if action == "undo":
                last = self.undo()
                if last is not None:
                    done.discard(last)
                    i = self.todo.index(last) if last in self.todo else i
                continue
            status = "skipped" if action == "skip" else (
                "ball" if "pos" in self.marks else "ball_missed" if self.miss else "no_ball")
            self.save(status)
            done.add(idx)
            i += 1

        self.cap.release()
        cv2.destroyAllWindows()
        print_summary(self.frames_csv, self.labels_csv)


def print_summary(frames_csv, labels_csv):
    frames, labels = read_csv(frames_csv), read_csv(labels_csv)
    st = {s: sum(r["status"] == s for r in frames) for s in ["ball", "ball_missed", "no_ball", "skipped", "not_gameplay"]}
    lb = {s: sum(r["label"] == s for r in labels) for s in COLORS}
    print(f"frames: {len(frames)}  {st}")
    print(f"crops:  {len(labels)}  {lb}")
    if st["ball"] + st["ball_missed"]:
        print(f"candidate-stage recall: {st['ball'] / (st['ball'] + st['ball_missed']):.1%}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    default_video = next((p for p in [os.path.join(here, "Barca-Get.mp4"), os.path.join(here, "..", "Barca-Get.mp4")]
                          if os.path.exists(p)), "Barca-Get.mp4")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=default_video)
    ap.add_argument("--out", default=os.path.join(here, "dataset"))
    ap.add_argument("--step", type=int, default=30, help="label every Nth frame (30 = 0.5 s at 60 fps)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--width", type=int, default=1280, help="display width of the frame window")
    ap.add_argument("--extra-hud", action="append", default=[], metavar="x1,y1,x2,y2",
                    help="extra HUD box to ignore (1920x1080 coords), repeatable; e.g. a video's mini-map")
    ap.add_argument("--no-gameplay-filter", dest="gameplay_filter", action="store_false")
    ap.add_argument("--model", default=os.path.join(here, "models", "ball_svm.pkl"))
    ap.add_argument("--no-prelabel", dest="prelabel", action="store_false")
    ap.add_argument("--prelabel-threshold", type=float, default=0.0,
                    help="pre-mark the top candidate only if its SVM score is above this")
    Labeler(ap.parse_args()).run()
