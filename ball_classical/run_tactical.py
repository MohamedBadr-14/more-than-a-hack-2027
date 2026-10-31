"""Run the classical ball detector on the real-match tactical video and compare it to references.

Pipeline per frame
  homography pitch polygon -> candidates (white blobs, tactical-camera thresholds) ->
  keep candidates whose pitch position (via inverse homography) is inside the field ->
  SVM score -> best candidate if score > --threshold

References (neither is hand-made ground truth)
  ATD ball   Metrica's automatic tracking, projected to pixels through the homography
  GT label   public possession labels: H/A = ball alive, D = dead ball (window 11185-18684 only)
  YOLO       zero-shot YOLO run from ball_yolo/overlay.py, if its CSV exists

    python run_tactical.py                  # the public 5-minute window
    python run_tactical.py --limit 200      # smoke test

Output goes to <mtah>/data/ball_classical/ (gitignored there: derived from the match assets, must not
be published). Video legend: green box = our detection (+score), blue circle = ATD ball,
yellow = pitch outline from the homography. Also writes <stem>_candidates.csv (every in-field
candidate with its score), which track.py turns into a smoothed ball track.
"""
import argparse
import csv
import os
import pickle
import sys
import time

import cv2
import numpy as np

from candidates import CFG, find_candidates, square_crop
from features import extract

HERE = os.path.dirname(os.path.abspath(__file__))
MTAH = os.path.dirname(HERE)  # this folder sits at the root of the more-than-a-hack-2027 repo

# Tactical camera: ball is ~4 px (far side) to ~10 px (near side) across, so no opening and smaller
# areas. Tuned on frames 25000-40000 (outside the public window) against the ATD ball.
TACTICAL_CFG = dict(CFG, hud_boxes=[], open_kernel=1, close_kernel=3,
                    white_lower=(0, 0, 170), white_upper=(179, 60, 255),
                    area_far=(4, 120), area_near=(15, 400))
FIELD_MARGIN = 0.005  # normalised pitch units (~0.5 m) beyond the lines still counts as in the field
MASK_DILATE = 41      # px; lets blobs sitting on the touchline be found before the in-field check
MATCH_PX = 25         # detection agrees with ATD if within this many pixels (ATD is offset up to ~15 px)

OURS, ATD, PITCH, TEXT = (0, 255, 0), (255, 128, 0), (0, 255, 255), (255, 255, 255)


def load_gt(path):
    with open(path) as f:
        return {int(r["frame"]): r["label"] for r in csv.DictReader(f)}


def load_yolo(path):
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["yolo_x1"]:
                out[int(r["video_frame"])] = ((float(r["yolo_x1"]) + float(r["yolo_x2"])) / 2,
                                              (float(r["yolo_y1"]) + float(r["yolo_y2"])) / 2)
            else:
                out[int(r["video_frame"])] = None
    return out


def detect(frame, H, model, threshold, project, unproject):
    poly = np.array([project(H, x, y) for x, y in [(0, 0), (1, 0), (1, 1), (0, 1)]], np.float64)
    pm = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(pm, [poly.round().astype(np.int32)], 255)
    pm = cv2.dilate(pm, np.ones((MASK_DILATE, MASK_DILATE), np.uint8))

    cands = []
    for c in find_candidates(frame, TACTICAL_CFG, pm):
        x1, y1, x2, y2 = c["bbox"]
        u, v = unproject(H, (x1 + x2) / 2, (y1 + y2) / 2)
        if -FIELD_MARGIN <= u <= 1 + FIELD_MARGIN and -FIELD_MARGIN <= v <= 1 + FIELD_MARGIN:
            cands.append(dict(c, u=float(u), v=float(v)))
    if not cands:
        return None, [], poly
    X = [extract(square_crop(frame, c["bbox"]), dict(zip(("x1", "y1", "x2", "y2"), c["bbox"]), **c),
                 model["feature_set"]) for c in cands]
    scores = model["model"].decision_function(np.array(X))
    for c, s in zip(cands, scores):
        c["score"] = float(s)
    k = int(np.argmax(scores))
    best = cands[k] if scores[k] > threshold else None
    return best, cands, poly


def draw(frame, video_frame, label, poly, atd_px, det, agree):
    cv2.polylines(frame, [poly.round().astype(np.int32)], True, PITCH, 1, cv2.LINE_AA)
    if atd_px is not None:
        cv2.circle(frame, (int(atd_px[0]), int(atd_px[1])), 14, ATD, 2)
    if det is not None:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(frame, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), OURS, 2)
        cv2.putText(frame, f"{det['score']:.2f}", (x1 - 6, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, OURS, 1, cv2.LINE_AA)
    status = (f"frame {video_frame}  GT {label}  ATD {'yes' if atd_px is not None else 'no'}  "
              f"OURS {'yes %.2f' % det['score'] if det else 'no'}"
              + ("" if agree is None else f"  agree {'yes' if agree else 'NO'}"))
    cv2.rectangle(frame, (0, 1040), (900, 1080), (0, 0, 0), -1)
    cv2.putText(frame, status, (10, 1068), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT, 2, cv2.LINE_AA)


def summarise(rows, yolo):
    def rate(num, den):
        return f"{num}/{den} = {num / den:.1%}" if den else "n/a"

    alive = [r for r in rows if r["gt"] in ("H", "A")]
    dead = [r for r in rows if r["gt"] == "D"]
    with_atd = [r for r in rows if r["atd_x"] != ""]
    det_atd = [r for r in with_atd if r["det_x"] != ""]
    print("\n=== OURS ===")
    print("frames                       ", len(rows))
    print("detection rate, ball alive   ", rate(sum(r["det_x"] != "" for r in alive), len(alive)))
    print("detection rate, dead ball    ", rate(sum(r["det_x"] != "" for r in dead), len(dead)))
    print("ATD ball present             ", rate(len(with_atd), len(rows)))
    print("  found it (recall vs ATD)   ", rate(sum(r["agree"] == "1" for r in with_atd), len(with_atd)))
    print("  our det agrees (precision) ", rate(sum(r["agree"] == "1" for r in det_atd), len(det_atd)))
    no_atd = [r for r in rows if r["atd_x"] == "" and r["gt"] in ("H", "A")]
    print("ball alive, ATD missing:     ", len(no_atd), "frames; we output a ball on",
          rate(sum(r["det_x"] != "" for r in no_atd), len(no_atd)))

    if yolo:
        ys = [r for r in with_atd if int(r["frame"]) in yolo]
        found = sum(yolo[int(r["frame"])] is not None and
                    np.hypot(yolo[int(r["frame"])][0] - float(r["atd_x"]),
                             yolo[int(r["frame"])][1] - float(r["atd_y"])) < MATCH_PX for r in ys)
        dets = [r for r in ys if yolo[int(r["frame"])] is not None]
        ours_same = sum(r["agree"] == "1" for r in ys)
        print(f"\n=== YOLO (best box, same {len(ys)} ATD frames it covers) ===")
        print("  found it (recall vs ATD)   ", rate(found, len(ys)), f"   ours on same frames: {rate(ours_same, len(ys))}")
        print("  its det agrees (precision) ", rate(found, len(dets)))


def main(args):
    sys.path[:0] = [args.mtah, os.path.join(args.mtah, "starter"), os.path.join(args.mtah, "ball_yolo")]
    import load_homography as LH
    import overlay as OV

    with open(args.model, "rb") as f:
        model = pickle.load(f)
    matrices, _, valid = LH.load()
    balls = OV.atd_ball(args.first, args.last)
    gt = load_gt(os.path.join(args.mtah, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
    yolo = load_yolo(os.path.join(args.mtah, "data", "ball_yolo", "yolo_vs_atd_11185_18684.csv"))

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"classical_vs_atd_{args.first}_{args.last}"
    print(f"skipping to frame {args.first} ...")
    capture = OV.open_video_at(args.first)  # grab() not seek: seeking can land a few frames off
    w, h = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None if args.no_video else cv2.VideoWriter(
        os.path.join(args.out_dir, stem + ".mp4"), cv2.VideoWriter_fourcc(*"mp4v"), capture.get(cv2.CAP_PROP_FPS), (w, h))
    cand_file = open(os.path.join(args.out_dir, stem + "_candidates.csv"), "w", newline="")
    cand_csv = csv.writer(cand_file)
    cand_csv.writerow(["frame", "x1", "y1", "x2", "y2", "u", "v", "score"])  # u, v = normalised pitch coords

    rows, started = [], time.time()
    for video_frame in range(args.first, args.last + 1):
        ok, frame = capture.read()
        if not ok:
            print("video ended early at", video_frame)
            break
        H = matrices[video_frame]
        det, cands, poly = (None, [], np.zeros((4, 2))) if not valid[video_frame] else \
            detect(frame, H, model, args.threshold, LH.project, LH.unproject)
        n_cands = len(cands)
        for c in cands:
            cand_csv.writerow([video_frame, *c["bbox"], f"{c['u']:.5f}", f"{c['v']:.5f}", f"{c['score']:.3f}"])

        atd_px = None
        bx, by = balls.get(video_frame, (np.nan, np.nan))
        if valid[video_frame] and not np.isnan(bx):
            atd_px = tuple(float(v) for v in LH.project(H, bx, by))
        dx = dy = None
        if det:
            x1, y1, x2, y2 = det["bbox"]
            dx, dy = (x1 + x2) / 2, (y1 + y2) / 2
        agree = None if atd_px is None else (det is not None and np.hypot(dx - atd_px[0], dy - atd_px[1]) < MATCH_PX)

        rows.append(dict(frame=video_frame, gt=gt.get(video_frame, "X"), n_cands=n_cands,
                         atd_x="" if atd_px is None else f"{atd_px[0]:.1f}", atd_y="" if atd_px is None else f"{atd_px[1]:.1f}",
                         det_x="" if det is None else f"{dx:.1f}", det_y="" if det is None else f"{dy:.1f}",
                         score="" if det is None else f"{det['score']:.3f}",
                         agree="" if agree is None else str(int(agree))))
        if writer:
            draw(frame, video_frame, rows[-1]["gt"], poly, atd_px, det, agree)
            writer.write(frame)

        done = len(rows)
        if done % 250 == 0:
            fps = done / (time.time() - started)
            print(f"  {done}/{args.last - args.first + 1} frames  {fps:.1f} fps  ~{(args.last - video_frame) / fps / 60:.0f} min left")

    capture.release()
    cand_file.close()
    if writer:
        writer.release()
    with open(os.path.join(args.out_dir, stem + ".csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    summarise([{k: str(v) for k, v in r.items()} for r in rows], yolo)
    print(f"\nwrote {os.path.join(args.out_dir, stem)}{'' if args.no_video else '.mp4 / '}.csv, _candidates.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mtah", default=MTAH, help="path to the more-than-a-hack-2027 repo")
    ap.add_argument("--first", type=int, default=11185)
    ap.add_argument("--last", type=int, default=18684)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default=os.path.join(HERE, "models", "ball_svm.pkl"))
    ap.add_argument("--threshold", type=float, default=0.0, help="min SVM score to output a ball")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-video", action="store_true", help="only write the CSVs (candidates for track.py)")
    a = ap.parse_args()
    a.out_dir = a.out_dir or os.path.join(a.mtah, "data", "ball_classical")
    if a.limit:
        a.last = min(a.last, a.first + a.limit - 1)
    main(a)
