"""Debug view of the ball tracker + possession logic, frame by frame.

render: writes <out-dir>/possession_<first>_<last><tag>_debug.mp4 (1920x1080) and a _debug_marks.csv of the
frames worth jumping to (hand-overs, their backdated starts, starts of disagreement with the GT).
view: steps through that video with the keyboard.

    python debug_view.py render --track ../data/ball_classical/boot_attach/tracked_11185_18684.csv --tag _v2
    python debug_view.py view --tag _v2

Layout: match video top left; right = minimap (top-down pitch) + the numbers behind the current frame;
bottom = timeline of +/-5 s around the current frame (cursor in the middle).
Video: players = team dots, nearest player ringed (white = evidence frame, grey = not), thin lines from the
ball to the nearest player (team colour) and nearest opponent (yellow), dashed circle = control radius R.
Ball: green = detection, orange = interpolated, cyan = attached to a carrier, magenta = ATD fallback.
Timeline rows: ours, GT, ball source, near (team of the nearest player when near; yellow = contested),
evidence (team colour where a control frame counts towards a hand-over), hits (count of the team that is
NOT in possession, out of K; filled at K = hand-over). White line = hand-over confirmed, dashed = where
it was backdated to.

Viewer keys: space pause/play, right/left (or d/a) step one frame, n/p next/previous hand-over,
m/M next/previous disagreement with GT, g go to frame (typed in the terminal), +/- speed, q quit.
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

import possession as P

W, H = 1920, 1080
VID_W, VID_H = 1440, 810
PANEL_X = VID_W
HALF = 125  # timeline frames each side of the cursor (5 s)
TL_X0, TL_Y0 = 130, VID_H + 12
ROW_H = 36
LABEL_C = {"H": P.TEAM_C[P.HOME], "A": P.TEAM_C[P.AWAY], "D": (110, 110, 110), "X": (45, 45, 45)}
WHITE, GREY, YELLOW, DARK = (255, 255, 255), (128, 128, 128), (0, 255, 255), (25, 25, 25)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(img, s, xy, scale=0.6, colour=WHITE, thick=1):
    cv2.putText(img, s, xy, FONT, scale, colour, thick, cv2.LINE_AA)


def minimap(canvas, d, i, x0, y0, w):
    """Top-down pitch: players by team, nearest ringed, ball, control radius."""
    sx = w / P.PITCH[0]
    h = int(P.PITCH[1] * sx)
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (40, 90, 40), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), WHITE, 1)
    cv2.line(canvas, (x0 + w // 2, y0), (x0 + w // 2, y0 + h), WHITE, 1)
    cv2.circle(canvas, (x0 + w // 2, y0 + h // 2), int(9.15 * sx), WHITE, 1)

    def pt(uv):
        return int(x0 + uv[0] * w), int(y0 + uv[1] * h)

    for s, uv in enumerate(d["players_n"][i]):
        if not np.isnan(uv[0]):
            cv2.circle(canvas, pt(uv), 4, P.TEAM_C[d["teams"][s]], -1)
            if s == d["slot"][i]:
                cv2.circle(canvas, pt(uv), 8, WHITE if d["evidence"][i] else GREY, 1)
    b = d["ball_n"][i]
    if not np.isnan(b[0]):
        cv2.circle(canvas, pt(b), max(2, int(P.R * sx)), YELLOW, 1)
        cv2.circle(canvas, pt(b), 3, P.BALL_C[d["status"][i]], -1)
    return h


def overlay(frame, d, i, Hm, LH):
    """Draw on the full-resolution match frame."""
    b = d["ball_n"][i]
    has_ball = not np.isnan(b[0])
    px = {}
    for s, uv in enumerate(d["players_n"][i]):
        if not np.isnan(uv[0]):
            px[s] = tuple(int(round(c)) for c in LH.project(Hm, *uv))
            cv2.circle(frame, px[s], 5, P.TEAM_C[d["teams"][s]], -1)
    if has_ball:
        bp = tuple(int(round(c)) for c in LH.project(Hm, *b))
        # nearest player / nearest opponent, found again here so the lines show even beyond R
        dist = np.hypot(*((d["players_n"][i] - b) * P.PITCH).T)
        if not np.all(np.isnan(dist)):
            s = int(np.nanargmin(dist))
            opp = np.where(d["teams"] != d["teams"][s], dist, np.nan)
            if s in px:
                cv2.line(frame, bp, px[s], P.TEAM_C[d["teams"][s]], 1, cv2.LINE_AA)
            if not np.all(np.isnan(opp)) and int(np.nanargmin(opp)) in px:
                cv2.line(frame, bp, px[int(np.nanargmin(opp))], YELLOW, 1, cv2.LINE_AA)
        ring = [LH.project(Hm, b[0] + P.R * np.cos(a) / P.PITCH[0], b[1] + P.R * np.sin(a) / P.PITCH[1])
                for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
        for k in range(0, 24, 2):
            cv2.line(frame, tuple(int(c) for c in ring[k]), tuple(int(c) for c in ring[k + 1]), YELLOW, 1, cv2.LINE_AA)
        cv2.circle(frame, bp, 9, P.BALL_C[d["status"][i]], 2)
    s = d["slot"][i]
    if s >= 0 and s in px:
        cv2.circle(frame, px[s], 16, WHITE if d["evidence"][i] else GREY, 2)
        if d["contested"][i]:
            cv2.drawMarker(frame, (px[s][0], px[s][1] - 26), YELLOW, cv2.MARKER_TILTED_CROSS, 14, 2)


def panel(canvas, d, i, ev_at):
    t = d["frames"][i]
    x = PANEL_X + 10
    y = 10 + minimap(canvas, d, i, x, 10, W - PANEL_X - 20) + 30
    lab, gt = d["labels"][i], d["gt"].get(int(t), "X")
    cv2.rectangle(canvas, (x, y - 24), (x + 200, y + 10), LABEL_C[lab], -1)
    text(canvas, f"OURS {lab}", (x + 8, y + 2), 0.75, WHITE, 2)
    cv2.rectangle(canvas, (x + 240, y - 24), (x + 440, y + 10), LABEL_C[gt], -1)
    text(canvas, f"GT {gt}", (x + 248, y + 2), 0.75, WHITE, 2)
    y += 40
    mm, ss = divmod((t - d["frames"][0]) / P.FPS, 60)
    rows = [f"frame {t}   +{int(mm)}:{ss:04.1f}",
            f"ball      {d['status'][i]}" + ("  (evidence)" if d["evidence"][i] else "  (no evidence)"),
            f"nearest   {fmt(d['d_near'][i])} m" + (f"  team {P.LABEL[d['teams'][d['slot'][i]]]}" if d["slot"][i] >= 0 else ""),
            f"opponent  {fmt(d['d_opp'][i])} m   margin {fmt(d['d_opp'][i] - d['d_near'][i])} (need {P.MARGIN})",
            f"v_rel     {fmt(d['v_rel'][i])} m/s (need < {P.V_REL})",
            f"near {yn(d['near'][i])}  contested {yn(d['contested'][i])}  control {yn(d['ctrl'][i])}"]
    for r in rows:
        text(canvas, r, (x, y), 0.55)
        y += 26
    y += 6
    state = P.LABEL.index(lab) if lab in P.LABEL else None
    for team in (P.HOME, P.AWAY):
        n = d["counts"][i, team]
        cv2.rectangle(canvas, (x + 230, y - 14), (x + 230 + 200, y + 4), GREY, 1)
        cv2.rectangle(canvas, (x + 230, y - 14), (x + 230 + int(200 * min(n, P.K) / P.K), y + 4), P.TEAM_C[team], -1)
        text(canvas, f"{P.LABEL[team]} hits {n}/{P.K}" + ("  (has ball)" if team == state else ""), (x, y), 0.5)
        y += 26
    if ev_at is not None:
        conf, start, team = ev_at
        cv2.rectangle(canvas, (x - 4, y), (W - 10, y + 52), P.TEAM_C[team], -1)
        text(canvas, f"HAND-OVER -> {P.LABEL[team]}  confirmed {d['frames'][conf]}", (x + 4, y + 20), 0.55, WHITE, 2)
        text(canvas, f"backdated to {d['frames'][start]} ({(conf - start) / P.FPS:.1f} s earlier)", (x + 4, y + 44), 0.55, WHITE, 1)


def fmt(v):
    return "  -  " if v is None or not np.isfinite(v) else f"{v:5.2f}"


def yn(b):
    return "Y" if b else "n"


def timeline(canvas, d, i):
    n = len(d["frames"])
    pxf = (W - TL_X0 - 10) / (2 * HALF + 1)
    cv2.rectangle(canvas, (0, VID_H), (W, H), DARK, -1)
    names = ["ours", "GT", "ball", "near", "evidence", "hits"]
    for r, name in enumerate(names):
        text(canvas, name, (10, TL_Y0 + r * ROW_H + 24), 0.55)
    src_c = dict(P.BALL_C, none=(45, 45, 45))
    for k in range(-HALF, HALF + 1):
        j = i + k
        if not 0 <= j < n:
            continue
        x1 = int(TL_X0 + (k + HALF) * pxf)
        x2 = int(TL_X0 + (k + HALF + 1) * pxf)

        def cell(row, colour, frac=1.0):
            y2 = TL_Y0 + row * ROW_H + ROW_H - 6
            cv2.rectangle(canvas, (x1, int(y2 - (ROW_H - 6) * frac)), (x2, y2), colour, -1)

        cell(0, LABEL_C[d["labels"][j]])
        cell(1, LABEL_C[d["gt"].get(int(d["frames"][j]), "X")])
        cell(2, src_c[d["status"][j]])
        if d["contested"][j]:
            cell(3, YELLOW)
        elif d["near"][j]:
            cell(3, P.TEAM_C[d["teams"][d["slot"][j]]])
        if d["evidence"][j] and d["team_ctrl"][j] >= 0:
            cell(4, P.TEAM_C[d["team_ctrl"][j]])
        lab = d["labels"][j]
        other = [tm for tm in (P.HOME, P.AWAY) if lab not in P.LABEL or P.LABEL[tm] != lab]
        c = max(other, key=lambda tm: d["counts"][j, tm])
        if d["counts"][j, c]:
            cell(5, P.TEAM_C[c], min(d["counts"][j, c], P.K) / P.K)
    y_top, y_bot = TL_Y0, TL_Y0 + len(names) * ROW_H
    for conf, start, team in d["events"]:
        for f, dashed in ((conf, False), (start, True)):
            if abs(f - i) <= HALF:
                x = int(TL_X0 + (f - i + HALF + 0.5) * pxf)
                for y in range(y_top, y_bot, 8 if dashed else y_bot):
                    cv2.line(canvas, (x, y), (x, min(y + (4 if dashed else y_bot), y_bot)), WHITE, 2)
    xc = int(TL_X0 + (HALF + 0.5) * pxf)
    cv2.line(canvas, (xc, y_top - 6), (xc, y_bot + 2), YELLOW, 1)
    text(canvas, "-5 s", (TL_X0, H - 8), 0.45, GREY)
    text(canvas, "+5 s", (W - 50, H - 8), 0.45, GREY)


def compose(frame, d, i, Hm, LH):
    overlay(frame, d, i, Hm, LH)
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[:VID_H, :VID_W] = cv2.resize(frame, (VID_W, VID_H), interpolation=cv2.INTER_AREA)
    recent = [e for e in d["events"] if 0 <= i - e[0] < 3 * P.FPS]  # banner for 3 s after a hand-over
    panel(canvas, d, i, recent[-1] if recent else None)
    timeline(canvas, d, i)
    return canvas


def stem(args):
    return os.path.join(args.out_dir, f"possession_{args.first}_{args.last}{args.tag}_debug")


def render(args):
    import load_homography as LH
    import overlay as OV
    d = P.compute(args.first, args.last, args.track)
    labels, gt_lab = d["labels"], np.array([d["gt"].get(int(t), "X") for t in d["frames"]])
    os.makedirs(args.out_dir, exist_ok=True)
    with open(stem(args) + "_marks.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["kind", "frame"])
        for conf, start, team in d["events"]:
            wr.writerow(["handover", int(d["frames"][conf])])
        bad = (labels != gt_lab) & (gt_lab != "X")
        for j in np.flatnonzero(bad & ~np.concatenate(([False], bad[:-1]))):
            wr.writerow(["mismatch", int(d["frames"][j])])
    matrices, _, _ = LH.load()
    print(f"rendering, skipping to frame {args.first} ...")
    cap = OV.open_video_at(args.first)
    path = stem(args) + ".mp4"
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), P.FPS, (W, H))
    started = time.time()
    for i, t in enumerate(d["frames"]):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(compose(frame, d, i, matrices[t], LH))
        if (i + 1) % 500 == 0:
            fps = (i + 1) / (time.time() - started)
            print(f"  {i + 1}/{len(d['frames'])} frames  {fps:.1f} fps  ~{(len(d['frames']) - i - 1) / fps / 60:.0f} min left")
    cap.release()
    writer.release()
    print("wrote", path)


def view(args):
    path = stem(args) + ".mp4"
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"cannot open {path}: run 'render' first")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    marks = {"handover": [], "mismatch": []}
    if os.path.exists(stem(args) + "_marks.csv"):
        with open(stem(args) + "_marks.csv") as f:
            for r in csv.DictReader(f):
                marks[r["kind"]].append(int(r["frame"]) - args.first)
    win = "possession debug"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1600, 900)
    i, playing, speed, img = 0, False, 1.0, None
    seek = True
    print(__doc__.split("Viewer keys: ")[1])
    while True:
        if seek:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            seek = False
        if img is None or playing:
            ok, img = cap.read()
            if not ok:
                playing, i, seek, img = False, max(0, n - 1), True, None
                continue
            i = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        cv2.imshow(win, img)
        key = cv2.waitKeyEx(int(1000 / (P.FPS * speed)) if playing else 0)
        if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        target = None
        if key == ord(" "):
            playing = not playing
        elif key in (2555904, ord("d")):
            target = i + 1
        elif key in (2424832, ord("a")):
            target = i - 1
        elif key in (ord("n"), ord("p"), ord("m"), ord("M")):
            lst = marks["handover" if key in (ord("n"), ord("p")) else "mismatch"]
            fwd = key in (ord("n"), ord("m"))
            cand = [f for f in lst if (f > i + 2 * P.FPS if fwd else f < i - 2 * P.FPS)]
            if cand:
                target = (min(cand) if fwd else max(cand)) - 2 * P.FPS  # land 2 s before it
        elif key == ord("g"):
            try:
                target = int(input("go to frame: ")) - args.first
            except ValueError:
                pass
        elif key in (ord("+"), ord("=")):
            speed = min(speed * 2, 8)
        elif key == ord("-"):
            speed = max(speed / 2, 0.125)
        if target is not None:
            i, seek, img, playing = int(np.clip(target, 0, n - 1)), True, None, False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["render", "view"])
    ap.add_argument("--first", type=int, default=11185)
    ap.add_argument("--last", type=int, default=18684)
    ap.add_argument("--track", default=os.path.join(P.MTAH, "data", "ball_classical", "strict_r04_gap2_carried", "tracked_11185_18684.csv"))
    ap.add_argument("--out-dir", default=os.path.join(P.MTAH, "data", "possession"))
    ap.add_argument("--tag", default="_v2")
    a = ap.parse_args()
    render(a) if a.mode == "render" else view(a)
