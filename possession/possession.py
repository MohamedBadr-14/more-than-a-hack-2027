"""Team in possession per frame: hand-over state machine on the classical ball + ATD players.

Dead ball is ignored (v1): every frame after the first confirmed controller is H or A.

1. Ball per frame: the classical tracker (ball_classical/track.py output, det or interp), else the ORIGINAL
   ATD ball (Metrica, BAR-ATH_tactical_atd_original.txt -- the live file now carries our ball in this window).
2. Near frame: the nearest player is within R of the ball and the nearest opponent is at least MARGIN
   further away (closer = contested 50/50, no evidence). Control frame: a near frame where the ball moves
   with him: |v_ball - v_player| < V_REL. Velocities are central differences over +/-VEL_HALF frames.
3. Evidence: control frames where the ball is a tracked detection, a gap carried by one player (track.py), or
   ATD fallback. Interpolated balls hold the state but never change it (their path is invented).
4. State machine: possession stays with the current team until the other team has K evidence control
   frames within the last M frames; the label switch is backdated to the start of the unbroken run of the
   new team's near frames that holds the first of them (the ball reaching him, before he slows it).
   Before the first confirmation the label is X (the scorer back-fills it).

    python possession.py                  # 11185-18684 -> ../data/possession/possession_11185_18684.{csv,mp4}
    python possession.py --no-video --tag _r2

Video legend: team bar + label top left, GT (checking only) next to it. Players: dots by team, controller
ringed (white ring = evidence frame, grey = interp frame), yellow X = contested. Ball: green = detection,
orange = interpolated, cyan = attached to a carrier, magenta = ATD fallback.
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MTAH = os.path.dirname(HERE)
sys.path[:0] = [MTAH, os.path.join(MTAH, "starter"), os.path.join(MTAH, "ball_yolo")]
from games import GAME, GAMES, require  # noqa: E402
from baseline import team_of_each_track  # noqa: E402

FPS = 25
PITCH = GAMES[GAME]["pitch"]  # metres
R = 1.5          # m, control radius
V_REL = 4.0      # m/s, max ball speed relative to the controller
VEL_HALF = 2     # frames each side for central-difference velocities
MARGIN = 0.3     # m, nearest opponent must be this much further than the nearest player, else contested
K, M = 8, 25     # evidence control frames needed within the last M frames to hand over
HOME, AWAY = 0, 1
LABEL = "HA"

TEAM_C = {HOME: (160, 60, 20), AWAY: (40, 40, 220)}  # BGR: Barca blue, Athletic red
BALL_C = {"det": (0, 255, 0), "interp": (0, 165, 255), "carried": (0, 215, 255), "attached": (255, 255, 0), "atd": (255, 0, 255)}
TEXT, CONTESTED = (255, 255, 255), (0, 255, 255)


def read_atd(path, first, last):
    """players (n, slots, 2) and ball (n, 2), normalised 0-1, NaN where missing; row 0 = video frame `first`."""
    players, ball = [], []
    with open(path) as f:
        for line in f:
            fid = int(line[:line.index(":")])
            if fid - 1 < first:
                continue
            if fid - 1 > last:
                break
            _, pl, bl = line.rstrip().split(":")
            players.append([[float(v) for v in g.split(",")[:2]] for g in pl.split(";")])
            ball.append([float(v) for v in bl.split(",")[:2]])
    return np.array(players), np.array(ball)


def read_track(path, first, last):
    """status list and (n, 2) normalised ball from track.py's CSV."""
    status, uv = ["none"] * (last - first + 1), np.full((last - first + 1, 2), np.nan)
    with open(path) as f:
        for r in csv.DictReader(f):
            t = int(r["frame"]) - first
            if 0 <= t < len(status) and r["status"] != "none":
                status[t], uv[t] = r["status"], (float(r["u"]), float(r["v"]))
    return status, uv


def velocity(pos):
    """Central difference over +/-VEL_HALF frames along axis 0, m/s; NaN where an end is missing."""
    v = np.full(pos.shape, np.nan)
    v[VEL_HALF:-VEL_HALF] = (pos[2 * VEL_HALF:] - pos[:-2 * VEL_HALF]) * FPS / (2 * VEL_HALF)
    return v


def control(ball_m, players_m, teams):
    """Per frame: nearest slot (-1 none), contested / near / control flags, and for the debug view the
    distance (m) to the nearest player and nearest opponent, and the relative speed (m/s), NaN if unknown."""
    n = len(ball_m)
    vb, vp = velocity(ball_m), velocity(players_m)
    slot, contested, near, ctrl = np.full(n, -1), np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    d_near, d_opp, v_rel = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for t in range(n):
        d = np.hypot(*(players_m[t] - ball_m[t]).T)
        if np.all(np.isnan(d)):
            continue
        s = int(np.nanargmin(d))
        opp = d[teams != teams[s]]
        d_near[t], d_opp[t] = d[s], np.nanmin(opp, initial=np.inf)
        v_rel[t] = np.hypot(*(vb[t] - vp[t, s]))
        if d[s] >= R:
            continue
        slot[t] = s
        if d_opp[t] - d[s] < MARGIN:
            contested[t] = True
            continue
        near[t] = True
        ctrl[t] = v_rel[t] < V_REL  # NaN velocity -> False
    return dict(slot=slot, contested=contested, near=near, ctrl=ctrl, d_near=d_near, d_opp=d_opp, v_rel=v_rel)


def hand_over(team_ctrl, evidence, team_near):
    """Labels 'H'/'A'/'X' from per-frame controlling team, near team (-1 none) and evidence flags.

    Also returns, for the debug view, the hit count of each team per frame (n, 2) and the hand-overs
    as (confirmed frame, backdated start, team), as indices into the arrays.
    """
    n = len(team_ctrl)
    labels = np.full(n, "X")
    counts, events = np.zeros((n, 2), int), []
    state, switched = None, 0
    hits = {HOME: [], AWAY: []}  # evidence control frames of each team, within the last M frames
    for t in range(n):
        if evidence[t] and team_ctrl[t] >= 0:
            hits[team_ctrl[t]].append(t)
        for team in hits:
            hits[team] = [h for h in hits[team] if h > t - M]
        for team in (HOME, AWAY):
            if team != state and len(hits[team]) >= K:
                start = hits[team][0]
                while start - 1 >= switched and team_near[start - 1] == team:
                    start -= 1
                state, switched = team, start
                labels[start:t] = LABEL[team]
                events.append((t, start, team))
                hits = {HOME: [], AWAY: []}
                break
        if state is not None:
            labels[t] = LABEL[state]
        counts[t] = len(hits[HOME]), len(hits[AWAY])
    return labels, counts, events


def render(args, frames, labels, status, ball_n, players_n, teams, slot, contested, evidence, gt):
    sys.path.insert(0, os.path.join(MTAH, "ball_yolo"))
    import load_homography as LH
    import overlay as OV
    matrices, _, _ = LH.load()
    print(f"rendering video, skipping to frame {frames[0]} ...")
    cap = OV.open_video_at(frames[0])
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    path = os.path.join(args.out_dir, f"possession_{args.first}_{args.last}{args.tag}.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    started = time.time()
    for i, t in enumerate(frames):
        ok, frame = cap.read()
        if not ok:
            break
        H = matrices[t]
        for s, (u, v) in enumerate(players_n[i]):
            if not np.isnan(u):
                x, y = (int(round(c)) for c in LH.project(H, u, v))
                cv2.circle(frame, (x, y), 5, TEAM_C[teams[s]], -1)
                if s == slot[i]:
                    cv2.circle(frame, (x, y), 16, TEXT if evidence[i] else (128, 128, 128), 2)
                    if contested[i]:
                        cv2.drawMarker(frame, (x, y - 26), CONTESTED, cv2.MARKER_TILTED_CROSS, 14, 2)
        if not np.isnan(ball_n[i, 0]):
            x, y = (int(round(c)) for c in LH.project(H, *ball_n[i]))
            cv2.circle(frame, (x, y), 9, BALL_C[status[i]], 2)
        lab = labels[i]
        cv2.rectangle(frame, (0, 0), (420, 44), TEAM_C[LABEL.index(lab)] if lab in LABEL else (60, 60, 60), -1)
        cv2.putText(frame, f"POSS {lab}   GT {gt.get(t, 'X')}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 1040), (700, 1080), (0, 0, 0), -1)
        cv2.putText(frame, f"frame {t}  ball {status[i]}" + ("  contested" if contested[i] else ""),
                    (10, 1068), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT, 2, cv2.LINE_AA)
        writer.write(frame)
        if (i + 1) % 500 == 0:
            fps = (i + 1) / (time.time() - started)
            print(f"  {i + 1}/{len(frames)} frames  {fps:.1f} fps  ~{(len(frames) - i - 1) / fps / 60:.0f} min left")
    cap.release()
    writer.release()
    print("wrote", path)


def compute(first, last, track):
    """Everything the labels come from, per frame of first..last (index 0 = frame `first`)."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    orig = raw_path.replace("_atd.txt", "_atd_original.txt")
    if not os.path.exists(orig):
        sys.exit(f"missing {orig}: the fallback must be Metrica's ball, not the spliced file")
    teams = team_of_each_track(meta_path)
    frames = np.arange(first, last + 1)
    scale = np.array(PITCH)

    players_n, atd_ball = read_atd(orig, first, last)
    status, ball_n = read_track(track, first, last)
    for i in range(len(frames)):
        if status[i] == "none" and not np.isnan(atd_ball[i, 0]):
            status[i], ball_n[i] = "atd", atd_ball[i]

    c = control(ball_n * scale, players_n * scale, teams)
    evidence = np.array([s in ("det", "carried", "attached", "atd") for s in status])
    team_of_slot = teams[np.maximum(c["slot"], 0)]
    team_ctrl = np.where(c["ctrl"], team_of_slot, -1)
    team_near = np.where(c["near"], team_of_slot, -1)
    labels, counts, events = hand_over(team_ctrl, evidence, team_near)

    with open(os.path.join(MTAH, "evaluation", "gt", f"{GAME}_possession_gt_public.csv")) as f:
        gt = {int(r["frame"]): r["label"] for r in csv.DictReader(f) if first <= int(r["frame"]) <= last}
    return dict(c, frames=frames, teams=teams, players_n=players_n, ball_n=ball_n, status=status, evidence=evidence,
                team_ctrl=team_ctrl, team_near=team_near, labels=labels, counts=counts, events=events, gt=gt)


def main(args):
    d = compute(args.first, args.last, args.track)
    frames, labels, status, slot, teams = d["frames"], d["labels"], d["status"], d["slot"], d["teams"]
    contested, ctrl, evidence = d["contested"], d["ctrl"], d["evidence"]

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"possession_{args.first}_{args.last}{args.tag}")
    with open(stem + ".csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "label", "ball_src", "ctrl_slot", "ctrl_team", "contested", "control", "evidence"])
        for i, t in enumerate(frames):
            wr.writerow([t, labels[i], status[i], slot[i], LABEL[teams[slot[i]]] if slot[i] >= 0 else "",
                         int(contested[i]), int(ctrl[i]), int(evidence[i])])
    full = np.full(GAMES[GAME]["n_video_frames"], "X")
    full[frames] = labels
    with open(stem + "_submission.csv", "w", newline="") as f:
        f.write("frame,label\n")
        f.writelines(f"{t},{lab}\n" for t, lab in enumerate(full))

    runs = [lab for i, lab in enumerate(labels) if lab in LABEL and (i == 0 or labels[i - 1] != lab)]
    src = {s: status.count(s) for s in ("det", "interp", "carried", "attached", "atd", "none")}
    print(f"ball source: {src}")
    print(f"control frames {ctrl.sum()} (evidence {(ctrl & evidence).sum()}), contested {contested.sum()}")
    print(f"possessions: {len(runs)} (H {runs.count('H')}, A {runs.count('A')}), "
          f"mean {np.isin(labels, list(LABEL)).sum() / max(len(runs), 1) / FPS:.1f} s, X frames {(labels == 'X').sum()}")
    print("wrote", stem + ".csv", "and", stem + "_submission.csv")
    if not args.no_video:
        render(args, frames, labels, status, d["ball_n"], d["players_n"], teams, slot, contested, evidence, d["gt"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--first", type=int, default=11185)
    ap.add_argument("--last", type=int, default=18684)
    ap.add_argument("--track", default=os.path.join(MTAH, "data", "ball_classical", "strict_r04_gap2_carried", "tracked_11185_18684.csv"))
    ap.add_argument("--out-dir", default=os.path.join(MTAH, "data", "possession"))
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the output names (keeps earlier outputs)")
    main(ap.parse_args())
