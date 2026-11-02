"""Offline ball tracking on top of the per-frame candidates from run_tactical.py.

1. Path selection (dynamic programming / Viterbi-style longest path over all candidates)
   Every in-field candidate is a node worth (SVM score - BIAS): weak detections cost, strong ones earn,
   so a long chain of weak boot detections loses value instead of slowly accumulating it.
   A path links candidates forward in time (up to FREE_RESTART frames apart):
     - normal link if the ball could travel that far: distance <= V_MAX * k / FPS + R0 (pitch metres)
       consecutive-frame links earn CONTINUITY, each metre costs LAMBDA. Skipped frames are free: a frame
       without a ball must cost the same whether it lies inside a track or in a long gap between tracks,
       otherwise a sparsely seen ball (a hit every few frames) loses to skipping the whole stretch.
     - otherwise a jump: costs NEW_TRACK + JUMP_PER_M per metre beyond that reach, so the track can leave
       a wrong object as soon as a strong enough ball track appears elsewhere, while a jump onto a weak
       track never pays off. Jumps and gaps longer than MAX_GAP are not interpolated.
     - after FREE_RESTART frames with no ball, a track may restart anywhere for NEW_TRACK
   The best total over the whole clip is found exactly; frames not on the path have no detection.
   This replaces "top-scoring candidate per frame": a boot that wins one frame loses to the
   slightly lower-scoring ball that fits the motion of its neighbours.

2. Gap filling
   Between two linked detections up to MAX_GAP frames apart, the ball is interpolated linearly in
   pitch coordinates and projected back with each frame's homography. Gaps between unrelated track
   segments, or longer than MAX_GAP, stay "no ball".
   --interp hermite: a cubic curve that leaves A and enters B with the ball's velocity there (fitted over
   up to VEL_FIT path detections on each side), falling back to the straight line if the curve would
   exceed INTERP_VMAX or a side has too few detections.

Optional (--vel-w > 0): links pay VEL_W per m/s that the link's velocity differs from the velocity the
ball arrived with (smoothed, via the predecessor's best path), beyond VEL_TOL m/s of jitter. Greedy
approximation of a second-order model: each node keeps the velocity of its own best path only.

    python track.py                               # uses the public-window outputs of run_tactical.py
    python track.py --no-video                    # numbers only
    python track.py --first 11185 --last 13184    # sub-range of the same run

3. Boots and carried balls (need ATD player positions, from the original Metrica file; slot ids never used)
   --boot-mode strict (default): the path is chosen with every candidate, then path detections within BOOT_R
   of a player become gaps (no other candidate is picked) and are filled like any other gap.
   Other modes: penalty (BOOT_PEN off the score before path selection), reject (dropped before), off.
   carried: an interpolated gap whose detections on both sides are within ATTACH_R of one player, followed
   frame to frame by position, is flagged "carried" (same curve; possession may count it as evidence).
   --attach (off by default): such gaps, any length the path links, follow the player's ATD position instead.
   Approved batch3hc output: --boot-mode off --max-gap 25 --no-carried.

Video legend: green box = tracked detection, orange circle = interpolated, orange circle + white dot = carried,
cyan circle = attached, magenta ring = boot zone (BOOT_R around each ATD player), thin magenta box + "boot" = path
detection turned into a gap (strict), "before>after" = candidate penalised / rejected, thin red box = raw top
candidate the tracker rejected, blue circle = ATD ball (Metrica automatic tracking, NOT ground truth).
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

from run_tactical import MATCH_PX, MTAH

FPS = 25
PITCH = (105.3, 68.0)  # metres, as games.py
V_MAX = 30.0           # m/s, about the fastest a ball travels
R0 = 1.0               # m, slack for homography noise / blob centre jitter
MAX_GAP = 50           # frames (2 s) that may be skipped/interpolated
FREE_RESTART = 125     # frames (5 s) without a ball after which a track may restart anywhere for NEW_TRACK
BIAS = 0.0             # subtracted from every SVM score; 0.4 / 0.2 / 0.1 dropped too many real balls
                       # (they often score 0.0-0.3 on real footage)
SKIP_COST = 0.0        # per skipped frame inside a track; 0.1 (capped at NEW_TRACK for long gaps) made skipping a
                       # sparsely seen ball cheaper than following it (frames 11855-11995 all dropped)
CONTINUITY = 0.1       # bonus for linking consecutive frames
LAMBDA = 0.05          # per metre moved
NEW_TRACK = 4.0        # base cost of a jump / restart; must exceed the cost of the longest allowed gap
                       # (SKIP_COST * MAX_GAP + LAMBDA * max distance), or gaps never get bridged
JUMP_PER_M = 0.2       # extra jump cost per metre beyond the physically reachable distance
MIN_SCORE = -1.0       # candidates below this are ignored entirely
VEL_W = 0.0            # per m/s of velocity change beyond VEL_TOL (0 = off, --vel-w)
VEL_TOL = 8.0          # m/s; blob-centre jitter at the feet alone is ~0.3 m/frame = 7.5 m/s
VEL_SMOOTH = 0.5       # node velocity = VEL_SMOOTH * link velocity + (1 - VEL_SMOOTH) * predecessor's
INTERP_MODE = "hermite"  # or "linear" (--interp); hermite = curve matching the ball's velocity at both ends
VEL_FIT = 5            # path detections used to fit the velocity at each end of a gap
INTERP_VMAX = 20.0     # m/s; hermite curves faster than this anywhere fall back to the straight line
BOOT_MODE = "strict"   # strict / penalty / reject / off (--boot-mode), see 3. above
BOOT_R = 0.4           # m; a candidate this close to an ATD player (about body radius) is likely a boot
BOOT_PEN = 0.5         # penalty mode: subtracted from the SVM score; a ball that fits the path still wins
CARRIED = True         # flag interpolated gaps carried by one player (--no-carried)
ATTACH = False         # --attach: such gaps follow the player's ATD position instead, status "attached"
ATTACH_R = 1.0         # m; detections on both sides of the gap must be this close to the followed player
FOLLOW_STEP = 0.5      # m/frame (12.5 m/s) the followed player may move; two players within it = ambiguous
FOLLOW_MISS = 5        # frames the followed player may be missing from the ATD

TRACKED, INTERP, REJECTED, ATD_C, PITCH_C, TEXT = (0, 255, 0), (0, 165, 255), (0, 0, 255), (255, 128, 0), (0, 255, 255), (255, 255, 255)
ATTACHED = (255, 255, 0)
BOOT_C, BOOTED = (255, 0, 255), (255, 0, 255)  # boot zone circles; penalised candidates


def load_candidates(path):
    by_frame = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            by_frame[int(r["frame"])].append(dict(bbox=tuple(int(r[k]) for k in ("x1", "y1", "x2", "y2")),
                                                  u=float(r["u"]), v=float(r["v"]), score=float(r["score"])))
    return by_frame


def load_players(path, first, last):
    """ATD player positions in pitch metres, (n_frames, slots, 2), NaN where missing; row 0 = video frame `first`.
    Slot ids are not used anywhere: they swap between players."""
    rows = []
    with open(path) as f:
        for line in f:
            fid = int(line[:line.index(":")])
            if fid - 1 < first:
                continue
            if fid - 1 > last:
                break
            rows.append([[float(v) for v in g.split(",")[:2]] for g in line.split(":")[1].split(";")])
    return np.array(rows) * np.array(PITCH)


def nearest_player(players, t, first, xy):
    """(distance m, position) of the ATD player closest to xy (metres) at frame t; (inf, None) if none."""
    d = np.linalg.norm(players[t - first] - xy, axis=1)
    if np.all(np.isnan(d)):
        return np.inf, None
    i = int(np.nanargmin(d))
    return float(d[i]), players[t - first, i]


def penalise_boots(by_frame, players, first, last):
    """Penalty / reject mode: lower the score of candidates within BOOT_R of a player, or drop them."""
    n = 0
    for t in range(first, last + 1):
        for c in by_frame.get(t, []):
            if nearest_player(players, t, first, np.array([c["u"], c["v"]]) * PITCH)[0] < BOOT_R:
                c["svm"], c["score"] = c["score"], -np.inf if BOOT_MODE == "reject" else c["score"] - BOOT_PEN  # -inf < MIN_SCORE
                n += 1
    return n


def gap_boots(path, players, first):
    """Strict mode: remove path detections within BOOT_R of a player, relinking the path across them.
    Returns the removed frames."""
    removed = []
    for t in sorted(path):
        c, pt, jump = path[t]
        if nearest_player(players, t, first, np.array([c["u"], c["v"]]) * PITCH)[0] < BOOT_R:
            c["boot_gap"] = True
            removed.append(t)
    for t in removed:
        _, pt, jump = path.pop(t)
        for f, (c2, p2, j2) in path.items():
            if p2 == t:
                path[f] = (c2, pt, j2 or jump)
    return removed


def follow(players, first, a, b, ball_a, ball_b):
    """Positions (metres) of the player carrying the ball from path frame a to b, or None.

    Followed by position, frame to frame (never by slot id): the one player within FOLLOW_STEP of where he
    was; missing for up to FOLLOW_MISS frames holds his last position; two candidates = ambiguous = None.
    """
    d, pos = nearest_player(players, a, first, ball_a)
    if d > ATTACH_R:
        return None
    out, miss = [], 0
    for s in range(a, b + 1):
        near = np.linalg.norm(players[s - first] - pos, axis=1) <= FOLLOW_STEP * (miss + 1)
        if near.sum() > 1:
            return None
        if near.sum() == 1:
            pos, miss = players[s - first][near][0], 0
        else:
            miss += 1
            if miss > FOLLOW_MISS:
                return None
        out.append(pos)
    return out if np.linalg.norm(out[-1] - ball_b) <= ATTACH_R else None


def best_path(by_frame, first, last):
    """Returns {frame: (node, prev_frame or None, jumped)} for the frames on the optimal path."""
    nodes = []          # (frame, cand)
    frame_nodes = {}    # frame -> np.array of node ids
    for t in range(first, last + 1):
        ids = []
        for c in by_frame.get(t, []):
            if c["score"] >= MIN_SCORE:
                ids.append(len(nodes))
                nodes.append((t, c))
        frame_nodes[t] = np.array(ids, int)
    n = len(nodes)
    pos = np.array([[c["u"] * PITCH[0], c["v"] * PITCH[1]] for _, c in nodes]).reshape(n, 2)
    gain = np.array([c["score"] for _, c in nodes]) - BIAS
    best = np.full(n, -np.inf)
    prev = np.full(n, -1)          # linked predecessor node, -1 = none
    jumped = np.zeros(n, bool)     # reached from prev by a jump (not physically reachable)
    start_from = np.full(n, -1)    # for new segments: best node before (or -1 = empty prefix)
    vel = np.full((n, 2), np.nan)  # smoothed arrival velocity (m/s) along each node's best path

    prefix = {}                    # frame -> (best path value over frames <= frame, its end node); empty path = 0
    run_val, run_node = 0.0, -1
    for t in range(first, last + 1):
        cur = frame_nodes[t]
        if len(cur):
            free_val, free_node = prefix.get(t - FREE_RESTART - 1, (0.0, -1))
            cand_val = np.full(len(cur), free_val - NEW_TRACK)
            cand_prev = np.full(len(cur), -1)
            cand_jump = np.zeros(len(cur), bool)
            cand_v = np.full((len(cur), 2), np.nan)
            for k in range(1, FREE_RESTART + 1):
                pk = frame_nodes.get(t - k)
                if pk is None or not len(pk):
                    continue
                d = np.linalg.norm(pos[cur][:, None, :] - pos[pk][None, :, :], axis=2)
                reach = V_MAX * k / FPS + R0
                gap_cost = SKIP_COST * (k - 1) if k <= MAX_GAP + 1 else min(SKIP_COST * (k - 1), NEW_TRACK)
                link = best[pk][None, :] - gap_cost + (CONTINUITY if k == 1 else 0) - LAMBDA * d
                v_link = (pos[cur][:, None, :] - pos[pk][None, :, :]) * FPS / k
                if VEL_W > 0 and k <= MAX_GAP + 1:
                    dv = np.nan_to_num(np.linalg.norm(v_link - vel[pk][None, :, :], axis=2), nan=0.0)
                    link = link - VEL_W * np.maximum(0.0, dv - VEL_TOL)
                jump = best[pk][None, :] - NEW_TRACK - JUMP_PER_M * (d - reach)
                is_jump = d > reach
                val = np.where(is_jump, jump, link)
                j = np.argmax(val, axis=1)
                rows = np.arange(len(cur))
                v = val[rows, j]
                better = v > cand_val
                cand_val[better] = v[better]
                cand_prev[better] = pk[j[better]]
                cand_jump[better] = is_jump[rows, j][better]
                ok = ~is_jump[rows, j] & (k <= MAX_GAP + 1)  # velocity only means something over short real links
                cand_v[better] = np.where(ok[better][:, None], v_link[rows, j][better], np.nan)
            best[cur] = gain[cur] + cand_val
            prev[cur] = cand_prev
            jumped[cur] = cand_jump
            pv = np.where(cand_prev[:, None] >= 0, vel[np.maximum(cand_prev, 0)], np.nan)
            vel[cur] = np.where(np.isnan(pv), cand_v, VEL_SMOOTH * cand_v + (1 - VEL_SMOOTH) * pv)
            start_from[cur] = np.where(cand_prev < 0, free_node, -1)
        if len(cur) and best[cur].max() > run_val:
            run_val, run_node = float(best[cur].max()), int(cur[np.argmax(best[cur])])
        prefix[t] = (run_val, run_node)

    path = {}
    node = run_node
    while node >= 0:
        t, c = nodes[node]
        p = int(prev[node])
        path[t] = (c, nodes[p][0] if p >= 0 else None, bool(jumped[node]))
        node = p if p >= 0 else int(start_from[node])
    return path


def build_track(path, matrices, project, first, last, players=None):
    """Per-frame dict: status det/interp/carried/attached/none, u, v (normalised), px (x, y), score, bbox.

    attached: a gap (any length the path links, not a jump) whose detections on both sides sit at the same
    position-followed player; the ball keeps its offset from him, blended linearly from one end to the other.
    """
    track = {t: dict(status="none") for t in range(first, last + 1)}
    frames = sorted(path)
    scale = np.array(PITCH)
    for t, (c, pt, jump) in path.items():
        x1, y1, x2, y2 = c["bbox"]
        track[t] = dict(status="det", u=c["u"], v=c["v"], px=((x1 + x2) / 2, (y1 + y2) / 2), score=c["score"], bbox=c["bbox"])
        if ATTACH and players is not None and pt is not None and not jump and t - pt > 1:
            a = path[pt][0]
            ball_a, ball_b = np.array([a["u"], a["v"]]) * scale, np.array([c["u"], c["v"]]) * scale
            carrier = follow(players, first, pt, t, ball_a, ball_b)
            if carrier is not None:
                off_a, off_b = ball_a - carrier[0], ball_b - carrier[-1]
                for s in range(pt + 1, t):
                    w = (s - pt) / (t - pt)
                    u, v = (carrier[s - pt] + off_a + w * (off_b - off_a)) / scale
                    track[s] = dict(status="attached", u=float(u), v=float(v), px=tuple(float(z) for z in project(matrices[s], u, v)))
                continue
        if pt is not None and not jump and 1 < t - pt <= MAX_GAP + 1:  # jumps and long gaps stay "no ball"
            a = path[pt][0]
            curve = hermite(path, frames, pt, t) if INTERP_MODE == "hermite" else None
            status = "interp"
            if CARRIED and players is not None and follow(players, first, pt, t, np.array([a["u"], a["v"]]) * scale,
                                                          np.array([c["u"], c["v"]]) * scale) is not None:
                status = "carried"
            for s in range(pt + 1, t):
                if curve is not None:
                    u, v = curve[s - pt - 1]
                else:
                    w = (s - pt) / (t - pt)
                    u, v = a["u"] + w * (c["u"] - a["u"]), a["v"] + w * (c["v"] - a["v"])
                track[s] = dict(status=status, u=u, v=v, px=tuple(float(z) for z in project(matrices[s], u, v)))
    return track


def end_velocity(path, frames, t, side):
    """Ball velocity (normalised pitch units/frame) at path frame t, fitted over up to VEL_FIT detections
    of the same linked segment before (side -1) or after (side +1) it. None if fewer than 2."""
    i = frames.index(t)
    pts = [t]
    while len(pts) < VEL_FIT:
        if side < 0:
            f = pts[-1]
            p = path[f][1]
            if p is None or path[f][2] or f - p > MAX_GAP + 1:
                break
            pts.append(p)
        else:
            i += 1
            if i >= len(frames):
                break
            f = frames[i]
            if path[f][1] != pts[-1] or path[f][2] or f - pts[-1] > MAX_GAP + 1:
                break
            pts.append(f)
    if len(pts) < 2:
        return None
    ts = np.array(pts, float)
    uv = np.array([[path[f][0]["u"], path[f][0]["v"]] for f in pts])
    return np.polyfit(ts, uv, 1)[0]


def hermite(path, frames, a, b):
    """Cubic Hermite points (u, v) for frames a+1..b-1, or None to fall back to the straight line."""
    va, vb = end_velocity(path, frames, a, -1), end_velocity(path, frames, b, +1)
    if va is None or vb is None:
        return None
    k = b - a
    A = np.array([path[a][0]["u"], path[a][0]["v"]])
    B = np.array([path[b][0]["u"], path[b][0]["v"]])
    s = np.arange(0, k + 1)[:, None] / k
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    pts = h00 * A + h10 * k * va + h01 * B + h11 * k * vb
    step = np.diff(pts, axis=0) * np.array(PITCH) * FPS
    if np.linalg.norm(step, axis=1).max() > INTERP_VMAX:
        return None
    return [tuple(map(float, p)) for p in pts[1:-1]]


def raw_top(by_frame, t):
    cs = by_frame.get(t, [])
    best = max(cs, key=lambda c: c.get("svm", c["score"])) if cs else None  # svm = score before the boot penalty
    return best if best and best.get("svm", best["score"]) > 0 else None


def summarise(ref, by_frame, track):
    def rate(a, b):
        return f"{a}/{b} = {a / b:.1%}" if b else "n/a"

    def agrees(px, r):
        return px is not None and r["atd_x"] != "" and np.hypot(px[0] - float(r["atd_x"]), px[1] - float(r["atd_y"])) < MATCH_PX

    outputs = {
        "raw top-1 (before)": {t: (lambda c: None if c is None else ((c["bbox"][0] + c["bbox"][2]) / 2, (c["bbox"][1] + c["bbox"][3]) / 2))(raw_top(by_frame, t)) for t in ref},
        "tracked, detections only": {t: track[t]["px"] if track[t]["status"] == "det" else None for t in ref},
        "tracked + interpolated": {t: track[t].get("px") for t in ref},
    }
    alive = [t for t, r in ref.items() if r["gt"] in ("H", "A")]
    dead = [t for t, r in ref.items() if r["gt"] == "D"]
    atd = [t for t, r in ref.items() if r["atd_x"] != ""]
    print(f"\n{'':28s} {'out: alive':>22s} {'out: dead':>22s} {'agrees w/ ATD':>22s} {'outputs agreeing':>22s}")
    for name, out in outputs.items():
        outs_atd = [t for t in atd if out[t] is not None]
        print(f"{name:28s} {rate(sum(out[t] is not None for t in alive), len(alive)):>22s} "
              f"{rate(sum(out[t] is not None for t in dead), len(dead)):>22s} "
              f"{rate(sum(agrees(out[t], ref[t]) for t in atd), len(atd)):>22s} "
              f"{rate(sum(agrees(out[t], ref[t]) for t in outs_atd), len(outs_atd)):>22s}")
    n_interp = sum(track[t]["status"] == "interp" for t in ref)
    n_attached = sum(track[t]["status"] == "attached" for t in ref)
    n_carried = sum(track[t]["status"] == "carried" for t in ref)
    changed = sum(1 for t in ref if raw_top(by_frame, t) is not None and
                  (track[t]["status"] != "det" or track[t]["bbox"] != raw_top(by_frame, t)["bbox"]))
    print(f"\ninterpolated frames: {n_interp + n_carried} (carried {n_carried})   attached frames: {n_attached}   raw top-1 replaced/removed by tracker: {changed}")
    print("(ATD = Metrica automatic tracking, not ground truth: 'agrees' is agreement, not accuracy)")


def boot_ring(frame, H, xy, project):
    """BOOT_R circle around a player (pitch metres), projected to pixels."""
    pts = [project(H, (xy[0] + BOOT_R * np.cos(a)) / PITCH[0], (xy[1] + BOOT_R * np.sin(a)) / PITCH[1])
           for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
    cv2.polylines(frame, [np.array(pts).round().astype(np.int32)], True, BOOT_C, 1, cv2.LINE_AA)


def render(args, track, by_frame, ref, matrices, LH, OV, players=None):
    print(f"rendering video, skipping to frame {args.first} ...")
    cap = OV.open_video_at(args.first)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    path = os.path.join(args.out_dir, f"tracked_{args.first}_{args.last}{args.tag}.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    started = time.time()
    for t in range(args.first, args.last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        cv2.polylines(frame, [np.array(LH.pitch_corners(matrices[t])).round().astype(np.int32)], True, PITCH_C, 1, cv2.LINE_AA)
        if players is not None and BOOT_MODE != "off":
            for xy in players[t - args.first]:
                if not np.isnan(xy[0]):
                    boot_ring(frame, matrices[t], xy, LH.project)
            for c in by_frame.get(t, []):
                if c.get("boot_gap"):  # strict: path detection turned into a gap
                    x1, y1, x2, y2 = c["bbox"]
                    cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), BOOTED, 1)
                    cv2.putText(frame, "boot", (x2 + 5, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BOOTED, 1, cv2.LINE_AA)
                if "svm" in c:  # penalised: thin magenta box + its score before -> after
                    x1, y1, x2, y2 = c["bbox"]
                    cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), BOOTED, 1)
                    cv2.putText(frame, f"{c['svm']:.2f}>{c['score']:.2f}", (x2 + 5, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BOOTED, 1, cv2.LINE_AA)
        r = ref.get(t)
        if r and r["atd_x"] != "":
            cv2.circle(frame, (int(float(r["atd_x"])), int(float(r["atd_y"]))), 14, ATD_C, 2)
        raw, tr = raw_top(by_frame, t), track[t]
        if raw is not None and (tr["status"] != "det" or tr["bbox"] != raw["bbox"]):
            x1, y1, x2, y2 = raw["bbox"]
            cv2.rectangle(frame, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), REJECTED, 1)
        if tr["status"] == "det":
            x1, y1, x2, y2 = tr["bbox"]
            cv2.rectangle(frame, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), TRACKED, 2)
        elif tr["status"] in ("interp", "carried", "attached"):
            cv2.circle(frame, (int(tr["px"][0]), int(tr["px"][1])), 10, ATTACHED if tr["status"] == "attached" else INTERP, 2)
            if tr["status"] == "carried":
                cv2.circle(frame, (int(tr["px"][0]), int(tr["px"][1])), 3, TEXT, -1)
        status = f"frame {t}  GT {r['gt'] if r else 'X'}  BALL {tr['status'].upper()}" + \
                 (f" {tr['score']:.2f}" if tr["status"] == "det" else "") + ("  (raw pick rejected)" if raw is not None and (tr["status"] != "det" or tr["bbox"] != raw["bbox"]) else "")
        cv2.rectangle(frame, (0, 1040), (900, 1080), (0, 0, 0), -1)
        cv2.putText(frame, status, (10, 1068), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT, 2, cv2.LINE_AA)
        writer.write(frame)
        done = t - args.first + 1
        if done % 500 == 0:
            fps = done / (time.time() - started)
            print(f"  {done}/{args.last - args.first + 1} frames  {fps:.1f} fps  ~{(args.last - t) / fps / 60:.0f} min left")
    cap.release()
    writer.release()
    print("wrote", path)


def main(args):
    sys.path[:0] = [args.mtah, os.path.join(args.mtah, "starter"), os.path.join(args.mtah, "ball_yolo")]
    import load_homography as LH
    import overlay as OV

    stem = os.path.join(args.out_dir, args.stem)
    by_frame = load_candidates(stem + "_candidates.csv")
    with open(stem + ".csv") as f:
        ref = {int(r["frame"]): r for r in csv.DictReader(f) if args.first <= int(r["frame"]) <= args.last}
    matrices, _, _ = LH.load()
    from games import require
    players = None
    if BOOT_MODE != "off" or ATTACH or CARRIED:
        # Metrica's original ATD: the live file carries our own ball in the public window (players identical)
        players = load_players(require(["atd_raw"])[0].replace("_atd.txt", "_atd_original.txt"), args.first, args.last)
    if BOOT_MODE in ("penalty", "reject"):
        print(f"boot {'rejected' if BOOT_MODE == 'reject' else 'penalty on'} {penalise_boots(by_frame, players, args.first, args.last)} candidates")

    path = best_path(by_frame, args.first, args.last)
    if BOOT_MODE == "strict":
        print(f"boot strict: {len(gap_boots(path, players, args.first))} path detections turned into gaps")
    track = build_track(path, matrices, LH.project, args.first, args.last, players)

    with open(os.path.join(args.out_dir, f"tracked_{args.first}_{args.last}{args.tag}.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "status", "u", "v", "px_x", "px_y", "score"])
        for t in range(args.first, args.last + 1):
            tr = track[t]
            wr.writerow([t, tr["status"]] + ([f"{tr['u']:.5f}", f"{tr['v']:.5f}", f"{tr['px'][0]:.1f}", f"{tr['px'][1]:.1f}",
                                               f"{tr['score']:.3f}" if "score" in tr else ""] if tr["status"] != "none" else [""] * 5))
    summarise(ref, by_frame, track)
    if not args.no_video:
        render(args, track, by_frame, ref, matrices, LH, OV, players)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mtah", default=MTAH)
    ap.add_argument("--first", type=int, default=11185)
    ap.add_argument("--last", type=int, default=18684)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--stem", default="classical_vs_atd_11185_18684",
                    help="run_tactical.py output to read; --first/--last may select a sub-range of it")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the output names, e.g. _skip0 (keeps earlier outputs)")
    ap.add_argument("--interp", choices=["linear", "hermite"], default=INTERP_MODE)
    ap.add_argument("--vel-w", type=float, default=VEL_W, help="velocity-change penalty per m/s (0 = off)")
    ap.add_argument("--boot-mode", choices=["strict", "penalty", "reject", "off"], default=BOOT_MODE)
    ap.add_argument("--boot-pen", type=float, default=BOOT_PEN, help="penalty mode: score penalty near an ATD player")
    ap.add_argument("--max-gap", type=int, default=MAX_GAP, help="longest gap (frames) filled by interpolation")
    ap.add_argument("--boot-r", type=float, default=BOOT_R, help="boot zone radius around a player, m")
    ap.add_argument("--no-carried", action="store_true", help="don't flag gaps carried by one player")
    ap.add_argument("--attach", action="store_true", help="carried gaps follow the player's ATD position")
    a = ap.parse_args()
    INTERP_MODE, VEL_W, BOOT_MODE, BOOT_PEN, BOOT_R = a.interp, a.vel_w, a.boot_mode, a.boot_pen, a.boot_r
    MAX_GAP, ATTACH, CARRIED = a.max_gap, a.attach, not a.no_carried
    a.out_dir = a.out_dir or os.path.join(a.mtah, "data", "ball_classical")
    main(a)
