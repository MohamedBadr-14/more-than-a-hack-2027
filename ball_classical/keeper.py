"""Keeper holds: a goalkeeper has the ball and our detector has lost it (ball in his hands, against his body).

During such a hold the tracker jumps to boots far away (e.g. 26690-26970). track.py --keeper-hold uses the holds
found here to keep the ball near the keeper; this file only finds them.

Rules (all from our ball track, our candidates and ATD player positions; ATD ball only as a release safety net):
- Only an open-play catch, never a dead ball: the catch frame must not be D in the v5f labels (--dead-csv, old
  track) and the hold ends where a D starts. (An arrival-speed test, arrives_fast(), was tried and dropped: it is
  measured on the first-pass track, which is on the boots exactly then, and rejected real catches.)
- Catch: our tracked ball (a detection) within CATCH_R of a goalkeeper standing in a penalty box, and within LOST
  frames after that the detector loses it: for LOST frames in a row no candidate within GATE_R of him scores > SEEN.
  A leave chain before the loss (save / parry / shot flying past) = no hold.
- Keeper: the ATD goalkeeper slot at the catch frame only; after that followed by position (slot ids swap).
- Release, first of:
  kick / throw: a leave chain = LEAVE_N linked candidates starting within GATE_R of him, moving away from him at
                >= LEAVE_V m/s each step (each step physically reachable), ending beyond GATE_R;
  seen again:   >= LOST - 2 of the last LOST frames have a candidate within GATE_R scoring > SEEN (the tracker can
                follow the ball on its own again; this is not the keeper releasing it);
  ATD away:     the raw ATD ball >= ATD_FAR from him for ATD_HOLD frames (safety net; the hold ends at the first of
                those frames). ATD never gives a ball position.
- Dead ball: the ball seen out (track status "out") from OUT_BEFORE before the catch to the end of the hold = the
  keeper fetching / placing a ball that went out, not a catch: no hold.
- Holds shorter than LOST frames are dropped; holds by the same keeper less than MERGE frames apart are merged.

    python keeper.py --first 11185 --last 90651 --track ../data/ball_classical/full_h1/tracked_11185_90651.csv \
        --stem ../data/ball_classical/full_h1/classical_vs_atd_11185_90651
"""
import argparse
import csv
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

FPS = 25
PITCH = np.array((105.3, 68.0))  # metres, as games.py
CATCH_R = 4.0          # m; a ball held ~1.5 m up projects up to ~4 m from the keeper on the pitch
GATE_R = 5.0           # m; candidates this close to the keeper belong to the hold
SEEN = 0.0             # candidate score above this near the keeper = the detector sees the ball
LOST = FPS // 2        # frames: detector-lost stretch that starts a hold / window of the seen-again release
LEAVE_V = 8.0          # m/s, each step of a leave chain (= touches.KICK_OUT)
LEAVE_N = 3            # steps in a leave chain
V_MAX, R0 = 30.0, 1.0  # as track.py: fastest ball, homography slack
ATD_FAR, ATD_HOLD = 6.0, FPS  # m, frames: ATD ball this far for this long = released
MERGE = FPS // 2       # frames: two holds by the same keeper closer than this = one hold
FOLLOW_STEP = 0.5      # m per frame the followed keeper may move (as track.py)
MIN_SCORE = -1.0       # as track.py: candidates below are ignored
OUT_BEFORE = 3 * FPS   # frames before the catch checked for a ball out (see hold_from)
ARRIVE, ARRIVE_V = FPS, 5.0  # frames, m/s: the ball must reach the keeper at least this fast (shot / cross / pass)
BOX_X, BOX_HW = 16.5, 20.16  # m, penalty box depth and half width


def goalkeeper_slots(meta_path):
    """ATD slot indices of the goalkeepers (same as possession/tagging.goalkeeper_slots)."""
    root = ET.parse(meta_path).getroot()
    team_ids = {team.get("id") for team in root.iter("Team")}
    slots = [p for p in root.iter("Player") if p.get("teamId") in team_ids]
    return [s for s, p in enumerate(slots)
            if any(q.findtext("Name") == "position_type" and q.findtext("Value") == "Goalkeeper"
                   for q in p.iter("ProviderParameter"))]


def load_atd(path, first, last):
    """players (n, slots, 2) and ball (n, 2) in metres, NaN where missing; row 0 = video frame `first`."""
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
    return np.array(players) * PITCH, np.array(ball) * PITCH


def in_box(p):
    return bool(np.isfinite(p).all() and abs(p[1] - PITCH[1] / 2) <= BOX_HW
                and (p[0] <= BOX_X or p[0] >= PITCH[0] - BOX_X))


class Scene:
    """Inputs of the hold search for frames first..last (index 0 = first)."""

    def __init__(self, first, status, ball, by_frame, players, atd_ball, gk_slots, dead=None):
        self.first, self.status, self.ball = first, status, ball  # ball: (n, 2) metres, NaN = none
        self.players, self.atd_ball, self.gk = players, atd_ball, gk_slots
        self.n = len(status)
        self.dead = np.zeros(self.n, bool) if dead is None else dead  # D in the v5f labels (old track)
        self.cands, self.scores = {}, {}
        for t, cs in by_frame.items():
            cs = [c for c in cs if c["score"] >= MIN_SCORE and not c.get("off")]
            if cs and 0 <= t - first < self.n:
                self.cands[t - first] = np.array([[c["u"], c["v"]] for c in cs]) * PITCH
                self.scores[t - first] = np.array([c["score"] for c in cs])

    def follow(self, i, pos):
        """Keeper position at i, i+1, ... followed by position (nearest player within FOLLOW_STEP per frame;
        missing = last position kept)."""
        miss = 0
        while i < self.n:
            d = np.linalg.norm(self.players[i] - pos, axis=1)
            if np.isfinite(d).any() and np.nanmin(d) <= FOLLOW_STEP * (miss + 1):
                pos, miss = self.players[i][np.nanargmin(d)], 0
            else:
                miss += 1
            yield pos
            i += 1

    def seen(self, i, kp):
        c = self.cands.get(i)
        return c is not None and bool(((np.linalg.norm(c - kp, axis=1) <= GATE_R) & (self.scores[i] > SEEN)).any())

    def leave_chain(self, i, kp):
        c0 = self.cands.get(i)
        if c0 is None:
            return False
        for cur in c0[np.linalg.norm(c0 - kp, axis=1) <= GATE_R]:
            for k in range(1, LEAVE_N + 1):
                c = self.cands.get(i + k)
                if c is None:
                    break
                step = np.linalg.norm(c - cur, axis=1)
                good = (step * FPS >= LEAVE_V) & (step <= V_MAX / FPS + R0) & \
                       (np.linalg.norm(c - kp, axis=1) > np.linalg.norm(cur - kp) + 0.1)
                if not good.any():
                    break
                cur = c[good][np.argmin(step[good])]
            else:
                if np.linalg.norm(cur - kp) > GATE_R:
                    return True
        return False

    def arrives_fast(self, i):
        """The ball reached frame i at >= ARRIVE_V: our track's mean speed over the last ARRIVE frames (a shot, cross
        or pass; a ball rolled / carried to the keeper at a stoppage arrives slowly)."""
        prev = [k for k in range(max(i - ARRIVE, 0), i - ARRIVE // 2) if np.isfinite(self.ball[k]).all()]
        if not prev:
            return False
        k = prev[0]
        return np.linalg.norm(self.ball[i] - self.ball[k]) / (i - k) * FPS >= ARRIVE_V

    def catch_slot(self, i):
        if self.status[i] != "det" or self.dead[i]:
            return None
        for s in self.gk:
            kp = self.players[i, s]
            if in_box(kp) and np.linalg.norm(self.ball[i] - kp) <= CATCH_R:
                return s
        return None

    def hold_from(self, i, s):
        """(end index, why, keeper positions i..end) of a hold caught at i by slot s, or None."""
        look = [(j, kp) for j, kp in zip(range(i, min(i + 2 * LOST + 1, self.n)), self.follow(i, self.players[i, s]))]
        sn = [self.seen(j, kp) for j, kp in look]
        lost_at = next((k for k in range(1, LOST + 1) if k + LOST <= len(sn) and not any(sn[k:k + LOST])), None)
        if lost_at is None or any(self.leave_chain(j, kp) for j, kp in look[:lost_at + 1]):
            return None
        kps, hist, far, why = [], [], 0, "end of range"
        j = i
        for kp in self.follow(i, self.players[i, s]):
            kps.append(kp)
            if j - i > lost_at and self.leave_chain(j, kp):
                why = "kick/throw"
                break
            if self.dead[j]:  # a dead ball starts: the catch logic does nothing inside it
                j -= 1
                why = "dead ball starts"
                break
            hist.append(self.seen(j, kp))
            if j - i > LOST and sum(hist[-LOST:]) >= LOST - 2:
                j -= LOST - 1
                why = "seen again"
                break
            a = self.atd_ball[j]
            far = far + 1 if np.isfinite(a).all() and np.linalg.norm(a - kp) > ATD_FAR else 0
            if far >= ATD_HOLD:
                j -= ATD_HOLD - 1
                why = "ATD away"
                break
            j += 1
            if j >= self.n:
                j -= 1
                break
        if "out" in self.status[max(i - OUT_BEFORE, 0):j + 1]:
            return j, "dead ball", None  # ball seen out: a keeper fetching / placing a dead ball, not a catch
        return j, why, kps[:j - i + 1]


def find_holds(scene):
    """[dict(a, b, why, slot, keeper)]: hold over indices a..b (inclusive), keeper = (b-a+1, 2) positions."""
    holds, i = [], 0
    while i < scene.n:
        s = scene.catch_slot(i)
        h = scene.hold_from(i, s) if s is not None else None
        if h is not None and h[2] is None:  # dead ball: skip the whole stretch (no catch inside it either)
            i = h[0] + 1
            continue
        if h is None or h[0] - i < LOST:
            i += 1
            continue
        b, why, kps = h
        prev = holds[-1] if holds else None
        if prev and prev["slot"] == s and i - prev["b"] <= MERGE:  # one catch, the ball glimpsed in between
            gap = [kps[0]] * (i - prev["b"] - 1)
            prev.update(b=b, why=why, keeper=np.vstack([prev["keeper"]] + gap + [np.array(kps)]) if gap else
                        np.vstack([prev["keeper"], np.array(kps)]), merged=prev.get("merged", 0) + 1)
        else:
            holds.append(dict(a=i, b=b, why=why, slot=s, keeper=np.array(kps)))
        i = b + 1
    return holds


def read_track(path, first, last):
    n = last - first + 1
    status, ball = ["none"] * n, np.full((n, 2), np.nan)
    with open(path) as f:
        for r in csv.DictReader(f):
            t = int(r["frame"]) - first
            if 0 <= t < n and r["status"] != "none":
                status[t], ball[t] = r["status"], (float(r["u"]) * PITCH[0], float(r["v"]) * PITCH[1])
    return status, ball


def read_dead(path, first, last):
    """Bool per frame first..last: label D in a possession CSV (v5f, old track); frames it lacks = False."""
    dead = np.zeros(last - first + 1, bool)
    with open(path) as f:
        for r in csv.DictReader(f):
            t = int(r["frame"]) - first
            if 0 <= t < len(dead):
                dead[t] = r["label"] == "D"
    return dead


def main(args):
    sys.path[:0] = [args.mtah]
    from games import require
    from track import load_candidates
    meta, raw = require(["atd_meta", "atd_raw"])
    players, atd_ball = load_atd(raw.replace("_atd.txt", "_atd_original.txt"), args.first, args.last)
    status, ball = read_track(args.track, args.first, args.last)
    by_frame = load_candidates(args.stem + "_candidates.csv")
    dead = read_dead(args.dead_csv, args.first, args.last) if args.dead_csv else None
    holds = find_holds(Scene(args.first, status, ball, by_frame, players, atd_ball, goalkeeper_slots(meta), dead))
    rows = []
    for k, h in enumerate(holds, 1):
        a, b = h["a"] + args.first, h["b"] + args.first
        t = h["a"] / FPS
        rows.append([k, a, b, round((b - a + 1) / FPS, 1), f"{int(t // 60)}:{t % 60:04.1f}", h["why"], h["slot"],
                     h.get("merged", 0)])
        print(f"{k:3d}. {a}-{b} ({(b - a + 1) / FPS:4.1f} s, video {rows[-1][4]}) release {h['why']}, gk slot {h['slot']}"
              + (f", merged {h['merged']}" if h.get("merged") else ""))
    print(len(holds), "holds")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n", "first", "last", "seconds", "video", "release", "gk_slot", "merged"])
            w.writerows(rows)
        print("wrote", args.out)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mtah", default=os.path.dirname(here))
    ap.add_argument("--first", type=int, default=11185)
    ap.add_argument("--last", type=int, default=90651)
    ap.add_argument("--track", required=True, help="track.py CSV (pass 1, without --keeper-hold)")
    ap.add_argument("--stem", required=True, help="run_tactical.py output stem (reads <stem>_candidates.csv)")
    ap.add_argument("--out", default=None, help="CSV of the holds")
    ap.add_argument("--dead-csv", default=None, help="possession CSV (v5f, old track): no catch inside its D")
    main(ap.parse_args())
