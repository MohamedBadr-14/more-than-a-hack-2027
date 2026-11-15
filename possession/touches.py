"""Touches: moments a team plays the ball (step 1 of the touch-based possession logic, used by possession_v3.py).

The ball's own path decides THAT something touched it; the players only decide WHO (ATD players are missing or
off by ~0.5 m too often to gate on). Interpolated and carried balls are treated exactly like detections.

0. Clean ball path: cut it where a step is impossible (> MAX_SPEED: ball in the air or tracker jump) or the source
   switches to / from ATD (different ball, metres apart); smooth each piece over +-SMOOTH frames. Velocities are a
   least-squares line through the window (one bad frame moves it little) and never cross a cut.
1. Ball change per frame: velocity in = over the W frames before, out = over the W frames after. The frame fires
   when any of:
   - turn: direction changes >= TURN deg, ball moving >= MIN_SPEED m/s in and out (slower = angle is noise),
     and the new direction holds over the next W frames too (a one-frame tracker jump does not)
   - slow down: in >= SLOW_IN m/s and out <= SLOW_FRAC * in (reception, block)
   - kick from slow: in < KICK_IN m/s and out >= KICK_OUT m/s
   A fast ball speeding up in a straight line does not fire: a ball in the air does that on screen.
2. Ball event: consecutive firing frames (holes <= BRIDGE) merged; its moment = the frame with the biggest
   velocity change.
3. Who: each team's nearest player over the event's frames.
   - one team within R: that team's touch (both within R: the nearer one, unless the two are within
     CONTEST_M of each other = contested, decided later by who comes out with the ball)
   - nobody within R: the same with the wider R_FAR, marked far (player position off by a metre or two)
   - nobody within R_FAR either: unknown, ignored (a bounce, a ball in the air, a missing player)
4. Hold: one player (followed by position, slot ids swap) within R for >= HOLD frames.
5. Bounce flag (review only, possession still counts these touches): a ball in the air is placed on the pitch too far along the
   camera's line of sight (the pitch direction of "up" in the image at the ball), so a bounce changes the
   velocity along that line. Ball events whose velocity change is within BOUNCE_COS of it are flagged.
"""
import numpy as np

import load_homography as LH
import possession as P

R = 1.5          # m, a player this close to a ball event made it
R_FAR = 3.0      # m, second chance when nobody is within R
CONTEST_M = 0.5  # m, both teams within R and this close in distance = contested
W = 3            # frames before / after for the ball velocity in / out
TURN = 25.0      # deg
MIN_SPEED = 2.0  # m/s, in and out, for the turn test
SLOW_IN, SLOW_FRAC = 5.0, 0.5  # m/s, fraction: slow down test
KICK_IN, KICK_OUT = 3.0, 8.0   # m/s: kick from slow test
BRIDGE = 1       # frames; firing frames this close are one event, reach holes this short are bridged in a hold
HOLD = 25        # frames (1 s)
SAME_PLAYER = 0.8  # m per frame; the nearest in-reach player moving more than this = another player
MAX_SPEED = 35.0   # m/s, a faster step between two frames is not the ball rolling (air / tracker jump)
SMOOTH = 2       # frames each side for the position smoothing
BOUNCE_COS = 0.98  # |cos| between the velocity change and the line of sight at or above this = bounce flag
NONE, BOTH = -1, 2
TESTS = ("turn", "slow down", "kick from slow")


def shift(x, k):
    """x moved k frames (x[t + k] at t), NaN beyond the ends."""
    out = np.full(x.shape, np.nan)
    if k >= 0:
        out[:len(x) - k] = x[k:]
    else:
        out[-k:] = x[:k]
    return out


def angle(u, v):
    """Angle (deg) between rows of u and v, NaN if either is unknown or zero."""
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (u * v).sum(1) / (np.hypot(*u.T) * np.hypot(*v.T))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def pieces(ball_m, status):
    """Piece id per frame: a new piece after every missing ball, impossible step or switch to / from ATD or to / from
    a ball out of play (status "out", track.py)."""
    step = np.hypot(*(ball_m - shift(ball_m, -1)).T) * P.FPS
    atd = np.array([s == "atd" for s in status])
    out = np.array([s == "out" for s in status])
    with np.errstate(invalid="ignore"):
        cut = ~(step <= MAX_SPEED) | (atd != np.roll(atd, 1)) | (out != np.roll(out, 1))
    cut[0] = True
    return np.cumsum(cut), cut


def smooth(ball_m, piece):
    """Mean over +-SMOOTH frames within the same piece."""
    total, count = np.zeros_like(ball_m), np.zeros(len(ball_m))
    for k in range(-SMOOTH, SMOOTH + 1):
        b, ok = shift(ball_m, k), shift(piece.astype(float), k) == piece
        ok &= ~np.isnan(b[:, 0])
        total[ok] += b[ok]
        count += ok
    with np.errstate(invalid="ignore"):
        return np.where(count[:, None] > 0, total / count[:, None], np.nan)


def fit(b, piece, lo, hi):
    """Velocity (m/s) at each frame: least-squares line through frames t+lo .. t+hi, NaN if any is missing or
    the window crosses a cut."""
    ks = np.arange(lo, hi + 1)
    wts = (ks - ks.mean()) / ((ks - ks.mean()) ** 2).sum()
    v = sum(wk * shift(b, k) for wk, k in zip(wts, ks)) * P.FPS
    same = shift(piece.astype(float), lo) == shift(piece.astype(float), hi)
    v[~same] = np.nan
    return v


def ball_change(ball_m, piece):
    """Per frame: v_in, v_out (n, 2) m/s, speeds in / out, turn (deg), each test (bool arrays) and fire.
    ball_m is the smoothed path, piece its piece ids."""
    vin, vout = fit(ball_m, piece, -W, 0), fit(ball_m, piece, 0, W)
    vnext = fit(ball_m, piece, W, 2 * W)  # the W frames after v_out
    s_in, s_out, s_next = np.hypot(*vin.T), np.hypot(*vout.T), np.hypot(*vnext.T)
    turn = angle(vin, vout)
    with np.errstate(invalid="ignore"):
        tests = {"turn": (s_in >= MIN_SPEED) & (s_out >= MIN_SPEED) & (turn >= TURN)
                         & (s_next >= MIN_SPEED) & (angle(vout, vnext) < TURN),
                 "slow down": (s_in >= SLOW_IN) & (s_out <= SLOW_FRAC * s_in),
                 "kick from slow": (s_in < KICK_IN) & (s_out >= KICK_OUT)}
    fire = np.logical_or.reduce(list(tests.values()))
    return dict(v_in=vin, v_out=vout, s_in=s_in, s_out=s_out, turn=turn, tests=tests, fire=fire,
                dv=np.hypot(*(vout - vin).T))


def spans(flag, bridge):
    """(a, b) runs of True, holes of <= bridge frames bridged."""
    out = []
    for t in np.flatnonzero(flag):
        if out and t - out[-1][1] - 1 <= bridge:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [tuple(s) for s in out]


def distances(d):
    """Per frame: distance (m) to each team's nearest player (n, 2), and the position (m) of the nearest player
    within R (NaN when none)."""
    ball_m, players_m = d["ball_n"] * P.PITCH, d["players_n"] * P.PITCH
    n = len(ball_m)
    teams_f = d.get("teams_f", np.tile(d["teams"], (n, 1)))  # team per frame per slot (possession_v3)
    d_team, who = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    for t in range(n):
        dist = np.hypot(*(players_m[t] - ball_m[t]).T)
        if np.all(np.isnan(dist)):
            continue
        for team in (P.HOME, P.AWAY):
            d_team[t, team] = np.nanmin(np.where(teams_f[t] == team, dist, np.nan), initial=np.inf)
        if np.nanmin(dist) < R:
            who[t] = players_m[t, int(np.nanargmin(dist))]
    return d_team, who


def team_of(dh, da, r=R):
    """Team of an event from each team's nearest distance: HOME/AWAY, BOTH (contested) or NONE (unknown)."""
    inside = [team for team, dm in ((P.HOME, dh), (P.AWAY, da)) if dm < r]
    if not inside:
        return NONE
    if len(inside) == 2 and abs(dh - da) < CONTEST_M:
        return BOTH
    return P.HOME if dh <= da else P.AWAY


def holds(d_team, who):
    """(a, b, team): one player within R (followed by position) for >= HOLD frames, the other team not."""
    n = len(d_team)
    with np.errstate(invalid="ignore"):
        cls = np.where(np.isnan(who[:, 0]), NONE, np.where(d_team[:, P.HOME] <= d_team[:, P.AWAY], P.HOME, P.AWAY))
        cls[(d_team < R).all(1)] = BOTH
    out, a = [], None
    for t in range(n + 1):
        same = (t < n and a is not None and cls[t] == cls[a] and cls[t] not in (NONE, BOTH)
                and np.hypot(*(who[t] - who[t - 1])) <= SAME_PLAYER)
        if not same:
            if a is not None and cls[a] not in (NONE, BOTH) and t - a >= HOLD:
                out.append((a, t - 1, int(cls[a])))
            a = t if t < n else None
    return out


def line_of_sight_cos(d, events):
    """|cos| between each ball event's velocity change and the camera's line of sight at the ball (NaN if
    unknown), from that frame's homography: the pitch shift of the ball's pixel moved up the image."""
    matrices, _, _ = LH.load()
    out = []
    for e in events:
        b = d["ball_n"][e["t"]]
        H = matrices[d["frames"][e["t"]]]
        px, py = LH.project(H, *b)
        los = (np.array(LH.unproject(H, px, py - 20)) - b) * P.PITCH
        dv = e["v_out"] - e["v_in"]
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(abs(los @ dv) / (np.hypot(*los) * np.hypot(*dv)))
    return out


def touches(d):
    """Events, ordered by time: dicts with a, b (indices), t (the moment), team (HOME/AWAY/BOTH/NONE),
    kind (touch/contested/unknown/hold), why, d_h / d_a (nearest of each team, m), and for ball events the
    speed in / out (m/s), turn (deg) and velocity vectors at the moment.
    Also the per-frame arrays the debug view shows (ball_change(), d_team, speed)."""
    piece, cut = pieces(d["ball_n"] * P.PITCH, d["status"])
    ball_m = smooth(d["ball_n"] * P.PITCH, piece)
    bc = ball_change(ball_m, piece)
    d_team, who = distances(d)
    out = []
    for a, b in spans(bc["fire"], BRIDGE):
        t = a + int(np.nanargmax(bc["dv"][a:b + 1]))
        dh, da = np.nanmin(d_team[a:b + 1, P.HOME], initial=np.inf), np.nanmin(d_team[a:b + 1, P.AWAY], initial=np.inf)
        team = team_of(dh, da)
        far = team == NONE
        if far:
            team = team_of(dh, da, R_FAR)
        kind = {NONE: "unknown", BOTH: "contested"}.get(team, "touch")
        why = ", ".join(k for k in TESTS if bc["tests"][k][a:b + 1].any())
        out.append(dict(a=a, b=b, t=t, team=team, kind=kind, far=far and team != NONE, why=why, d_h=dh, d_a=da, s_in=bc["s_in"][t],
                        s_out=bc["s_out"][t], turn=bc["turn"][t], v_in=bc["v_in"][t], v_out=bc["v_out"][t]))
    for e, c in zip(out, line_of_sight_cos(d, out)):
        e["los_cos"], e["bounce"] = c, bool(c >= BOUNCE_COS)
    for a, b, team in holds(d_team, who):
        out.append(dict(a=a, b=b, t=a, team=team, kind="hold", far=False, bounce=False, why=f"{b - a + 1} frames",
                        d_h=np.nanmin(d_team[a:b + 1, P.HOME]), d_a=np.nanmin(d_team[a:b + 1, P.AWAY])))
    out = [e for e in out if d["status"][e["t"]] != "out"]  # ball out of play: nobody plays it (deadball.py uses it)
    out.sort(key=lambda e: e["t"])
    return out, dict(bc, d_team=d_team, speed=np.hypot(*fit(ball_m, piece, -2, 2).T), cut=cut)


def per_frame(events, n, kinds):
    """Index of the event of one of `kinds` each frame is in (-1 none)."""
    idx = np.full(n, -1)
    for k, e in enumerate(events):
        if e["kind"] in kinds:
            idx[e["a"]:e["b"] + 1] = k
    return idx


if __name__ == "__main__":
    import argparse
    import os
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--first", type=int, default=11185)
    ap.add_argument("--last", type=int, default=18684)
    ap.add_argument("--track", default=os.path.join(P.MTAH, "data", "ball_classical", "strict_r04_gap2_carried", "tracked_11185_18684.csv"))
    a = ap.parse_args()
    d = P.compute(a.first, a.last, a.track)
    ev, _ = touches(d)
    kinds = {k: [e for e in ev if e["kind"] == k] for k in ("touch", "contested", "unknown", "hold")}
    print({k: len(v) for k, v in kinds.items()})
    for k in ("touch", "hold"):
        print(k, {P.LABEL[tm]: sum(e["team"] == tm for e in kinds[k]) for tm in (P.HOME, P.AWAY)})
    print("tests", {k: sum(k in e["why"] for e in ev if e["kind"] != "hold") for k in TESTS})
