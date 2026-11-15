"""Dead ball (D): stoppages found from the restart backwards (used by possession_v4.py; terms in ../GLOSSARY.md).

No smart tagging here (only the kick-off detection of possession_v3 uses its kick-off tags).

1. Wait (the pause): players have slowed down (median player speed, 1 s mean, < SLOW) for >= WAIT frames, plus at
   least one of
   - ball still (ball speed, BALL_MEAN mean, < STILL),
   - crowded end (>= CROWD players between one goal line and the 16.5 m line, full width: the corner shape, incl.
     the taker and players beside the box; the ball is often tracked on feet there, so this does not need the ball),
   - whistled: a referee whistle (whistle.py) ended at most WH_LEAD before the players' slow stretch began (or while
     it was going on); the whole slow stretch after the whistle counts,
   - outed: the ball was seen over a line (track.py "out") and no real kick happened since; the first slow stretch
     after it counts, however long the players take to slow down (goal kicks: they jog back first).
   A fast ball (>= FAST, measured: not an interpolated / carried ball, whose speed the tracker invents) ends a
   wait: the restart kick. Corners: while the box is crowded, a ball inside that box is
   players' feet (the tracker), not the ball: it neither ends the wait nor counts as the restart kick. After a
   whistle or the ball out, only a real kick (touch "kick from slow" by one team) ends it: setting up, players
   move / carry the ball.
2. Restart = the first kick after the wait (a touch by one team within LOOK of the wait's end, not in the crowded box);
   possession goes to the kicker's team from the kick. to_mate = the next touch / hold is a teammate's (a kick to an
   opponent still gives the kicker's team the restart; the v3 rules change it after). Kick-off detected in the wait ->
   the kick-off. No kick -> restart = frame after the wait, team = first hold within LOOK, else the team nearest the
   ball, else no dead ball.
4. Dead ball start, walking back from the wait start, after the last controlled sequence was confirmed (and after the
   previous restart): the first frame the ball is beyond a line (seen out, or on the in -> out path, track.py); else
   the frame after the end of the last referee whistle (whistle.py); else when play visibly stops: the start of the
   ball-still run that leads into the wait, or the wait start (players slow), whichever is earlier. (Not the foul itself: dead ball starts
   when play stops.) Late is safe: it only ends the previous spell late.
5. Keeper: when the ball rests with a team's goalkeeper in its own box (a catch is in play), the dead ball needs a
   ball seen over a line or a whistle (step 4).
"""
import warnings

import numpy as np

import possession as P
import possession_v3 as V3
import touches as T

SLOW = 1.4       # m/s, median player speed (1 s mean) below this = players slowed down
STILL = 1.0      # m/s, ball speed (BALL_MEAN mean) below this = ball still
BALL_MEAN = P.FPS // 2  # frames (0.5 s, centred): short, so the still ball ends close to the kick
FAST = 8.0       # m/s, ball speed at or above this ends a wait (= touches.KICK_OUT)
CROWD = 14       # players in one end strip (goal line to the 16.5 m line, full width): ~2/3 of the 22
WAIT = 2 * P.FPS  # frames, shortest wait
BRIDGE = P.FPS // 2  # frames, holes this short inside a wait are bridged
LOOK = 2 * P.FPS  # frames after the restart searched for the restart touch
BOX_X, BOX_HALF_W = 16.5, 20.16  # m, penalty box depth and half width
WH_LEAD = 3 * P.FPS  # frames: players slowing down this soon after a referee whistle = a whistled pause


def mean_1s(x, n=P.FPS):
    """Centred mean over n frames (default 1 s) ignoring NaN (NaN where all n are NaN)."""
    k = np.ones(n)
    ok = np.isfinite(x)
    s = np.convolve(np.where(ok, x, 0), k, "same")
    c = np.convolve(ok.astype(float), k, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan)


def player_speed(players_m):
    """Per frame median speed (m/s) over the players seen in this and the previous frame."""
    step = np.hypot(*np.diff(players_m, axis=0).transpose(2, 0, 1)) * P.FPS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames
        med = np.nanmedian(step, axis=1)
    return np.r_[med[:1], med]


def in_box(xy, left):
    """Bool array: points inside the left (left=True) or right penalty box."""
    x, y = xy[..., 0], xy[..., 1]
    with np.errstate(invalid="ignore"):
        yb = np.abs(y - P.PITCH[1] / 2) < BOX_HALF_W
        return yb & ((x < BOX_X) if left else (x > P.PITCH[0] - BOX_X))


def whistled(d, slow):
    """Bool per frame: inside the players' slow stretch that follows a referee whistle (from the whistle's end)."""
    out = np.zeros(len(slow), bool)
    runs = [(a, b) for a, b in T.spans(slow, BRIDGE) if b - a + 1 >= WAIT]  # blips are not a slowdown
    for w in d.get("whistles", []):
        if not w["ref"]:
            continue
        e = w["b"] - int(d["frames"][0])
        run = [(a, b) for a, b in runs if b >= e and a <= e + WH_LEAD][:1]
        if run:
            out[max(e + 1, 0):run[0][1] + 1] = True
    return out


def outed(over_line, slow, kick):
    """Bool per frame: inside the first players' slow stretch after the ball was seen over a line, with no real kick
    between the ball going out and that stretch (from the first out frame)."""
    out = np.zeros(len(slow), bool)
    runs = [(a, b) for a, b in T.spans(slow, BRIDGE) if b - a + 1 >= WAIT]  # blips are not a slowdown
    for e, _ in T.spans(over_line, BRIDGE):
        run = [(a, b) for a, b in runs if b >= e][:1]
        if run and not kick[e:max(run[0][0], e)].any():
            out[e:run[0][1] + 1] = True
    return out


def kicks(d, feet):
    """Bool per frame: a real kick (touch 'kick from slow' by one team, not feet in a crowded box)."""
    out = np.zeros(len(feet), bool)
    for e in d["touches"]:
        if e["kind"] == "touch" and "kick from slow" in e["why"] and not feet[e["t"]]:
            out[e["a"]:e["b"] + 1] = True
    return out


def features(d):
    """Per-frame arrays: pspeed, slow, bspeed, still, crowd (busiest end strip count), crowded, feet, whistled, kick, fast,
    over_line."""
    ball, players = d["ball_n"] * P.PITCH, d["players_n"] * P.PITCH
    pspeed = mean_1s(player_speed(players))
    speed = d["tinfo"]["speed"]
    bspeed = mean_1s(speed, BALL_MEAN)
    with np.errstate(invalid="ignore"):
        n_left, n_right = (players[..., 0] < BOX_X).sum(1), (players[..., 0] > P.PITCH[0] - BOX_X).sum(1)
    crowd = np.maximum(n_left, n_right)
    crowded = crowd >= CROWD
    feet = crowded & np.where(n_left >= n_right, in_box(ball, True), in_box(ball, False))  # ball in that end's penalty box = feet
    slow = pspeed < SLOW
    wh, kick = whistled(d, slow), kicks(d, feet)
    with np.errstate(invalid="ignore"):
        to_line = np.minimum.reduce([ball[:, 0], P.PITCH[0] - ball[:, 0], ball[:, 1], P.PITCH[1] - ball[:, 1]])
        seen = np.asarray(d["status"]) != "none"
        over_line = (to_line < 0) & seen
        ob = outed(over_line, slow, kick)
        held = wh | ob  # set-piece pauses: only a real kick ends them
        measured = ~np.isin(np.asarray(d["status"]), ["interp", "carried"])  # invented speeds never end a wait
        fast = ((speed >= FAST) & measured & ~feet & (~held | kick)) | (held & kick)  # whistle / out: a real kick
        return dict(pspeed=pspeed, slow=slow, bspeed=bspeed, still=bspeed < STILL, crowd=crowd, crowded=crowded,
                    feet=feet, whistled=wh, outed=ob, kick=kick, fast=fast, over_line=over_line)


def waits(ft):
    """(a, b) index spans of waits; short holes are bridged, but never across a fast ball / kick (it ends the wait)."""
    flag = ft["slow"] & (ft["still"] | ft["crowded"] | ft["whistled"] | ft["outed"]) & ~ft["fast"]
    out = []
    for a, b in T.spans(flag, BRIDGE):
        fast = np.flatnonzero(ft["fast"][a:b + 1])
        if len(fast):
            b = a + int(fast[0]) - 1
        if b - a + 1 >= WAIT:
            out.append((a, b))
    return out


def keeper_rest(d, a, b, gk_slots, home_left):
    """True when, over most of the wait, the ball's nearest player is a goalkeeper within touches.R in its own box."""
    ball, players = d["ball_n"] * P.PITCH, d["players_n"] * P.PITCH
    hits = 0
    for t in range(a, b + 1):
        dist = np.hypot(*(players[t] - ball[t]).T)
        if np.all(np.isnan(dist)):
            continue
        s = int(np.nanargmin(dist))
        if s in gk_slots and dist[s] < T.R:
            own_left = bool(home_left[t]) == (d["teams_f"][t, s] == P.HOME)
            hits += bool(in_box(ball[t], own_left))
    return hits > (b - a + 1) / 2


def first_kick(d, ft, b):
    """(restart index, kicker's team, to_mate) of the first kick after a wait ending at b, or None."""
    team_ev = [e for e in d["touches"] if e["kind"] in ("touch", "hold") and e["team"] in (P.HOME, P.AWAY)]
    for k, e in enumerate(team_ev):
        if e["kind"] == "touch" and b - BRIDGE <= e["t"] <= b + 1 + LOOK and not ft["feet"][e["t"]]:
            nxt = [x for x in team_ev[k + 1:] if not ft["feet"][x["t"]]][:1]
            return e["a"], e["team"], bool(nxt) and nxt[0]["team"] == e["team"]
    return None


def restart_team(d, r):
    """Fallback when no kick: (team, source) of the restart at index r, or (None, '')."""
    for e in d["touches"]:
        if r - BRIDGE <= e["t"] <= r + LOOK and e["kind"] == "hold" and e["team"] in (P.HOME, P.AWAY):
            return e["team"], "hold"
    dt = d["tinfo"]["d_team"][min(r, len(d["frames"]) - 1)]
    if np.isfinite(dt).any():
        return int(np.nanargmin(dt)), "nearest"
    return None, ""


def walk_back(d, ft, a, floor):
    """(start index, why): dead ball start for a wait starting at a; floor = earliest allowed index."""
    lo = floor
    for team, evs in V3.sequences([e for e in d["touches"] if e["t"] < a]):
        c = V3.controlled_at(evs)
        if c is not None and c < a:
            lo = max(lo, c)
    line = np.flatnonzero(ft["over_line"][lo:a])
    if len(line):
        return lo + int(line[0]), "out"
    wh = [w["b"] - d["frames"][0] + 1 for w in d.get("whistles", []) if w["ref"]]
    wh = [t for t in wh if lo <= t <= a]
    if wh:
        return wh[-1], "whistle"
    t = a
    while t > lo and ft["still"][t - 1] and ft["still"][a]:
        t -= 1
    return (t, "ball rest") if t < a else (a, "wait")


def dead_balls(d, kickoffs, gk_slots, home_left):
    """[dict(s, r, a, b, team, src, why, keeper, to_mate)]: dead ball s..r-1 (indices), restart r, wait a..b."""
    ft = features(d)
    d["dfeat"] = ft
    f, out, floor = d["frames"], [], 0
    for a, b in waits(ft):
        if a < floor:
            continue
        r = min(b + 1, len(f) - 1)
        kick = [k for k in kickoffs if a <= k["t"] <= b + V3.KICK_MARGIN]
        to_mate, fk = None, first_kick(d, ft, b)
        if kick:
            r = kick[0]["t"]
            team, src = kick[0]["team"], "kick-off"
        elif fk is not None:
            r, team, to_mate = fk
            src = "kick"
        else:
            team, src = restart_team(d, r)
        if team is None:
            continue
        s, why = walk_back(d, ft, a, floor)
        keeper = keeper_rest(d, a, b, gk_slots, home_left)
        if keeper and why not in ("out", "whistle"):
            continue
        if s >= r:
            continue
        out.append(dict(s=s, r=r, a=a, b=b, team=team, src=src, why=why, keeper=keeper, to_mate=to_mate))
        floor = r
    return out
