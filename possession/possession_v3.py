"""
    python possession_v3.py                 # -> ../data/possession/possession_11185_18684_v3_r04_gap2{.csv,_submission.csv}
"""
import argparse
import csv
import os

import numpy as np

import possession as P
import tagging as TG
from games import half_video_blocks
import touches as T

MIN_TOUCHES = 2
KICK_SPOT = 2.0   # m, ball this close to the centre spot = on the spot
KICK_AWAY = 5.0   # m, ball this far from the centre spot = kicked
KICK_STILL = 0.5  # m, kickoffs(closest=True): ball this close to its nearest distance to the spot = not moving yet
KICK_MARGIN = P.FPS  # frames searched after the end of a kick-off tag


def teams_per_frame(d, gk_slots, home_left):
    """(n, slots) team per frame: the slot's team, switched for a goalkeeper slot in the half its team attacks."""
    teams_f = np.tile(d["teams"], (len(d["frames"]), 1))
    x = d["players_n"][:, :, 0]
    for s in gk_slots:
        own_left = home_left == (d["teams"][s] == P.HOME)
        with np.errstate(invalid="ignore"):
            wrong = np.where(own_left, x[:, s] > 0.5, x[:, s] < 0.5)
        teams_f[wrong, s] = 1 - d["teams"][s]
    return teams_f


def kickoffs(d, kicks, home_left, spot=KICK_SPOT, closest=False):
    """[dict(t, near, into, team)] for the kick-off tags inside the window: t = the kick (index), near = team
    nearest the centre spot, into = team whose half the ball moves into, team = both agree, else None.
    spot: ball this close to the centre spot = on the spot (possession_v5 passes a wider one).
    closest: t = the last frame of the on-spot stretch before the kick where the ball is within KICK_STILL of its
    closest distance to the spot (where it starts to move), not the last frame on the spot."""
    centre = np.array(P.PITCH) / 2
    ball, players = d["ball_n"] * P.PITCH, d["players_n"] * P.PITCH
    dist = np.hypot(*(ball - centre).T)
    f0, out = d["frames"][0], []
    for _, a, b in kicks:
        lo, hi = max(a - f0, 0), min(b + KICK_MARGIN - f0, len(dist) - 1)
        if lo > hi:
            continue
        on = [i for i in range(lo, hi + 1) if dist[i] <= spot]
        away = [i for i in range(on[0], hi + 1) if dist[i] >= KICK_AWAY] if on else []
        if not away:
            continue
        t = max(i for i in on if i < away[0])
        if closest:
            a0 = t
            while a0 - 1 >= lo and dist[a0 - 1] <= spot:
                a0 -= 1
            near_min = np.nanmin(dist[a0:t + 1]) + KICK_STILL
            t = max(i for i in range(a0, t + 1) if dist[i] <= near_min)
        dp = np.hypot(*(players[t] - centre).T)
        near = int(d["teams_f"][t, int(np.nanargmin(dp))])
        right = ball[away[0], 0] > centre[0]
        into = P.HOME if right != home_left[t] else P.AWAY
        out.append(dict(t=t, near=near, into=into, team=near if near == into else None))
    return out


def sequences(events):
    """[(team, [events])]: team events in time order, split where the team changes."""
    out = []
    for e in events:
        if e["kind"] not in ("touch", "hold"):
            continue
        if out and out[-1][0] == e["team"]:
            out[-1][1].append(e)
        else:
            out.append((e["team"], [e]))
    return out


def controlled_at(evs):
    """Index (frame) at which a sequence becomes controlled, None if it never does."""
    touch_t = [e["t"] for e in evs if e["kind"] == "touch"]
    when = [e["a"] + T.HOLD - 1 for e in evs if e["kind"] == "hold"]
    if len(touch_t) >= MIN_TOUCHES:
        when.append(touch_t[MIN_TOUCHES - 1])
    return min(when) if when else None


def hand_over(events, n, kicks=()):
    """Labels 'H'/'A'/'X' and the changes as (confirmed, start, team) indices (the debug view's format).
    kicks: kickoffs() -- possession goes to the kicking team at the kick."""
    labels, changes, state = np.full(n, "X"), [], None
    pending = sorted((k["t"], k["team"]) for k in kicks if k["team"] is not None)
    for team, evs in sequences(events) + [(None, [dict(t=n)])]:
        while pending and pending[0][0] <= evs[0]["t"]:
            t, kteam = pending.pop(0)
            if kteam != state:
                state = kteam
                changes.append((t, t, kteam))
        conf = controlled_at(evs) if team is not None else None
        if team != state and conf is not None:
            state = team
            changes.append((conf, evs[0]["t"], team))
    for k, (_, start, team) in enumerate(changes):
        end = changes[k + 1][1] if k + 1 < len(changes) else n
        labels[start:end] = P.LABEL[team]
    return labels, changes


def compute(first, last, track):
    """possession.compute()'s dict with labels / events replaced by v3, plus the touches and their info."""
    d = P.compute(first, last, track)
    meta_path, raw_path, tag_path = P.require(["atd_meta", "atd_raw", "smart_tagging"])
    kicks = TG.kickoffs(tag_path)
    home_left = TG.home_left(d["frames"], TG.sides(kicks, raw_path.replace("_atd.txt", "_atd_original.txt"), d["teams"]))
    d["gk_slots"], d["home_left"] = TG.goalkeeper_slots(meta_path), home_left  # also used by deadball.py
    d["teams_f"] = teams_per_frame(d, d["gk_slots"], home_left)
    d["touches"], d["tinfo"] = T.touches(d)
    d["kickoffs"] = kickoffs(d, kicks, home_left)
    d["labels"], d["events"] = hand_over(d["touches"], len(d["frames"]), d["kickoffs"])
    return d


def main(args, compute_fn=None, ungraded="X"):
    """Writes the CSVs and prints a summary; compute_fn: a compute() with the same output (possession_v5);
    ungraded: label written on the frames outside both halves (never scored)."""
    d = (compute_fn or compute)(args.first, args.last, args.track)
    frames, labels, status, ev = d["frames"], d["labels"], d["status"], d["touches"]
    at = {}
    for e in ev:
        if e["kind"] != "hold":
            at[e["t"]] = e
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"possession_{args.first}_{args.last}{args.tag}")
    with open(stem + ".csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "label", "ball_src", "event", "event_team", "far"])
        for i, t in enumerate(frames):
            e = at.get(i)
            team = "" if e is None else {T.BOTH: "both", T.NONE: ""}.get(e["team"]) or P.LABEL[e["team"]]
            wr.writerow([t, labels[i], status[i], e["kind"] if e else "", team, int(e["far"]) if e else ""])
    full = np.full(P.GAMES[P.GAME]["n_video_frames"], ungraded)
    for _, a, b in half_video_blocks(P.GAME):
        full[a:b + 1] = "X"
    full[frames] = labels
    with open(stem + "_submission.csv", "w", newline="") as f:
        f.write("frame,label\n")
        f.writelines(f"{t},{lab}\n" for t, lab in enumerate(full))

    for k in d["kickoffs"]:
        print(f"kick-off at {frames[k['t']]}: nearest the spot {P.LABEL[k['near']]}, ball into the half of "
              f"{P.LABEL[k['into']]} -> {P.LABEL[k['team']] if k['team'] is not None else 'disagree, not used'}")
    print("goalkeeper slot frames switched:", int((d["teams_f"] != d["teams"]).sum()))
    kinds = [e["kind"] for e in ev]
    print({k: kinds.count(k) for k in ("touch", "contested", "unknown", "hold")},
          "far touches", sum(e["far"] for e in ev))
    runs = [lab for i, lab in enumerate(labels) if lab in P.LABEL and (i == 0 or labels[i - 1] != lab)]
    print(f"possessions: {len(runs)} (H {runs.count('H')}, A {runs.count('A')}), "
          f"mean {np.isin(labels, list(P.LABEL)).sum() / max(len(runs), 1) / P.FPS:.1f} s, X frames {(labels == 'X').sum()}")
    print("wrote", stem + ".csv", "and", stem + "_submission.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--first", type=int, default=11185)
    ap.add_argument("--last", type=int, default=18684)
    ap.add_argument("--track", default=os.path.join(P.MTAH, "data", "ball_classical", "strict_r04_gap2_carried", "tracked_11185_18684.csv"))
    ap.add_argument("--out-dir", default=os.path.join(P.MTAH, "data", "possession"))
    ap.add_argument("--tag", default="_v3_r04_gap2", help="suffix for the output names (keeps earlier outputs)")
    main(ap.parse_args())
