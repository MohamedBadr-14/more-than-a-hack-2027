"""Team in possession per frame, v5: v4 plus the edges of each half (terms in ../GLOSSARY.md).

Runs on any range inside one half (a whole half or a segment of it).
- Referee whistles: whistle.referee_whistles (steps 1-5 or the pair detector).
- Sides (which end HOME defends): from the players, not the tagging. Kick-off formation (goalkeeper slots left out,
  ATD mixes them up): >= FORM_MIN outfield players seen per team, team means >= FORM_GAP apart, no player more than
  FORM_SLACK into the other half (many stand on the halfway line). Sides of a half = the first formation held for
  FORM_HOLD frames within FORM_SEARCH of the half's first frame (read from the ATD, whatever range is processed).
- Kick-offs (v3 kick test with a wider spot: ball within KICK_SPOT of the centre spot, then KICK_AWAY from it; the
  kick = where the ball is closest to the spot before it leaves, v3 kickoffs(closest=True)): searched inside each
  smart tagging kick-off tag (as v3) and, for a referee whistle with no tag around it, from the whistle to
  KICK_AFTER after it; a whistle kick-off also needs the kick-off formation at the kick.
- Half start: when the range starts at the half's first frame, D up to the first kick-off; touches before it ignored.
- Half end: when the range ends at the half's last frame, the final whistle = the first referee whistle after which
  the players never play again (no run of median player speed >= deadball.SLOW with >= PLAY_MIN players seen, for
  >= deadball.WAIT, that goes on past WH_LEAD after the whistle). D from the end of that whistle to the end of the half, whatever is touched.
  No such whistle: a wait still open at the end of the half -> D from its dead-ball start (deadball.walk_back).
- Dead ball (DEAD_ATD, v5e): the dead balls are found as in v5c (raw ATD ball kick ends a wait); then each restart
  moves later to the ATD kick or the players clearly playing again (deadball.later_restarts); its team = the first
  touch / hold after it (v5f, deadball.first_touch_team); the players' restart is ignored when ATD sees the ball within
  deadball.ATD_GAP of the players' run start (v5i: ball still being placed, wait for its kick)
  (deadball.atd_fast / atd_kicker); the dead ball's start and open play keep our ball.
- Submission: frames outside both halves (never scored) are D.

    python possession_v5.py --first 110808 --last 114500 --track ../data/ball_classical/seg_110808_114500/tracked_110808_114500.csv --tag _v5_seg
"""
import argparse
import os

import numpy as np

import deadball as DB
import possession as P
import possession_v3 as V3
import tagging as TG
import touches as T
import whistle as WH
from games import half_video_blocks

FORM_SLACK = 2.0          # m from the halfway line: the kicker and his partner stand on it
FORM_MIN = 7              # outfield players seen per team
FORM_GAP = 10.0           # m between the two teams' mean x
FORM_HOLD = P.FPS         # frames the formation must hold to give the sides
FORM_SEARCH = 80000       # frames after the half's first frame searched for the sides (the whole half: half 1's
                          # tracking starts at the kick, so its first held formation is the first after-goal kick-off)
KICK_AFTER = 5 * P.FPS    # frames after a whistle searched for the kick-off kick
KICK_SPOT = 4.0           # m, ball this close to the centre spot = on the spot (v3: 2 m). The kicker stands on the
                          # ball, so the boot rule turns those detections into gaps; the track passes ~3 m from it.
PLAY_MIN = 14             # players seen for their speed to count as play (players leaving after the final whistle:
                          # 2-8 seen, their median speed means nothing)
TAG_LEAD = 5 * P.FPS     # a whistle this soon before a tag (or inside it) belongs to the tag
DEAD_ATD = True           # v5c/v5d: inside a dead ball the raw ATD ball (ATD original file) ends the wait and gives the
                          # restart (our tracker takes boots at set pieces for the ball); else only the players clearly playing again
                          # (deadball.PLAY_SPEED for PLAY_HOLD) gives the restart (v5e, deadball.later_restarts); --no-dead-atd = v5


def formation(x, teams, gk_slots):
    """HOME defends left (bool) when the players x (m, per slot) stand in kick-off formation, else None:
    no outfield player more than FORM_SLACK into the other half (many stand on the halfway line), >= FORM_MIN seen
    per team, team means >= FORM_GAP apart."""
    half, xs = P.PITCH[0] / 2, {}
    for tm in (P.HOME, P.AWAY):
        sel = (teams == tm) & np.isfinite(x)
        sel[list(gk_slots)] = False
        xs[tm] = x[sel]
        if len(xs[tm]) < FORM_MIN:
            return None
    gap = xs[P.AWAY].mean() - xs[P.HOME].mean()
    if abs(gap) < FORM_GAP:
        return None
    left, right = (P.HOME, P.AWAY) if gap > 0 else (P.AWAY, P.HOME)
    if (xs[left] > half + FORM_SLACK).any() or (xs[right] < half - FORM_SLACK).any():
        return None
    return left == P.HOME


def half_sides(orig, h_first, h_last, teams, gk_slots):
    """(HOME defends left, frame) of the half from the first kick-off formation held FORM_HOLD frames."""
    players, _ = P.read_atd(orig, h_first, min(h_first + FORM_SEARCH, h_last))
    x = players[:, :, 0] * P.PITCH[0]
    run, prev = 0, None
    for i in range(len(x)):
        f = formation(x[i], teams, gk_slots)
        run = run + 1 if f is not None and f == prev else (1 if f is not None else 0)
        prev = f
        if run >= FORM_HOLD:
            return f, h_first + i - FORM_HOLD + 1
    return None, None


def kickoffs(d, tags, whistles, gk_slots):
    """v3 kickoffs() plus src ('tag' / 'whistle'): tag windows as v3; whistle windows need the formation at the kick;
    tag windows (v5j) need it on at least one frame of the FORM_HOLD before the kick (a tag alone is not a kick-off)."""
    x = d["players_n"][:, :, 0] * P.PITCH[0]
    out = [dict(k, src="tag") for k in V3.kickoffs(d, tags, d["home_left"], KICK_SPOT, closest=True)
           if any(formation(x[i], d["teams"], gk_slots) == bool(d["home_left"][k["t"]])
                  for i in range(max(k["t"] - FORM_HOLD, 0), k["t"] + 1))]
    f0 = d["frames"][0]
    for w in whistles:
        if any(a - TAG_LEAD <= w["a"] <= b + V3.KICK_MARGIN for _, a, b in tags):
            continue
        for k in V3.kickoffs(d, [("whistle", w["a"], w["b"] + KICK_AFTER - V3.KICK_MARGIN)], d["home_left"], KICK_SPOT,
                             closest=True):
            if formation(x[k["t"]], d["teams"], gk_slots) == bool(d["home_left"][k["t"]]) \
                    and not any(abs(k["t"] - o["t"]) < P.FPS for o in out):
                out.append(dict(k, src="whistle", whistle=w["a"] - f0))
    return sorted(out, key=lambda k: k["t"])


def kickoff_dead(dead, kicks, k0):
    """v5j: an after-goal kick-off (after k0) ends any dead ball covering it (restart = the kick-off); with no dead ball
    of at least deadball.WAIT ending at it, the dead ball runs from the end of the most recent one to the kick-off
    (players' celebration and walk back after the goal), merged with it. Returns the new list."""
    dead = sorted(dead, key=lambda x: x["s"])
    for k in kicks:
        t = k["t"]
        if t <= k0:
            continue
        team = dict(team=k["team"], src="kick-off", to_mate=None)
        dead = [dict(x, r=t, b=t - 1, **team) if x["s"] < t < x["r"] else x for x in dead]  # kick-off ends D
        at = [x for x in dead if x["r"] == t]
        if at and t - at[-1]["s"] >= DB.WAIT:
            continue
        dead = [x for x in dead if x["r"] != t]  # too short: part of the stoppage below
        prev = [x for x in dead if x["r"] < t]
        if prev:
            p = prev[-1]
            dead = [x for x in dead if x is not p] + [dict(p, r=t, b=t - 1, why=p["why"] + ", to kick-off", **team)]
        else:
            dead.append(dict(s=k0, r=t, a=k0, b=t - 1, why="to kick-off", keeper=False, **team))
        dead.sort(key=lambda x: x["s"])
    return dead


def final_whistle(d, whistles):
    """Index of the end of the final whistle (players never play again after it), or None."""
    seen = np.isfinite(d["players_n"][:, :, 0]).sum(1)
    with np.errstate(invalid="ignore"):
        fast = (d["dfeat"]["pspeed"] >= DB.SLOW) & (seen >= PLAY_MIN)
    play = [(a, b) for a, b in T.spans(fast, DB.BRIDGE) if b - a + 1 >= DB.WAIT]
    f0, n = d["frames"][0], len(d["frames"])
    for w in sorted(whistles, key=lambda w: w["a"]):
        e = w["b"] - f0
        if 0 <= e < n and not any(b > e + DB.WH_LEAD for _, b in play):
            return e
    return None


def compute(first, last, track):
    """possession_v4.compute()'s dict, with v5's whistles, sides, kick-offs and half edges (d['edges'])."""
    (h, h_first, h_last), = [blk for blk in half_video_blocks(P.GAME) if blk[1] <= first <= blk[2]]
    if last > h_last:
        raise SystemExit(f"range {first}-{last} crosses the end of half {h} ({h_last})")
    d = P.compute(first, last, track)
    meta_path, raw_path, tag_path = P.require(["atd_meta", "atd_raw", "smart_tagging"])
    orig = raw_path.replace("_atd.txt", "_atd_original.txt")
    n = len(d["frames"])
    gk = TG.goalkeeper_slots(meta_path)
    left, side_frame = half_sides(orig, h_first, h_last, d["teams"], gk)
    if left is None:
        raise SystemExit(f"no kick-off formation within {FORM_SEARCH} frames of the start of half {h}")
    d["gk_slots"], d["home_left"] = gk, np.full(n, left)
    d["teams_f"] = V3.teams_per_frame(d, gk, d["home_left"])
    d["touches"], d["tinfo"] = T.touches(d)
    d["whistles"] = WH.referee_whistles(first, last)
    d["kickoffs"] = kickoffs(d, TG.kickoffs(tag_path), d["whistles"], gk)
    if DEAD_ATD:
        d["atd_ball"] = P.read_atd(orig, first, last)[1]
    dead = DB.dead_balls(d, d["kickoffs"], gk, d["home_left"])

    edges = dict(half=h, sides=(left, side_frame), start=None, end=None)
    k0 = 0
    if first == h_first:
        if d["kickoffs"]:
            k0 = d["kickoffs"][0]["t"]
            edges["start"] = (0, k0, d["kickoffs"][0]["src"])
        else:
            edges["start"] = (None, None, "no kick-off found")
    end = n
    if last == h_last:
        e = final_whistle(d, d["whistles"])
        if e is not None:
            end = min(e + 1, n)
            edges["end"] = (end, n, "final whistle")
        else:
            ws = DB.waits(d["dfeat"])
            if ws and ws[-1][1] >= n - 1 - DB.BRIDGE:
                floor = max([db["r"] for db in dead if db["r"] <= ws[-1][0]], default=k0)
                end, why = DB.walk_back(d, d["dfeat"], ws[-1][0], floor)
                edges["end"] = (end, n, f"open wait, start by {why}")
            else:
                edges["end"] = (None, None, "no final whistle, no open wait")
    d["edges"] = edges
    dead = kickoff_dead(dead, d["kickoffs"], k0)
    d["dead"] = [db for db in dead if db["r"] > k0 and db["s"] < end]

    inside = np.zeros(n + 1, bool)
    inside[:k0] = True
    inside[end:] = True
    for db in d["dead"]:
        inside[db["s"]:db["r"]] = True
    events = [e for e in d["touches"] if not inside[e["t"]]]
    restarts = [dict(t=db["r"], team=db["team"]) for db in d["dead"]]
    kicks = [k for k in d["kickoffs"] if k["t"] >= k0
             and not any(db["src"] == "kick-off" and db["r"] == k["t"] for db in d["dead"])]
    d["labels"], d["events"] = V3.hand_over(events, n, kicks + restarts)
    for db in d["dead"]:
        d["labels"][db["s"]:db["r"]] = "D"
    d["labels"][:k0] = "D"
    d["labels"][end:] = "D"
    return d


def main(args):
    global DEAD_ATD
    DEAD_ATD = not args.no_dead_atd
    d = {}

    def keep(*a):
        d.update(compute(*a))
        return d
    V3.main(args, keep, ungraded="D")
    f, ed = d["frames"], d["edges"]
    left, sf = ed["sides"]
    print(f"half {ed['half']}: HOME defends the {'left' if left else 'right'} end (kick-off formation at {sf})")
    for w in d["whistles"]:
        print(f"referee whistle {w['a']}-{w['b']} by {w['src']}")
    for k in d["kickoffs"]:
        print(f"kick-off at {f[k['t']]} ({k['src']}): team {P.LABEL[k['team']] if k['team'] is not None else 'disagree'}")
    for name in ("start", "end"):
        if ed[name] is not None:
            a, b, why = ed[name]
            print(f"half {name}: " + (f"D {f[a]}-{f[b - 1]} ({why})" if a is not None else why))
    for db in d["dead"]:
        print(f"dead ball {f[db['s']]}-{f[db['r']] - 1} (start by {db['why']}), restart {f[db['r']]} -> "
              f"{P.LABEL[db['team']]} ({db['src']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--first", type=int, default=11185)
    ap.add_argument("--last", type=int, default=18684)
    ap.add_argument("--track", default=os.path.join(P.MTAH, "data", "ball_classical", "out3", "tracked_11185_18684.csv"))
    ap.add_argument("--out-dir", default=os.path.join(P.MTAH, "data", "possession"))
    ap.add_argument("--no-dead-atd", action="store_true", help="v5: our ball inside dead balls too")
    ap.add_argument("--tag", default="_v5j", help="suffix for the output names (keeps earlier outputs)")
    main(ap.parse_args())
