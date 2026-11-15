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
NOT in possession, out of K; filled at K = hand-over). White line = hand-over confirmed, dashed = where it
was backdated to.

--layout touch (touches.py only): video with the touches.R circle, lines to players within it, ball speed,
event / hold labels; panel = (1) ball change at this frame (speed in/out, turn, tests, in/out arrows),
(2) nearest of each team, (3) result; timeline rows: ball source, speed, change (firing frames), event
(team colour, yellow = contested, low grey = unknown; white tick = its moment), hold.

--layout v3 (possession_v3.py): the touch layout plus ours / GT rows on top of the timeline, white line =
possession change confirmed (2nd touch / hold), dashed = the change point (first touch), and an ours / GT
box with a change banner in the panel. Outer grey circle = touches.R_FAR.

--layout v4 (possession_v4.py): the v3 layout plus a "dead" timeline row (grey = our dead ball, low white =
wait, cyan = referee whistle) with a yellow line at each restart, and the dead-ball
features in the panel (player speed, ball speed, players in the busiest box, tag, ball near a line).

Viewer keys: space pause/play, right/left (or d/a) step one frame, l/j jump 5 s forward/back, n/p next/previous hand-over,
m/M next/previous disagreement with GT, t/T next/previous touch, b/B next/previous bounce-flagged touch,
r/R next/previous restart (our dead ball), g go to frame (typed in the terminal), +/- speed, q quit.
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

import possession as P
import possession_v3 as V3
import possession_v4 as V4
import touches as T

W, H = 1920, 1080
VID_W, VID_H = 1440, 810
PANEL_X = VID_W
HALF = 125  # timeline frames each side of the cursor (5 s)
TL_X0, TL_Y0 = 130, VID_H + 12
ROW_H = 36
LABEL_C = {"H": P.TEAM_C[P.HOME], "A": P.TEAM_C[P.AWAY], "D": (110, 110, 110), "X": (45, 45, 45)}
WHITE, GREY, YELLOW, DARK = (255, 255, 255), (128, 128, 128), (0, 255, 255), (25, 25, 25)
GREEN = (80, 220, 80)
FONT = cv2.FONT_HERSHEY_SIMPLEX
BOUNCE_C = (255, 0, 255)  # touch flagged as a possible bounce
TOUCH_STYLE = {"touch": None, "hold": None, "contested": (0, 255, 255), "unknown": (128, 128, 128)}  # None = team colour


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


def outlined(img, s, xy, scale, colour, thick=2):
    cv2.putText(img, s, xy, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, xy, FONT, scale, colour, thick, cv2.LINE_AA)


def ev_label(e):
    team = {T.BOTH: "both", T.NONE: "?"}.get(e["team"]) or P.LABEL[e["team"]]
    colour = TOUCH_STYLE[e["kind"]] or P.TEAM_C[e["team"]]
    if e.get("bounce") and e["kind"] == "touch":
        colour = BOUNCE_C
    return f"{e['kind'].upper()} {team}" + (" far" if e.get("far") else "") + (" ? bounce" if e.get("bounce") else ""), colour


def touch_overlay(frame, d, i, Hm, LH):
    """Touch layout: players, the R circle, lines to the players within R, ball speed, event / hold labels."""
    b = d["ball_n"][i]
    teams = d["teams_f"][i] if "teams_f" in d else d["teams"]
    px = {}
    for s, uv in enumerate(d["players_n"][i]):
        if not np.isnan(uv[0]):
            px[s] = tuple(int(round(c)) for c in LH.project(Hm, *uv))
            cv2.circle(frame, px[s], 5, P.TEAM_C[teams[s]], -1)
    if np.isnan(b[0]):
        return
    bp = tuple(int(round(c)) for c in LH.project(Hm, *b))
    dist = np.hypot(*((d["players_n"][i] - b) * P.PITCH).T)
    for s in np.flatnonzero(dist < T.R):
        if s in px:
            cv2.line(frame, bp, px[s], P.TEAM_C[teams[s]], 2, cv2.LINE_AA)
            cv2.circle(frame, px[s], 12, WHITE, 2)
    for r, colour in ((T.R, YELLOW), (T.R_FAR, GREY)):
        ring = [LH.project(Hm, b[0] + r * np.cos(a) / P.PITCH[0], b[1] + r * np.sin(a) / P.PITCH[1])
                for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
        for k in range(0, 24, 2):
            cv2.line(frame, tuple(int(c) for c in ring[k]), tuple(int(c) for c in ring[k + 1]), colour, 1, cv2.LINE_AA)
    cv2.circle(frame, bp, 9, P.BALL_C[d["status"][i]], 2)
    outlined(frame, f"{fmt(d['tinfo']['speed'][i]).strip()} m/s", (bp[0] + 14, bp[1] + 26), 0.6, WHITE, 1)
    y = bp[1] - 14
    for key in ("eidx", "hidx"):
        k = d[key][i]
        if k >= 0:
            e = d["touches"][k]
            label, colour = ev_label(e)
            outlined(frame, label, (bp[0] + 14, y), 0.8, colour)
            y -= 30


def touch_panel(canvas, d, i):
    """Touch layout: (1) does the ball change at this frame, (2) who is within R, (3) the result."""
    x, y = PANEL_X + 16, 44
    ti = d["tinfo"]
    text(canvas, f"frame {d['frames'][i]}", (x, y), 0.7, WHITE, 2)
    y += 30
    text(canvas, f"ball: {d['status'][i]}   speed now {fmt(ti['speed'][i])} m/s", (x, y), 0.6)
    y += 44
    ek, hk = d["eidx"][i], d["hidx"][i]
    e = d["touches"][ek] if ek >= 0 else None
    shown = e or (d["touches"][hk] if hk >= 0 else None)
    label, colour = ev_label(shown) if shown else ("no touch", GREY)
    cv2.rectangle(canvas, (x - 6, y - 30), (W - 16, y + 12), colour, -1)
    text(canvas, label, (x + 4, y), 0.85, WHITE if shown and shown["kind"] != "unknown" else DARK, 2)
    y += 50

    text(canvas, "1. ball change at this frame", (x, y), 0.6, WHITE, 2)
    y += 28
    text(canvas, f"   speed in {fmt(ti['s_in'][i])} -> out {fmt(ti['s_out'][i])} m/s", (x, y), 0.6)
    y += 26
    turn = ti["turn"][i]
    text(canvas, f"   turn {turn:.0f} deg" if np.isfinite(turn) else "   turn  -", (x, y), 0.6)
    y += 30
    tests = [(f"turn    >= {T.TURN:g} deg, >= {T.MIN_SPEED:g} m/s, holds after", "turn"),
             (f"slow    in >= {T.SLOW_IN:g}, out <= {T.SLOW_FRAC:g} x in", "slow down"),
             (f"kick    in < {T.KICK_IN:g}, out >= {T.KICK_OUT:g} m/s", "kick from slow")]
    for s, key in tests:
        ok = bool(ti["tests"][key][i])
        text(canvas, ("   OK  " if ok else "   --  ") + s, (x, y), 0.52, (80, 220, 80) if ok else GREY, 2 if ok else 1)
        y += 26
    y += 14

    text(canvas, f"2. who: nearest within {T.R:g} m (else {T.R_FAR:g} m = far)", (x, y), 0.6, WHITE, 2)
    y += 28
    for team, name in ((P.HOME, "Barca"), (P.AWAY, "Athletic")):
        dm = ti["d_team"][i, team]
        ok = dm < T.R_FAR
        text(canvas, f"   {name:9s} {fmt(dm)} m", (x, y), 0.6, P.TEAM_C[team] if ok else GREY, 2 if ok else 1)
        y += 26
    y += 14

    text(canvas, "3. result", (x, y), 0.6, WHITE, 2)
    y += 28
    if e:
        f = d["frames"]
        text(canvas, f"   ball event {f[e['a']]}-{f[e['b']]}, moment {f[e['t']]}", (x, y), 0.52)
        y += 26
        text(canvas, f"   by: {e['why']}", (x, y), 0.52)
        y += 26
        text(canvas, f"   change vs line of sight |cos| {fmt(e['los_cos'])} (bounce flag >= {T.BOUNCE_COS:g})", (x, y), 0.52,
             BOUNCE_C if e["bounce"] else GREY)
        y += 26
        text(canvas, f"   nearest H {fmt(e['d_h'])}  A {fmt(e['d_a'])} m -> {label}", (x, y), 0.52, colour, 1)
        y += 26
    if hk >= 0:
        h = d["touches"][hk]
        text(canvas, f"   hold {d['frames'][h['a']]}-{d['frames'][h['b']]} ({h['why']})", (x, y), 0.52,
             P.TEAM_C[h["team"]], 2)
        y += 26
    if not e and hk < 0:
        text(canvas, "   no ball change, no hold", (x, y), 0.52, GREY)
    arrows(canvas, ti["v_in"][i], ti["v_out"][i], (PANEL_X + 400, 300))
    if "v3" in d:
        poss_block(canvas, d, i, x, 670)
    if "v4" in d:
        dead_block(canvas, d, i, x, 772)


def dead_block(canvas, d, i, x, y):
    """v4 layout: the dead-ball features at this frame, and the dead ball / wait it is in."""
    ft, f = d["dfeat"], d["frames"]
    on = lambda k: GREEN if ft[k][i] else GREY
    fi = int(d["frames"][i])
    whistling = any(w["ref"] and w["a"] <= fi <= w["b"] for w in d.get("whistles", []))
    items = [(f"players(1s) {fmt(ft['pspeed'][i])}", on("slow")), (f"ball(0.5s) {fmt(ft['bspeed'][i])}", on("still")),
             (f"end {ft['crowd'][i]}", on("crowded")), ("whistled", on("whistled")), ("outed", on("outed")), ("out", on("over_line")), ("feet", on("feet")), ("whistle", GREEN if whistling else GREY)]
    xx = x
    for s, c in items:
        text(canvas, s, (xx, y), 0.5, c, 1)
        xx += cv2.getTextSize(s, FONT, 0.5, 1)[0][0] + 18
    here = [db for db in d["dead"] if db["s"] <= i <= db["r"] + 3 * P.FPS]
    if here:
        db = here[-1]
        text(canvas, f"D {f[db['s']]}-{f[db['r']] - 1} by {db['why']} | wait {f[db['a']]}-{f[db['b']]} | "
             f"restart {f[db['r']]} {P.LABEL[db['team']]} ({db['src']})", (x, y + 26), 0.45, WHITE, 1)


def poss_block(canvas, d, i, x, y):
    """v3 layout: our label vs GT, and a banner for 3 s after a possession change is confirmed."""
    lab, gt = d["labels"][i], d["gt"].get(int(d["frames"][i]), "X")
    cv2.rectangle(canvas, (x - 6, y - 26), (x + 200, y + 10), LABEL_C[lab], -1)
    text(canvas, f"OURS {lab}", (x + 4, y), 0.75, WHITE, 2)
    cv2.rectangle(canvas, (x + 234, y - 26), (x + 440, y + 10), LABEL_C[gt], -1)
    text(canvas, f"GT {gt}", (x + 244, y), 0.75, WHITE, 2)
    recent = [e for e in d["events"] if 0 <= i - e[0] < 3 * P.FPS]
    if recent:
        conf, start, team = recent[-1]
        f = d["frames"]
        cv2.rectangle(canvas, (x - 6, y + 22), (W - 16, y + 80), P.TEAM_C[team], -1)
        text(canvas, f"CHANGE -> {P.LABEL[team]} from {f[start]} (first touch)", (x + 4, y + 44), 0.55, WHITE, 2)
        text(canvas, f"confirmed {f[conf]} (2nd touch / hold)", (x + 4, y + 70), 0.55, WHITE, 1)


def arrows(canvas, vin, vout, c):
    """Ball direction in (grey) and out (white) at this frame, both from the centre (angle between = turn),
    length by speed, pitch seen from above."""
    cv2.circle(canvas, c, 55, (60, 60, 60), 1)
    for v, colour, name in ((vin, GREY, "in"), (vout, WHITE, "out")):
        sp = np.hypot(*v)
        if not np.isfinite(sp) or sp < 0.1:
            continue
        L = 15 + 35 * min(sp, 25) / 25
        tip = (int(c[0] + v[0] / sp * L), int(c[1] + v[1] / sp * L))
        cv2.arrowedLine(canvas, c, tip, colour, 2, cv2.LINE_AA, tipLength=0.2)
        text(canvas, name, (tip[0] + (6 if v[0] >= 0 else -30), tip[1] + (16 if v[1] >= 0 else -6)), 0.5, colour)


def touch_timeline(canvas, d, i):
    """Touch layout rows: ball source, ball speed, ball change (firing frames), events, holds.
    v3 layout: ours and GT rows on top, white line = change confirmed, dashed = change point."""
    n = len(d["frames"])
    pxf = (W - TL_X0 - 10) / (2 * HALF + 1)
    poss, dead = "v3" in d, "v4" in d
    r0, row_h = (2, 30 if dead else 36) if poss else (0, 48)
    cv2.rectangle(canvas, (0, VID_H), (W, H), DARK, -1)
    names = (["ours", "GT"] if poss else []) + ["ball", "speed", "change", "event", "hold"] + (["dead"] if dead else [])
    for r, name in enumerate(names):
        text(canvas, name, (10, TL_Y0 + r * row_h + row_h - 14), 0.6)
    src_c = dict(P.BALL_C, none=(45, 45, 45))
    ti = d["tinfo"]
    for k in range(-HALF, HALF + 1):
        j = i + k
        if not 0 <= j < n:
            continue
        x1, x2 = int(TL_X0 + (k + HALF) * pxf), int(TL_X0 + (k + HALF + 1) * pxf)

        def cell(row, colour, frac=1.0):
            y2 = TL_Y0 + (r0 + row) * row_h + row_h - 8
            cv2.rectangle(canvas, (x1, int(y2 - (row_h - 8) * frac)), (x2, y2), colour, -1)

        if poss:
            cell(-2, LABEL_C[d["labels"][j]])
            cell(-1, LABEL_C[d["gt"].get(int(d["frames"][j]), "X")])
        cell(0, (0, 0, 255) if ti["cut"][j] and d["status"][j] != "none" else src_c[d["status"][j]], 0.5)
        if np.isfinite(ti["speed"][j]):
            cell(1, (200, 200, 200), min(ti["speed"][j], 25) / 25)
        if ti["fire"][j]:
            cell(2, (80, 220, 80))
        for row, key in ((3, "eidx"), (4, "hidx")):
            if d[key][j] >= 0:
                e = d["touches"][d[key][j]]
                cell(row, ev_label(e)[1], 0.35 if e["kind"] == "unknown" else 1.0)
        if dead:
            if d["labels"][j] == "D":
                cell(5, LABEL_C["D"])
            if any(db["a"] <= j <= db["b"] for db in d["dead"]):
                cell(5, WHITE, 0.4)
            fj = int(d["frames"][j])
            if any(w["ref"] and w["a"] <= fj <= w["b"] for w in d.get("whistles", [])):
                cell(5, (255, 255, 0), 1.0)
    for e in d["touches"]:  # white tick at each event's moment
        if e["kind"] != "hold" and abs(e["t"] - i) <= HALF:
            xs = int(TL_X0 + (e["t"] - i + HALF + 0.5) * pxf)
            cv2.line(canvas, (xs, TL_Y0 + (r0 + 3) * row_h), (xs, TL_Y0 + (r0 + 4) * row_h - 8), WHITE, 2)
    y_bot = TL_Y0 + len(names) * row_h
    if poss:
        for conf, start, team in d["events"]:
            for f, dashed in ((conf, False), (start, True)):
                if abs(f - i) <= HALF:
                    xs = int(TL_X0 + (f - i + HALF + 0.5) * pxf)
                    for y in range(TL_Y0, y_bot, 8 if dashed else y_bot):
                        cv2.line(canvas, (xs, y), (xs, min(y + (4 if dashed else y_bot), y_bot)), WHITE, 2)
    for db in d.get("dead", []) if dead else []:
        if abs(db["r"] - i) <= HALF:
            xs = int(TL_X0 + (db["r"] - i + HALF + 0.5) * pxf)
            cv2.line(canvas, (xs, TL_Y0), (xs, y_bot), YELLOW, 3)
    xc = int(TL_X0 + (HALF + 0.5) * pxf)
    cv2.line(canvas, (xc, TL_Y0 - 6), (xc, y_bot), YELLOW, 1)
    text(canvas, "-5 s", (TL_X0, H - 6), 0.45, GREY)
    text(canvas, "+5 s", (W - 50, H - 6), 0.45, GREY)


def compose(frame, d, i, Hm, LH, layout="full"):
    if layout in ("touch", "v3", "v4"):
        touch_overlay(frame, d, i, Hm, LH)
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[:VID_H, :VID_W] = cv2.resize(frame, (VID_W, VID_H), interpolation=cv2.INTER_AREA)
        touch_panel(canvas, d, i)
        touch_timeline(canvas, d, i)
        return canvas
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
    if args.layout in ("v3", "v4"):
        d = (V3 if args.layout == "v3" else V4).compute(args.first, args.last, args.track)
        d["v3"] = True
        if args.layout == "v4":
            d["v4"] = True
    else:
        d = P.compute(args.first, args.last, args.track)
        d["touches"], d["tinfo"] = T.touches(d)
    d["eidx"] = T.per_frame(d["touches"], len(d["frames"]), ("touch", "contested", "unknown"))
    d["hidx"] = T.per_frame(d["touches"], len(d["frames"]), ("hold",))
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
        for e in d["touches"]:
            if e["kind"] != "unknown":
                wr.writerow(["touch", int(d["frames"][e["t"]])])
            if e["kind"] == "touch" and e["bounce"]:
                wr.writerow(["bounce", int(d["frames"][e["t"]])])
        for db in d.get("dead", []):
            wr.writerow(["restart", int(d["frames"][db["r"]])])
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
        writer.write(compose(frame, d, i, matrices[t], LH, args.layout))
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
    marks = {"handover": [], "mismatch": [], "touch": [], "bounce": [], "restart": []}
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
        elif key in (ord("l"), ord("j")):  # jump 5 s forward / back
            target = i + (5 if key == ord("l") else -5) * P.FPS
        elif key in (ord("n"), ord("p"), ord("m"), ord("M"), ord("t"), ord("T"), ord("b"), ord("B"),
                     ord("r"), ord("R")):
            kind = {"n": "handover", "p": "handover", "m": "mismatch", "M": "mismatch",
                    "b": "bounce", "B": "bounce", "r": "restart", "R": "restart"}.get(chr(key), "touch")
            fwd = key in (ord("n"), ord("m"), ord("t"), ord("b"), ord("r"))
            lead = P.FPS // 2 if kind in ("touch", "bounce") else 2 * P.FPS  # land 0.5 s / 2 s before it
            cand = [f for f in marks[kind] if (f > i + lead if fwd else f < i - lead)]
            if cand:
                target = (min(cand) if fwd else max(cand)) - lead
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
    ap.add_argument("--layout", choices=["full", "touch", "v3", "v4"], default="full",
                    help="touch = only what decides a touch (ball, reach, tests, touch row); v3 = touch + possession v3; v4 = v3 + dead ball")
    a = ap.parse_args()
    render(a) if a.mode == "render" else view(a)
