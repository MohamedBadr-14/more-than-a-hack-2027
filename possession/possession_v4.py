"""Team in possession per frame, v4: possession v3 plus dead ball (deadball.py).

- Dead balls come from deadball.py (wait -> restart, walk back to the start). Frames from the start to the frame
  before the restart are D.
- Touches and holds inside a dead ball are ignored (fetching the ball, feet near a still ball).
- At the restart the possession goes to the restart team straight away (like the kick-off), then the v3 rules go on.

    python possession_v4.py                 # -> ../data/possession/possession_11185_18684_v4_out3_r04_gap2{.csv,_submission.csv}
"""
import argparse
import os

import numpy as np

import deadball as DB
import possession as P
import possession_v3 as V3
import whistle as WH


def compute(first, last, track):
    """possession_v3.compute()'s dict with dead balls (d['dead']) and labels incl. D."""
    d = V3.compute(first, last, track)
    n = len(d["frames"])
    d["whistles"] = WH.whistles(first, last)
    d["dead"] = DB.dead_balls(d, d["kickoffs"], d["gk_slots"], d["home_left"])
    inside = np.zeros(n + 1, bool)
    for db in d["dead"]:
        inside[db["s"]:db["r"]] = True
    events = [e for e in d["touches"] if not inside[e["t"]]]
    restarts = [dict(t=db["r"], team=db["team"]) for db in d["dead"]]
    kicks = [k for k in d["kickoffs"] if not any(db["src"] == "kick-off" and db["r"] == k["t"] for db in d["dead"])]
    d["labels"], d["events"] = V3.hand_over(events, n, kicks + restarts)
    for db in d["dead"]:
        d["labels"][db["s"]:db["r"]] = "D"
    return d


def main(args):
    d = {}

    def keep(*a):
        d.update(compute(*a))
        return d
    V3.main(args, keep)
    f = d["frames"]
    for db in d["dead"]:
        print(f"dead ball {f[db['s']]}-{f[db['r']] - 1} (start by {db['why']}), wait {f[db['a']]}-{f[db['b']]}, "
              f"restart {f[db['r']]} -> {P.LABEL[db['team']]} ({db['src']}"
              + {True: ", to a teammate", False: ", to an opponent / none"}.get(db["to_mate"], "") + ")"
              + (", keeper rest" if db["keeper"] else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--first", type=int, default=11185)
    ap.add_argument("--last", type=int, default=18684)
    ap.add_argument("--track", default=os.path.join(P.MTAH, "data", "ball_classical", "out3", "tracked_11185_18684.csv"),
                    help="track.py output; out3 = with the ball-out detection (run_tactical.py --out-margin 3)")
    ap.add_argument("--out-dir", default=os.path.join(P.MTAH, "data", "possession"))
    ap.add_argument("--tag", default="_v4_out3_r04_gap2", help="suffix for the output names (keeps earlier outputs)")
    main(ap.parse_args())
