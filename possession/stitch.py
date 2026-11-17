"""Joins per-range submissions (possession_v5.py on each half) into one full-match submission.

Each input is a full-length submission CSV (frame,label) that is X on the graded frames it did not process. Per frame
the output takes the one input that is not X there; frames outside both halves (never scored) are D. Two inputs with
a label on the same graded frame is an error (ranges must not overlap).

    python stitch.py ../data/possession/possession_11185_90651_v5_full_submission.csv \
        ../data/possession/possession_110808_190197_v5_full_submission.csv --out ../data/possession/BAR-ATH_v5_full_submission.csv
"""
import argparse

import numpy as np

import possession as P
from games import half_video_blocks


def read(path):
    with open(path) as f:
        next(f)
        return np.array([line.rstrip().split(",")[1] for line in f])


def main(args):
    n = P.GAMES[P.GAME]["n_video_frames"]
    graded = np.zeros(n, bool)
    for _, a, b in half_video_blocks(P.GAME):
        graded[a:b + 1] = True
    out = np.where(graded, "X", "D")
    for path in args.inputs:
        lab = read(path)
        if len(lab) != n:
            raise SystemExit(f"{path}: {len(lab)} rows, expected {n}")
        mine = graded & (lab != "X")
        clash = mine & (out != "X")
        if clash.any():
            raise SystemExit(f"{path}: overlaps an earlier input at {int(np.flatnonzero(clash)[0])}")
        out[mine] = lab[mine]
    with open(args.out, "w", newline="") as f:
        f.write("frame,label\n")
        f.writelines(f"{t},{lab}\n" for t, lab in enumerate(out))
    print(f"wrote {args.out}: graded frames still X: {int((graded & (out == 'X')).sum())}, "
          + ", ".join(f"{k} {int((out[graded] == k).sum())}" for k in "HAD"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
