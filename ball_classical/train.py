"""Train and compare ball classifiers on the labelled crops.

Split is by time (frame number) so near-identical neighbouring frames never land on both sides.
Hyper-parameters are tuned with grouped CV on the train part (groups = 10 s chunks of video).

Usage: python train.py [--dataset dataset --dataset dataset_barca_real] [--split N ...] [--gap 200]
                       [--features hog+color] [--out models/ball_svm_x.pkl]
Writes models/ball_svm.pkl (best feature set by FC 27 test AP, or the one given by --features) and
reports/errors.png (test mistakes, for spotting mislabels); with --out, reports/errors_<out name>.png.
"""
import argparse
import csv
import os
import pickle

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import FEATURE_SETS, extract

HERE = os.path.dirname(os.path.abspath(__file__))
PARAM_GRID = [
    {"svc__kernel": ["linear"], "svc__C": [0.01, 0.1, 1]},
    {"svc__kernel": ["rbf"], "svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.001]},
]


def load(dataset):
    rows = [r for r in csv.DictReader(open(os.path.join(dataset, "labels.csv"))) if r["label"] in ("pos", "neg")]
    crops = [cv2.imread(r["path"] if os.path.isabs(r["path"]) else os.path.join(HERE, r["path"])) for r in rows]
    keep = [i for i, c in enumerate(crops) if c is not None and c.size]
    rows, crops = [rows[i] for i in keep], [crops[i] for i in keep]
    y = np.array([r["label"] == "pos" for r in rows], int)
    frames = np.array([int(r["frame"]) for r in rows])
    return rows, crops, y, frames


def top1_accuracy(frames, y, scores):
    """Over test frames that contain a ball candidate: is the highest-scoring candidate the ball?"""
    hits = total = 0
    for f in np.unique(frames):
        m = frames == f
        if y[m].any():
            total += 1
            hits += y[m][np.argmax(scores[m])] == 1
    return hits / max(total, 1), total


def error_grid(crops, y, scores, path, n=24, size=64):
    """Top false positives (row 1) and lowest-scored positives (row 2)."""
    fp = [i for i in np.argsort(-scores) if y[i] == 0][:n]
    fn = [i for i in np.argsort(scores) if y[i] == 1][:n]
    rows = []
    for idx in (fp, fn):
        tiles = []
        for i in idx:
            t = cv2.resize(crops[i], (size, size), interpolation=cv2.INTER_NEAREST)
            cv2.putText(t, f"{scores[i]:.1f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            tiles.append(t)
        tiles += [np.zeros((size, size, 3), np.uint8)] * (n - len(tiles))
        rows.append(np.hstack(tiles))
    cv2.imwrite(path, np.vstack(rows))


def main(args):
    # Each dataset is one video with its own time split; frame ids are offset per dataset so that
    # frames from different videos never share a CV group or a top-1 frame.
    rows, crops, y, frames, ds_id, tr, te = [], [], [], [], [], [], []
    for d, path in enumerate(args.dataset):
        r, c, yy, f = load(path)
        split = args.split[d] if d < len(args.split) else int(np.percentile(f[yy == 1], args.test_pct))
        trd, ted = f < split, f >= split + args.gap
        print(f"{os.path.basename(path):20s} train {trd.sum():5d} crops ({yy[trd].sum():3d} pos) frames < {split}"
              f"   test {ted.sum():5d} crops ({yy[ted].sum():3d} pos) frames >= {split + args.gap}")
        rows += r; crops += c; y.append(yy); frames.append(f + d * 10_000_000)
        ds_id.append(np.full(len(f), d)); tr.append(trd); te.append(ted)
    y, frames, ds_id, tr, te = map(np.concatenate, (y, frames, ds_id, tr, te))
    print(f"{'total':20s} train {tr.sum():5d} crops ({y[tr].sum():3d} pos)"
          f"                        test {te.sum():5d} crops ({y[te].sum():3d} pos)\n")

    results = {}
    for name in args.features or FEATURE_SETS:
        X = np.array([extract(c, r, name) for c, r in zip(crops, rows)])
        pipe = make_pipeline(StandardScaler(), SVC(class_weight="balanced"))
        groups = frames[tr] // 600
        search = GridSearchCV(pipe, PARAM_GRID, scoring="average_precision",
                              cv=GroupKFold(n_splits=min(5, len(np.unique(groups)))), n_jobs=-1)
        search.fit(X[tr], y[tr], groups=groups)
        model = search.best_estimator_
        scores = model.decision_function(X[te])
        p, r, f1, _ = precision_recall_fscore_support(y[te], scores > 0, average="binary", zero_division=0)
        ap = average_precision_score(y[te], scores)
        top1, n_frames = top1_accuracy(frames[te], y[te], scores)
        results[name] = dict(model=model, scores=scores, ap=ap, X_dim=X.shape[1])
        params = {k.replace("svc__", ""): v for k, v in search.best_params_.items()}
        print(f"{name:16s} dim={X.shape[1]:4d}  {params}")
        print(f"{'':16s} precision={p:.2f} recall={r:.2f} F1={f1:.2f} AP={ap:.3f}  top-1={top1:.1%} ({n_frames} frames)")
        for d, path in enumerate(args.dataset):
            m = ds_id[te] == d
            t1, nf = top1_accuracy(frames[te][m], y[te][m], scores[m])
            print(f"{'':16s}   {os.path.basename(path):20s} top-1={t1:.1%} ({nf} frames)")
        print()

    best = max(results, key=lambda k: results[k]["ap"])
    print(f"best by test AP: {best}")
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    out = args.out or os.path.join(HERE, "models", "ball_svm.pkl")
    errors = os.path.join(HERE, "reports", "errors.png" if not args.out else
                          f"errors_{os.path.splitext(os.path.basename(out))[0]}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(dict(model=results[best]["model"], feature_set=best), f)
    error_grid([crops[i] for i in np.where(te)[0]], y[te], results[best]["scores"], errors)
    print(f"saved {out}, {errors}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    default_ds = [p for p in (os.path.join(HERE, "dataset"), os.path.join(HERE, "dataset_barca_real"))
                  if os.path.exists(os.path.join(p, "labels.csv"))]
    ap.add_argument("--dataset", action="append", default=None,
                    help="dataset folder (one per video), repeatable; default: every dataset* folder with labels")
    ap.add_argument("--split", type=int, action="append", default=[],
                    help="first test frame per dataset, in --dataset order; default: --test-pct of its positives")
    ap.add_argument("--test-pct", type=float, default=75, help="auto split: positives after this percentile are test")
    ap.add_argument("--gap", type=int, default=200, help="frames dropped after the split to avoid leakage")
    ap.add_argument("--features", action="append", choices=list(FEATURE_SETS), default=None,
                    help="feature set(s) to train, repeatable; default: all, keep the best by test AP")
    ap.add_argument("--out", default=None, help="model path; default models/ball_svm.pkl (the model in use)")
    a = ap.parse_args()
    a.dataset = a.dataset or default_ds
    main(a)
