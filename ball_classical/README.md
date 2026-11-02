# ball_classical — classical-ML ball detector + offline tracker

Finds the ball in the tactical video without deep learning: white-blob candidates → SVM classifier
(HOG + colour features) → offline tracker (best physically possible path + curved gap filling).

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Needs the match assets unpacked into `../data/` as described in `../DATA.md`, and `../ball_yolo/overlay.py`
(ATD reader + frame-exact video opener).

## Reproduce

Run from this folder. Outputs go to `../data/ball_classical/` (gitignored: derived from the match, never publish).

```
python run_tactical.py --no-video     # candidates + scores for frames 11185-18684 (--first/--last for other ranges)
python track.py                       # tracked ball -> tracked_11185_18684.{csv,mp4}
python track.py --first 11185 --last 13184 --stem classical_vs_atd_11185_18684   # sub-range of the same run
```

Video legend: green box = detection on the track, orange circle = filled gap (white dot = carried), magenta ring = boot zone, magenta box "boot" = detection turned into a gap, thin red box = top-scoring
candidate the tracker rejected, blue circle = ATD ball (shown for reference only, not ground truth).

## Files

| file | role |
|---|---|
| `candidates.py` | white-blob candidates: pitch mask → HSV white mask → morphology → perspective area filter → circularity |
| `features.py` | HOG / colour / shape features of a 32×32 candidate crop |
| `run_tactical.py` | per-frame detection on the tactical video (homography pitch mask, in-field filter, `TACTICAL_CFG`) |
| `track.py` | offline tracker: dynamic-programming path in pitch metres (≤30 m/s); detections at a player's feet (≤0.4 m of an ATD player) become gaps; gaps ≤2 s filled with a Hermite curve, flagged "carried" when one player is at both ends |
| `train.py` | trains the SVM (`--features hog+color --out ...`); time-split test, grouped CV |
| `label_tool.py` | click labeller used to build the datasets |
| `models/ball_svm.pkl` | model in use: RBF SVM, HOG+colour, trained on both datasets |
| `dataset/`, `dataset_barca_real/` | training data (labels CSV + crops) |

## Training data (RULES.md: what, how, by whom, how much)

- **What:** frames of EA SPORTS FC 27 gameplay video (a video game, not the provided match or any real match):
  `dataset/` = Barça gameplay (frames 0–30000), `dataset_barca_real/` = Barça vs Real Madrid gameplay (frames 10789–25789).
- **How:** `label_tool.py` shows the white-blob candidates of a frame; the labeller clicks the ball (pos), marks the rest
  negative, or ignores unclear blobs. Later batches were pre-labelled by the model and corrected by hand.
- **By whom:** Mohamed Badreldin.
- **How much:** `dataset/` 412 pos / 7079 neg crops; `dataset_barca_real/` 178 pos / 2464 neg crops.
- **Provided match:** no hand labels of any kind.

Retraining reproduces `models/ball_svm.pkl` exactly:
`python train.py --features hog+color --out models/ball_svm.pkl`
