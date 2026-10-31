# Handoff: classical ball detector

For the next agent working in `ball_classical/`. Read `README.md` first (setup, how to run, files, training-data
provenance). This file is the state, the decisions and the rules that README doesn't hold.

## Working rules (from the user)
- Be concise: telegraphic bullets, no filler, keep every fact.
- **ATD ball (Metrica automatic tracking) is NOT ground truth and NOT a reference.** The user found errors in it. Never
  judge, tune or recommend based on "agreement with ATD" or "frames where ATD has no ball". `run_tactical.py`/`track.py`
  still print ATD columns; ignore them. Judge changes by: the user watching rendered videos (main signal), possession GT
  H/A/D (frames 11185–18684, for checking only), physical plausibility (jumps, speeds).
- Before changing the candidate pipeline (thresholds/morphology in `candidates.py` / `TACTICAL_CFG`), propose it and get
  the user's OK.
- Never overwrite a model or an earlier output: new models get a unique `--out` name, new runs a `--tag` or `--out-dir`.
  The user compares videos side by side.
- RULES.md: no hand labels on the provided match in any amount. The public window 11185–18684 is not a tuning set; tune on
  frames outside it (25000–40000 was used). Outputs derived from the match go to `../data/ball_classical/` (gitignored),
  never commit or publish them.
- User REJECTED an inference-time "out of play" rule (ball last seen near a line then missing → stop). Don't propose it again.

## Current state (all verified, user approved the output)
- Model `models/ball_svm.pkl` = RBF SVM, HOG+colour, trained on `dataset/` + `dataset_barca_real/` (FC 27 gameplay).
  Retraining with `python train.py --features hog+color --out <path>` reproduces it exactly.
- Tracker `track.py` defaults: V_MAX 30 m/s, R0 1 m, MAX_GAP 25 frames, FREE_RESTART 125, BIAS 0, SKIP_COST 0, CONTINUITY 0.1,
  LAMBDA 0.05, NEW_TRACK 4, JUMP_PER_M 0.2, MIN_SCORE −1, interpolation = hermite, VEL_W 0 (off).
- Output the user approved ("good"): `../data/ball_classical/tracked_11185_18684_batch3hc.{mp4,csv}` (5-min window) and
  `tracked_11185_13184_batch3hc.*` (minute). The move to this repo reproduced the minute CSV byte-identically.
- Numbers on 11185–18684 (vs possession GT): ball-alive frames with output 71.9% detections / 94.6% with gap filling;
  dead-ball frames still get output 89.1% (known weakness).

## History: what was tried and why it ended up this way
- Models (old files remain in the original folder `C:\Users\Mohamed\Desktop\barca\gp\Computer_Vision_Fifa\ball_detection_classical\models\`):
  - batch1/batch2: Barça gameplay only, HOG+colour. batch2 was in use until now.
  - batch3: + Barça–Real gameplay, `train.py` auto-picked HOG+colour+**shape** (won FC 27 test AP by <0.001). On the real
    match it output far less (ball-alive raw 70.8 → 58.3%). Cause: shape features (blob area, vertical position in frame)
    are camera-specific; FC 27 balls are bigger/sharper than the 4–10 px tactical ball.
  - batch3 with HOG+colour only (= current model): coverage back to batch2 level, tracker rejects its top pick less often
    (873 vs 1136 frames on the 5-min window) → user preferred it. Lesson: `train.py`'s auto-pick uses FC 27 only; always pass `--features`.
  - More FC 27 labelling/retraining on all FC 27 data: recommended against (FC 27 gains plateaued 97.7→98.0% top-1, didn't transfer).
- Tracker:
  - Skip-cost fix: a per-frame skip cost (0.1) with long gaps capped at NEW_TRACK made a sparsely seen ball cheaper to skip
    than to follow (frames 11855–11995 all dropped). Now BIAS 0, SKIP_COST 0 (= every ball-less frame costs the same).
  - Jumps: a candidate links to any candidate up to 125 frames back; reach = 30 m/s × gap + 1 m. Beyond reach = jump
    (NEW_TRACK + 0.2/m), not interpolated. Restart anywhere only after 5 s unseen.
  - Hermite gap filling (`--interp hermite`, now default): cubic curve leaving/entering the gap with the ball's velocity,
    fitted over ≤5 track detections each side; falls back to a straight line if a side has <2 detections or the curve
    exceeds 20 m/s. Only the gap-filled path changes. User: "love it".
  - Velocity-change penalty (`--vel-w`, off): greedy approximation, link pays VEL_W per m/s of velocity change beyond 8 m/s.
    At 0.1 it removed fast gap-fills but dropped detections (+135 empty frames on the minute). Not adopted; keep available.

## Next step (user approved, not started)
**Far-side detection.** Ball on the far side (top of image) is detected less: detections 69–73% of tracked frames vs
79–83% near side; ball 6–7 px vs 9 px. Of far-side gap-filled frames ~half have no candidate near the ball (candidate
stage), ~half have one that scores low (classifier). Approved fix 1: a looser white threshold (now V≥170, S≤60) and a smaller min area
(area_far now (4,120)) for the far side only. It changes the candidate pipeline: show the user the exact proposed values
before running. Tune on frames 25000–40000, then render the minute and 5-min windows with a new `--tag`/`--out-dir` for the user to compare.
Other options if needed: 2× upscale of the far band before detection; lower score bar for far candidates in the tracker.

## Later
- Output during dead ball / out of play (89% of D frames still get a ball). Don't use the rejected "out of play" rule.
- Real-footage training data (classifier has only seen FC 27). Allowed by RULES.md: automatic pseudo-labels on the match
  if no person decides any label (e.g. high-score, physically consistent tracked detections as positives, rejected ones as
  negatives), or hand labels on *other* real games (document them). Pseudo-labels must come from outside 11185–18684.
  Avoid labelling from ATD (user: not a reference).

## Tips
- Run scripts from this folder with `-u` (unbuffered). Don't pipe long runs through grep.
- An mp4 can't be opened until writing finishes. Use `open_video_at` (grab) for frame-exact runs, not `cap.set`.
- `track.py --first/--last` can select a sub-range of an existing `run_tactical.py` output (`--stem`).
- `run_tactical.py` always writes `classical_vs_atd_<first>_<last>*`. Use `--out-dir` to keep earlier runs.
