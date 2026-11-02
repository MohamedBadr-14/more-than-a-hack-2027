# Handoff: classical ball detector

For the next agent working in `ball_classical/`. Read `README.md` first (setup, how to run, files, training-data
provenance). This file is the state, the decisions and the rules that README doesn't hold.

## Working rules (from the user)
- Be concise: telegraphic bullets, no filler, keep every fact.
- **ATD ball (Metrica automatic tracking) is NOT ground truth and NOT a reference.** The user found errors in it. Never
  judge, tune or recommend based on "agreement with ATD" or "frames where ATD has no ball". `run_tactical.py`/`track.py`
  still print ATD columns; ignore them. Judge changes by: the user watching rendered videos (main signal), possession GT
  H/A/D (frames 11185–18684, for checking only), physical plausibility (jumps, speeds).
- **ATD player slot ids swap between players: never rely on them.** Follow a player by position frame to frame.
- Before changing the candidate pipeline (thresholds/morphology in `candidates.py` / `TACTICAL_CFG`), propose it and get
  the user's OK.
- Never overwrite a model or an earlier output: new models get a unique `--out` name, new runs a `--tag` or `--out-dir`.
  The user compares videos side by side.
- RULES.md: no hand labels on the provided match in any amount. The public GT (11185–18684) is for checking only, never
  for choosing parameters (RULES.md:77-80). The user is fine with looking at that window's video. Outputs derived from
  the match go to `../data/ball_classical/` (gitignored), never commit or publish them.
- User REJECTED an inference-time "out of play" rule (ball last seen near a line then missing → stop). Don't propose it again.
- No `.venv` in this repo: run with `C:\Users\Mohamed\Desktop\barca\gp\Computer_Vision_Fifa\ball_detection_classical\.venv\Scripts\python.exe`.

## Current state (user approved, 2026-11-02)
- Model `models/ball_svm.pkl` = RBF SVM, HOG+colour, trained on `dataset/` + `dataset_barca_real/` (FC 27 gameplay).
  Retraining with `python train.py --features hog+color --out <path>` reproduces it exactly (needs the crop PNGs,
  which are not committed; they are in the original folder).
- `track.py` defaults = the output the user picked (`../data/ball_classical/strict_r04_gap2_carried/`):
  - path selection unchanged: V_MAX 30 m/s, R0 1 m, FREE_RESTART 125, BIAS 0, SKIP_COST 0, CONTINUITY 0.1, LAMBDA 0.05,
    NEW_TRACK 4, JUMP_PER_M 0.2, MIN_SCORE −1, VEL_W 0, hermite gap filling.
  - **boot rule `--boot-mode strict`, BOOT_R 0.4 m**: path chosen with every candidate, then path detections within
    0.4 m of an ATD player become gaps. No other candidate is picked for that frame.
  - **MAX_GAP 50 frames (2 s)** gap filling.
  - **carried flag**: a filled gap whose detections at both ends are within 1 m of one player (followed by position:
    ≤0.5 m/frame, missing ≤5 frames, two candidates = ambiguous = no flag) is status `carried` (same curve).
    possession.py counts `carried` frames as evidence; plain `interp` frames only hold.
  - player positions come from `data/tactical/BAR-ATH_tactical_atd_original.txt` (see ATD splice below).
- 5-min window (11185–18684) with the defaults: det 4979, interp 1541, carried 860, none 120; 318 detections turned into gaps.
- Earlier outputs reproduce exactly with flags:
  - approved batch3hc: `--boot-mode off --max-gap 25 --no-carried`
  - boot_attach: `--boot-mode penalty --boot-r 0.3 --max-gap 25 --attach --no-carried`
- **ATD splice**: `data/tactical/BAR-ATH_tactical_atd.txt` has our (batch3hc) ball in place of Metrica's for frames
  11185–18684 (players untouched). Original kept as `BAR-ATH_tactical_atd_original.txt`. Anything that needs Metrica's
  ball or the players reads the original.

## History: what was tried and why it ended up this way
- Models (old files remain in the original folder `C:\Users\Mohamed\Desktop\barca\gp\Computer_Vision_Fifa\ball_detection_classical\models\`):
  - batch1/batch2: Barça gameplay only, HOG+colour. batch2 was in use until batch3.
  - batch3: + Barça–Real gameplay, `train.py` auto-picked HOG+colour+**shape** (won FC 27 test AP by <0.001). On the real
    match it output far less (ball-alive raw 70.8 → 58.3%). Cause: shape features (blob area, vertical position in frame)
    are camera-specific; FC 27 balls are bigger/sharper than the 4–10 px tactical ball.
  - batch3 with HOG+colour only (= current model): coverage back to batch2 level → user preferred it. Lesson: `train.py`'s
    auto-pick uses FC 27 only; always pass `--features`.
  - More FC 27 labelling/retraining on all FC 27 data: recommended against (FC 27 gains plateaued 97.7→98.0% top-1, didn't transfer).
- Tracker:
  - Skip-cost fix: SKIP_COST 0.1 made a sparsely seen ball cheaper to skip than to follow. Now 0.
  - Jumps: links up to 125 frames back; reach = 30 m/s × gap + 1 m. Beyond reach = jump, not interpolated.
  - Hermite gap filling: user "love it".
  - Velocity-change penalty (`--vel-w`, off): dropped detections. Not adopted.
  - Boots (main tracker problem per user). Tried on the 5-min window, all kept as options:
    - penalty −0.5 within 0.3 m (`boot_attach/`): only 41 detections changed.
    - reject before path selection (`boot_reject*/`): never made the tracker pick another candidate on this window.
    - strict (current): user wants a boot frame interpolated, never replaced by another candidate.
    - radius: 0.35 vs 0.4 m; 0.4 m chosen after video review. Wider rejects real dribbles (ball 0.3–1 m ahead).
      Check uses the detection's centre point only.
    - `--attach` (ball follows the ATD player position through a carried gap): user rejected for the default; they
      want the ball from our detections/curve only, with a flag for possession.
    - gap limit 1 s / 1.5 s / 2 s: none frames 620 / 296 / 120 at 0.4 m; 2 s chosen.
  - Known blind spots of the boot rule (both ATD-based): ATD player point off by ~0.5 m from the real feet; ATD missing
    the player in that frame (no zone at all).

## Next steps (not started)
- Boots: detect players ourselves (person detector, reject at the bottom of the box in pixels) or add boots as hard
  negatives pseudo-labelled outside 11185–18684. Both fix the ATD blind spots above.
- **Far-side detection** (approved earlier): far-side ball detected less (6–7 px vs 9 px). Fix 1: looser white threshold
  (now V≥170, S≤60) and smaller min area (area_far now (4,120)) for the far side only. Changes the candidate pipeline:
  show the user exact values first.
- Run the pipeline on the full match (only 11185–18684 has candidates now; `run_tactical.py` ≈5 min per 7500 frames).
- Output during dead ball: with the 2 s gap limit 100% of D frames get a ball (ball-alive 98.0%). Don't use the rejected "out of play" rule.

## Tips
- Run scripts from this folder with `-u` (unbuffered). Don't pipe long runs through grep.
- An mp4 can't be opened until writing finishes. Use `open_video_at` (grab) for frame-exact runs, not `cap.set`.
- `track.py --first/--last` can select a sub-range of an existing `run_tactical.py` output (`--stem`).
- `run_tactical.py` always writes `classical_vs_atd_<first>_<last>*`. Use `--out-dir` to keep earlier runs; for a new
  tracker variant copy the two `classical_vs_atd_*` CSVs into a new folder and run `track.py --out-dir` there.
