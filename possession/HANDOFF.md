# Handoff: possession logic

For the next agent / colleague working in `possession/`. Also read `../ball_classical/HANDOFF.md` (ball tracker, out
detection) and `../GLOSSARY.md` (the words used below). Constants live in the module docstrings; this file says what
and why.

## Working rules (from the user)
- Be concise: telegraphic bullets, keep every fact. Explain simply when asked ("explain it simply").
- Target the ranking metric **spellF1** (possession spells matched by temporal IoU); see `../EVALUATION.md`.
- Judge versions by the **user's video review** (`debug_view.py`). The public GT (11185–18684, 17 spells) is a
  check only, never used to pick parameters (RULES.md:77-80). One spell ≈ 0.06 spellF1 there.
- Rules, not learning. **No hand-fixed frames or offsets**: every fix comes from a detector, expressed as a general rule.
  User listening / watching notes are for checking, never tuning.
- ATD is not ground truth; ATD player slot ids swap, never use them. Follow players by position. The raw ATD **ball**
  is used as an input inside dead balls (v5c+, organiser-provided input, user decision).
- **Propose before changing logic** (say what changes and why, then wait); the user decides. Commit only when asked;
  never push to `origin` (organisers' repo); private remote is `mine`, branch `ball-and-possession`.
- New runs: always a new `--tag`. Outputs go to `../data/possession/` (gitignored), never overwrite or delete them.
- Python: `C:\Users\Mohamed\Desktop\barca\gp\Computer_Vision_Fifa\ball_detection_classical\.venv\Scripts\python.exe`
  (has scipy, OpenCV, imageio-ffmpeg). The bare `python` is a different install without them. Run from `possession/`.
  Needs ~6 GB free RAM per run (whistle STFT): run halves one at a time.
- Smart tagging: not used for dead ball. Used only for kick-off windows (a tag → search the kick inside it).
- Whistles: never validated against smart tagging (user: tags not accurate).

## Current version: v5f (`possession_v5.py`, default tag `_v5f`)
Runs on any range inside ONE half; run each half, then `stitch.py`. Frames outside both halves (never scored,
DATA.md:262) are written D.
```
python possession_v5.py --first 11185 --last 90651 --track ../data/ball_classical/full_h1/tracked_11185_90651.csv --tag _v5f
python possession_v5.py --first 110808 --last 190197 --track ../data/ball_classical/full_h2/tracked_110808_190197.csv --tag _v5f
python stitch.py ../data/possession/possession_11185_90651_v5f_submission.csv \
    ../data/possession/possession_110808_190197_v5f_submission.csv --out ../data/possession/BAR-ATH_v5f_full_submission.csv
```
~5 min per half (ball tracks already made). Graded halves (organisers' frames, allowed): 11185–90651, 110808–190197.

### Data needed (`../data/`, gitignored)
- Organiser: `tactical/BAR-ATH_tactical.mp4`, `_atd.xml`, `_homography.csv`, `_smart_tagging.xml`.
- `tactical/BAR-ATH_tactical_atd_original.txt` = the organisers' original `_atd.txt`, renamed (v5 reads the ATD ball and
  the kick-off formation from it). `_atd_old.txt` is an identical copy (used by the viewer's `--atd`).
- `tactical/BAR-ATH_tactical_atd.txt` = the user's modified ATD (own ball on ~6k frames); ask the user for it.
- Ball tracks (~2 h per half to rebuild): `ball_classical/full_h1/tracked_11185_90651.csv`,
  `ball_classical/full_h2/tracked_110808_190197.csv` (`run_tactical.py --out-margin 3 --no-video`, then `track.py`).
- Optional: `audio/tactical_11185_90651.wav`, `audio/tactical_110808_190197.wav` (else cut from the mp4, ~1 min).

### What v5 adds to v4 (touches / possession / dead ball of v4 below still apply)
- **Referee whistles** (`whistle.referee_whistles`): v4 tone detector + **pair detector**: two simultaneous peaks at the
  ratio of the referee's two chambers (4275/3830 = 1.116 ±2 %), lower tone 3.5–4 kHz; hop counts at ≥ 18 dB,
  ≥ 10 hops, gaps ≤ 5 bridged, loudest hop ≥ 28 dB. 28 dB = lowest with **no detection in half-time** (fans whistle
  the same pitches). Strict on purpose: user checked half 1 by ear, 18 found, **no false positives**, some missed.
  `python whistle.py --first A --last B --pair`.
- **Sides** from the players, not tags: kick-off formation (≥ 7 outfield per team, means ≥ 10 m apart, nobody > 2 m into
  the other half; GK slots excluded) held 25 frames, searched over the whole half. H1: Barça defends left; H2: right.
- **Kick-offs**: in each smart-tagging kick-off tag window, else from a referee whistle to 5 s after it (+ formation).
  Kick = ball within 4 m of the centre spot (boot rule removes the ball under the kicker), then 5 m away; kick frame =
  where the ball is closest to the spot (`V3.kickoffs(closest=True)`).
- **Half start**: D from the half's first frame to the first kick-off.
- **Half end**: final whistle = first referee whistle after which players never play again (play = median player speed
  ≥ 1.4 m/s with ≥ 14 players seen, ≥ 2 s, ending > 3 s after the whistle). D from it to the half's end.
  H1: D 88703–90651; H2: D 189151–190197 (user confirmed 126:05).
- **Dead ball with the ATD ball** (`DEAD_ATD`, `--no-dead-atd` = v5). Reason: at set pieces our tracker locks on boots,
  so the dead ball ended too early.
  1. v5c: dead balls found as in v4, but a wait ends only on an **ATD kick** (raw ATD ball, no interpolation, ≥ 8 m/s
     between two ATD detections ≤ 0.5 s apart; 8 m/s = `touches.KICK_OUT`).
  2. v5e (`deadball.later_restarts`): same dead balls (same starts); each restart moved later to the first ATD kick or
     the start of players clearly playing (median ≥ 2 m/s for 2 s) after the wait start. A restart moved past the next
     dead ball's start merges the two. Kick-offs keep their restart. (v5d, ending inside `waits()`, made 23 new false
     D in open play: rejected.)
  3. v5f (`deadball.first_touch_team`): restart team = team of the first touch / hold within 5 s after the restart;
     else the player nearest the ATD ball if within `touches.R`; else the v5e team. (Old: nearest player to the ATD
     ball, wrong at corners / throw-ins where the taker is not in the ATD.)

## Results (public window, check only)
| version | macroF1 | spellF1 | spells (GT 17) |
|---|---|---|---|
| baseline_smooth50 | 0.621 | 0.326 | 25 |
| v3c (touches, no D) | 0.563 | 0.558 | 14 |
| v4m (outed pause) | 0.932 | 0.676 | 12 |
| v5 (half edges, pair whistles) | 0.932 | 0.676 | 12 |
| **v5c / v5e / v5f (ATD ball in dead balls)** | **0.941** | **0.667** | 13 |

v5e/v5f changes are all after frame 18996, so the public window cannot judge them: video only.
Full match v5 (before ATD): Barça 69.4 % / Athletic 30.6 % of live frames, D 22.6 % of graded frames.

## Open
- **Tried after v5f, reverted (pending video review)** — v5g/v5h, see commit 4e3827d:
  - 33895: first touch was a weak `far` touch by A, Barça held from 34018 → restart team skipping far touches.
  - 37185 goal kick: players' fallback fired while the keeper placed the ball (ATD ball 1.4 m/s); real kick 37389 →
    players' fallback only when ATD has no ball.
  - 37052: "ATD kick" = ball first seen 7 m behind the goal line at 44 m/s → fast run must start on the pitch (±1 m).
    Keeping only the first fast step (v5g) lost 9 real dead balls (fast steps during passes stop waits forming):
    public 0.789. v5h (drop whole run if it starts off pitch) lost none.
- Long D to check on video: 70193–73950 (46:47, ~150 s, ATD sees the ball 0–30 %, players slow), 58073–61661 (144 s,
  merged into the 39:49 goal stoppage), 86547–88500 (78 s).
- **Goal celebrations**: only 1 of 4 goals has a D before its kick-off (33:08, 50:21, 108:43 missing). Idea (not
  agreed): D from the goal (last ball out / walk-back) to the after-goal kick-off.
- Whistles with no D near them (H1: 22:36, 35:26, 41:36, 45:01, 58:47; H2: 8) — fouls with quick restart / advantage?
- Half 2 not yet run with v5c+; half-2 whistles (22) not checked by ear.
- `HANDOFF.md` of v4 open items still open: restart kicks not detected as touches, 3 short Barça spells missed in the
  public window, change points off at 11351 / 13704.

## Debug view (`debug_view.py`)
- Render: `python debug_view.py render --layout v5 --first A --last B --track <tracked.csv> --tag <tag>`
  → `../data/possession/possession_A_B<tag>_debug.mp4` (+ `_marks.csv`). Full half ≈ 55 min at ~26 fps.
- **Cache**: the computed dict is cached in `<out-dir>/cache/` keyed by range, layout, track and the logic code
  (`*.py` here + `render_compute`); drawing-only changes keep it. First render of a code version ≈ 5 min extra.
- **Clips**: `--clip A B` (repeatable) renders only those frames; labels still computed on `--first..--last`.
- **One video of many ranges**: `--clip ... --join NAME` → `possession_<first>_<last><tag>_NAME_debug.mp4` +
  `_frames.csv` (match frame per video row). Latest: `../data/possession/clips_v5f/…_v5f_deadballs_debug.mp4`
  (every H1 dead ball, start → restart + 10 s, 32 segments).
- View: `python debug_view.py view --layout v5 --first A --last B --track <csv> --tag <tag> [--out-dir D] [--join NAME] --atd`
  (needs a window; the user runs it). `--atd` draws the raw ATD ball (magenta ✕, from `_atd_old.txt`), `o` toggles.
- Keys: space play, ←/→ step, l/j ±5 s, n/p changes, m/M GT disagreements, t/T touches, b/B bounce flags,
  r/R restarts, k/K kick-offs, w/W whistles, s/S segments (joined video), o ATD ball, g go to match frame
  (typed in the terminal), +/- speed, q quit.
- v4/v5 panel bottom: dead-ball signs at this frame; "D <start>-<end> by <why> | wait | restart <frame> <team>
  (<source>)"; v5 adds a magenta line at half start/end D and kick-offs. Timeline "dead" row: grey = D, low white =
  pause, cyan = referee whistle, yellow line = restart, magenta = kick-off.
- Match video time = frame / 25 (e.g. 33900 = 22:36).
