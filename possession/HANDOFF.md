# Handoff: possession logic

For the next agent working in `possession/`. Also read `../ball_classical/HANDOFF.md` (ball tracker, out detection) and
`../GLOSSARY.md` (the words used below). Constants live in the module docstrings; this file says what and why.

## Working rules (from the user)
- Be concise: telegraphic bullets, keep every fact. Explain simply when asked ("explain it simply").
- Target the ranking metric **spellF1** (possession spells matched by temporal IoU); see `../EVALUATION.md`.
- Judge versions by the **user's video review** (`debug_view.py`). The public GT (11185–18684, 17 spells) is a
  check only, never used to pick parameters (RULES.md:77-80). One spell ≈ 0.06 spellF1 there.
- Rules, not learning. **No hand-fixed frames or offsets**: every fix comes from a detector, expressed as a general rule.
- ATD is not a reference; ATD player slot ids swap, never use them. Follow players by position.
- **Propose before changing logic** (say what changes and why, then wait); the user decides. Commit only when asked;
  never push to `origin` (organisers' repo); private remote is `mine`, branch `ball-and-possession`.
- New runs: always a new `--tag`. Outputs go to `../data/possession/` (gitignored), never overwrite or delete them.
- Python: `C:\Users\Mohamed\Desktop\barca\gp\Computer_Vision_Fifa\ball_detection_classical\.venv\Scripts\python.exe`
  (has scipy, OpenCV, imageio-ffmpeg). The bare `python` is a different install without them. Run from `possession/`.
- Smart tagging: **not used for dead ball** (user decision). Only the v3 kick-off detection and half sides use the
  kick-off tags.

## Current version: v4 (`possession_v4.py` = v3 + `deadball.py` + `whistle.py`)
`python possession_v4.py` → `../data/possession/possession_11185_18684_v4_out3_r04_gap2{.csv,_submission.csv}`
(identical to the reviewed `_v4m_out3_r04_gap2`). Default ball track: `../data/ball_classical/out3/` (with ball-out
detection). Interpolated / carried / out frames all have positions; touches ignore `out` frames.

### Touches (`touches.py`, unchanged from v3)
The ball path decides THAT a touch happened, players decide WHO. Clean path (cut at > 35 m/s steps, at ATD switches and
at in/out switches; ±2 frame mean); turn ≥ 25° / slow down / kick from slow (in < 3, out ≥ 8 m/s). Who: one team
within 1.5 m, contested if both within 1.5 m and < 0.5 m apart, else 3 m `far`, else unknown. Hold = one player within
1.5 m ≥ 1 s. Bounce flag (line of sight) is review only.

### Possession (`possession_v3.py`, unchanged)
Change only when the other team has a controlled sequence (≥ 2 touches or a hold); change point = its first event.
Kick-off (smart-tagging tag window): kicking team when nearest-to-spot and ball-into-half agree. Goalkeeper slot in the
half its team attacks gets the other team (`teams_f`).

### Dead ball (`deadball.py`), three questions, scanned once through the match
1. **Is play paused?** Pause = players slow (median speed, 1 s mean, < 1.4 m/s) ≥ 2 s, plus one sign:
   - ball still (ball speed, 0.5 s mean, < 1 m/s);
   - crowded end: ≥ 14 players between one goal line and the 16.5 m line, full width (corner shape);
   - whistled: a referee whistle, then players slow (a ≥ 2 s slow stretch starting ≤ 3 s after it);
   - outed: ball seen over a line, then the first ≥ 2 s slow stretch, however late (goal kicks), no real kick between.
   Ends at a fast ball (≥ 8 m/s, measured: interpolated / carried speeds are invented and never count). After a
   whistle or the ball out, only a **real kick** ends it (touch "kick from slow"). In a crowded end, a ball inside that
   end's penalty box is **feet** (tracker on boots): never ends it, never the restart kick.
2. **When does play restart?** First kick after the pause (not feet). Kicker's team gets the ball from the kick, even
   if it goes to an opponent (v3 rules take over). No kick: hold right after, else nearest team, else no D.
3. **When did play stop?** Walking back from the pause start, not before the last confirmed controlled sequence or the
   previous restart: ball seen over a line → end of the last referee whistle → ball came to rest → pause start.
   (User decisions: D starts when play stops, not at the foul; at the **end** of the whistle — middle tried, same
   score, reverted; "near a line" removed as too weak.)
- Keeper check: ball resting with a keeper in his own box (a catch is play) → D only with ball out or a whistle.
- v4 labels: D from "play stopped" to the frame before the restart; touches inside D ignored; restart team set at the
  kick, then v3.
- Why pause and "play stopped" are separate: the pause is reliable but 1–3 s late; the stop moment is exact but easy to
  fake (fans, ball near lines). Pause confirms, walk-back dates.

### Referee whistle (`whistle.py`)
Audio cut from the tactical video with imageio-ffmpeg's ffmpeg, cached in `../data/audio/`. Whistle = ≥ 2 perfectly flat
tones starting together (multi-chamber whistle); fans = single gliding tones. Referee tones 3830 / 4275 / 4730 Hz,
**measured at the smart-tagging kick-off whistles outside the public window** (113707, 163072), ±2 %. In the window:
2 referee whistles (12835–12848, 15880–15896), 5 fan whistles rejected. `python whistle.py --first A --last B` lists them.

## Results (public window, check only)
| version | macroF1 | spellF1 | spells (GT 17) |
|---|---|---|---|
| baseline_smooth50 | 0.621 | 0.326 | 25 |
| v3c (touches, no D) | 0.563 | 0.558 | 14 |
| v4 first dead ball (old track) | 0.775 | 0.556 | 14 |
| + ball-out detection (out3 track) | 0.812 | 0.594 | 12 |
| v4j: whistle pause, corner end strip, no invented speeds | 0.884 | 0.635 | 12 |
| **v4m: + outed pause (goal kick)** | **0.932** | **0.676** | 12 |

v4m dead balls vs GT: free kick 12849–13026 (GT 12851–12988), free kick 15897–16397 (GT 15889–16399), corner
17339–17854 (GT 17380–17837), goal kick 18025–18482 (GT 18027–18388). All 4 found.

## Open (most spellF1 now lost on possession changes, not dead ball)
- **Restart kicks not detected as touches**: free kick at ~12975 (we restart 13027) and goal kick at ~18389 (we restart
  18483). Ball speeds up gradually in the tracking, never passes "kick from slow". Idea (not agreed): ball still, then
  ≥ 4 m/s within 0.5 s.
- **3 short Barca spells missed** (15346–15364, 15766–15798, 17293–17379): need 2 touches / hold; each miss merges A
  spells around it (costs ~2 GT spells).
- Change points off: 11351 (ours 11447, Barca player missing in ATD), 13704 (ours 13671).
- 14837: Barca touch by distance (0.27 vs 1.13 m), GT goes A at 14846; user to check video. Possible rule: contested
  when both teams within R on a kick from slow.
- Full match: needs `run_tactical.py --out-margin 3` + `track.py` on all frames first.

## Debug view (`debug_view.py`)
- `python debug_view.py render --layout v4 --track ../data/ball_classical/out3/tracked_11185_18684.csv --tag <tag>`
  → `../data/possession/possession_11185_18684<tag>_debug.mp4` (~5 min). `view --tag <tag>` to step through it
  (needs a window; the user runs it or asks you to). Latest rendered: `_v4m_out3_r04_gap2`.
- Keys: space play, ←/→ step, l/j ±5 s, n/p changes, m/M GT disagreements, t/T touches, b/B bounce flags,
  r/R restarts, g go to frame (terminal only), +/- speed, q quit.
- v4 panel bottom: dead-ball signs at this frame (green = on): players(1s), ball(0.5s), end N, whistled, outed, out,
  feet, whistle; then "D <start>-<end> by <why> | wait | restart <frame> <team> (<source>)". Timeline "dead" row: grey
  = D, low white = pause, cyan = referee whistle, thick yellow line = restart. Ball row white = ball out.
- Video time = (frame − 11185) / 25.
- Other layouts: `v3`, `touch`, `full` (v2).
