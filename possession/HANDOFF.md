# Handoff: possession logic

For the next agent working in `possession/`. Also read `../ball_classical/HANDOFF.md` (working rules, ball tracker).

## Working rules (from the user)
- Be concise: telegraphic bullets, keep every fact.
- Target the ranking metric **spellF1** (possession spells matched by temporal IoU); see `../EVALUATION.md`.
- Judge versions by the **user's video review** (`debug_view.py`). The public GT (11185–18684, 17 possessions) is a
  check only, never used to pick parameters (RULES.md:77-80). One possession ≈ 0.06 spellF1 there: noise.
- Rules, not learning: no model trained on this match. **No hand-fixed frames**: every fix must come from a detector.
- ATD is not a reference; ATD player slot ids swap, never use them. Follow players by position.
- Dead ball (D) not done yet: every frame after the kick-off is H or A.
- New runs: always a new `--tag` / `--out-dir`. Outputs go to `../data/possession/` (gitignored).
- Propose before changing logic; the user decides. Commit only when asked; never push to `origin` (organisers' repo),
  private remote is `mine`.
- Python: `C:\Users\Mohamed\Desktop\barca\gp\Computer_Vision_Fifa\ball_detection_classical\.venv\Scripts\python.exe`.
  Run from `possession/`.

## Current logic: v3 (`touches.py` + `possession_v3.py` + `tagging.py`, agreed with the user)
Ball input: our tracker CSV (default `../data/ball_classical/strict_r04_gap2_carried/`), ATD original ball where ours
is missing (`possession.compute()`). Interp and carried count like detections.

**Touches (`touches.py`)**: the ball's path decides THAT something touched it, players only decide WHO.
0. Clean path: cut at steps > MAX_SPEED 35 m/s (air / tracker jump) and at switches to/from ATD ball; mean over ±2
   frames inside each piece; velocities = least-squares line, never across a cut.
1. Ball change per frame, v_in over W=3 frames before, v_out over 3 after. Fires on any of:
   turn ≥ 25° with both speeds ≥ 2 m/s and the new direction holding the next 3 frames; slow down (in ≥ 5, out ≤ 0.5·in);
   kick from slow (in < 3, out ≥ 8). Fast ball speeding up straight never fires (long balls in the air).
2. Event = consecutive firing frames; moment = biggest velocity change.
3. Who: nearest per team over the event. One team within R=1.5 m → its touch; both within R and < 0.5 m apart →
   contested; nobody within R → same test at R_FAR=3 m, marked `far`; nobody within 3 m → unknown (ignored).
4. Hold: one player (followed by position) within R ≥ 25 frames, other team not within R.
5. Bounce flag: velocity change along the camera line of sight (|cos| ≥ 0.98, from the homography) → `bounce`.
   **Review only**: possession still counts these touches. Skipping them (v3d) lost 2 real changes (12674, 17360).

**Possession (`possession_v3.py`)**
- Sequence = consecutive touches/holds of one team, unbroken by the other team; contested/unknown skipped (neither add
  nor break). Controlled = ≥ 2 touches or a hold.
- Change only when the other team has a controlled sequence; change point = its first event (so a deflection to a
  teammate puts the change on the deflection — accepted). Single touch back to the old team = no change. No gap limit.
- Kick-off: inside each smart-tagging kick-off tag (+1 s), kick = last frame ball ≤ 2 m from centre spot before it is
  ≥ 5 m away (no fixed offset). Two checks: team nearest the spot, team whose half the ball moves into. Agree →
  possession theirs from the kick; disagree → not used.
- Sides: per half at the smart-tagging "Kick Off (Start)" tag, from each team's mean x (`tagging.sides`).
- Goalkeepers: ATD metadata has one Goalkeeper slot per team (23 Barca, 45 Athletic). A keeper slot in the half its
  team attacks gets the other team on that frame (`teams_f`). Fixes a phantom Athletic keeper at Barca's goal
  11185–12661.

**Smart tagging facts**: kick-off tags have Team N/A; no team-side field ("Side" = wing in final third). Set pieces
(throw-in, corner, goal kick, free kick) carry a team. Tag time × 25 = video frame; tag starts a few s before the kick.

## Results (public window, check only)
| version | spellF1 | spellF1 noD | macroF1 | H/A acc | spells (GT 17) |
|---|---|---|---|---|---|
| baseline_smooth50 | 0.326 | 0.371 | 0.621 | 83.5% | 25 |
| v2 (proximity + K/M hand-over) | 0.373 | 0.497 | 0.508 | 84.7% | 12 |
| v3 (touches) | 0.560 | 0.691 | 0.541 | 90.0% | 16 |
| **v3b/v3c** (+ kick-off, keeper fix) | **0.558** | 0.658 | 0.563 | **94.5%** | 14 |
| v3d (bounce touches skipped, reverted) | 0.529 | 0.621 | 0.542 | 91.1% | 12 |
- v3→v3b spellF1 dip: removing a false Barca spell (15649–15762) merged two of our A spells; GT has 3 A spells there,
  split by short H spells we miss (15346–15365, 15766–15799).
- Remaining errors (v3c): **1469 GT-D frames (20%) all predicted H/A** (D class F1 = 0); 331 wrong-team frames (4%).

## Debug view (`debug_view.py`)
- `render --layout v3 --tag _v3c_r04_gap2` → `../data/possession/possession_<first>_<last><tag>_debug.mp4` + marks.
- `view --tag <t>`: space play, ←/→ step, n/p changes, m/M GT disagreements, t/T touches, b/B bounce-flagged
  touches, g go to frame (needs a terminal: run it yourself).
- v3 layout: R and R_FAR circles, event labels, speed; panel = ball change (speeds, turn, tests, in/out arrows), who,
  result (+ line-of-sight |cos|), ours/GT + change banner; timeline = ours, GT, ball (red = path cut), speed, change,
  event, hold; dashed line = change point, solid = confirmed.
- Other layouts: `touch` (touches only), `full` (v2).

## Open
- **Dead ball** (next, the big one): what starts D (out of play, foul, goal), what ends it (restart touch); smart
  tagging set pieces with team as inputs. Design with the user first (`/grill-with-docs`).
- Late changes: 11351 vs ours 11448 (Barca player not detected — ATD), 15346, 15766; short H spells missed.
- Bounces: line of sight alone flags 37 touches incl. real ones; needs a better signal if revisited.
- Full match: needs `run_tactical.py` on all frames first.
