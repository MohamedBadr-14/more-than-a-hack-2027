# Handoff: possession logic

For the next agent working in `possession/`. Also read `../ball_classical/HANDOFF.md` (working rules, ball tracker).

## Working rules (from the user)
- Be concise: telegraphic bullets, keep every fact.
- Target the ranking metric **spellF1** (possession spells matched by temporal IoU); see `../EVALUATION.md`.
- Judge versions by the **user's video review** (`debug_view.py`). The public GT (11185–18684, 17 possessions) is a
  check only, never used to pick parameters (RULES.md:77-80). One possession ≈ 0.06 spellF1 there: noise.
- Rules, not learning: no model trained on this match.
- ATD is not a reference; ATD player slot ids swap, never use them. Follow players by position.
- Dead ball (D) is ignored for now (user's call). Every frame after the first confirmed controller is H or A.
- New runs: always a new `--tag` / `--out-dir`. Outputs go to `../data/possession/` (gitignored).
- Python: `C:\Users\Mohamed\Desktop\barca\gp\Computer_Vision_Fifa\ball_detection_classical\.venv\Scripts\python.exe`.

## Current logic (`possession.py`, agreed with the user)
1. Ball per frame: our tracker CSV (`--track`, default `../data/ball_classical/strict_r04_gap2_carried/`): det, interp,
   carried. Where our track has no ball: Metrica's ball from `BAR-ATH_tactical_atd_original.txt` (`atd`). The live ATD
   file carries our own ball in the window, hence the original.
2. Near frame: nearest player within R = 1.5 m and the nearest opponent at least MARGIN = 0.3 m further away (else
   contested, no evidence). Control frame: near + |v_ball − v_player| < V_REL = 4 m/s (central differences ±2 frames).
   The speed gate stops a slow ball rolling past an opponent from counting.
3. Evidence: control frames on det, carried, attached or atd frames. Plain interp frames only hold the state.
4. Hand-over: the other team needs K = 8 evidence control frames within the last M = 25. The switch is backdated to the
   start of the unbroken run of that team's near frames holding the first of them (the ball reaching him).
5. X only before the first confirmation (scorer back-fills it). Loose ball holds indefinitely. X inside play is never
   better (EVALUATION.md: spellF1 forward-fills X; macroF1 counts it as a miss).

## Results (public window, check only)
| version | ball input | spellF1 | macroF1 | poss (GT 17) | lat |
|---|---|---|---|---|---|
| baseline_smooth50 | ATD | 0.326 | 0.621 | 26 | +4 |
| v1 (no opp within R, plain backdate) | batch3hc | 0.359 | 0.504 | 12 | +9 |
| v2 (margin, near-run backdate) | boot_attach | 0.388 | 0.509 | 16 | +1 |
| v2 | strict 0.35 m, 1.5 s, attach | 0.387 | 0.510 | 13 | +1 |
- With GT D frames removed from scoring (perfect dead-ball upper bound): v2 spellF1 0.533, H/A macroF1 0.858.
- Every ball change that removed real sightings (wider boot radius, longer gap fill replacing ATD fallback) cost 1–2
  hand-overs: possession is short of evidence.
- **Not yet run**: possession on the new default ball (`strict_r04_gap2_carried`, with `carried` as evidence).

## Debug view (`debug_view.py`)
- `python debug_view.py render --tag <t>` → `../data/possession/possession_<first>_<last><t>_debug.mp4` + `_marks.csv`.
- `python debug_view.py view --tag <t>`: space play, ←/→ step, n/p hand-overs, m/M GT disagreements, g go to frame.
- Shows: video with players, nearest/opponent lines, R circle; minimap; distances, margin, v_rel, near/control flags,
  hit counters; ±5 s timeline (ours, GT, ball source, near, evidence, hits, hand-over lines).

## Open
- Run v2 on the new ball default and review with the debug view.
- More evidence for hand-overs (ideas: count long fills that end at the other team; lower K slightly).
- Dead ball (the big one: +0.15 spellF1 upper bound on the public window).
- Full match: needs `run_tactical.py` on all frames first.
