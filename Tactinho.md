# Tactinho: ball possession, technical report

Match: BAR-ATH · Code: this repository

## 1. Approach

Every frame is labelled **H** (Barcelona), **A** (Athletic) or **D** (dead ball) by **hand-written rules, not a
trained possession model**. There is no external possession data to train on, and rules are easy to read and check.

The pipeline has four steps:

1. **Ball:** our own detector and tracker.
2. **Touches:** the moments a team plays the ball.
3. **Possession:** changes only when the other team really controls the ball.
4. **Dead ball:** from pauses, outs, referee whistles and kick-offs.

## 2. Assets used

| Asset | Use |
|---|---|
| Video frames | our ball detector |
| Video audio | referee-whistle detector |
| Homography | image ↔ pitch metres |
| ATD players | who is near the ball. Followed by position; slot ids are not used. |
| ATD ball | **inside dead balls only**: the restart kick and who takes it. Our tracker confuses boots with the ball at set pieces. |
| Smart tagging | **only kick-off tags**, as search windows. The players must also be in kick-off formation. |
| Public ground truth | format and alignment checks only, never for tuning |

## 3. Training data

The only trained part is a small ball classifier (`ball_classical/models/ball_svm.pkl`): an SVM on HOG and colour
features of ball-candidate crops.

| What | How | By whom | How much |
|---|---|---|---|
| **EA SPORTS FC 27 gameplay** (a video game) recorded by a team member, **not the provided match** | click labelling of white-blob candidates (`label_tool.py`); later batches pre-labelled by the model and corrected by hand | Mohamed Badreldin | 590 ball / 9,543 not-ball crops |

The label CSVs are in the repository (`ball_classical/dataset/`, `ball_classical/dataset_barca_real/`). The
10,133 crop images (21 MB) are published as a release asset:
https://github.com/MohamedBadr-14/more-than-a-hack-2027/releases/download/training-data-v1/tactinho_ball_crops.zip

To retrain: unzip it into `ball_classical/`, then run `python train.py`. The submission itself needs no retraining:
the trained model is committed.

**Rights:** EA's content policy (https://help.ea.com/en/articles/security-and-rules/ea-content-policy/) allows game
content, including gameplay, to be used for personal projects, as long as it is not used commercially, does not imply
EA endorsement, and is not combined with other brands. We believe our non-commercial use for this competition falls
within this.

## 4. Method

**Ball.**
- White blobs on the pitch are scored by the SVM.
- An offline tracker picks the most plausible path (≤ 30 m/s).
- Detections on a player's boot are dropped, and short gaps are filled.

**Touches.**
- A touch is a sharp change in the ball's path: a turn, a stop, or a kick from slow.
- It belongs to the team with a player within 1.5 m (3 m as a second chance).
- A hold is a player staying with the ball.

**Keeper catches.**
- When a goalkeeper in the box catches the ball and our detector loses it, a second ball track keeps the ball on
  the keeper until a kick or throw, so the hold counts as the keeper's team's touch.
- Open play only: catches inside a dead ball are ignored.
- The touches come from this track. Dead balls come from the normal track, so they are unchanged.

**Possession.** It stays with the current team, through loose and contested balls, until the other team has a
controlled sequence (2 touches or a hold).

**Dead ball (D).**
- **Pause:** players slow down for ≥ 2 s, together with one of: the ball still, a crowded box (corner), a referee
  whistle, or the ball out.
- **Start:** when the ball went out, else the whistle, else when the ball stopped.
- **Restart:** the first kick of the ATD ball, else the players clearly playing again.
- **Restart team:** the team of the first touch after the restart.
- **A clear out is a dead ball by itself** (quick throw-ins). The ATD ball can veto an out our tracker sees by mistake,
  for example a long ball in the air drawn past the line.
- **Referee whistle:** two tones at the ratio of the referee's whistle.
- **Kick-offs:** the ball on the centre spot and the players in kick-off formation. A kick-off ends D. After a goal, D
  covers the celebration up to the kick-off.
- **Half edges:** D before each half's kick-off and after its final whistle.

## 5. Compliance with the rules clarification (17 Nov 2026)

| Rule | Our work |
|---|---|
| No hand labels or manual data on the game | **Complies.** Nothing on the match was labelled or corrected by hand, and no rule refers to a specific frame. |
| No training or tuning on the ground truth | **Complies.** The public window was used only to check format and alignment. |
| No training or calibrating on the provided video | **Complies.** All adjustments were based on logical and physical considerations to ensure accurate and consistent operation, without training or calibrating on the provided video. |
| No uploading provided data to AI services | **Complies** No video or audio file was uploaded. |
| Allowed inputs, intermediate files, licensed datasets | **Complies.** All inputs are on the allowed list.Training data is our own FC 27 recordings. |

## 6. Limitations

- **Set pieces:** where ATD also has no ball, restarts can be late.
- **Balls in the air:** they are projected in the wrong place, which can create false outs.
- **Missing restart taker:** when the taker is not tracked, the restart team comes from the receiver.

## 7. Reproduction

**Setup:** Python 3.13, `pip install -r requirements.txt` (all versions pinned). Unpack
the data bundle into `data/` (`DATA.md`), and copy `data/tactical/BAR-ATH_tactical_atd.txt` to
`BAR-ATH_tactical_atd_original.txt`.

```bash
cd ball_classical          # ~2 h per half
python run_tactical.py --first 11185  --last 90651  --out-margin 3 --no-video --out-dir ../data/ball_classical/full_h1
python track.py        --first 11185  --last 90651  --stem classical_vs_atd_11185_90651   --out-dir ../data/ball_classical/full_h1
python run_tactical.py --first 110808 --last 190197 --out-margin 3 --no-video --out-dir ../data/ball_classical/full_h2
python track.py        --first 110808 --last 190197 --stem classical_vs_atd_110808_190197 --out-dir ../data/ball_classical/full_h2
cd ../possession           # dead balls (v5k) on the normal track; ~5 min per half, ~9 GB RAM, one at a time
python possession_v5.py --first 11185  --last 90651  --track ../data/ball_classical/full_h1/tracked_11185_90651.csv   --tag _v5k
python possession_v5.py --first 110808 --last 190197 --track ../data/ball_classical/full_h2/tracked_110808_190197.csv --tag _v5k
cd ../ball_classical       # keeper track: ball kept on the goalkeeper during open-play catches (outside v5k D)
python track.py --first 11185  --last 90651  --stem classical_vs_atd_11185_90651   --out-dir ../data/ball_classical/full_h1 \
                --no-video --keeper-hold --dead-csv ../data/possession/possession_11185_90651_v5k.csv   --tag _keeper
python track.py --first 110808 --last 190197 --stem classical_vs_atd_110808_190197 --out-dir ../data/ball_classical/full_h2 \
                --no-video --keeper-hold --dead-csv ../data/possession/possession_110808_190197_v5k.csv --tag _keeper
cd ../possession           # final labels: touches from the keeper track, dead balls from the normal track
python possession_v5.py --first 11185  --last 90651  --track ../data/ball_classical/full_h1/tracked_11185_90651_keeper.csv \
                        --dead-track ../data/ball_classical/full_h1/tracked_11185_90651.csv   --tag _v5k_keeper
python possession_v5.py --first 110808 --last 190197 --track ../data/ball_classical/full_h2/tracked_110808_190197_keeper.csv \
                        --dead-track ../data/ball_classical/full_h2/tracked_110808_190197.csv --tag _v5k_keeper
python stitch.py ../data/possession/possession_11185_90651_v5k_keeper_submission.csv \
                 ../data/possession/possession_110808_190197_v5k_keeper_submission.csv --out ../Tactinho.csv
python ../evaluation/validate_submission.py ../Tactinho.csv
```
