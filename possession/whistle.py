"""Referee whistle from the tactical video's audio track (dead-ball evidence for deadball.py).

A referee whistle is two or more perfectly flat tones that start together (a multi-chamber whistle), about
0.2-2 s long. Fans whistle too, but a fan's whistle is one tone that glides up and down (curved line in the
spectrogram), and a crowd is many such lines starting at different times. So:

1. Spectrogram (STFT, NPERSEG samples, hop HOP s) in FMIN..FMAX.
2. Peaks: bins at least PROM dB above the local median of the spectrum (over +-MED_HZ) and a local maximum.
3. Lines: peaks linked frame to frame when within LINK_REL of the line's last frequency (gaps <= GAP_HOPS hops).
   A line is flat when it lasts >= MIN_DUR s and its frequency spread is <= FLAT_REL of its mean.
4. Whistle: >= MIN_LINES flat lines whose onsets are within ONSET s of each other; start / end = earliest onset /
   latest offset; freqs = the lines' mean frequencies. Whistles closer than MERGE s are one whistle.
5. Referee, not fans: >= MIN_LINES of its tones within TONE_REL of the referee's tones REF_TONES. REF_TONES were
   measured with steps 1-4 at kick-off whistles of the provided match, outside the public window.

Pair detector (pair_whistles, used by possession_v5 through referee_whistles): the referee's whistle wobbles, fades
and comes in short blasts, and its pitch moves with how hard it is blown (lower tone ~3.69-3.87 kHz), so tone
following (steps 3-5) misses many. But both chambers move together: the RATIO of its two tones stays ~REF_TONES[1] /
REF_TONES[0] (1.116). Fans: one tone (overtone at ratio 2).
6. Every hop: score = max over peak pairs (lower peak in PAIR_LOW, upper peak at RATIO +- PAIR_TOL) of the weaker
   peak's prominence.
7. Whistle: hops with score >= PAIR_ON, gaps <= PAIR_GAP hops bridged, >= PAIR_MIN hops, loudest hop >= PAIR_PEAK.
   PAIR_PEAK = the lowest value with no detection over the whole halftime (players off the pitch, fans whistling
   at the same pitches); weaker real whistles are missed on purpose (no false positives).

The audio is cut from the video with the ffmpeg binary bundled by imageio-ffmpeg and cached in ../data/audio/.

    python whistle.py --first 11185 --last 18684      # list whistles in a frame range
    python whistle.py --first 11185 --last 18684 --pair   # referee whistles: steps 1-5 or the pair detector
"""
import argparse
import os
import subprocess

import numpy as np

import possession as P

SR = 22050
NPERSEG = 2048          # samples (~93 ms, ~10.8 Hz bins)
HOP = 0.01              # s
FMIN, FMAX = 2000, 8000  # Hz searched
PROM = 15.0             # dB above the local median
MED_HZ = 200            # Hz each side for the local median
LINK_REL = 0.01         # a peak joins a line within 1 % of its last frequency
GAP_HOPS = 2            # hops a line may miss
MIN_DUR = 0.2           # s
FLAT_REL = 0.02         # max - min frequency over mean
MIN_LINES = 2
ONSET = 0.1             # s, onsets of a whistle's tones this close
MERGE = 0.15            # s, whistles this close are one
REF_TONES = (3830, 4275, 4730)  # Hz, the referee's whistle (see 5.)
TONE_REL = 0.02
PAIR_RATIO = REF_TONES[1] / REF_TONES[0]  # 1.116, the two chambers
PAIR_TOL = 0.02         # relative
PAIR_LOW = (3500, 4000)  # Hz, the lower tone
PAIR_ON = 18.0          # dB, a hop counts
PAIR_PEAK = 28.0        # dB, loudest hop of a whistle (halftime: no detection)
PAIR_MIN = 10           # hops (0.1 s)
PAIR_GAP = 5            # hops bridged
PAIR_CHUNK = 60.0       # s of audio per STFT


def audio(first, last):
    """(samples float mono, sample rate) of video frames first..last, cut once and cached."""
    import imageio_ffmpeg
    from scipy.io import wavfile
    os.makedirs(os.path.join(P.MTAH, "data", "audio"), exist_ok=True)
    path = os.path.join(P.MTAH, "data", "audio", f"tactical_{first}_{last}.wav")
    if not os.path.exists(path):
        video = P.require(["video"])[0]
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", f"{first / P.FPS:.3f}", "-t", f"{(last - first + 1) / P.FPS:.3f}", "-i", video,
                        "-vn", "-ac", "1", "-ar", str(SR), path], check=True)
    sr, x = wavfile.read(path)
    return x.astype(float) / 32768, sr


def flat_lines(x, sr):
    """[(t0, t1, mean Hz)] in seconds from the start of x."""
    from scipy.ndimage import median_filter
    from scipy.signal import stft
    hop = int(round(HOP * sr))
    f, t, Z = stft(x, sr, nperseg=NPERSEG, noverlap=NPERSEG - hop)
    band = (f >= FMIN) & (f <= FMAX)
    f, S = f[band], 20 * np.log10(np.abs(Z[band]) + 1e-9)
    med = median_filter(S, size=(2 * int(MED_HZ / (f[1] - f[0])) + 1, 1), mode="nearest")
    prom = S - med
    peak = (prom >= PROM) & (S >= np.roll(S, 1, 0)) & (S >= np.roll(S, -1, 0))
    active, done = [], []  # line = [first hop, last hop, [freqs]]
    for k in range(S.shape[1]):
        for fq in f[peak[:, k]]:
            near = [ln for ln in active if abs(fq - ln[2][-1]) <= LINK_REL * ln[2][-1] and ln[1] < k]
            if near:
                ln = min(near, key=lambda ln: abs(fq - ln[2][-1]))
                ln[1] = k
                ln[2].append(fq)
            else:
                active.append([k, k, [fq]])
        keep = []
        for ln in active:
            (keep if k - ln[1] <= GAP_HOPS else done).append(ln)
        active = keep
    out = []
    for a, b, fs in done + active:
        fs = np.array(fs)
        if t[b] - t[a] >= MIN_DUR and (fs.max() - fs.min()) <= FLAT_REL * fs.mean():
            out.append((float(t[a]), float(t[b]), float(fs.mean())))
    return sorted(out)


def whistles(first, last):
    """[dict(a, b, freqs, ref)]: a / b = video frames of the whistle's start / end, ref = referee's tones."""
    x, sr = audio(first, last)
    lines, out = flat_lines(x, sr), []
    used = set()
    for i, (t0, t1, fq) in enumerate(lines):
        if i in used:
            continue
        group = [j for j, (u0, _, _) in enumerate(lines) if j not in used and abs(u0 - t0) <= ONSET]
        if len(group) >= MIN_LINES:
            used.update(group)
            out.append(dict(a=first + int(round(min(lines[j][0] for j in group) * P.FPS)),
                            b=first + int(round(max(lines[j][1] for j in group) * P.FPS)),
                            freqs=sorted(round(lines[j][2]) for j in group)))
    merged = []
    for w in sorted(out, key=lambda w: w["a"]):
        if merged and w["a"] - merged[-1]["b"] <= MERGE * P.FPS:
            merged[-1]["b"] = max(merged[-1]["b"], w["b"])
            merged[-1]["freqs"] = sorted(set(merged[-1]["freqs"]) | set(w["freqs"]))
        else:
            merged.append(w)
    for w in merged:
        w["ref"] = sum(any(abs(fq - r) <= TONE_REL * r for r in REF_TONES) for fq in w["freqs"]) >= MIN_LINES
    return merged


def pair_score(x, sr):
    """(hop times s, score dB) per hop: step 6."""
    from scipy.ndimage import median_filter
    from scipy.signal import stft
    hop, n = int(round(HOP * sr)), int(PAIR_CHUNK * sr)
    ts, sc = [], []
    for s in range(0, len(x), n):
        s0 = max(s - NPERSEG, 0)
        f, t, Z = stft(x[s0:s + n + NPERSEG], sr, nperseg=NPERSEG, noverlap=NPERSEG - hop, boundary=None, padded=False)
        band = (f >= 3200) & (f <= 4900)
        f, S = f[band], 20 * np.log10(np.abs(Z[band]) + 1e-9)
        prom = S - median_filter(S, size=(2 * int(MED_HZ / (f[1] - f[0])) + 1, 1), mode="nearest")
        peak = (prom >= PAIR_ON / 2) & (S >= np.roll(S, 1, 0)) & (S >= np.roll(S, -1, 0))
        t = t + s0 / sr
        for k in np.flatnonzero((t >= s / sr) & (t < (s + n) / sr)):
            pi = np.flatnonzero(peak[:, k])
            best = 0.0
            for i in pi[(f[pi] >= PAIR_LOW[0]) & (f[pi] <= PAIR_LOW[1])]:
                m = np.abs(f[pi] / f[i] / PAIR_RATIO - 1) <= PAIR_TOL
                if m.any():
                    best = max(best, min(prom[i, k], prom[pi[m], k].max()))
            ts.append(t[k])
            sc.append(best)
    return np.array(ts), np.array(sc)


def pair_whistles(first, last):
    """[dict(a, b, freqs=[], ref=True, src='pair')]: step 7."""
    x, sr = audio(first, last)
    t, s = pair_score(x, sr)
    idx, spans = np.flatnonzero(s >= PAIR_ON), []
    for i in idx:
        if spans and i - spans[-1][1] <= PAIR_GAP + 1:
            spans[-1][1] = i
        else:
            spans.append([i, i])
    return [dict(a=first + int(round(t[a] * P.FPS)), b=first + int(round(t[b] * P.FPS)), freqs=[], ref=True, src="pair")
            for a, b in spans if (s[a:b + 1] >= PAIR_ON).sum() >= PAIR_MIN and s[a:b + 1].max() >= PAIR_PEAK]


def referee_whistles(first, last):
    """Referee whistles from steps 1-5 (ref only) or the pair detector, merged when closer than MERGE."""
    ws = [dict(w, src="tones") for w in whistles(first, last) if w["ref"]] + pair_whistles(first, last)
    out = []
    for w in sorted(ws, key=lambda w: w["a"]):
        if out and w["a"] - out[-1]["b"] <= MERGE * P.FPS:
            out[-1]["b"] = max(out[-1]["b"], w["b"])
            out[-1]["freqs"] = sorted(set(out[-1]["freqs"]) | set(w["freqs"]))
            out[-1]["src"] = "+".join(sorted(set(out[-1]["src"].split("+")) | {w["src"]}))
        else:
            out.append(dict(w))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--first", type=int, default=11185)
    ap.add_argument("--last", type=int, default=18684)
    ap.add_argument("--pair", action="store_true", help="referee whistles incl. the pair detector")
    a = ap.parse_args()
    if a.pair:
        for w in referee_whistles(a.first, a.last):
            m, s = divmod(w["a"] / P.FPS, 60)
            print(f"referee whistle {w['a']}-{w['b']} ({int(m)}:{s:04.1f}, {(w['b'] - w['a']) / P.FPS:.2f} s) by {w['src']}")
    else:
        for w in whistles(a.first, a.last):
            print(f"whistle {w['a']}-{w['b']} ({(w['b'] - w['a']) / P.FPS:.2f} s) tones {w['freqs']} Hz -> "
                  + ("referee" if w["ref"] else "not the referee's tones"))
