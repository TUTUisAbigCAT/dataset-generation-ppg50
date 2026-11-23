"""DREAMT 2.2.0 PPG at 50 Hz and the PPG -> 25 Hz IBI path shared by real and generated PPG.

PPG -> IBI (ppg_to_ibi; identical for real and generated PPG):
  1. Contiguous blocks: consecutive 30-s epochs are joined; nothing is processed per 30-s chunk.
  2. Each block is resampled to 100 Hz (50 Hz PPG is band-limited, so this adds no content),
     cleaned with nk.ppg_clean(method="elgendi"), peaks from nk.ppg_peaks, and every peak is
     refined by 3-point parabolic interpolation (sub-sample timing, no 500 Hz upsampling).
  3. IBI = diff(peak times). Intervals outside 0.3-2.0 s (optionally also those far from the local
     median) are bad. A short run of bad intervals (<= 2 intervals and <= 3 s, with good intervals
     on both sides) is filled on the 25 Hz grid by linear interpolation between the neighbouring
     good intervals; longer runs, runs at a block edge and samples outside the beats are 0.
     Good intervals are written as in WatchSleepNet: each 25 Hz sample holds the interval of the
     beat pair around it.
  4. The 25 Hz series is cut into 30-s epochs of 750 samples; an epoch with less than half of its
     samples valid (measured or filled) gets epoch_ok = False (zero it or mask it in the loss).

Extraction of one participant (extract_subject) builds on the PSG 30-s grid:
  * PPG: E4 BVP band-passed 0.5-8 Hz on the whole night (zero phase), anti-aliased to 50 Hz,
    stored per epoch [n, 1500] and not normalized;
  * stages (W=0, N1=1, N2=2, N3=3, R=4) and respiratory events at 50 Hz;
  * IBI from the stored 50 Hz PPG with the path above (the one generated PPG will take), the same
    with the extra relative rule, without any filling, from the raw 100 Hz BVP (the teacher's
    original route), with WatchSleepNet's per-epoch method, and from the PSG ECG (reference);
  * PPG-to-ECG lag and its drift, beat sensitivity/PPV and errors of every IBI variant against
    the ECG IBI, and per-epoch quality (template SQI, wrist motion, flat and missing fractions).
Raw CSVs are only read; flags never remove anything from the stored arrays.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np

SCHEMA = "dreamt_ppg50_extraction_v2"
FS_RAW = 100
FS_PPG = 50
FS_DETECT = 100
FS_IBI = 25
EPOCH_SECONDS = 30
EPOCH_ROWS = FS_RAW * EPOCH_SECONDS          # 3000
EPOCH_PPG = FS_PPG * EPOCH_SECONDS           # 1500
EPOCH_IBI = FS_IBI * EPOCH_SECONDS           # 750
STAGE_CODES = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
EVENT_COLUMNS = ("Obstructive_Apnea", "Central_Apnea", "Hypopnea", "Multiple_Events")
IBI_RANGE = (0.3, 2.0)
# 25 Hz sample status
MEASURED, INTERPOLATED, ARTIFACT, NO_BEAT = 0, 1, 2, 3
DEFAULT_CLEANING = dict(ibi_min=0.3, ibi_max=2.0, max_fill_intervals=2, max_fill_s=3.0,
                        relative_threshold=None, relative_window=11, min_valid_fraction=0.5)
DEFAULT_RELATIVE_THRESHOLD = 0.3


def sha256_file(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def cleaning_options(**overrides):
    unknown = set(overrides) - set(DEFAULT_CLEANING)
    if unknown:
        raise ValueError(f"Unknown cleaning options {sorted(unknown)}")
    options = dict(DEFAULT_CLEANING, **overrides)
    if not 0 < options["ibi_min"] < options["ibi_max"]:
        raise ValueError("Need 0 < ibi_min < ibi_max")
    return options


# --------------------------------------------------------------------------- epochs
def epoch_grid(stage_labels, *, epoch_rows=EPOCH_ROWS):
    """PSG 30-s grid from the stage column: offset, epoch start rows, codes, validity.

    The offset is the most common position of stage changes modulo one epoch;
    on_grid_fraction reports how many changes fall on it (1.0 for PSG scoring).
    An epoch is valid when all its rows carry the same scored stage.
    """
    labels = np.asarray([str(value) for value in stage_labels], dtype=object)
    codes = np.fromiter((STAGE_CODES.get(value, -1) for value in labels), dtype=np.int64, count=len(labels))
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    if changes.size:
        values, counts = np.unique(changes % epoch_rows, return_counts=True)
        offset, on_grid = int(values[np.argmax(counts)]), float(counts.max() / changes.size)
    else:
        offset, on_grid = 0, 1.0
    epochs = max((len(labels) - offset) // epoch_rows, 0)
    starts = offset + epoch_rows * np.arange(epochs, dtype=np.int64)
    if epochs:
        block = codes[offset:offset + epochs * epoch_rows].reshape(epochs, epoch_rows)
        first = block[:, 0]
        valid = (first >= 0) & np.all(block == first[:, None], axis=1)
    else:
        first, valid = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
    return dict(offset=offset, starts=starts, codes=first.astype(np.int64), valid=valid,
                on_grid_fraction=on_grid, changes=int(changes.size))


def epoch_runs(epoch_start_s, epoch_seconds=EPOCH_SECONDS):
    """Index arrays of epochs that follow each other without a gap (the contiguous blocks)."""
    starts = np.asarray(epoch_start_s, dtype=np.float64)
    if not starts.size:
        return []
    breaks = np.flatnonzero(np.abs(np.diff(starts) - epoch_seconds) > 1e-6) + 1
    return np.split(np.arange(starts.size), breaks)


def epoch_sample_times(epoch_start_s, fs, samples):
    return np.asarray(epoch_start_s, dtype=np.float64)[:, None] + np.arange(samples, dtype=np.float64)[None, :] / fs


# --------------------------------------------------------------------------- beats
def _fill_nan(signal):
    signal = np.asarray(signal, dtype=np.float64).copy()
    bad = ~np.isfinite(signal)
    if bad.all():
        return np.zeros_like(signal), bad
    if bad.any():
        index = np.arange(signal.size)
        signal[bad] = np.interp(index[bad], index[~bad], signal[~bad])
    return signal, bad


def refine_peaks_parabolic(signal, peaks, sampling_rate):
    """Sub-sample peak positions by 3-point parabolic interpolation across y[p-1], y[p], y[p+1].

    Returns (exact_peak_samples, exact_peak_times_s); the shift is clipped to +-0.5 sample and
    flat tops are left unchanged.
    """
    y = np.asarray(signal, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=np.int64)
    exact = peaks.astype(np.float64)
    valid = (peaks > 0) & (peaks < y.size - 1)
    p = peaks[valid]
    y_prev, y_curr, y_next = y[p - 1], y[p], y[p + 1]
    denominator = y_prev - 2.0 * y_curr + y_next
    safe = denominator < -1e-12
    delta = np.zeros(p.size, dtype=np.float64)
    delta[safe] = 0.5 * (y_prev[safe] - y_next[safe]) / denominator[safe]
    exact[valid] += np.clip(delta, -0.5, 0.5)
    return exact, exact / float(sampling_rate)


def ppg_peaks(signal, fs):
    """(cleaned, integer peaks): nk.ppg_clean(method="elgendi") and nk.ppg_peaks at fs."""
    import neurokit2 as nk

    signal, _ = _fill_nan(signal)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaned = np.asarray(nk.ppg_clean(signal, sampling_rate=fs, method="elgendi"), dtype=np.float64)
        try:
            if hasattr(nk, "ppg_peaks"):
                peaks = nk.ppg_peaks(cleaned, sampling_rate=fs, method="elgendi")[1]["PPG_Peaks"]
            else:                                        # neurokit2 < 0.2.4
                peaks = nk.ppg_findpeaks(cleaned, sampling_rate=fs, method="elgendi")["PPG_Peaks"]
            peaks = np.asarray(peaks, dtype=np.int64)
        except Exception:                                # flat or too short signal
            peaks = np.zeros(0, dtype=np.int64)
    return cleaned, peaks


def resample_to(signal, fs, target_fs):
    """Band-limited polyphase resampling; sample i of the result sits at time i / target_fs."""
    from scipy.signal import resample_poly

    if fs == target_fs:
        return np.asarray(signal, dtype=np.float64)
    ratio = Fraction(target_fs).limit_denominator(1000) / Fraction(fs).limit_denominator(1000)
    return resample_poly(np.asarray(signal, dtype=np.float64), ratio.numerator, ratio.denominator)


def ecg_is_inverted(cleaned, fs, *, window_s=2.0):
    """neurokit2's ecg_invert test on non-overlapping windows: the median of each window's
    largest-magnitude value is negative for an inverted ECG (its sliding version needs ~6 GB per night)."""
    x = np.asarray(cleaned, dtype=np.float64)
    x = x - np.nanmean(x)
    width = int(window_s * fs)
    blocks = x[: x.size // width * width].reshape(-1, width)
    if not blocks.size:
        return False
    extreme = blocks[np.arange(blocks.shape[0]), np.abs(blocks).argmax(axis=1)]
    return bool(np.nanmedian(extreme) < 0)


def ecg_beats(ecg, fs):
    """R-peak times in seconds (neurokit2, polarity checked, parabolic refinement)."""
    import neurokit2 as nk

    signal, _ = _fill_nan(ecg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaned = np.asarray(nk.ecg_clean(signal, sampling_rate=fs), dtype=np.float64)
        inverted = ecg_is_inverted(cleaned, fs)
        if inverted:
            cleaned = -cleaned
        peaks = np.asarray(nk.ecg_peaks(cleaned, sampling_rate=fs)[1]["ECG_R_Peaks"], dtype=np.int64)
    return refine_peaks_parabolic(cleaned, peaks, fs)[1], inverted


# --------------------------------------------------------------------------- PPG -> IBI
def ppg_blocks(epochs, fs, epoch_start_s, *, detect_fs=FS_DETECT, epoch_seconds=EPOCH_SECONDS):
    """Step 2 on every contiguous block: list of dicts (epochs, beat times, unrefined times, cleaned SQI)."""
    epochs = np.asarray(epochs, dtype=np.float64)
    samples = int(round(fs * epoch_seconds))
    if epochs.ndim != 2 or epochs.shape[1] != samples:
        raise ValueError(f"epochs must be [n, {samples}] at {fs} Hz; got {epochs.shape}")
    starts = np.asarray(epoch_start_s, dtype=np.float64)
    if starts.shape != (epochs.shape[0],):
        raise ValueError("epoch_start_s needs one start time per epoch")
    per_epoch = int(round(detect_fs * epoch_seconds))
    blocks = []
    for run in epoch_runs(starts, epoch_seconds):
        signal = resample_to(epochs[run].reshape(-1), fs, detect_fs)
        cleaned, peaks = ppg_peaks(signal, detect_fs)
        _, times = refine_peaks_parabolic(cleaned, peaks, detect_fs)
        bounds = [(k * per_epoch, (k + 1) * per_epoch) for k in range(run.size)]
        blocks.append(dict(epochs=run, beat_times_s=times + starts[run[0]],
                           unrefined_s=peaks / float(detect_fs) + starts[run[0]],
                           sqi=template_sqi(cleaned, peaks, detect_fs, bounds),
                           beats_per_epoch=np.bincount(peaks // per_epoch, minlength=run.size)[:run.size]))
    return blocks


def classify_intervals(beat_times, options):
    """Step 3 on one block: (ibi, status, fill) per interval; status 0 good, 1 filled, 2 zeroed.

    fill holds (t_start, ibi_before, t_end, ibi_after) for filled intervals (NaN elsewhere).
    """
    t = np.asarray(beat_times, dtype=np.float64)
    ibi = np.diff(t)
    n = ibi.size
    status = np.zeros(n, dtype=np.uint8)
    fill = tuple(np.full(n, np.nan) for _ in range(4))
    if not n:
        return ibi, status, fill
    bad = (ibi < options["ibi_min"]) | (ibi >= options["ibi_max"])
    if options.get("relative_threshold"):
        import pandas as pd

        local = pd.Series(np.where(bad, np.nan, ibi)).rolling(
            int(options["relative_window"]), center=True, min_periods=3).median().to_numpy()
        bad |= np.isfinite(local) & (np.abs(ibi - local) > options["relative_threshold"] * local)
    edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(np.int8), 0]))
    for a, b in zip(edges[::2], edges[1::2]):           # bad intervals a .. b-1, from beat a to beat b
        fillable = (b - a <= options["max_fill_intervals"] and t[b] - t[a] <= options["max_fill_s"]
                    and a > 0 and b < n)
        if fillable:
            status[a:b] = 1
            for array, value in zip(fill, (t[a], ibi[a - 1], t[b], ibi[b])):
                array[a:b] = value
        else:
            status[a:b] = 2
    return ibi, status, fill


def intervals_to_samples(sample_times, beat_times, ibi, status, fill):
    """25 Hz values and sample status from classified intervals (WatchSleepNet hold for good ones)."""
    sample_times = np.asarray(sample_times, dtype=np.float64)
    values = np.zeros(sample_times.shape, dtype=np.float64)
    code = np.full(sample_times.shape, NO_BEAT, dtype=np.uint8)
    if ibi.size == 0:
        return values, code
    k = np.searchsorted(beat_times, sample_times, side="right") - 1
    inside = (k >= 0) & (k < ibi.size)
    kk, tau = k[inside], sample_times[inside]
    state = status[kk]
    t_a, v_a, t_b, v_b = (array[kk] for array in fill)
    with np.errstate(invalid="ignore", divide="ignore"):
        filled = v_a + (v_b - v_a) * (tau - t_a) / (t_b - t_a)
    values[inside] = np.where(state == 0, ibi[kk], np.where(state == 1, filled, 0.0))
    code[inside] = np.array([MEASURED, INTERPOLATED, ARTIFACT], dtype=np.uint8)[state]
    return values, code


def blocks_to_ibi(blocks, epoch_start_s, options, *, out_fs=FS_IBI, epoch_seconds=EPOCH_SECONDS):
    """Steps 3-4: IBI [n, out_fs*30], sample status, valid fraction and epoch_ok for all epochs."""
    starts = np.asarray(epoch_start_s, dtype=np.float64)
    samples = int(round(out_fs * epoch_seconds))
    ibi = np.zeros((starts.size, samples), dtype=np.float32)
    status = np.full((starts.size, samples), NO_BEAT, dtype=np.uint8)
    for block in blocks:
        run = block["epochs"]
        times = epoch_sample_times(starts[run], out_fs, samples)
        beats = block["beat_times_s"]
        values, code = intervals_to_samples(times, beats, *classify_intervals(beats, options))
        ibi[run], status[run] = values, code
    valid_fraction = np.isin(status, (MEASURED, INTERPOLATED)).mean(axis=1).astype(np.float32)
    return dict(ibi=ibi, status=status, valid_fraction=valid_fraction,
                epoch_ok=valid_fraction >= options["min_valid_fraction"])


def ppg_to_ibi(epochs, fs, epoch_start_s, *, detect_fs=FS_DETECT, **cleaning):
    """PPG epochs [n, fs*30] (real or generated) -> WatchSleepNet-format 25 Hz IBI [n, 750] plus beats.

    Consecutive epochs (by epoch_start_s) form one block; see the module docstring for the steps.
    """
    options = cleaning_options(**cleaning)
    blocks = ppg_blocks(epochs, fs, epoch_start_s, detect_fs=detect_fs)
    result = blocks_to_ibi(blocks, epoch_start_s, options)
    n = len(epoch_start_s)
    join = lambda key: np.concatenate([b[key] for b in blocks]) if blocks else np.zeros(0)
    sqi, beats_per_epoch = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.int32)
    for block in blocks:
        sqi[block["epochs"]], beats_per_epoch[block["epochs"]] = block["sqi"], block["beats_per_epoch"]
    result.update(beat_times_s=join("beat_times_s"), beat_times_unrefined_s=join("unrefined_s"),
                  sqi=sqi, beats_per_epoch=beats_per_epoch, options=options)
    return result


def ibi_at(times_s, beat_times, *, ibi_range=IBI_RANGE):
    """Plain WatchSleepNet hold at arbitrary times (no filling): out-of-range intervals and edges are 0."""
    times_s = np.asarray(times_s, dtype=np.float64)
    beats = np.asarray(beat_times, dtype=np.float64)
    values = np.zeros(times_s.shape, dtype=np.float64)
    if beats.size < 2:
        return values.astype(np.float32)
    index = np.searchsorted(beats, times_s, side="right") - 1
    inside = (index >= 0) & (index < beats.size - 1)
    k = index[inside]
    interval = beats[k + 1] - beats[k]
    values[inside] = np.where((interval >= ibi_range[0]) & (interval < ibi_range[1]), interval, 0.0)
    return values.astype(np.float32)


def watchsleepnet_epoch_ibi(bvp_100, epoch_starts_rows):
    """The published WatchSleepNet per-epoch IBI at 25 Hz, plus its absolute peak times (s) and epochs."""
    import neurokit2 as nk

    out = np.zeros((len(epoch_starts_rows), EPOCH_IBI), dtype=np.float32)
    peaks_all, epoch_of_peak = [], []
    for row, start in enumerate(epoch_starts_rows):
        segment = np.asarray(bvp_100[start:start + EPOCH_ROWS], dtype=np.float64)
        if not np.isfinite(segment).all():
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", nk.misc.NeuroKitWarning)
            try:
                _, info = nk.ppg_process(segment, sampling_rate=FS_RAW)
            except Exception:
                continue
            if any("Too few peaks detected to compute the rate" in str(w.message) for w in caught):
                continue
        peaks = np.asarray(info["PPG_Peaks"], dtype=np.int64)
        ibi = np.zeros(EPOCH_ROWS)
        for k in range(len(peaks) - 1):
            ibi[peaks[k]:peaks[k + 1]] = (peaks[k + 1] - peaks[k]) / FS_RAW
        ibi[(ibi < IBI_RANGE[0]) | (ibi >= IBI_RANGE[1])] = 0.0
        out[row] = ibi[::4]
        peaks_all.append((start + peaks) / FS_RAW)
        epoch_of_peak.append(np.full(len(peaks), row))
    if peaks_all:
        return out, np.concatenate(peaks_all), np.concatenate(epoch_of_peak)
    return out, np.zeros(0), np.zeros(0, dtype=np.int64)


# --------------------------------------------------------------------------- alignment and matching
def hr_cross_lag(ecg_t, ppg_t, *, max_lag_s=60.0, fs=10.0):
    """Coarse PPG-minus-ECG lag from cross-correlating instantaneous heart rate series."""
    if ecg_t.size < 20 or ppg_t.size < 20:
        return None, None
    start, stop = max(ecg_t[0], ppg_t[0]), min(ecg_t[-1], ppg_t[-1])
    if stop - start < 4 * max_lag_s:
        return None, None
    grid = np.arange(start, stop, 1.0 / fs)

    def series(t):
        rr = np.diff(t)
        keep = (rr >= IBI_RANGE[0]) & (rr < IBI_RANGE[1])
        mid = (t[1:] + t[:-1]) / 2
        value = np.interp(grid, mid[keep], 60.0 / rr[keep]) if keep.sum() > 2 else np.zeros_like(grid)
        value = value - value.mean()
        return value / (value.std() or 1.0)

    a, b = series(ecg_t), series(ppg_t)
    n = 1 << int(np.ceil(np.log2(2 * grid.size)))
    corr = np.fft.irfft(np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n)), n) / grid.size
    lags = np.r_[np.arange(0, n // 2), np.arange(-n // 2, 0)] / fs
    best = np.argmax(np.where(np.abs(lags) <= max_lag_s, corr, -np.inf))
    return float(lags[best]), float(corr[best])


def pulse_arrival(ecg_t, ppg_t, *, max_s=1.0):
    """Delay from each PPG beat back to the preceding R peak (0 < d < max_s) and the PPG beat time."""
    if ecg_t.size == 0 or ppg_t.size == 0:
        return np.zeros(0), np.zeros(0)
    index = np.searchsorted(ecg_t, ppg_t, side="right") - 1
    ok = index >= 0
    delay = ppg_t[ok] - ecg_t[index[ok]]
    keep = (delay > 0) & (delay < max_s)
    return delay[keep], ppg_t[ok][keep]


def mode_lag(lags, *, width=0.05, bin_s=0.01):
    """(estimate, concentration): the densest lag (10-ms histogram) refined by the median near it."""
    lags = np.asarray(lags, dtype=np.float64)
    if lags.size < 50:
        return None, 0.0
    counts, edges = np.histogram(lags, np.arange(lags.min(), lags.max() + 2 * bin_s, bin_s))
    peak = edges[int(np.argmax(np.convolve(counts, np.ones(5), mode="same")))] + bin_s / 2
    estimate = float(np.median(lags[np.abs(lags - peak) <= width]))
    return estimate, float(np.mean(np.abs(lags - estimate) <= width))


def lag_track(ecg_t, ppg_t, *, coarse=None, block_s=1800.0, min_beats=300):
    """PPG-minus-ECG beat lag: whole-night estimate, concentration and a 30-min track.

    Each PPG beat is paired with the last R peak before it after shifting ECG by a base lag; two
    bases are tried (-0.2 s for an aligned recording, coarse - 0.5 s from the heart-rate
    cross-correlation) and the one whose lags concentrate best is kept.
    """
    best = None
    for base in sorted({-0.2} | ({round(coarse - 0.5, 3)} if coarse is not None else set())):
        delays, at = pulse_arrival(ecg_t + base, ppg_t)
        estimate, concentration = mode_lag(base + delays)
        if estimate is not None and (best is None or concentration > best[1]):
            best = (estimate, concentration, base + delays, at, base)
    if best is None:
        return dict(lag_s=None, concentration=0.0, track_t=[], track_lag_s=[], base_s=None)
    estimate, concentration, lags, at, base = best
    track_t, track = [], []
    near = np.abs(lags - estimate) <= 0.3
    for lo in np.arange(at.min(), at.max() + 1e-9, block_s):
        inside = near & (at >= lo) & (at < lo + block_s)
        if inside.sum() >= min_beats:
            block, block_concentration = mode_lag(lags[inside])
            if block is not None and block_concentration >= 0.3:
                track_t.append(float(lo + block_s / 2))
                track.append(block)
    return dict(lag_s=estimate, concentration=concentration, track_t=track_t, track_lag_s=track, base_s=base)


def lag_at(times, track):
    """Lag (s) at the given times from lag_track's result (constant outside the track)."""
    if track["track_t"]:
        return np.interp(np.asarray(times, dtype=np.float64), track["track_t"], track["track_lag_s"])
    return np.full(np.shape(times), track["lag_s"], dtype=np.float64)


def match_beats(ecg_t, ppg_t, lag, *, tolerance=0.15):
    """For each ECG beat, the index of the PPG beat at ecg + lag within tolerance (-1 if none), one-to-one."""
    matched = np.full(ecg_t.size, -1, dtype=np.int64)
    if ecg_t.size == 0 or ppg_t.size == 0:
        return matched
    expected = ecg_t + lag
    right = np.clip(np.searchsorted(ppg_t, expected), 0, ppg_t.size - 1)
    left = np.clip(right - 1, 0, ppg_t.size - 1)
    pick = np.where(np.abs(ppg_t[left] - expected) <= np.abs(ppg_t[right] - expected), left, right)
    close = np.abs(ppg_t[pick] - expected) <= tolerance
    matched[close] = pick[close]
    order = np.argsort(np.abs(ppg_t[np.maximum(matched, 0)] - expected) + (matched < 0) * 1e9)
    seen = set()
    for i in order:
        j = matched[i]
        if j < 0:
            continue
        if j in seen:
            matched[i] = -1
        else:
            seen.add(j)
    return matched


def ibi_errors(ecg_t, ppg_t, matched, *, pair_ok=None):
    """PPG minus ECG interval (s) for consecutive matched beats that are also consecutive in PPG."""
    i = np.flatnonzero((matched[:-1] >= 0) & (matched[1:] >= 0) & (matched[1:] == matched[:-1] + 1))
    if pair_ok is not None:
        i = i[pair_ok[matched[i]]]
    rr = ecg_t[i + 1] - ecg_t[i]
    pp = ppg_t[matched[i + 1]] - ppg_t[matched[i]]
    keep = (rr >= IBI_RANGE[0]) & (rr < IBI_RANGE[1])
    return pp[keep] - rr[keep]


def error_summary(errors):
    errors = np.asarray(errors, dtype=np.float64)
    if not errors.size:
        return dict(pairs=0, mae_ms=None, median_abs_ms=None, bias_ms=None, p95_abs_ms=None)
    absolute = np.abs(errors) * 1000.0
    return dict(pairs=int(errors.size), mae_ms=float(absolute.mean()), median_abs_ms=float(np.median(absolute)),
                bias_ms=float(errors.mean() * 1000.0), p95_abs_ms=float(np.quantile(absolute, 0.95)))


def beat_accuracy(ecg_t, beats, track, *, pair_ok=None):
    """(matched, metrics) of PPG beats against ECG R peaks shifted by the lag track."""
    matched = match_beats(ecg_t, beats, lag_at(ecg_t, track))
    metrics = dict(sensitivity=float((matched >= 0).mean()) if matched.size else None,
                   ppv=float(np.unique(matched[matched >= 0]).size / beats.size) if beats.size else None,
                   **error_summary(ibi_errors(ecg_t, beats, matched, pair_ok=pair_ok)))
    return matched, metrics


def series_accuracy(ibi, reference, *, big_error_s=0.1):
    """25 Hz IBI against the lag-aligned ECG IBI, on samples where the ECG IBI is valid."""
    ibi, reference = np.asarray(ibi, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    ref_ok = reference > 0
    both = ref_ok & (ibi > 0)
    if not ref_ok.any():
        return dict(coverage=None, mae_ms=None, big_error_fraction=None)
    error = np.abs(ibi[both] - reference[both])
    return dict(coverage=float(both.sum() / ref_ok.sum()),
                mae_ms=float(error.mean() * 1000.0) if error.size else None,
                big_error_fraction=float((error > big_error_s).mean()) if error.size else None)


# --------------------------------------------------------------------------- per-epoch quality
def template_sqi(cleaned, peak_samples, fs, epoch_bounds):
    """Mean correlation of each beat with the epoch's average beat (Orphanidou-style), per epoch."""
    cleaned = np.asarray(cleaned, dtype=np.float64)
    peaks = np.asarray(peak_samples, dtype=np.int64)
    out = np.zeros(len(epoch_bounds), dtype=np.float32)
    for e, (lo, hi) in enumerate(epoch_bounds):
        inside = peaks[(peaks >= lo) & (peaks < hi)]
        if inside.size < 3:
            continue
        half = int(round(np.median(np.diff(inside)) / 2))
        if half < 2:
            continue
        inside = inside[(inside - half >= 0) & (inside + half < cleaned.size)]
        if inside.size < 3:
            continue
        beats = np.stack([cleaned[p - half:p + half] for p in inside])
        centred = beats.mean(axis=0)
        centred = centred - centred.mean()
        beats = beats - beats.mean(axis=1, keepdims=True)
        denominator = np.linalg.norm(beats, axis=1) * (np.linalg.norm(centred) or 1.0)
        out[e] = float(np.mean((beats @ centred) / np.where(denominator > 0, denominator, np.inf)))
    return out


def flat_fraction(signal, *, min_run=20):
    """Fraction of samples inside runs of identical consecutive values at least min_run long."""
    signal = np.asarray(signal)
    if signal.size < 2:
        return 0.0
    same = np.r_[False, np.diff(signal) == 0]
    flat = np.zeros(signal.size, dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[0, same.astype(np.int8), 0]))
    for begin, end in zip(edges[::2], edges[1::2]):
        if end - begin + 1 >= min_run:
            flat[max(begin - 1, 0):end] = True
    return float(flat.mean())


# --------------------------------------------------------------------------- one participant
def read_subject_csv(csv_path, ecg_column="ECG"):
    import pandas as pd

    wanted = ("TIMESTAMP", "BVP", ecg_column, "ACC_X", "ACC_Y", "ACC_Z", "Sleep_Stage") + EVENT_COLUMNS
    frame = pd.read_csv(csv_path, usecols=lambda name: name in wanted, low_memory=False)
    missing = [c for c in ("BVP", "Sleep_Stage") + EVENT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{Path(csv_path).name} lacks columns {missing}; has {list(frame.columns)}")
    if "TIMESTAMP" in frame.columns and len(frame) > 1:
        step = float(np.median(np.diff(frame["TIMESTAMP"].to_numpy(dtype=np.float64)[:10001])))
        if abs(step - 1.0 / FS_RAW) > 0.05 / FS_RAW:
            raise ValueError(f"{Path(csv_path).name}: TIMESTAMP step {step:.5f} s is not 100 Hz")
    return frame


def extract_subject(csv_path, *, ecg_column="ECG", watchsleepnet=True, cleaning=None,
                    relative_threshold=DEFAULT_RELATIVE_THRESHOLD, ahi=None):
    """Return (arrays, report) for one raw DREAMT CSV; see the module docstring."""
    from scipy.signal import butter, resample_poly, sosfiltfilt

    from apnea_labels import event_flags

    options = cleaning_options(**(cleaning or {}))
    csv_path = Path(csv_path)
    frame = read_subject_csv(csv_path, ecg_column)

    # PSG 30-s grid
    grid = epoch_grid(frame["Sleep_Stage"].to_numpy(dtype=object))
    keep = np.flatnonzero(grid["valid"])
    if not keep.size:
        raise ValueError(f"{csv_path.name}: no scored 30-s epoch")
    starts, stages, epoch_index = grid["starts"][keep], grid["codes"][keep], keep.astype(np.int64)
    start_s = starts / FS_RAW
    rows100 = starts[:, None] + np.arange(EPOCH_ROWS)[None, :]
    rows50 = starts[:, None] + 2 * np.arange(EPOCH_PPG)[None, :]          # 100 Hz row of each 50 Hz sample
    sleep = np.flatnonzero(stages > 0)
    in_sleep = np.zeros(keep.size, dtype=bool)
    if sleep.size:
        in_sleep[sleep[0]:sleep[-1] + 1] = True

    # stored PPG: whole-night zero-phase 0.5-8 Hz band-pass at 100 Hz, anti-aliased 2:1 on the epoch phase
    bvp_raw = frame["BVP"].to_numpy(dtype=np.float64)
    bvp, bvp_nan = _fill_nan(bvp_raw)
    filtered = sosfiltfilt(butter(3, [0.5, 8.0], btype="bandpass", fs=FS_RAW, output="sos"), bvp)
    phase = grid["offset"] % 2
    ppg50 = resample_poly(filtered[phase:], 1, 2)                          # sample m sits on row phase + 2m
    take50 = (rows50 - phase) // 2
    if take50.max() >= ppg50.size:
        raise ValueError(f"{csv_path.name}: 50 Hz signal shorter than the last epoch")
    ppg = ppg50[take50].astype(np.float32)

    # PPG -> IBI: the generated-PPG route (stored 50 Hz -> 100 Hz) and the original 100 Hz route
    blocks50 = ppg_blocks(ppg, FS_PPG, start_s)
    blocks100 = ppg_blocks(bvp[rows100], FS_RAW, start_s)
    variants = dict(
        ppg50_clean=blocks_to_ibi(blocks50, start_s, options),
        ppg50_relative=blocks_to_ibi(blocks50, start_s, dict(options, relative_threshold=relative_threshold)),
        ppg50_raw=blocks_to_ibi(blocks50, start_s, dict(options, max_fill_intervals=0)),
        ppg100_clean=blocks_to_ibi(blocks100, start_s, options),
    )
    main = variants["ppg50_clean"]
    join = lambda blocks, key: np.concatenate([b[key] for b in blocks]) if blocks else np.zeros(0)
    ppg_t, ppg100_t = join(blocks50, "beat_times_s"), join(blocks100, "beat_times_s")
    sqi = np.zeros(keep.size, dtype=np.float32)
    beats_per_epoch = np.zeros(keep.size, dtype=np.int32)
    for block in blocks50:
        sqi[block["epochs"]], beats_per_epoch[block["epochs"]] = block["sqi"], block["beats_per_epoch"]

    if watchsleepnet:
        ibi_wsn, wsn_t, wsn_epoch = watchsleepnet_epoch_ibi(bvp_raw, starts)
    else:
        ibi_wsn, wsn_t, wsn_epoch = np.zeros((keep.size, EPOCH_IBI), np.float32), np.zeros(0), np.zeros(0, np.int64)

    ecg_values = frame[ecg_column].to_numpy(dtype=np.float64) if ecg_column in frame.columns else None
    have_ecg = ecg_values is not None and np.isfinite(ecg_values).mean() > 0.5
    ecg_inverted = None
    if have_ecg:
        ecg_t, ecg_inverted = ecg_beats(ecg_values, FS_RAW)
        ecg_nan = ~np.isfinite(ecg_values)
    else:
        ecg_t, ecg_nan = np.zeros(0), np.ones(len(frame), dtype=bool)
    times25 = epoch_sample_times(start_s, FS_IBI, EPOCH_IBI)
    ibi_ecg = ibi_at(times25, ecg_t)

    # alignment of wrist PPG to PSG ECG
    alignment = dict(ecg_available=bool(have_ecg), ecg_inverted=ecg_inverted, aligned=False)
    track = None
    if have_ecg and ecg_t.size > 100 and ppg_t.size > 100:
        coarse, coarse_r = hr_cross_lag(ecg_t, ppg_t)
        track = lag_track(ecg_t, ppg_t, coarse=coarse)
        lags = track["track_lag_s"]
        drift = float(max(lags) - min(lags)) if len(lags) >= 2 else 0.0
        if track["lag_s"] is None:
            track = None
        alignment.update(
            coarse_hr_lag_s=coarse, coarse_hr_corr=coarse_r,
            pulse_arrival_s=None if track is None else track["lag_s"],
            pulse_arrival_concentration=0.0 if track is None else track["concentration"],
            lag_track_t_s=[] if track is None else track["track_t"], lag_track_s=lags, pulse_arrival_drift_s=drift,
            aligned=bool(track is not None and track["concentration"] >= 0.4 and drift <= 0.2
                         and 0.1 <= track["lag_s"] <= 0.8))

    # accuracy against ECG: beats (inside kept epochs) and the 25 Hz IBI series (lag-aligned)
    beats, series = {}, {}
    epoch_match = np.zeros(keep.size, dtype=np.float32)
    if track is not None:
        ecg_epoch = np.floor((ecg_t * FS_RAW - grid["offset"]) / EPOCH_ROWS).astype(np.int64)
        inside = np.isin(ecg_epoch, epoch_index)
        ecg_in, position = ecg_t[inside], np.searchsorted(epoch_index, ecg_epoch[inside])
        matched, beats["ppg50"] = beat_accuracy(ecg_in, ppg_t, track)
        _, beats["ppg50_unrefined"] = beat_accuracy(ecg_in, join(blocks50, "unrefined_s"), track)
        _, beats["ppg100"] = beat_accuracy(ecg_in, ppg100_t, track)
        _, beats["ppg100_unrefined"] = beat_accuracy(ecg_in, join(blocks100, "unrefined_s"), track)
        if wsn_t.size:
            order = np.argsort(wsn_t, kind="stable")
            same_epoch = np.r_[wsn_epoch[order][1:] == wsn_epoch[order][:-1], False]
            _, beats["watchsleepnet"] = beat_accuracy(ecg_in, wsn_t[order], track, pair_ok=same_epoch)
        hits = np.bincount(position, weights=(matched >= 0), minlength=keep.size)
        totals = np.bincount(position, minlength=keep.size)
        epoch_match = np.where(totals > 0, hits / np.maximum(totals, 1), 0.0).astype(np.float32)
        reference = ibi_at(times25 - lag_at(times25, track), ecg_t)
        for name, variant in variants.items():
            series[name] = series_accuracy(variant["ibi"], reference)
        if watchsleepnet:
            series["watchsleepnet"] = series_accuracy(ibi_wsn, reference)

    # quality per epoch (stored for every epoch)
    if all(c in frame.columns for c in ("ACC_X", "ACC_Y", "ACC_Z")):
        acc = np.sqrt(sum(np.nan_to_num(frame[c].to_numpy(dtype=np.float64)) ** 2
                          for c in ("ACC_X", "ACC_Y", "ACC_Z"))) / 64.0       # E4 units are 1/64 g
        acc_std = acc[rows100].std(axis=1).astype(np.float32)
    else:
        acc_std = np.full(keep.size, np.nan, dtype=np.float32)
    flat = np.array([flat_fraction(bvp_raw[s:s + EPOCH_ROWS]) for s in starts], dtype=np.float32)
    bvp_nan_fraction = bvp_nan[rows100].mean(axis=1).astype(np.float32)
    ecg_nan_fraction = ecg_nan[rows100].mean(axis=1).astype(np.float32)
    events = event_flags(frame)[0][rows50]                               # [n, 1500, 4] at 50 Hz
    status = main["status"]

    report = dict(
        subject=csv_path.name.split("_")[0].split(".")[0], csv=csv_path.name, rows=int(len(frame)),
        epochs=int(keep.size), epochs_on_grid=int(grid["valid"].size), grid_offset_rows=int(grid["offset"]),
        stage_changes_on_grid_fraction=grid["on_grid_fraction"], sleep_period_epochs=int(in_sleep.sum()),
        epoch_runs=len(epoch_runs(start_s)),
        stage_epochs={name: int((stages == code).sum()) for name, code in STAGE_CODES.items()},
        ppg_beats=int(ppg_t.size), ecg_beats=int(ecg_t.size), cleaning=options,
        relative_threshold=relative_threshold, alignment=alignment, beat_accuracy=beats, series_accuracy=series,
        epoch_ok_fraction=float(main["epoch_ok"].mean()),
        epoch_ok_fraction_sleep_period=float(main["epoch_ok"][in_sleep].mean()) if in_sleep.any() else None,
        epoch_ok_fraction_relative=float(variants["ppg50_relative"]["epoch_ok"].mean()),
        measured_sample_fraction=float((status == MEASURED).mean()),
        interpolated_sample_fraction=float((status == INTERPOLATED).mean()),
        artifact_sample_fraction=float((status == ARTIFACT).mean()),
        no_beat_sample_fraction=float((status == NO_BEAT).mean()),
        median_sqi=float(np.median(sqi)),
        median_acc_std_g=float(np.nanmedian(acc_std)) if np.isfinite(acc_std).any() else None,
        event_sample_fraction=float(events.max(axis=2).mean()), participant_ahi=None if ahi is None else float(ahi),
    )
    arrays = dict(
        schema=np.array(SCHEMA), subject=np.array(report["subject"]), source_csv_name=np.array(csv_path.name),
        source_csv_sha256=np.array(sha256_file(csv_path)), fs_ppg=np.array(FS_PPG), fs_ibi=np.array(FS_IBI),
        epoch_seconds=np.array(EPOCH_SECONDS), grid_offset_rows=np.array(grid["offset"]),
        epoch_index=epoch_index, epoch_start_s=start_s, stages=stages.astype(np.int8), in_sleep_period=in_sleep,
        ppg=ppg, events=events, event_columns=np.array(EVENT_COLUMNS),
        ibi_ppg=main["ibi"], ibi_ppg_status=status, ibi_ppg_valid_fraction=main["valid_fraction"],
        epoch_ok=main["epoch_ok"],
        ibi_ppg_relative=variants["ppg50_relative"]["ibi"],
        ibi_ppg_relative_valid_fraction=variants["ppg50_relative"]["valid_fraction"],
        ibi_ppg_raw=variants["ppg50_raw"]["ibi"], ibi_ppg100=variants["ppg100_clean"]["ibi"],
        ibi_wsn=ibi_wsn, ibi_ecg=ibi_ecg,
        ppg_peaks_s=ppg_t, ppg100_peaks_s=ppg100_t, ecg_peaks_s=ecg_t, wsn_peaks_s=np.sort(wsn_t),
        sqi=sqi, acc_std_g=acc_std, flat_fraction=flat, bvp_nan_fraction=bvp_nan_fraction,
        ecg_nan_fraction=ecg_nan_fraction, ecg_match_fraction=epoch_match, ppg_beats_per_epoch=beats_per_epoch,
        report=np.array(json.dumps(report, allow_nan=True)),
    )
    return arrays, report
