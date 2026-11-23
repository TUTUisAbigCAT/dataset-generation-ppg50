"""DREAMT 2.2.0 PPG extraction at 50 Hz with PSG-ECG checks (read-only on the raw CSVs).

For every participant this builds, on the PSG 30-second grid:
  * the wrist PPG (E4 BVP) band-passed 0.5-8 Hz on the whole night (zero phase)
    and resampled with anti-aliasing to 50 Hz (1500 samples per epoch);
  * sleep stage per epoch (W=0, N1=1, N2=2, N3=3, R=4), respiratory-event
    flags at 50 Hz (Obstructive_Apnea, Central_Apnea, Hypopnea, Multiple_Events);
  * beat times from the 50 Hz PPG (the same path generated PPG will take), from
    the 100 Hz PPG, and ECG R peaks from the PSG ECG, each refined on a 500 Hz
    cubic-spline grid around every detected peak;
  * the PPG-to-ECG lag (pulse arrival time), its stability over the night and
    a coarse heart-rate cross-correlation lag, which flag a misaligned night;
  * beat matching PPG vs ECG (sensitivity, PPV) and the IBI error in ms for the
    50 Hz, 100 Hz and WatchSleepNet per-epoch pipelines;
  * per-epoch quality: PPG template-match SQI, wrist motion (ACC), flat-line
    fraction, missing samples, share of ECG beats matched by PPG beats;
  * IBI at 25 Hz in the WatchSleepNet sample-and-hold format (values outside
    0.3-2.0 s written as 0) from the stored 50 Hz PPG epochs with ppg_to_ibi
    (consecutive epochs joined, so no artificial zeros at epoch edges; generated
    PPG goes through the same function), from ECG (reference) and with the
    WatchSleepNet per-epoch method (for comparison with the published pipeline).

Nothing is discarded: every metric is stored per epoch, and the thresholds that
decide which epochs count as good are applied in the summary only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import warnings

import numpy as np

SCHEMA = "dreamt_ppg50_extraction_v1"
FS_RAW = 100
FS_PPG = 50
FS_IBI = 25
EPOCH_SECONDS = 30
EPOCH_ROWS = FS_RAW * EPOCH_SECONDS          # 3000
EPOCH_PPG = FS_PPG * EPOCH_SECONDS           # 1500
EPOCH_IBI = FS_IBI * EPOCH_SECONDS           # 750
STAGE_CODES = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
EVENT_COLUMNS = ("Obstructive_Apnea", "Central_Apnea", "Hypopnea", "Multiple_Events")
IBI_RANGE = (0.3, 2.0)
DEFAULT_THRESHOLDS = dict(sqi=0.86, ecg_match=0.9, acc_std_g=0.05, flat_fraction=0.05, nan_fraction=0.0)


def sha256_file(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


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
        residues = changes % epoch_rows
        values, counts = np.unique(residues, return_counts=True)
        offset = int(values[np.argmax(counts)])
        on_grid = float(counts.max() / changes.size)
    else:
        offset, on_grid = 0, 1.0
    epochs = (len(labels) - offset) // epoch_rows
    starts = offset + epoch_rows * np.arange(max(epochs, 0), dtype=np.int64)
    block = codes[offset:offset + epochs * epoch_rows].reshape(epochs, epoch_rows) if epochs else np.zeros((0, epoch_rows))
    first = block[:, 0] if epochs else np.zeros(0, dtype=np.int64)
    valid = (first >= 0) & np.all(block == first[:, None], axis=1) if epochs else np.zeros(0, dtype=bool)
    return dict(offset=offset, starts=starts, codes=first.astype(np.int64), valid=valid,
                on_grid_fraction=on_grid, changes=int(changes.size))


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


def refine_peaks(signal, peaks, fs, *, target_fs=500, half_window_s=0.04):
    """Peak times in seconds, refined on a target_fs cubic-spline grid around each detected peak."""
    from scipy.interpolate import CubicSpline

    signal = np.asarray(signal, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=np.int64)
    half = max(2, int(round(half_window_s * fs)))
    times = peaks.astype(np.float64) / fs
    step = 1.0 / target_fs
    for k, p in enumerate(peaks):
        lo, hi = max(0, p - half), min(signal.size, p + half + 1)
        if hi - lo < 4:
            continue
        x = np.arange(lo, hi, dtype=np.float64) / fs
        spline = CubicSpline(x, signal[lo:hi])
        grid = np.arange(x[0], x[-1] + step / 2, step)
        times[k] = grid[int(np.argmax(spline(grid)))]
    return times


def ppg_beats(ppg, fs, *, refine=True):
    """(peak_times_s, peak_samples, cleaned) with neurokit2's Elgendi cleaning and detector.

    The same function serves real and generated PPG, at any sampling rate.
    """
    import neurokit2 as nk

    signal, _ = _fill_nan(ppg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaned = np.asarray(nk.ppg_clean(signal, sampling_rate=fs, method="elgendi"), dtype=np.float64)
        try:
            peaks = np.asarray(nk.ppg_findpeaks(cleaned, sampling_rate=fs, method="elgendi")["PPG_Peaks"], dtype=np.int64)
        except Exception:                                  # flat or too short signal
            peaks = np.zeros(0, dtype=np.int64)
    times = refine_peaks(cleaned, peaks, fs) if refine else peaks / float(fs)
    return times, peaks, cleaned


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


def ecg_beats(ecg, fs, *, refine=True):
    """R-peak times in seconds (neurokit2), refined on the polarity-corrected cleaned ECG."""
    import neurokit2 as nk

    signal, _ = _fill_nan(ecg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaned = np.asarray(nk.ecg_clean(signal, sampling_rate=fs), dtype=np.float64)
        inverted = ecg_is_inverted(cleaned, fs)
        if inverted:
            cleaned = -cleaned
        peaks = np.asarray(nk.ecg_peaks(cleaned, sampling_rate=fs)[1]["ECG_R_Peaks"], dtype=np.int64)
    times = refine_peaks(cleaned, peaks, fs) if refine else peaks / float(fs)
    return times, peaks, bool(inverted)


def artifact_beats(peak_samples, fs):
    """Bool per beat: ectopic, missed, extra or long/short beats (Kubios method); beats are flagged, never moved."""
    import neurokit2 as nk

    flags = np.zeros(len(peak_samples), dtype=bool)
    if len(peak_samples) < 10:
        return flags
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            info, _ = nk.signal_fixpeaks(np.asarray(peak_samples), sampling_rate=fs, iterative=False, method="Kubios")
        except Exception:
            return flags
    for key in ("ectopic", "missed", "extra", "longshort"):
        index = np.asarray(info.get(key, []), dtype=np.int64)
        index = index[(index >= 0) & (index < flags.size)]
        flags[index] = True
    return flags


# --------------------------------------------------------------------------- IBI at 25 Hz
def ibi_at(times_s, beat_times, *, flagged=None, ibi_range=IBI_RANGE):
    """WatchSleepNet sample-and-hold IBI at arbitrary times: (values, flag).

    Each time takes the interval of the beat pair around it; before the first
    and after the last beat, or outside ibi_range, the value is 0.
    """
    times_s = np.asarray(times_s, dtype=np.float64)
    beats = np.asarray(beat_times, dtype=np.float64)
    values = np.zeros(times_s.shape, dtype=np.float64)
    flag = np.zeros(times_s.shape, dtype=np.uint8)
    if beats.size < 2:
        return values.astype(np.float32), flag
    index = np.searchsorted(beats, times_s, side="right") - 1
    inside = (index >= 0) & (index < beats.size - 1)
    k = index[inside]
    interval = beats[k + 1] - beats[k]
    interval = np.where((interval >= ibi_range[0]) & (interval < ibi_range[1]), interval, 0.0)
    values[inside] = interval
    if flagged is not None:
        flagged = np.asarray(flagged, dtype=bool)
        flag[inside] = (flagged[k] | flagged[k + 1]).astype(np.uint8)
    return values.astype(np.float32), flag


def epoch_sample_times(epoch_start_s, fs, samples):
    return np.asarray(epoch_start_s, dtype=np.float64)[:, None] + np.arange(samples, dtype=np.float64)[None, :] / fs


def epoch_runs(epoch_start_s, epoch_seconds=EPOCH_SECONDS):
    """Index arrays of epochs that follow each other without a gap."""
    starts = np.asarray(epoch_start_s, dtype=np.float64)
    if not starts.size:
        return []
    breaks = np.flatnonzero(np.abs(np.diff(starts) - epoch_seconds) > 1e-6) + 1
    return np.split(np.arange(starts.size), breaks)


def ppg_to_ibi(epochs, fs, epoch_start_s, *, refine=True, flag_artifacts=True, out_fs=FS_IBI,
               epoch_seconds=EPOCH_SECONDS):
    """PPG epochs [n, fs*30] -> WatchSleepNet-format IBI [n, out_fs*30] plus beats and quality.

    Consecutive epochs (by epoch_start_s) are joined into one signal before beat
    detection, so IBI is continuous inside a run of epochs; across a gap the
    interval is longer than 2 s and therefore 0, as in WatchSleepNet. Use the
    same call for real and generated PPG so both take an identical path.
    """
    epochs = np.asarray(epochs, dtype=np.float64)
    if epochs.ndim != 2 or epochs.shape[1] != int(round(fs * epoch_seconds)):
        raise ValueError(f"epochs must be [n, {int(round(fs * epoch_seconds))}] at {fs} Hz; got {epochs.shape}")
    starts = np.asarray(epoch_start_s, dtype=np.float64)
    if starts.shape != (epochs.shape[0],):
        raise ValueError("epoch_start_s needs one start time per epoch")
    n, samples = epochs.shape
    times, unrefined, artifacts = [], [], []
    sqi = np.zeros(n, dtype=np.float32)
    beats_per_epoch = np.zeros(n, dtype=np.int32)
    for run in epoch_runs(starts, epoch_seconds):
        t, peaks, cleaned = ppg_beats(epochs[run].reshape(-1), fs, refine=refine)
        times.append(t + starts[run[0]])
        unrefined.append(peaks / float(fs) + starts[run[0]])
        artifacts.append(artifact_beats(peaks, fs) if flag_artifacts else np.zeros(peaks.size, dtype=bool))
        sqi[run] = template_sqi(cleaned, peaks, fs, [(k * samples, (k + 1) * samples) for k in range(run.size)])
        beats_per_epoch[run] = np.bincount(peaks // samples, minlength=run.size)[:run.size]
    join = lambda parts, dtype: np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)
    times, unrefined, artifacts = join(times, np.float64), join(unrefined, np.float64), join(artifacts, bool)
    ibi, flag = ibi_at(epoch_sample_times(starts, out_fs, int(round(out_fs * epoch_seconds))), times, flagged=artifacts)
    return dict(ibi=ibi, ibi_flag=flag, beat_times_s=times, beat_times_unrefined_s=unrefined,
                artifact_beats=artifacts, sqi=sqi, beats_per_epoch=beats_per_epoch)


def watchsleepnet_epoch_ibi(bvp_100, epoch_starts_rows):
    """The published WatchSleepNet per-epoch IBI at 25 Hz, plus its absolute peak times (s)."""
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
    window = np.abs(lags) <= max_lag_s
    best = np.argmax(np.where(window, corr, -np.inf))
    return float(lags[best]), float(corr[best])


def pulse_arrival(ecg_t, ppg_t, *, max_s=1.0):
    """Delay from each PPG beat back to the preceding R peak (0 < d < max_s)."""
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
    edges = np.arange(lags.min(), lags.max() + 2 * bin_s, bin_s)
    counts, edges = np.histogram(lags, edges)
    smooth = np.convolve(counts, np.ones(5), mode="same")
    peak = edges[int(np.argmax(smooth))] + bin_s / 2
    estimate = float(np.median(lags[np.abs(lags - peak) <= width]))
    return estimate, float(np.mean(np.abs(lags - estimate) <= width))


def lag_track(ecg_t, ppg_t, *, coarse=None, block_s=1800.0, min_beats=300):
    """PPG-minus-ECG beat lag: whole-night estimate, concentration and a 30-min track.

    Each PPG beat is paired with the last R peak before it after shifting ECG by
    a base lag; two bases are tried (-0.2 s for an aligned recording, and
    coarse - 0.5 s from the heart-rate cross-correlation) and the one whose lags
    concentrate best is kept.
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
    # keep one ECG beat per PPG beat (the closest)
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
        template = beats.mean(axis=0)
        beats = beats - beats.mean(axis=1, keepdims=True)
        centred = template - template.mean()
        denominator = np.linalg.norm(beats, axis=1) * (np.linalg.norm(centred) or 1.0)
        r = (beats @ centred) / np.where(denominator > 0, denominator, np.inf)
        out[e] = float(np.mean(r))
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
def beat_accuracy(ecg_t, beats, track, *, pair_ok=None):
    """(matched, metrics) of PPG beats against ECG R peaks shifted by the lag track."""
    matched = match_beats(ecg_t, beats, lag_at(ecg_t, track))
    metrics = dict(sensitivity=float((matched >= 0).mean()) if matched.size else None,
                   ppv=float(np.unique(matched[matched >= 0]).size / beats.size) if beats.size else None,
                   **error_summary(ibi_errors(ecg_t, beats, matched, pair_ok=pair_ok)))
    return matched, metrics


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


def extract_subject(csv_path, *, ecg_column="ECG", watchsleepnet=True, thresholds=None, ahi=None):
    """Return (arrays, report) for one raw DREAMT CSV; see the module docstring."""
    from scipy.signal import butter, resample_poly, sosfiltfilt

    from apnea_labels import event_flags

    thresholds = dict(DEFAULT_THRESHOLDS, **(thresholds or {}))
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

    # PPG: whole-night zero-phase 0.5-8 Hz band-pass at 100 Hz, anti-aliased 2:1 to 50 Hz on the epoch phase
    bvp_raw = frame["BVP"].to_numpy(dtype=np.float64)
    bvp, bvp_nan = _fill_nan(bvp_raw)
    filtered = sosfiltfilt(butter(3, [0.5, 8.0], btype="bandpass", fs=FS_RAW, output="sos"), bvp)
    phase = grid["offset"] % 2
    ppg50 = resample_poly(filtered[phase:], 1, 2)                          # sample m sits on row phase + 2m
    take50 = (rows50 - phase) // 2
    if take50.max() >= ppg50.size:
        raise ValueError(f"{csv_path.name}: 50 Hz signal shorter than the last epoch")
    ppg = ppg50[take50].astype(np.float32)

    # beats: 50 Hz (the generated-PPG path), 100 Hz, WatchSleepNet per-epoch, ECG
    p50 = ppg_to_ibi(ppg, FS_PPG, start_s)
    p100 = ppg_to_ibi(filtered[rows100], FS_RAW, start_s, flag_artifacts=False)
    ppg_t = p50["beat_times_s"]
    ibi_wsn, wsn_t, wsn_epoch = (watchsleepnet_epoch_ibi(bvp_raw, starts) if watchsleepnet
                                 else (np.zeros((keep.size, EPOCH_IBI), np.float32), np.zeros(0), np.zeros(0, np.int64)))
    ecg_values = frame[ecg_column].to_numpy(dtype=np.float64) if ecg_column in frame.columns else None
    have_ecg = ecg_values is not None and np.isfinite(ecg_values).mean() > 0.5
    ecg_inverted = None
    if have_ecg:
        ecg_t, _, ecg_inverted = ecg_beats(ecg_values, FS_RAW)
        ecg_nan = ~np.isfinite(ecg_values)
    else:
        ecg_t, ecg_nan = np.zeros(0), np.ones(len(frame), dtype=bool)

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

    # beat accuracy against ECG, on ECG beats inside the kept epochs
    accuracy = {}
    epoch_match = np.zeros(keep.size, dtype=np.float32)
    if track is not None:
        ecg_epoch = np.floor((ecg_t * FS_RAW - grid["offset"]) / EPOCH_ROWS).astype(np.int64)
        inside = np.isin(ecg_epoch, epoch_index)
        ecg_in, position = ecg_t[inside], np.searchsorted(epoch_index, ecg_epoch[inside])
        matched, accuracy["ppg50_refined"] = beat_accuracy(ecg_in, ppg_t, track)
        _, accuracy["ppg50_unrefined"] = beat_accuracy(ecg_in, p50["beat_times_unrefined_s"], track)
        _, accuracy["ppg100_refined"] = beat_accuracy(ecg_in, p100["beat_times_s"], track)
        _, accuracy["ppg100_unrefined"] = beat_accuracy(ecg_in, p100["beat_times_unrefined_s"], track)
        if wsn_t.size:
            order = np.argsort(wsn_t, kind="stable")
            same_epoch = np.r_[wsn_epoch[order][1:] == wsn_epoch[order][:-1], False]
            _, accuracy["watchsleepnet"] = beat_accuracy(ecg_in, wsn_t[order], track, pair_ok=same_epoch)
        hits = np.bincount(position, weights=(matched >= 0), minlength=keep.size)
        totals = np.bincount(position, minlength=keep.size)
        epoch_match = np.where(totals > 0, hits / np.maximum(totals, 1), 0.0).astype(np.float32)

    ibi_ecg = ibi_at(epoch_sample_times(start_s, FS_IBI, EPOCH_IBI), ecg_t)[0]

    # quality per epoch (stored for every epoch; thresholds only decide good_epoch)
    if all(c in frame.columns for c in ("ACC_X", "ACC_Y", "ACC_Z")):
        acc = np.sqrt(sum(np.nan_to_num(frame[c].to_numpy(dtype=np.float64)) ** 2
                          for c in ("ACC_X", "ACC_Y", "ACC_Z"))) / 64.0       # E4 units are 1/64 g
        acc_std = acc[rows100].std(axis=1).astype(np.float32)
    else:
        acc_std = np.full(keep.size, np.nan, dtype=np.float32)
    flat = np.array([flat_fraction(bvp_raw[s:s + EPOCH_ROWS]) for s in starts], dtype=np.float32)
    bvp_nan_fraction = bvp_nan[rows100].mean(axis=1).astype(np.float32)
    ecg_nan_fraction = ecg_nan[rows100].mean(axis=1).astype(np.float32)
    sqi = p50["sqi"]
    good = ((sqi >= thresholds["sqi"]) & (flat <= thresholds["flat_fraction"])
            & (bvp_nan_fraction <= thresholds["nan_fraction"])
            & (np.nan_to_num(acc_std, nan=0.0) <= thresholds["acc_std_g"]))
    if track is not None:
        good &= epoch_match >= thresholds["ecg_match"]

    events = event_flags(frame)[0][rows50]                               # [n, 1500, 4] at 50 Hz

    report = dict(
        subject=csv_path.name.split("_")[0].split(".")[0], csv=csv_path.name, rows=int(len(frame)),
        epochs=int(keep.size), epochs_on_grid=int(grid["valid"].size), grid_offset_rows=int(grid["offset"]),
        stage_changes_on_grid_fraction=grid["on_grid_fraction"], sleep_period_epochs=int(in_sleep.sum()),
        epoch_runs=len(epoch_runs(start_s)),
        stage_epochs={name: int((stages == code).sum()) for name, code in STAGE_CODES.items()},
        ppg_beats=int(ppg_t.size), ecg_beats=int(ecg_t.size),
        ppg_artifact_beat_fraction=float(p50["artifact_beats"].mean()) if ppg_t.size else None,
        alignment=alignment, accuracy=accuracy, thresholds=thresholds,
        good_epoch_fraction=float(good.mean()),
        good_epoch_fraction_sleep_period=float(good[in_sleep].mean()) if in_sleep.any() else None,
        median_sqi=float(np.median(sqi)),
        median_acc_std_g=float(np.nanmedian(acc_std)) if np.isfinite(acc_std).any() else None,
        ppg_ibi_zero_fraction=float((p50["ibi"] == 0).mean()),
        ecg_ibi_zero_fraction=float((ibi_ecg == 0).mean()),
        wsn_ibi_zero_fraction=float((ibi_wsn == 0).mean()) if watchsleepnet else None,
        event_sample_fraction=float(events.max(axis=2).mean()), participant_ahi=None if ahi is None else float(ahi),
    )
    arrays = dict(
        schema=np.array(SCHEMA), subject=np.array(report["subject"]), source_csv_name=np.array(csv_path.name),
        source_csv_sha256=np.array(sha256_file(csv_path)), fs_ppg=np.array(FS_PPG), fs_ibi=np.array(FS_IBI),
        epoch_seconds=np.array(EPOCH_SECONDS), grid_offset_rows=np.array(grid["offset"]),
        epoch_index=epoch_index, epoch_start_s=start_s, stages=stages.astype(np.int8), in_sleep_period=in_sleep,
        ppg=ppg, events=events, event_columns=np.array(EVENT_COLUMNS),
        ppg_peaks_s=ppg_t, ppg_artifact_beats=p50["artifact_beats"], ppg100_peaks_s=p100["beat_times_s"],
        ecg_peaks_s=ecg_t, wsn_peaks_s=np.sort(wsn_t),
        ibi_ppg=p50["ibi"], ibi_ppg_flag=p50["ibi_flag"], ibi_ecg=ibi_ecg, ibi_wsn=ibi_wsn,
        sqi=sqi, acc_std_g=acc_std, flat_fraction=flat, bvp_nan_fraction=bvp_nan_fraction,
        ecg_nan_fraction=ecg_nan_fraction, ecg_match_fraction=epoch_match,
        ppg_beats_per_epoch=p50["beats_per_epoch"], good_epoch=good,
        report=np.array(json.dumps(report, allow_nan=True)),
    )
    return arrays, report
