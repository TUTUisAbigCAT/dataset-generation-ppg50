"""DREAMT 2.2.0 PPG at 50 Hz and the PPG -> 25 Hz IBI path shared by real and generated PPG.

PPG -> IBI (ppg_to_ibi; identical for real and generated PPG):
  1. Contiguous blocks: consecutive 30-s epochs are joined; nothing is processed per 30-s chunk.
  2. Each block is resampled to 100 Hz (50 Hz PPG is band-limited, so this adds no content),
     cleaned with nk.ppg_clean(method="elgendi"), peaks from nk.ppg_peaks (Elgendi) on the cleaned
     block divided by its 5-s running RMS (Elgendi's threshold is global otherwise, and a long block
     then loses beats next to loud artifacts), and every peak is refined by 3-point parabolic
     interpolation (sub-sample timing, no 500 Hz upsampling).
  3. IBI = diff(peak times). Intervals outside 0.3-2.0 s (optionally also those far from the local
     median) are bad. A short run of bad intervals (<= 2 intervals and <= 3 s, with good intervals
     on both sides) is filled on the 25 Hz grid by linear interpolation between the neighbouring
     good intervals; longer runs, runs at a block edge and samples outside the beats are 0.
     Good intervals are written as in WatchSleepNet: each 25 Hz sample holds the interval of the
     beat pair around it.
  4. The 25 Hz series is cut into 30-s epochs of 750 samples; an epoch with less than half of its
     samples valid (measured or filled) gets epoch_ok = False (zero it or mask it in the loss).

Extraction of one participant (extract_subject) builds on the PSG 30-s grid:
  * PPG: the E4 BVP anti-aliased to 50 Hz and nothing else, stored per epoch [n, 1500], not normalized
    (nk.ppg_clean in the PPG -> IBI path is then the only band-pass, as on the raw 100 Hz BVP);
  * stages (W=0, N1=1, N2=2, N3=3, R=4) and respiratory events at 50 Hz;
  * IBI from the stored 50 Hz PPG with the path above (the one generated PPG will take), the same
    with the extra relative rule, without any filling, from the raw 100 Hz BVP (the teacher's
    original route), with WatchSleepNet's per-epoch method, and from the PSG ECG (reference);
  * the PPG-to-ECG lag tracked beat by beat (in DREAMT the E4 clock drifts and wanders against the PSG
    by up to seconds), beat sensitivity/PPV and errors of every IBI variant against the lag-aligned ECG
    IBI, an ECG rhythm flag (e.g. bigeminy, where wrist pulses do not follow ECG beats), the agreement
    of the 50 Hz and 100 Hz routes on the same clock, and per-epoch quality (template SQI, wrist
    motion, flat and missing fractions).
Raw CSVs are only read; flags never remove anything from the stored arrays.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np

SCHEMA = "dreamt_ppg50_extraction_v4"
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


PEAK_METHODS = ("elgendi_local", "elgendi", "charlton")


def local_rms(signal, fs, window_s):
    from scipy.ndimage import uniform_filter1d

    power = uniform_filter1d(np.asarray(signal, dtype=np.float64) ** 2, size=max(int(window_s * fs), 1), mode="nearest")
    rms = np.sqrt(np.maximum(power, 0.0))
    floor = 1e-6 * (float(np.median(rms)) or 1.0)
    return np.maximum(rms, floor)


def ppg_peaks(signal, fs, *, method="elgendi_local", rms_window_s=5.0):
    """(cleaned, integer peaks) on one contiguous block.

    The block is cleaned with nk.ppg_clean(method="elgendi"). Peaks:
      elgendi_local  nk.ppg_peaks (Elgendi) on the cleaned block divided by its 5-s running RMS. Elgendi's
                     threshold uses the mean of the squared signal over the whole input, so on a long block
                     loud artifact periods raise it everywhere and quiet stretches lose beats; the local
                     normalization keeps the block processing (no edge effects) with local thresholds.
      elgendi        nk.ppg_peaks on the cleaned block as is.
      charlton       nk.ppg_findpeaks(method="charlton"), MSPTDfast (Charlton et al.).
    Peak refinement and SQI always use the cleaned block itself.
    """
    import neurokit2 as nk

    if method not in PEAK_METHODS:
        raise ValueError(f"peak method must be one of {PEAK_METHODS}")
    signal, _ = _fill_nan(signal)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaned = np.asarray(nk.ppg_clean(signal, sampling_rate=fs, method="elgendi"), dtype=np.float64)
        try:
            if method == "charlton":
                peaks = nk.ppg_findpeaks(cleaned, sampling_rate=fs, method="charlton")["PPG_Peaks"]
            else:
                x = cleaned / local_rms(cleaned, fs, rms_window_s) if method == "elgendi_local" else cleaned
                if hasattr(nk, "ppg_peaks"):
                    peaks = nk.ppg_peaks(x, sampling_rate=fs, method="elgendi")[1]["PPG_Peaks"]
                else:                                    # neurokit2 < 0.2.4
                    peaks = nk.ppg_findpeaks(x, sampling_rate=fs, method="elgendi")["PPG_Peaks"]
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
def ppg_blocks(epochs, fs, epoch_start_s, *, detect_fs=FS_DETECT, epoch_seconds=EPOCH_SECONDS,
               peak_method="elgendi_local"):
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
        cleaned, peaks = ppg_peaks(signal, detect_fs, method=peak_method)
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


def ppg_to_ibi(epochs, fs, epoch_start_s, *, detect_fs=FS_DETECT, peak_method="elgendi_local", **cleaning):
    """PPG epochs [n, fs*30] (real or generated) -> WatchSleepNet-format 25 Hz IBI [n, 750] plus beats.

    Consecutive epochs (by epoch_start_s) form one block; see the module docstring for the steps.
    """
    options = cleaning_options(**cleaning)
    blocks = ppg_blocks(epochs, fs, epoch_start_s, detect_fs=detect_fs, peak_method=peak_method)
    result = blocks_to_ibi(blocks, epoch_start_s, options)
    n = len(epoch_start_s)
    join = lambda key: np.concatenate([b[key] for b in blocks]) if blocks else np.zeros(0)
    sqi, beats_per_epoch = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.int32)
    for block in blocks:
        sqi[block["epochs"]], beats_per_epoch[block["epochs"]] = block["sqi"], block["beats_per_epoch"]
    result.update(beat_times_s=join("beat_times_s"), beat_times_unrefined_s=join("unrefined_s"),
                  sqi=sqi, beats_per_epoch=beats_per_epoch, options=options, peak_method=peak_method)
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


def _nearest(sorted_t, target):
    """(index, distance) of the value in sorted_t closest to target (-1, inf if empty)."""
    r = int(np.searchsorted(sorted_t, target))
    best, gap = -1, np.inf
    for q in (r - 1, r):
        if 0 <= q < sorted_t.size and abs(target - sorted_t[q]) < gap:
            best, gap = q, abs(target - sorted_t[q])
    return best, gap


def track_beats(ecg_t, ppg_t, *, priors=(0.3,), window=0.2, alpha=0.3, beta=0.1, confirm=5, look_ahead=8,
                reacquire_after=8, max_rate=0.03, centre=None, max_dev=None):
    """Follow a slowly changing PPG-minus-ECG lag beat by beat (E4 and PSG clocks drift and wander).

    A track starts at a PPG beat by trying, for every prior lag, the shift of whole RR intervals that
    brings the beat closest to that prior, and is accepted when at least `confirm` of the next
    `look_ahead` PPG beats also land within `window` of an R peak. While tracking, each PPG beat is
    paired with the R peak nearest to (beat - predicted lag) within `window`; lag and its rate of change
    follow the observed delay as an alpha-beta filter (gains `alpha`, `beta`), so steady ramps are
    followed without lag (the rate is limited to `max_rate` s per s; the largest wander seen in DREAMT
    is about 1 s per minute); after more than `reacquire_after` misses, or once the lag leaves
    `centre` +- `max_dev`, the track is restarted (beats are nearly periodic, so a track can otherwise
    slide from beat to beat and run away).
    Finally every continuous run is shifted by up to two beats to the ECG assignment whose RR
    intervals best match the PP intervals (the clock offset itself cannot tell beats apart).

    Returns dict(matched=ECG index per PPG beat or -1, lag=PPG minus matched R (s) or NaN, run=run id or -1).
    """
    ecg_t = np.asarray(ecg_t, dtype=np.float64)
    ppg_t = np.asarray(ppg_t, dtype=np.float64)
    n = ppg_t.size
    matched = np.full(n, -1, dtype=np.int64)
    run = np.full(n, -1, dtype=np.int64)
    if ecg_t.size < 3 or n < 3:
        return dict(matched=matched, lag=np.full(n, np.nan), run=run)
    lag, rate, last, misses, run_id = None, 0.0, 0.0, 0, -1
    for i in range(n):
        p = ppg_t[i]
        if lag is None:
            k = int(np.searchsorted(ecg_t, p, side="right")) - 1
            if k < 1:
                continue
            rr = ecg_t[k] - ecg_t[k - 1]
            if not IBI_RANGE[0] <= rr < IBI_RANGE[1]:
                continue
            base = p - ecg_t[k]
            best, best_hits = None, -1
            for prior in priors:
                trial = base - np.round((base - prior) / rr) * rr
                hits = sum(_nearest(ecg_t, ppg_t[j] - trial)[1] <= window
                           for j in range(i + 1, min(i + 1 + look_ahead, n)))
                if hits > best_hits:
                    best, best_hits = trial, hits
            if best_hits < confirm:
                continue
            lag, rate, last, misses, run_id = best, 0.0, p, 0, run_id + 1
        predicted = lag + rate * (p - last)
        q, gap = _nearest(ecg_t, p - predicted)
        if gap <= window:
            matched[i], run[i] = q, run_id
            step = max(p - last, 1e-3)
            innovation = (p - ecg_t[q]) - predicted
            lag = predicted + alpha * innovation
            rate = float(np.clip(rate + beta * innovation / step, -max_rate, max_rate))
            last, misses = p, 0
            if centre is not None and max_dev is not None and abs(lag - centre) > max_dev:
                lag = None
        else:
            misses += 1
            if misses > reacquire_after:
                lag = None
    for r in np.unique(run[run >= 0]):
        index = np.flatnonzero(run == r)
        if index.size < 10:
            continue
        best_shift, best_error = 0, np.inf
        for shift in (-2, -1, 0, 1, 2):
            m = matched[index] + shift
            if m.min() < 0 or m.max() >= ecg_t.size:
                continue
            pairs = np.flatnonzero((index[1:] == index[:-1] + 1) & (m[1:] == m[:-1] + 1))
            if pairs.size < 5:
                continue
            pp = ppg_t[index[pairs + 1]] - ppg_t[index[pairs]]
            rr = ecg_t[m[pairs + 1]] - ecg_t[m[pairs]]
            error = float(np.median(np.abs(pp - rr)))
            if error < best_error - 1e-9:
                best_shift, best_error = shift, error
        matched[index] += best_shift
    lag_out = np.full(n, np.nan)
    hit = matched >= 0
    lag_out[hit] = ppg_t[hit] - ecg_t[matched[hit]]
    return dict(matched=matched, lag=lag_out, run=run)


def align_beats(ecg_t, ppg_t, *, coarse=None, max_dev=4.0, min_pairs=100):
    """Two-pass tracking: find the night's typical lag, then track around it.

    Pass 1 tracks from a 0.3 s prior and, if the heart-rate cross-correlation suggests a large offset,
    from that offset too; the offset replaces the 0.3 s prior only if its PP intervals follow the RR
    intervals clearly better (median |PP - RR| below 80 %). The chosen track's median lag is the centre. Pass 2 tracks with every run kept within centre +- max_dev.
    Returns (track, centre); centre is None when nothing could be tracked.
    """
    candidates = [0.3] + ([float(coarse)] if coarse is not None and abs(coarse - 0.3) > 1.0 else [])
    best, best_score = None, np.inf
    for k, prior in enumerate(candidates):
        track = track_beats(ecg_t, ppg_t, priors=(prior,))
        accuracy = tracked_accuracy(ecg_t, ppg_t, track)
        score = accuracy["median_abs_ms"] if accuracy["pairs"] >= min_pairs else np.inf
        # a far-away offset (e.g. DREAMT 2.1.0's ~25 s) must be clearly better than the usual small lag
        if score < (best_score if k == 0 else 0.8 * best_score):
            best, best_score = track, score
    if best is None:
        return track_beats(ecg_t, ppg_t, priors=(0.3,), centre=0.3, max_dev=max_dev), None
    centre = float(np.median(best["lag"][best["matched"] >= 0]))
    return track_beats(ecg_t, ppg_t, priors=(centre,), centre=centre, max_dev=max_dev), centre


def lag_series(times, ppg_t, track, *, max_gap_s=5.0):
    """Lag at arbitrary times, interpolated between matched beats of the same run (NaN elsewhere)."""
    times = np.asarray(times, dtype=np.float64)
    hit = track["matched"] >= 0
    t, lag, run = np.asarray(ppg_t, dtype=np.float64)[hit], track["lag"][hit], track["run"][hit]
    out = np.full(times.shape, np.nan)
    if t.size < 2:
        return out
    k = np.searchsorted(t, times)
    left, right = np.clip(k - 1, 0, t.size - 1), np.clip(k, 0, t.size - 1)
    ok = (k > 0) & (k < t.size) & (run[left] == run[right]) & (t[right] - t[left] <= max_gap_s)
    span = np.where(t[right] > t[left], t[right] - t[left], 1.0)
    weight = (times - t[left]) / span
    out[ok] = (lag[left] + weight * (lag[right] - lag[left]))[ok]
    return out


def tracked_accuracy(ecg_t, ppg_t, track, *, ecg_inside=None, pair_ok=None):
    """Beat metrics of a tracked PPG beat sequence against ECG."""
    matched = track["matched"]
    ecg_inside = np.ones(ecg_t.size, dtype=bool) if ecg_inside is None else ecg_inside
    covered = np.zeros(ecg_t.size, dtype=bool)
    covered[matched[matched >= 0]] = True
    pairs = np.flatnonzero((matched[:-1] >= 0) & (matched[1:] >= 0) & (matched[1:] == matched[:-1] + 1))
    if pair_ok is not None:
        pairs = pairs[pair_ok[pairs]]
    rr = ecg_t[matched[pairs + 1]] - ecg_t[matched[pairs]]
    pp = ppg_t[pairs + 1] - ppg_t[pairs]
    keep = (rr >= IBI_RANGE[0]) & (rr < IBI_RANGE[1])
    lag = track["lag"][matched >= 0]
    return dict(sensitivity=float(covered[ecg_inside].mean()) if ecg_inside.any() else None,
                ppv=float((matched >= 0).mean()) if matched.size else None,
                lag_median_s=float(np.median(lag)) if lag.size else None,
                **error_summary(pp[keep] - rr[keep]))


def ecg_rhythm(ecg_t, *, jump=0.3):
    """Share of consecutive RR intervals that change by more than `jump` (relative).

    Sinus rhythm in sleep stays well below 0.1; ventricular bigeminy (short-long pairs) is close to 1,
    and then wrist pulses do not follow ECG beats, so ECG is no reference for PPG beats.
    """
    rr = np.diff(np.asarray(ecg_t, dtype=np.float64))
    ok = (rr[1:] >= IBI_RANGE[0]) & (rr[1:] < IBI_RANGE[1]) & (rr[:-1] >= IBI_RANGE[0]) & (rr[:-1] < IBI_RANGE[1])
    if not ok.any():
        return None
    ratio = rr[1:][ok] / rr[:-1][ok]
    return float(np.mean((ratio < 1 - jump) | (ratio > 1 / (1 - jump))))


def route_agreement(reference_t, other_t, *, tolerance=0.05):
    """Beats of two PPG routes on the same clock (e.g. 50->100 Hz against the raw 100 Hz BVP)."""
    reference_t = np.asarray(reference_t, dtype=np.float64)
    other_t = np.asarray(other_t, dtype=np.float64)
    matched = match_beats(reference_t, other_t, 0.0, tolerance=tolerance)
    hit = matched >= 0
    if not hit.any():
        return dict(reference_matched=0.0, other_matched=0.0, dt_median_abs_ms=None, dt_p95_abs_ms=None,
                    ibi_median_abs_ms=None, ibi_p95_abs_ms=None)
    dt = np.abs(other_t[matched[hit]] - reference_t[hit]) * 1000.0
    errors = np.abs(ibi_errors(reference_t, other_t, matched)) * 1000.0
    return dict(reference_matched=float(hit.mean()), other_matched=float(np.unique(matched[hit]).size / other_t.size),
                dt_median_abs_ms=float(np.median(dt)), dt_p95_abs_ms=float(np.quantile(dt, 0.95)),
                ibi_median_abs_ms=float(np.median(errors)) if errors.size else None,
                ibi_p95_abs_ms=float(np.quantile(errors, 0.95)) if errors.size else None)


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
                    relative_threshold=DEFAULT_RELATIVE_THRESHOLD, peak_method="elgendi_local", ahi=None):
    """Return (arrays, report) for one raw DREAMT CSV; see the module docstring."""
    from scipy.signal import resample_poly

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

    # stored PPG: the E4 BVP anti-aliased 2:1 to 50 Hz on the epoch phase, nothing else (nk.ppg_clean in
    # the PPG -> IBI path does the band-pass, once, exactly as on the raw 100 Hz BVP)
    bvp_raw = frame["BVP"].to_numpy(dtype=np.float64)
    bvp, bvp_nan = _fill_nan(bvp_raw)
    phase = grid["offset"] % 2
    ppg50 = resample_poly(bvp[phase:], 1, 2)                               # sample m sits on row phase + 2m
    take50 = (rows50 - phase) // 2
    if take50.max() >= ppg50.size:
        raise ValueError(f"{csv_path.name}: 50 Hz signal shorter than the last epoch")
    ppg = ppg50[take50].astype(np.float32)

    # PPG -> IBI: the generated-PPG route (stored 50 Hz -> 100 Hz) and the original 100 Hz route
    blocks50 = ppg_blocks(ppg, FS_PPG, start_s, peak_method=peak_method)
    blocks100 = ppg_blocks(bvp[rows100], FS_RAW, start_s, peak_method=peak_method)
    others = [m for m in PEAK_METHODS if m != peak_method]
    blocks_other = {m: ppg_blocks(ppg, FS_PPG, start_s, peak_method=m) for m in others}
    variants = dict(
        ppg50_clean=blocks_to_ibi(blocks50, start_s, options),
        ppg50_relative=blocks_to_ibi(blocks50, start_s, dict(options, relative_threshold=relative_threshold)),
        ppg50_raw=blocks_to_ibi(blocks50, start_s, dict(options, max_fill_intervals=0)),
        ppg100_clean=blocks_to_ibi(blocks100, start_s, options),
        **{f"ppg50_{m}_clean": blocks_to_ibi(blocks_other[m], start_s, options) for m in others},
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

    # wrist PPG against PSG ECG: the lag drifts and wanders, so it is tracked beat by beat
    alignment = dict(ecg_available=bool(have_ecg), ecg_inverted=ecg_inverted)
    beats, series = {}, {}
    epoch_match = np.zeros(keep.size, dtype=np.float32)
    ecg_epoch = np.floor((ecg_t * FS_RAW - grid["offset"]) / EPOCH_ROWS).astype(np.int64)
    ecg_inside = np.isin(ecg_epoch, epoch_index)
    rhythm = ecg_rhythm(ecg_t[ecg_inside]) if have_ecg else None
    alignment.update(ecg_rr_jump_fraction=rhythm, ecg_irregular=bool(rhythm is not None and rhythm > 0.2))
    if have_ecg and ecg_t.size > 100 and ppg_t.size > 100:
        coarse, coarse_r = hr_cross_lag(ecg_t, ppg_t)
        _, centre = align_beats(ecg_t, ppg_t, coarse=coarse)
        track_options = dict(priors=(0.3 if centre is None else centre,), centre=0.3 if centre is None else centre,
                             max_dev=4.0)
        order = np.argsort(wsn_t, kind="stable")
        wsn_sorted = wsn_t[order]
        same_epoch = np.r_[wsn_epoch[order][1:] == wsn_epoch[order][:-1], False]
        routes = dict(ppg50=(ppg_t, None), ppg50_unrefined=(join(blocks50, "unrefined_s"), None),
                      ppg100=(ppg100_t, None), ppg100_unrefined=(join(blocks100, "unrefined_s"), None),
                      **{f"ppg50_{m}": (join(blocks_other[m], "beat_times_s"), None) for m in others})
        if wsn_sorted.size:
            routes["watchsleepnet"] = (wsn_sorted, same_epoch)
        tracks = {}
        for name, (route_t, pair_ok) in routes.items():
            tracks[name] = track_beats(ecg_t, route_t, **track_options)
            beats[name] = tracked_accuracy(ecg_t, route_t, tracks[name], ecg_inside=ecg_inside, pair_ok=pair_ok)
        track = tracks["ppg50"]
        lag = track["lag"][track["matched"] >= 0]
        at = ppg_t[track["matched"] >= 0]
        hours = np.floor(at / 3600.0)
        alignment.update(
            coarse_hr_lag_s=coarse, coarse_hr_corr=coarse_r, lag_centre_s=centre,
            tracked_fraction=float((track["matched"] >= 0).mean()),
            lag_median_s=float(np.median(lag)) if lag.size else None,
            lag_p5_s=float(np.quantile(lag, 0.05)) if lag.size else None,
            lag_p95_s=float(np.quantile(lag, 0.95)) if lag.size else None,
            lag_hourly_median_s={int(h): float(np.median(lag[hours == h])) for h in np.unique(hours)
                                 if (hours == h).sum() >= 100})
        covered = np.zeros(ecg_t.size, dtype=bool)
        covered[track["matched"][track["matched"] >= 0]] = True
        position = np.searchsorted(epoch_index, ecg_epoch[ecg_inside])
        hits = np.bincount(position, weights=covered[ecg_inside], minlength=keep.size)
        totals = np.bincount(position, minlength=keep.size)
        epoch_match = np.where(totals > 0, hits / np.maximum(totals, 1), 0.0).astype(np.float32)
        shift = lag_series(times25, ppg_t, track)
        reference = np.where(np.isfinite(shift), ibi_at(times25 - np.nan_to_num(shift), ecg_t), 0.0)
        for name, variant in variants.items():
            series[name] = series_accuracy(variant["ibi"], reference)
        if watchsleepnet:
            series["watchsleepnet"] = series_accuracy(ibi_wsn, reference)
        series["reference_coverage"] = float((reference > 0).mean())
    agreement = route_agreement(ppg100_t, ppg_t)

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
        ppg_beats=int(ppg_t.size), ppg100_beats=int(ppg100_t.size), ecg_beats=int(ecg_t.size), cleaning=options,
        peak_method=peak_method,
        relative_threshold=relative_threshold, alignment=alignment, beat_accuracy=beats, series_accuracy=series,
        route_agreement_50_vs_100=agreement,
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
