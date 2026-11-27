r"""KDE, spectrum, heart rate and HRV of real vs generated IBI, overall and per sleep stage (CPU only, reads only).

  python scripts/ibi_kde_spectrum.py --ibi NIGHTS/ibi_elgendi_local_fix [--output-dir DIR] [--workers N]
  python scripts/ibi_kde_spectrum.py --plot-only DIR          # re-draw the figures from DIR's CSV/JSON only
  ... --stages 4    W, Light (N1 + N2), Deep and REM, as the classifier's four classes, in IBI/kde_spectrum_4stages

Reads IBI_DIR/subjects/S*.npz from scripts/ppg_nights_to_ibi.py: the 25 Hz IBI (s) of every 30-s window,
generated and real (both from PPG through the same route), the sample status (0 measured, 1 filled,
2 artifact, 3 no beat) and the PSG stage. Valid samples are IBI > 0. Output in --output-dir
(default IBI_DIR/kde_spectrum):

  The figures, each for all stages together (_all) and for W/N1/N2/Deep/REM side by side (_stages):
  fig_kde_ibi_*.png     KDE of the IBI values (s), real vs generated
  fig_fft_*.png         IBI spectrum inside 30-s windows (median and IQR over participants), real vs generated
  fig_hr_*.png          KDE of heart rate (60 / IBI, bpm), real vs generated
  They are drawn only from kde_curves.csv, psd_stage.csv and kde_spectrum_summary.json (see --plot-only).

  With --extra-figures also the detailed figures:
  kde_ibi_hr.png        KDE of IBI (s) and of heart rate (60 / IBI, bpm), all stages and W/N1/N2/Deep/REM,
                        pooled valid samples (5 per second), Gaussian kernel SD = 0.03 x SD of the real values
                        (bw_method=0.03 as in compare_night_generations.py)
  spectrum_night.png    whole-night Welch PSD as in compare_night_generations.py (contiguous runs of epochs,
                        values outside 0.3-2.0 s filled linearly, 10-min Hann segments, 50 % overlap), mean over
                        participants, and per participant VLF/LF/HF power generated - real (dB)
  spectrum_stage.png    PSD of single 30-s windows (paired windows, both >= 90 % valid; the rest filled linearly
                        inside the window, linear detrend, Hann), median over participants per stage, and the
                        per-participant generated - real difference (dB). Generated windows are independent, so
                        this is the fair spectral comparison; frequency resolution 1/30 Hz.
  stage_hr_hrv.png      per participant and stage: heart rate, RMSSD and SD inside a window, LF and HF power,
                        HF centre frequency (breathing rate), real vs generated (median and IQR over participants)
  stage_contrast.png    per participant, stage minus N2 of heart rate and HF power, real vs generated
  participant_stage_metrics.csv, night_band_power.csv, psd_stage.csv, kde_curves.csv, kde_spectrum_summary.json

RMSSD uses successive measured intervals only (a run of identical 25 Hz values with status 0 is one interval).
The HF centre is the power-weighted mean frequency of 0.15-0.4 Hz of the mean window PSD (breathing rate; the
peak bin alone moves in steps of 2 per minute at 30-s resolution).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

FS = 25
EPOCH = 750
MEASURED = 0
STAGE_SETS = {"5": (("W", 0), ("N1", 1), ("N2", 2), ("Deep", 3), ("REM", 4)),
              "4": (("W", 0), ("Light", 1), ("Deep", 3), ("REM", 4))}    # 4: N2 windows counted as Light (code 1)
BASE_STAGES = {"5": "N2", "4": "Light"}                                  # the stage the contrasts compare against
STAGES = STAGE_SETS["5"]
GROUPS = ("ALL",) + tuple(name for name, _ in STAGES)
BASE_STAGE = BASE_STAGES["5"]


def use_stages(key):
    """Stage set for this process: "5" (W, N1, N2, Deep, REM) or "4" (W, Light = N1 + N2, Deep, REM)."""
    global STAGES, GROUPS, BASE_STAGE
    STAGES = STAGE_SETS[key]
    GROUPS = ("ALL",) + tuple(name for name, _ in STAGES)
    BASE_STAGE = BASE_STAGES[key]
NIGHT_BANDS = dict(VLF=(0.0033, 0.04), LF=(0.04, 0.15), HF=(0.15, 0.4))
LF, HF = (0.04, 0.15), (0.15, 0.4)
NPERSEG = 15000                     # 10 min at 25 Hz
KDE_STEP = 5                        # pooled samples: 5 per second
MIN_VALID = 0.9                     # window spectra: valid share needed in both windows of a pair
MIN_WINDOWS = 5                     # windows needed for a participant's stage metrics
MAX_FREQUENCY = 0.5
METRICS = ("hr_bpm", "ibi_ms", "sd_ms", "rmssd_ms", "lf_ms2", "hf_ms2", "lf_hf", "hf_centre_bpm")
LOG_METRICS = ("lf_ms2", "hf_ms2", "lf_hf")
SOURCES = ("real", "gen")
REAL_COLOR, GEN_COLOR = "#52514e", "#2a78d6"
INK, MUTED, GRID = "#52514e", "#8a8985", "#e4e3df"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ibi", type=Path, help="a ppg_nights_to_ibi.py output folder")
    parser.add_argument("--output-dir", type=Path, help="default: IBI/kde_spectrum")
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--exclude", nargs="*", default=[], help="participants to leave out")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--stages", choices=sorted(STAGE_SETS), default="5",
                        help="5: W, N1, N2, Deep, REM (default); 4: W, Light (N1 + N2), Deep, REM")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--extra-figures", action="store_true",
                        help="also the detailed figures (night spectrum, HR/HRV by stage, stage contrasts)")
    parser.add_argument("--plot-only", type=Path, metavar="DIR",
                        help="only re-draw the figures from an earlier output folder (no IBI data needed)")
    args = parser.parse_args(argv)
    if args.ibi is None and args.plot_only is None:
        parser.error("give --ibi (or --plot-only DIR)")
    return args


# --------------------------------------------------------------------------- per participant
def fill_drops(values, low=0.3, high=2.0):
    """Linear fill of zeros and out-of-range values (same rule as compare_night_generations.py)."""
    values = np.asarray(values, dtype=np.float64).copy()
    good = np.isfinite(values) & (values >= low) & (values < high)
    if good.sum() < 2:
        return None
    index = np.arange(values.size)
    values[~good] = np.interp(index[~good], index[good], values[good])
    return values


def night_psd(ibi, epoch_index):
    """Whole-night Welch PSD (ms^2/Hz) over contiguous runs of epochs, weighted by their segment count."""
    from scipy.signal import welch

    cuts = np.flatnonzero(np.diff(epoch_index) != 1) + 1
    total, weight, frequency = None, 0, None
    for run in np.split(np.arange(len(epoch_index)), cuts):
        if run.size * EPOCH < NPERSEG:
            continue
        filled = fill_drops(ibi[run].reshape(-1))
        if filled is None:
            continue
        frequency, psd = welch((filled - filled.mean()) * 1000.0, fs=FS, window="hann", nperseg=NPERSEG,
                               noverlap=NPERSEG // 2, detrend="constant", scaling="density")
        segments = 1 + (filled.size - NPERSEG) // (NPERSEG // 2)
        total = psd * segments if total is None else total + psd * segments
        weight += segments
    return (None, None) if total is None else (frequency, total / weight)


def trapezoid_power(frequency, psd, band):
    selected = (frequency >= band[0]) & (frequency < band[1])
    if selected.sum() < 2:
        return None
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(psd[selected], frequency[selected]))


def window_psd(rows):
    """Periodogram (ms^2/Hz) of 30-s windows [m, 750] (s): zeros filled inside the window, linear detrend, Hann."""
    from scipy.signal import periodogram

    filled = np.empty(rows.shape, dtype=np.float64)
    index = np.arange(rows.shape[1])
    for k, row in enumerate(rows):
        good = row > 0
        filled[k] = np.interp(index, index[good], row[good]) if not good.all() else row
    frequency, psd = periodogram(filled * 1000.0, fs=FS, window="hann", detrend="linear", scaling="density", axis=-1)
    keep = frequency <= MAX_FREQUENCY + 1e-9
    return frequency[keep], psd[:, keep]


def rect_power(frequency, psd, band):
    selected = (frequency >= band[0]) & (frequency < band[1])
    return float(psd[..., selected].sum(axis=-1) * (frequency[1] - frequency[0])) if selected.any() else None


def window_rmssd(values, status, min_pairs=10):
    """RMSSD (s) of one window from successive measured intervals (one hold run = one interval)."""
    measured = status == MEASURED
    change = np.ones(values.size, dtype=bool)
    change[1:] = (values[1:] != values[:-1]) | (measured[1:] != measured[:-1])
    starts = np.flatnonzero(change)
    run_values, run_measured = values[starts].astype(np.float64), measured[starts]
    pairs = run_measured[:-1] & run_measured[1:]
    steps = np.diff(run_values)[pairs]
    return float(np.sqrt(np.mean(steps ** 2))) if steps.size >= min_pairs else np.nan


def window_sd(values, min_samples=125):
    good = values[values > 0]
    return float(good.std()) if good.size >= min_samples else np.nan


def stage_metrics(ibi, status, rows, frequency=None, psd=None):
    """Metrics of one source over the windows `rows` (and mean window PSD of its paired windows)."""
    values = ibi[rows]
    valid = values[values > 0].astype(np.float64)
    out = dict(windows=int(rows.sum()))
    if out["windows"] < MIN_WINDOWS or valid.size < 2:
        return None
    out["hr_bpm"] = float(np.mean(60.0 / valid))
    out["ibi_ms"] = float(valid.mean() * 1000)
    sd = [window_sd(v) for v in values]
    rmssd = [window_rmssd(v, s) for v, s in zip(values, status[rows])]
    out["sd_ms"] = float(np.nanmedian(sd) * 1000) if np.isfinite(sd).any() else None
    out["rmssd_ms"] = float(np.nanmedian(rmssd) * 1000) if np.isfinite(rmssd).any() else None
    out.update(lf_ms2=None, hf_ms2=None, lf_hf=None, hf_centre_bpm=None, spectrum_windows=0)
    if psd is not None and len(psd) >= MIN_WINDOWS:
        mean = psd.mean(axis=0)
        lf, hf = rect_power(frequency, mean, LF), rect_power(frequency, mean, HF)
        band = (frequency >= HF[0]) & (frequency < HF[1])
        out.update(lf_ms2=lf, hf_ms2=hf, lf_hf=(lf / hf if hf else None), spectrum_windows=int(len(psd)),
                   hf_centre_bpm=float(np.sum(frequency[band] * mean[band]) / np.sum(mean[band]) * 60)
                   if mean[band].sum() > 0 else None, mean_psd=mean)
    return out


def analyze_subject(task):
    sid, folder, stage_set = task if len(task) == 3 else (*task, "5")
    use_stages(stage_set)
    with np.load(Path(folder) / "subjects" / f"{sid}.npz") as z:
        data = dict(gen=(z["ibi_generated"].astype(np.float64), z["ibi_generated_status"]),
                    real=(z["ibi_real"].astype(np.float64), z["ibi_real_status"]))
        stages = z["stages"].astype(np.int64)
        epoch_index = z["epoch_index"].astype(np.int64)
    if stage_set == "4":
        stages = np.where(stages == 2, 1, stages)                     # N1 and N2 together as Light
    valid_fraction = {s: (data[s][0] > 0).mean(axis=1) for s in SOURCES}
    paired = (valid_fraction["gen"] >= MIN_VALID) & (valid_fraction["real"] >= MIN_VALID)
    result = dict(subject=sid, epochs=int(len(stages)), paired_windows=int(paired.sum()), metrics={}, pooled={},
                  night={}, frequency=None, windows={g: int(len(stages)) if g == "ALL" else int((stages == c).sum())
                                                     for g, c in (("ALL", None),) + STAGES})
    spectra = {}
    for source in SOURCES:
        ibi, status = data[source]
        sampled = ibi[:, ::KDE_STEP]
        keep = sampled > 0
        result["pooled"][source] = (sampled[keep].astype(np.float32),
                                    np.repeat(stages, sampled.shape[1]).reshape(sampled.shape)[keep].astype(np.int8))
        frequency, psd = night_psd(ibi, epoch_index)
        if psd is not None:
            result["night"][source] = {band: trapezoid_power(frequency, psd, limits)
                                       for band, limits in NIGHT_BANDS.items()}
            result["night"][source]["psd"] = psd.astype(np.float32)
            result["night_frequency"] = frequency
        if paired.any():
            result["frequency"], spectra[source] = window_psd(ibi[paired])
    for group in GROUPS:
        codes = [c for _, c in STAGES] if group == "ALL" else [dict(STAGES)[group]]
        rows = np.isin(stages, codes)
        for source in SOURCES:
            ibi, status = data[source]
            psd = spectra[source][np.isin(stages[paired], codes)] if source in spectra else None
            metrics = stage_metrics(ibi, status, rows, result["frequency"], psd)
            if metrics is not None:
                result["metrics"][(group, source)] = metrics
    return result


# --------------------------------------------------------------------------- pooled statistics
def binned_kde(values, grid, sigma):
    step = grid[1] - grid[0]
    edges = np.concatenate([grid - step / 2, [grid[-1] + step / 2]])
    counts, _ = np.histogram(values, edges)
    half = int(math.ceil(4 * sigma / step))
    offsets = np.arange(-half, half + 1) * step
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(counts, kernel, mode="same") / (values.size * step)


def compare_samples(real, gen, scale):
    from scipy.stats import ks_2samp, wasserstein_distance

    if real.size < 2 or gen.size < 2:
        return None
    return dict(samples_real=int(real.size), samples_gen=int(gen.size),
                w1=float(wasserstein_distance(real, gen) * scale), ks=float(ks_2samp(real, gen).statistic),
                median_real=float(np.median(real) * scale), median_gen=float(np.median(gen) * scale),
                sd_real=float(real.std() * scale), sd_gen=float(gen.std() * scale))


def paired_summary(real, gen, log):
    from scipy.stats import wilcoxon

    real, gen = np.asarray(real, float), np.asarray(gen, float)
    keep = np.isfinite(real) & np.isfinite(gen) & ((real > 0) & (gen > 0) if log else True)
    real, gen = real[keep], gen[keep]
    if real.size < 3:
        return dict(participants=int(real.size))
    diff = 10 * np.log10(gen / real) if log else gen - real
    try:
        p = float(wilcoxon(diff).pvalue) if real.size >= 10 and np.any(diff != 0) else None
    except ValueError:
        p = None
    return dict(participants=int(real.size), real_median=float(np.median(real)), gen_median=float(np.median(gen)),
                real_q1=float(np.percentile(real, 25)), real_q3=float(np.percentile(real, 75)),
                gen_q1=float(np.percentile(gen, 25)), gen_q3=float(np.percentile(gen, 75)),
                median_diff=float(np.median(diff)), diff_unit="dB" if log else "metric", wilcoxon_p=p,
                r=float(np.corrcoef(real, gen)[0, 1]) if real.size >= 3 and real.std() > 0 and gen.std() > 0 else None)


def contrast_summary(real, gen):
    real, gen = np.asarray(real, float), np.asarray(gen, float)
    keep = np.isfinite(real) & np.isfinite(gen)
    real, gen = real[keep], gen[keep]
    if real.size < 3:
        return dict(participants=int(real.size))
    return dict(participants=int(real.size), real_median=float(np.median(real)), gen_median=float(np.median(gen)),
                real_positive=float((real > 0).mean()), gen_positive=float((gen > 0).mean()),
                same_sign=float((np.sign(real) == np.sign(gen)).mean()),
                r=float(np.corrcoef(real, gen)[0, 1]) if real.std() > 0 and gen.std() > 0 else None)


def table(results, group, source, metric):
    return np.array([np.nan if r["metrics"].get((group, source), {}).get(metric) is None
                     else r["metrics"][(group, source)][metric] for r in results], dtype=float)


def contrast(results, stage, metric):
    out = {}
    for source in SOURCES:
        value, base = table(results, stage, source, metric), table(results, BASE_STAGE, source, metric)
        out[source] = 10 * np.log10(value / base) if metric in LOG_METRICS else value - base
    return out


def summarize(results, pooled, args, ibi_dir):
    summary = dict(participants=len(results), ibi_dir=str(ibi_dir), excluded=sorted(args.exclude),
                   stage_set=args.stages, stages=[name for name, _ in STAGES], contrast_base=BASE_STAGE,
                   windows={g: int(sum(r["windows"][g] for r in results)) for g in GROUPS},
                   peak_method=None, kde={}, night_spectrum={}, window_spectrum={}, stage_metrics={},
                   stage_contrasts={}, settings=dict(kde_samples_per_second=FS // KDE_STEP, kde_sigma="0.03 x SD real",
                                                     night_nperseg_s=NPERSEG / FS, window_min_valid=MIN_VALID,
                                                     min_windows=MIN_WINDOWS, lf_hz=LF, hf_hz=HF))
    meta = ibi_dir / "ibi_summary.json"
    if meta.is_file():
        stored = json.loads(meta.read_text(encoding="utf-8"))
        summary.update(peak_method=stored.get("peak_method"), checkpoint_epoch=stored.get("checkpoint_epoch"),
                       nights=stored.get("nights"))
    for group in GROUPS:
        code = None if group == "ALL" else dict(STAGES)[group]
        sel = {s: pooled[s][0] if code is None else pooled[s][0][pooled[s][1] == code] for s in SOURCES}
        summary["kde"][group] = dict(ibi_ms=compare_samples(sel["real"].astype(float), sel["gen"].astype(float), 1000),
                                     hr_bpm=compare_samples(60 / sel["real"].astype(float),
                                                            60 / sel["gen"].astype(float), 1))
    for band in NIGHT_BANDS:
        real = [r["night"]["real"][band] for r in results if "real" in r["night"] and "gen" in r["night"]]
        gen = [r["night"]["gen"][band] for r in results if "real" in r["night"] and "gen" in r["night"]]
        summary["night_spectrum"][band] = paired_summary(real, gen, log=True)
    for group in GROUPS:
        summary["stage_metrics"][group] = {m: paired_summary(table(results, group, "real", m),
                                                             table(results, group, "gen", m), m in LOG_METRICS)
                                           for m in METRICS}
        summary["window_spectrum"][group] = dict(
            participants=int(np.isfinite(table(results, group, "real", "hf_ms2")).sum()),
            windows=int(np.nansum(table(results, group, "real", "spectrum_windows"))))
    for stage, _ in STAGES:
        if stage == BASE_STAGE:
            continue
        summary["stage_contrasts"][stage] = {m: contrast_summary(*(contrast(results, stage, m)[s] for s in SOURCES))
                                             for m in ("hr_bpm", "rmssd_ms", "hf_ms2", "lf_hf")}
    return summary


def write_tables(output, results):
    with (output / "participant_stage_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("subject", "stage", "source", "windows", "spectrum_windows", *METRICS))
        for r in results:
            for (group, source), m in sorted(r["metrics"].items()):
                writer.writerow((r["subject"], group, source, m["windows"], m.get("spectrum_windows"),
                                 *("" if m.get(k) is None else f"{m[k]:.6g}" for k in METRICS)))
    with (output / "night_band_power.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("subject", "band", "real_ms2", "gen_ms2", "gen_minus_real_db"))
        for r in results:
            if "real" not in r["night"] or "gen" not in r["night"]:
                continue
            for band in NIGHT_BANDS:
                a, b = r["night"]["real"][band], r["night"]["gen"][band]
                db = 10 * np.log10(b / a) if a and b and a > 0 and b > 0 else None
                writer.writerow((r["subject"], band, a, b, "" if db is None else f"{db:.4f}"))
    frequency = next((r["frequency"] for r in results if r["frequency"] is not None), None)
    if frequency is None:
        return
    with (output / "psd_stage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("stage", "source", "frequency_hz", "participants", "median_ms2_per_hz", "q1", "q3"))
        for group in GROUPS:
            for source in SOURCES:
                stack = stage_psd(results, group, source)
                if stack is None:
                    continue
                for k, f in enumerate(frequency):
                    q1, med, q3 = np.percentile(stack[:, k], [25, 50, 75])
                    writer.writerow((group, source, f"{f:.4f}", len(stack), f"{med:.6g}", f"{q1:.6g}", f"{q3:.6g}"))


def stage_psd(results, group, source, both=True):
    rows = [r["metrics"][(group, source)]["mean_psd"] for r in results
            if "mean_psd" in r["metrics"].get((group, source), {})
            and (not both or all("mean_psd" in r["metrics"].get((group, s), {}) for s in SOURCES))]
    return np.array(rows) if rows else None


KDE_SPECS = (("ibi_s", np.linspace(0.3, 2.0, 851), lambda v: v), ("hr_bpm", np.linspace(30, 200, 851), lambda v: 60 / v))


def write_kde_curves(output, pooled):
    """KDE curves of IBI (s) and heart rate (bpm) per group; kernel SD = 0.03 x SD of all real values."""
    real_all = pooled["real"][0].astype(np.float64)
    with (output / "kde_curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("variable", "group", "x", "real", "gen"))
        for variable, grid, transform in KDE_SPECS:
            sigma = 0.03 * float(np.std(transform(real_all)))
            for group in GROUPS:
                curves = {}
                for source in SOURCES:
                    values, codes = pooled[source]
                    values = values if group == "ALL" else values[codes == dict(STAGES)[group]]
                    curves[source] = (binned_kde(transform(values.astype(np.float64)), grid, sigma)
                                      if values.size > 1 else np.full(grid.size, np.nan))
                for x, a, b in zip(grid, curves["real"], curves["gen"]):
                    writer.writerow((variable, group, f"{x:.4f}", f"{a:.6g}", f"{b:.6g}"))


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def simple_figures(output):
    """KDE of IBI, FFT and heart rate, all stages (_all) and per stage (_stages), from the CSV/JSON outputs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    summary = json.loads((output / "kde_spectrum_summary.json").read_text(encoding="utf-8"))
    kde = {}
    for row in read_rows(output / "kde_curves.csv"):
        kde.setdefault((row["variable"], row["group"]), []).append((float(row["x"]), float(row["real"]),
                                                                    float(row["gen"])))
    kde = {key: np.array(value).T for key, value in kde.items()}
    psd = {}
    for row in read_rows(output / "psd_stage.csv"):
        psd.setdefault((row["stage"], row["source"]), []).append(
            (float(row["frequency_hz"]), float(row["median_ms2_per_hz"]), float(row["q1"]), float(row["q3"]),
             int(row["participants"])))
    psd = {key: np.array(value).T for key, value in psd.items()}
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.edgecolor": MUTED, "font.size": 10,
                         "axes.titlesize": 11, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.axisbelow": True})
    n = summary["participants"]
    stages = summary.get("stages") or [name for name, _ in STAGE_SETS["5"]]    # a summary from before --stages: 5
    series = (("real", REAL_COLOR, "Real", 2.0), ("gen", GEN_COLOR, "Generated", 1.8))

    def kde_panel(ax, group, variable, key, unit, limits, label):
        if (variable, group) not in kde:
            return
        x, *curves = kde[(variable, group)]
        for (source, color, name, width), y in zip(series, curves):
            ax.plot(x, y, color=color, lw=width, label=name)
        stats = summary["kde"][group][key]
        if stats:
            digits = 1 if unit == "ms" else 2
            middle = ".0f" if unit == "ms" else ".1f"
            ax.text(0.97, 0.96, f"W1 {stats['w1']:.{digits}f} {unit}\nmedian {stats['median_real']:{middle}} / "
                    f"{stats['median_gen']:{middle}} {unit}", transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color=INK)
        ax.set_xlim(*limits)
        ax.set_xlabel(label)

    def fft_panel(ax, group):
        if (group, "real") not in psd:
            return
        for low, high, band in (*LF, "LF"), (*HF, "HF"):
            ax.axvspan(low, high, color="#f4f3f0" if band == "LF" else "#ebeae6", zorder=0)
            ax.text((low + high) / 2, 0.98, band, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=9, color=INK)
        for source, color, name, width in series:
            f, median, q1, q3, _ = psd[(group, source)]
            shown = f > 0
            ax.fill_between(f[shown], 10 * np.log10(q1[shown]), 10 * np.log10(q3[shown]), color=color, alpha=0.14,
                            lw=0)
            ax.plot(f[shown], 10 * np.log10(median[shown]), color=color, lw=width, label=name)
        ax.set_xlim(0.02, MAX_FREQUENCY)
        ax.set_xlabel("frequency (Hz)")

    def subtitle(group, kind):
        if kind == "fft" and (group, "real") in psd:
            windows = summary["window_spectrum"][group]["windows"]
            return f"{int(psd[(group, 'real')][4][0])} participants, {windows} windows"
        return f"{summary['windows'][group]} windows"

    kinds = (
        ("kde_ibi", "IBI distribution (KDE)", "density",
         lambda ax, g: kde_panel(ax, g, "ibi_s", "ibi_ms", "ms", (0.4, 1.7), "IBI (s)")),
        ("fft", "IBI spectrum inside 30-s windows (median and IQR over participants)", "PSD (dB re 1 ms²/Hz)",
         fft_panel),
        ("hr", "Heart rate distribution (KDE)", "density",
         lambda ax, g: kde_panel(ax, g, "hr_bpm", "hr_bpm", "bpm", (35, 125), "Heart rate (bpm)")),
    )
    written = []
    for kind, title, ylabel, draw in kinds:
        fig, ax = plt.subplots(figsize=(6.6, 4.3))
        draw(ax, "ALL")
        ax.set_ylabel(ylabel)
        detail = subtitle("ALL", kind) if kind == "fft" else f"{n} participants, {subtitle('ALL', kind)}"
        ax.set_title(f"{title}\nall stages; {detail}", color=INK)
        legend(fig, [ax], loc="lower left" if kind == "fft" else "upper left", fontsize=9)
        fig.tight_layout()
        fig.savefig(output / f"fig_{kind}_all.png", dpi=170)
        plt.close(fig)
        fig, axes = plt.subplots(1, len(stages), figsize=(3.5 * len(stages), 4.0), sharex=True, sharey=True)
        for ax, stage in zip(axes, stages):
            draw(ax, stage)
            ax.set_title(f"{stage}\n{subtitle(stage, kind)}", color=INK)
        axes[0].set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=10)
        fig.suptitle(f"{title}, by sleep stage ({n} participants)", color=INK, fontsize=12, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(output / f"fig_{kind}_stages.png", dpi=170)
        plt.close(fig)
        written += [f"fig_{kind}_all.png", f"fig_{kind}_stages.png"]
    return written


def legend(fig, axes, **options):
    """One legend from the labelled artists of any panel (a panel can be empty)."""
    seen = {}
    for ax in np.ravel(axes):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(label, handle)
    if seen:
        np.ravel(axes)[0].legend(list(seen.values()), list(seen), frameon=False, **options)


# --------------------------------------------------------------------------- figures
def detail_figures(output, results, pooled, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.edgecolor": MUTED, "font.size": 10,
                         "axes.titlesize": 11, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.axisbelow": True})
    stages = dict(STAGES)
    n = summary["participants"]

    # KDE of IBI and heart rate
    fig, axes = plt.subplots(2, len(GROUPS), figsize=(3.3 * len(GROUPS), 6.6), sharey="row")
    real_all = pooled["real"][0].astype(np.float64)
    specs = (("ibi_ms", np.linspace(0.3, 2.0, 851), lambda v: v, 1000, "IBI (s)", (0.4, 1.7), "ms"),
             ("hr_bpm", np.linspace(30, 200, 851), lambda v: 60 / v, 1, "Heart rate (bpm)", (35, 125), "bpm"))
    for row, (key, grid, transform, _, label, limits, unit) in enumerate(specs):
        sigma = 0.03 * float(np.std(transform(real_all)))
        for col, group in enumerate(GROUPS):
            ax = axes[row, col]
            for source, color, name in (("real", REAL_COLOR, "Real"), ("gen", GEN_COLOR, "Generated")):
                values, codes = pooled[source]
                values = values if group == "ALL" else values[codes == stages[group]]
                if values.size > 1:
                    ax.plot(grid, binned_kde(transform(values.astype(np.float64)), grid, sigma), color=color,
                            lw=2 if source == "real" else 1.8, label=name)
            stats = summary["kde"][group][key]
            if stats:
                ax.text(0.97, 0.96, f"W1 {stats['w1']:.1f} {unit}\nmedian {stats['median_real']:.0f} / "
                        f"{stats['median_gen']:.0f}" if key == "ibi_ms" else
                        f"W1 {stats['w1']:.2f} {unit}\nmedian {stats['median_real']:.1f} / {stats['median_gen']:.1f}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)
            ax.set_xlim(*limits)
            ax.set_xlabel(label)
            if row == 0:
                ax.set_title("All stages" if group == "ALL" else group, color=INK)
            if col == 0:
                ax.set_ylabel("density")
    legend(fig, axes[0], loc="upper left", fontsize=9)
    fig.suptitle(f"IBI and heart-rate distributions, real vs generated ({n} participants; "
                 "text: W1 and median real / generated)", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "kde_ibi_hr.png", dpi=160)
    plt.close(fig)

    # whole-night spectrum
    night = [r for r in results if "real" in r["night"] and "gen" in r["night"]]
    if night:
        frequency = night[0]["night_frequency"]
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw=dict(width_ratios=(1.7, 1)))
        for (low, high), shade in zip(NIGHT_BANDS.values(), ("#f4f3f0", "#ebeae6", "#f4f3f0")):
            ax.axvspan(low, high, color=shade, zorder=0)
        for band, (low, high) in NIGHT_BANDS.items():
            ax.text(np.sqrt(low * high), 0.98, band, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    color=INK, fontsize=9)
        selected = (frequency > 0) & (frequency <= 1.0)
        for source, color, name in (("real", REAL_COLOR, "Real"), ("gen", GEN_COLOR, "Generated")):
            mean = np.mean([r["night"][source]["psd"] for r in night], axis=0)
            ax.plot(frequency[selected], 10 * np.log10(mean[selected] + 1e-30), color=color, lw=2, label=name)
        ax.set_xscale("log")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("PSD (dB re 1 ms²/Hz)")
        ax.set_title(f"Whole-night IBI spectrum, mean of {len(night)} participants", color=INK)
        ax.legend(frameon=False, loc="lower left")
        rng = np.random.default_rng(0)
        for k, band in enumerate(NIGHT_BANDS):
            diffs = np.array([10 * np.log10(r["night"]["gen"][band] / r["night"]["real"][band]) for r in night
                              if r["night"]["gen"][band] and r["night"]["real"][band]])
            bx.scatter(k + rng.uniform(-0.18, 0.18, diffs.size), diffs, s=14, color=GEN_COLOR, alpha=0.45,
                       edgecolor="none")
            q1, med, q3 = np.percentile(diffs, [25, 50, 75])
            bx.plot([k - 0.28, k + 0.28], [med, med], color=INK, lw=2.2)
            bx.plot([k, k], [q1, q3], color=INK, lw=1.2)
            bx.text(k + 0.3, med, f"{med:+.1f}", va="center", fontsize=9, color=INK)
        bx.axhline(0, color=MUTED, lw=1, ls="--")
        bx.set_xticks(range(len(NIGHT_BANDS)), list(NIGHT_BANDS))
        bx.set_xlim(-0.6, len(NIGHT_BANDS) - 0.2)
        bx.set_ylabel("generated - real (dB)")
        bx.set_title("Band power per participant (line: median)", color=INK)
        fig.tight_layout()
        fig.savefig(output / "spectrum_night.png", dpi=160)
        plt.close(fig)

    # 30-s window spectrum per stage
    frequency = next((r["frequency"] for r in results if r["frequency"] is not None), None)
    if frequency is not None:
        names = [name for name, _ in STAGES]
        fig, axes = plt.subplots(2, len(names), figsize=(3.4 * len(names), 6.4), sharey="row",
                                 gridspec_kw=dict(height_ratios=(1.5, 1)))
        shown = frequency > 0
        for col, stage in enumerate(names):
            ax, bx = axes[0, col], axes[1, col]
            for low, high in (LF, HF):
                for axis in (ax, bx):
                    axis.axvspan(low, high, color="#f4f3f0" if low == LF[0] else "#ebeae6", zorder=0)
            stacks = {s: stage_psd(results, stage, s) for s in SOURCES}
            if stacks["real"] is None:
                continue
            for source, color, name in (("real", REAL_COLOR, "Real"), ("gen", GEN_COLOR, "Generated")):
                q1, med, q3 = (10 * np.log10(np.percentile(stacks[source][:, shown], q, axis=0)) for q in (25, 50, 75))
                ax.fill_between(frequency[shown], q1, q3, color=color, alpha=0.14, lw=0)
                ax.plot(frequency[shown], med, color=color, lw=2, label=name)
            diff = 10 * np.log10(stacks["gen"][:, shown] / stacks["real"][:, shown])
            q1, med, q3 = np.percentile(diff, [25, 50, 75], axis=0)
            bx.fill_between(frequency[shown], q1, q3, color=GEN_COLOR, alpha=0.14, lw=0)
            bx.plot(frequency[shown], med, color=GEN_COLOR, lw=2)
            bx.axhline(0, color=MUTED, lw=1, ls="--")
            windows = summary["window_spectrum"][stage]["windows"]
            ax.set_title(f"{stage}  ({len(stacks['real'])} participants, {windows} windows)", color=INK)
            ax.text(np.mean(LF), 0.98, "LF", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)
            ax.text(np.mean(HF), 0.98, "HF", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)
            bx.set_xlabel("frequency (Hz)")
            for axis in (ax, bx):
                axis.set_xlim(0.02, MAX_FREQUENCY)
        low, high = axes[0, 0].get_ylim()
        axes[0, 0].set_ylim(low, high + 0.12 * (high - low))
        axes[0, 0].set_ylabel("PSD (dB re 1 ms²/Hz)\nmedian and IQR")
        axes[1, 0].set_ylabel("generated - real (dB)\nper participant")
        legend(fig, axes[0], loc="lower left", fontsize=9)
        fig.suptitle("IBI spectrum inside 30-s windows by sleep stage (same windows real and generated)",
                     color=INK, fontsize=12)
        fig.tight_layout()
        fig.savefig(output / "spectrum_stage.png", dpi=160)
        plt.close(fig)

    # stage levels: heart rate and HRV
    panels = (("hr_bpm", "Heart rate (bpm)", False), ("rmssd_ms", "RMSSD in a window (ms)", False),
              ("sd_ms", "IBI SD in a window (ms)", False), ("lf_ms2", "LF power 0.04-0.15 Hz (ms²)", True),
              ("hf_ms2", "HF power 0.15-0.4 Hz (ms²)", True), ("hf_centre_bpm", "HF centre = breathing (per min)", False))
    names = [name for name, _ in STAGES]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6))
    for ax, (metric, label, log) in zip(axes.ravel(), panels):
        for offset, source, color, name in ((-0.13, "real", REAL_COLOR, "Real"), (0.13, "gen", GEN_COLOR, "Generated")):
            for k, stage in enumerate(names):
                values = table(results, stage, source, metric)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                q1, med, q3 = np.percentile(values, [25, 50, 75])
                ax.plot([k + offset] * 2, [q1, q3], color=color, lw=2.2, solid_capstyle="round")
                ax.scatter([k + offset], [med], s=64, color=color, edgecolor="white", lw=1.5, zorder=3,
                           label=name if k == 0 else None)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                medians = [np.nanmedian(table(results, s, source, metric)) for s in names]
            ax.plot(np.arange(len(names)) + offset, medians, color=color, lw=1, alpha=0.5)
        if log:
            ax.set_yscale("log")
        ax.set_xticks(range(len(names)), names)
        ax.set_title(label, color=INK)
    legend(fig, axes, fontsize=9)
    fig.suptitle(f"Heart rate and HRV by sleep stage, per participant (dot: median, bar: IQR; {n} participants)",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "stage_hr_hrv.png", dpi=160)
    plt.close(fig)

    # stage contrasts against N2 (Light with --stages 4), per participant
    others = [name for name in names if name != BASE_STAGE]
    rows = (("hr_bpm", f"heart rate, stage - {BASE_STAGE} (bpm)"), ("hf_ms2", f"HF power, stage / {BASE_STAGE} (dB)"))
    fig, axes = plt.subplots(len(rows), len(others), figsize=(3.6 * len(others), 7.4))
    for i, (metric, label) in enumerate(rows):
        for j, stage in enumerate(others):
            ax = axes[i, j]
            c = contrast(results, stage, metric)
            keep = np.isfinite(c["real"]) & np.isfinite(c["gen"])
            x, y = c["real"][keep], c["gen"][keep]
            if x.size == 0:
                continue
            lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
            pad = 0.06 * (hi - lo or 1)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=1, ls="--")
            ax.axhline(0, color=GRID, lw=1)
            ax.axvline(0, color=GRID, lw=1)
            ax.scatter(x, y, s=22, color=GEN_COLOR, alpha=0.75, edgecolor="white", lw=0.5, zorder=3)
            s = summary["stage_contrasts"][stage][metric]
            r_text = "n/a" if s.get("r") is None else f"{s['r']:.2f}"
            ax.set_title(f"{stage}: median {s['real_median']:+.1f} real / {s['gen_median']:+.1f} gen\n"
                         f"r = {r_text}, same sign {s['same_sign']:.0%}", color=INK, fontsize=10)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlabel(f"real: {label}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"generated: {label}", fontsize=9)
    fig.suptitle("Does the generated night keep each participant's stage differences? (one dot = one participant)",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "stage_contrast.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def print_summary(summary):
    print(f"\nparticipants {summary['participants']}  peak method {summary['peak_method']}")
    print("\nKDE (pooled valid samples)        IBI W1 ms  median R/G ms      HR W1 bpm  median R/G bpm   KS(IBI)")
    for group in GROUPS:
        a, b = summary["kde"][group]["ibi_ms"], summary["kde"][group]["hr_bpm"]
        if a and b:
            print(f"  {group:5s}                            {a['w1']:8.1f}  {a['median_real']:6.0f} / {a['median_gen']:<6.0f}"
                  f"   {b['w1']:8.2f}  {b['median_real']:5.1f} / {b['median_gen']:<5.1f}   {a['ks']:.3f}")
    print("\nwhole-night band power, generated - real (dB, per participant)")
    for band, s in summary["night_spectrum"].items():
        if "median_diff" in s:
            print(f"  {band:4s} median {s['median_diff']:+6.2f}  participants {s['participants']}  "
                  f"p {s['wilcoxon_p'] if s['wilcoxon_p'] is None else round(s['wilcoxon_p'], 4)}")
    print("\nper stage, median over participants real / generated  (diff: gen-real, dB for LF/HF/ratio; Wilcoxon p)")
    for metric in METRICS:
        print(f"  {metric}")
        for group in GROUPS:
            s = summary["stage_metrics"][group][metric]
            if "median_diff" not in s:
                continue
            p = "  -  " if s["wilcoxon_p"] is None else f"{s['wilcoxon_p']:.3g}"
            r = "  -  " if s["r"] is None else f"{s['r']:.2f}"
            print(f"    {group:5s} {s['real_median']:9.2f} / {s['gen_median']:<9.2f} diff {s['median_diff']:+8.2f}"
                  f"  p {p:>8s}  r {r}  n {s['participants']}")
    print(f"\nstage - {BASE_STAGE} contrasts per participant: median real / generated, same sign, r")
    for stage, metrics in summary["stage_contrasts"].items():
        for metric, s in metrics.items():
            if "same_sign" in s:
                r = "  -  " if s["r"] is None else f"{s['r']:.2f}"
                print(f"  {stage:4s} {metric:9s} {s['real_median']:+8.2f} / {s['gen_median']:+8.2f}  "
                      f"same sign {s['same_sign']:.0%}  r {r}  n {s['participants']}")


def main(argv=None):
    args = parse_args(argv)
    if args.plot_only is not None:
        written = simple_figures(args.plot_only.expanduser().resolve())
        print("written:", ", ".join(written))
        return None
    use_stages(args.stages)
    ibi_dir = args.ibi.expanduser().resolve()
    default_name = "kde_spectrum" if args.stages == "5" else "kde_spectrum_4stages"
    output = (args.output_dir or ibi_dir / default_name).expanduser().resolve()
    sids = sorted(p.stem for p in (ibi_dir / "subjects").glob("S*.npz"))
    if args.subjects:
        sids = [s for s in sids if s in set(args.subjects)]
    sids = [s for s in sids if s not in set(args.exclude)]
    if not sids:
        raise SystemExit(f"No participants in {ibi_dir / 'subjects'}")
    output.mkdir(parents=True, exist_ok=True)
    tasks = [(sid, str(ibi_dir), args.stages) for sid in sids]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(analyze_subject, tasks))
    else:
        results = [analyze_subject(task) for task in tasks]
    pooled = {s: (np.concatenate([r["pooled"][s][0] for r in results]),
                  np.concatenate([r["pooled"][s][1] for r in results])) for s in SOURCES}
    summary = summarize(results, pooled, args, ibi_dir)
    (output / "kde_spectrum_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    write_tables(output, results)
    write_kde_curves(output, pooled)
    if not args.no_figures:
        simple_figures(output)
        if args.extra_figures:
            detail_figures(output, results, pooled, summary)
    print_summary(summary)
    print(f"\noutput: {output}")
    return summary


if __name__ == "__main__":
    main()
