"""Whole-night 50 Hz PPG generation with a trained ppg_training model.

For every participant, each stored 30-s epoch is generated independently (as in the IBI work: one
window per epoch, stitched into the night) from the participant's real conditions: sleep stage,
AHI/BMI/AGE/gender and the per-sample respiratory-event flags.

Quality check (the supervisor's request for generated PPG): every generated window goes through the
same template-matching SQI as real PPG; windows below `quality_threshold` are generated again with new
noise, up to `max_retries` times, and the attempt with the highest SQI is kept. Per-window SQI and the
number of attempts are stored, so the effect of the check is visible.

IBI: windows are generated independently, so the interval across a seam between two windows has no
meaning; each window therefore goes through ppg_extraction.ppg_to_ibi on its own (WatchSleepNet 25 Hz
format), and the real PPG is processed exactly the same way for the comparison.

Respiratory events: a counterfactual on windows that contain events (same noise, event channel zeroed)
shows whether the model reacts to the event condition; the within-window IBI contrast (event samples
minus other samples) is reported for real and generated nights.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zlib

import numpy as np

import ppg_extraction as px
import ppg_training as pt

SCHEMA = "ppg50_night_generation_v1"
STAGE_MAPPING = dict(W=0, N1=1, N2=2, Deep=3, REM=4)            # codes of the extraction (and night NPZ)
STAGE_GROUPS = dict(W=(0,), Light=(1, 2), REM=(4,), Deep=(3,))


def sha256_file(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def subject_seed(base, sid):
    return (int(base) + zlib.crc32(str(sid).encode())) % (2 ** 31)


def load_generator(checkpoint_path, device):
    """(model, weights dict) with the EMA weights when the checkpoint has them."""
    import torch

    weights = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if weights.get("schema") != pt.SCHEMA:
        raise ValueError(f"{checkpoint_path} is not a {pt.SCHEMA} checkpoint")
    config = weights["config"]
    if config["window_epochs"] != 1:
        raise ValueError("Night generation supports models with window_epochs = 1")
    model = pt.build_model(config, len(weights["channels"]))
    model.load_state_dict(weights["ema"] if weights.get("ema") is not None else weights["model"])
    return model.to(device).eval(), weights


def generate_windows(model, conditions, schedule, device, generator, batch_size):
    import torch

    out = []
    for lo in range(0, len(conditions), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(conditions[lo:lo + batch_size])).to(device)
        out.append(pt.sample(model, batch, schedule, generator=generator).cpu().numpy())
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, conditions.shape[1]), np.float32)


def windowed(windows):
    """ppg_to_ibi on every window separately: (ibi [N, 750], status [N, 750], sqi [N], beats [N])."""
    windows = np.asarray(windows, dtype=np.float64)
    if not len(windows):
        empty = np.zeros((0, px.EPOCH_IBI))
        return empty, empty.astype(np.uint8), np.zeros(0), np.zeros(0)
    result = px.ppg_to_ibi(windows, pt.FS, np.arange(len(windows)) * 1000.0)
    return result["ibi"], result["status"], result["sqi"], result["beats_per_epoch"]


def generate_with_quality(model, conditions, schedule, device, generator, *, batch_size, threshold, max_retries):
    """Generate windows; regenerate those with SQI below threshold, keeping each window's best attempt."""
    ppg = generate_windows(model, conditions, schedule, device, generator, batch_size)
    _, _, sqi, _ = windowed(ppg)
    first_sqi = sqi.copy()
    attempts = np.ones(len(ppg), dtype=np.int16)
    for _ in range(max_retries if threshold > 0 else 0):
        redo = np.flatnonzero(sqi < threshold)
        if not redo.size:
            break
        candidate = generate_windows(model, conditions[redo], schedule, device, generator, batch_size)
        _, _, candidate_sqi, _ = windowed(candidate)
        attempts[redo] += 1
        better = candidate_sqi > sqi[redo]
        ppg[redo[better]] = candidate[better]
        sqi[redo[better]] = candidate_sqi[better]
    return ppg, sqi, first_sqi, attempts


def event_contrast(ibi, events25, *, min_samples=125):
    """Mean over windows of (IBI during event samples - IBI elsewhere), valid samples only, in ms."""
    values = []
    for row, flag in zip(ibi, events25.astype(bool)):
        inside, outside = row[flag & (row > 0)], row[~flag & (row > 0)]
        if inside.size >= min_samples and outside.size >= min_samples:
            values.append((inside.mean() - outside.mean()) * 1000.0)
    return (float(np.mean(values)) if values else None), len(values)


def ibi_comparison(generated, real):
    from scipy.stats import wasserstein_distance

    g, r = generated[generated > 0], real[real > 0]
    if not g.size or not r.size:
        return dict(w1_ms=None, mean_gen_ms=None, mean_real_ms=None, sd_gen_ms=None, sd_real_ms=None)
    return dict(w1_ms=float(wasserstein_distance(g, r) * 1000.0), mean_gen_ms=float(g.mean() * 1000.0),
                mean_real_ms=float(r.mean() * 1000.0), sd_gen_ms=float(g.std() * 1000.0),
                sd_real_ms=float(r.std() * 1000.0))


def counterfactual_windows(apnea, count):
    """Up to `count` windows with event samples, spread over the night."""
    with_events = np.flatnonzero(apnea.any(axis=1))
    if not with_events.size or count <= 0:
        return with_events[:0]
    return with_events[np.unique(np.round(np.linspace(0, with_events.size - 1, min(count, with_events.size))).astype(int))]


def generate_subject(sid, cohort, model, schedule, device, options):
    """(arrays, metrics) for one participant's whole night."""
    import torch

    record = cohort.records[sid]
    n = record["stages"].size
    conditions = np.stack([cohort.window_array(sid, k) for k in range(n)])
    generator = torch.Generator().manual_seed(subject_seed(options["seed"], sid))
    ppg, sqi, first_sqi, attempts = generate_with_quality(
        model, conditions, schedule, device, generator, batch_size=options["batch_size"],
        threshold=options["quality_threshold"], max_retries=options["max_retries"])
    ibi_gen, status_gen, _, beats_gen = windowed(ppg)
    ibi_real, status_real, sqi_real, beats_real = windowed(record["ppg"])
    events25 = record["apnea"][:, ::2]                                   # 50 Hz -> 25 Hz samples

    metrics = dict(subject=sid, epochs=int(n), checkpoint_epoch=options.get("checkpoint_epoch"),
                   quality_threshold=options["quality_threshold"], max_retries=options["max_retries"],
                   overall=ibi_comparison(ibi_gen, ibi_real),
                   valid_fraction_gen=float((ibi_gen > 0).mean()), valid_fraction_real=float((ibi_real > 0).mean()),
                   beats_per_epoch_gen=float(beats_gen.mean()), beats_per_epoch_real=float(beats_real.mean()),
                   sqi_median_gen=float(np.median(sqi)), sqi_median_real=float(np.median(sqi_real)),
                   sqi_below_threshold_first_attempt=float((first_sqi < options["quality_threshold"]).mean()),
                   sqi_below_threshold_final=float((sqi < options["quality_threshold"]).mean()),
                   sqi_below_threshold_real=float((sqi_real < options["quality_threshold"]).mean()),
                   retried_epochs=float((attempts > 1).mean()), mean_attempts=float(attempts.mean()),
                   by_stage={})
    for name, codes in STAGE_GROUPS.items():
        rows = np.isin(record["stages"], codes)
        if rows.any():
            metrics["by_stage"][name] = dict(epochs=int(rows.sum()), **ibi_comparison(ibi_gen[rows], ibi_real[rows]))
    contrast_real, windows_real = event_contrast(ibi_real, events25)
    contrast_gen, windows_gen = event_contrast(ibi_gen, events25)
    metrics["events"] = dict(event_sample_fraction=float(events25.mean()),
                             within_window_contrast_real_ms=contrast_real, contrast_windows_real=windows_real,
                             within_window_contrast_gen_ms=contrast_gen, contrast_windows_gen=windows_gen)

    chosen = counterfactual_windows(record["apnea"], options["counterfactual_windows"])
    counterfactual = {}
    if chosen.size:
        zeroed = conditions[chosen].copy()
        zeroed[:, :, -1] = 0.0
        seed = subject_seed(options["seed"] + 1, sid)
        with_events = generate_windows(model, conditions[chosen], schedule, device,
                                       torch.Generator().manual_seed(seed), options["batch_size"])
        without = generate_windows(model, zeroed, schedule, device, torch.Generator().manual_seed(seed),
                                   options["batch_size"])
        ibi_with, ibi_without = windowed(with_events)[0], windowed(without)[0]
        flags = events25[chosen].astype(bool)
        both = (ibi_with > 0) & (ibi_without > 0)
        diff_event = (ibi_with - ibi_without)[both & flags]
        diff_other = (ibi_with - ibi_without)[both & ~flags]
        counterfactual = dict(epochs=chosen, ppg_with_events=with_events, ppg_events_zeroed=without,
                              ibi_with_events=ibi_with.astype(np.float32), ibi_events_zeroed=ibi_without.astype(np.float32))
        metrics["events"].update(
            counterfactual_windows=int(chosen.size),
            counterfactual_ibi_change_event_samples_ms=float(diff_event.mean() * 1000) if diff_event.size else None,
            counterfactual_ibi_change_other_samples_ms=float(diff_other.mean() * 1000) if diff_other.size else None,
            counterfactual_mean_abs_ppg_change=float(np.abs(with_events - without).mean()),
            within_window_contrast_with_events_ms=event_contrast(ibi_with, events25[chosen])[0],
            within_window_contrast_events_zeroed_ms=event_contrast(ibi_without, events25[chosen])[0])
    arrays = dict(schema=np.array(SCHEMA), subject=np.array(sid), generated_ppg=ppg,
                  sqi_generated=sqi.astype(np.float32), sqi_first_attempt=first_sqi.astype(np.float32),
                  attempts=attempts, sqi_real=sqi_real.astype(np.float32),
                  ibi_generated=ibi_gen.astype(np.float32), ibi_generated_status=status_gen,
                  ibi_real_windowed=ibi_real.astype(np.float32), ibi_real_status=status_real,
                  stages=record["stages"].astype(np.int8), epoch_index=record["epoch_index"],
                  epoch_ok=record["epoch_ok"], apnea_25hz=events25.astype(np.uint8),
                  **{f"counterfactual_{k}": v for k, v in counterfactual.items()},
                  metrics=np.array(json.dumps(metrics, default=float)))
    return arrays, metrics


def night_arrays(ibi, record, ahi, *, synthetic, epoch_start_s=None):
    """DREAMT/WatchSleepNet night NPZ fields (as night_data.save_night_npz writes them)."""
    ibi = np.asarray(ibi, dtype=np.float32)
    arrays = dict(data=ibi.reshape(-1)[:, None], stages=np.repeat(record["stages"].astype(np.int64), px.EPOCH_IBI),
                  fs=np.array(px.FS_IBI), ahi=np.array(float(ahi)), synthetic=np.array(bool(synthetic)),
                  stage_mapping=np.array(json.dumps(STAGE_MAPPING, sort_keys=True)),
                  epoch_index=record["epoch_index"], apnea=record["apnea"][:, ::2].reshape(-1).astype(np.uint8),
                  ibi_processing=np.array("each 30-s window through ppg_extraction.ppg_to_ibi on its own"))
    if epoch_start_s is not None:
        arrays["epoch_start_seconds"] = np.asarray(epoch_start_s)
    return arrays
