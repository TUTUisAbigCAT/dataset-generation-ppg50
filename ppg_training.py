"""Conditional diffusion on 50 Hz PPG epochs extracted by ppg_extraction.py (DREAMT 2.2.0).

Model and diffusion are the ones of the IBI work: the SSSD-S4 imputer (TSImputers), a 200-step
linear_power schedule ending at alpha_bar = 1e-8, epsilon-MSE on the signal channel only, parameter
EMA for evaluation, and the DDPM reverse step with the posterior standard deviation. What changes:

  * data: one window = `window_epochs` consecutive 30-s epochs of the stored 50 Hz PPG (1500 samples
    per epoch); by default only epochs with epoch_ok (teacher's step 4) are used;
  * channels: PPG, stage one-hot (W, Light = N1+N2, REM, Deep), AHI/BMI/AGE z-scores, gender F/M and,
    optionally, a 0/1 respiratory-event channel per sample (any of the four DREAMT event columns);
  * normalization: per window, PPG minus its median divided by half its 1-99 % range, clipped to
    +-3 (wrist BVP amplitude is arbitrary; the PPG -> IBI path is scale invariant);
  * checkpoint selection: several validation participants (S092 is never one of them by default),
    chosen from the extraction report (regular ECG rhythm, good PPG, E4 clock within 3 s of the PSG)
    and spread over AHI. Every epoch
    the EMA model generates their windows, generated and real windows go through the same
    ppg_extraction.ppg_to_ibi, and the mean per-participant Wasserstein-1 distance between generated
    and real IBI values (ms) selects best.pt.
"""
from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path
import time

import numpy as np

import ppg_extraction as px

SCHEMA = "ppg50_training_v1"
FS = px.FS_PPG
EPOCH_SAMPLES = px.EPOCH_PPG
STAGE_NAMES = ("W", "Light", "REM", "Deep")
STAGE_TO_CLASS = np.array([0, 1, 1, 3, 2], dtype=np.int64)       # W, N1, N2, N3, R -> W, Light, Light, Deep, REM
CONTINUOUS = ("AHI", "BMI", "AGE")
BASE_CHANNELS = ("PPG",) + tuple("stage_" + s for s in STAGE_NAMES) + CONTINUOUS + ("gender_F", "gender_M")
APNEA_CHANNEL = "apnea_event"

DEFAULTS = dict(
    data_dir="/home/yxtao/links/scratch/WatchSleepProcessedDataset/DREAMT22_PPG50",
    metadata_path="/home/yxtao/links/scratch/dreamt/physionet.org/files/dreamt/2.2.0/participant_info.csv",
    output_root="/home/yxtao/links/scratch/dataset-generation-ppg50/runs",
    exclude_subjects=[],
    epoch_filter="epoch_ok",
    window_epochs=1,
    apnea_condition=True,
    normalization=dict(low_percentile=1.0, high_percentile=99.0, clip=3.0),
    validation=dict(subjects=None, n_subjects=8, never=["S092"], windows_per_subject=32, batch_size=64,
                    seed=20261122, held_out=False, min_median_sqi=0.9, min_epoch_ok_fraction=0.95,
                    max_abs_lag_s=3.0),
    training=dict(epochs=50, batch_size=32, lr=1e-4, ema_decay=0.9999, s4_dropout=0.03, seed=42, num_workers=4,
                  log_every=200, max_steps_per_epoch=None),
    model=dict(num_res_layers=4, res_channels=64, skip_channels=64, s4_dstate=64,
               diff_step_embed_in=32, diff_step_embed_mid=64, diff_step_embed_out=64),
    diffusion=dict(timesteps=200, beta_0=1e-4, beta_T=0.02, diffusion_schedule="linear_power",
                   terminal_alpha_bar=1e-8),
)


# --------------------------------------------------------------------------- configuration
def _merge(defaults, given, path=""):
    unknown = set(given) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown configuration keys {sorted(path + k for k in unknown)}")
    merged = {}
    for key, value in defaults.items():
        if key in given and isinstance(value, dict) and value and isinstance(given[key], dict):
            merged[key] = _merge(value, given[key], path + key + ".")
        else:
            merged[key] = copy.deepcopy(given.get(key, value))
    return merged


def load_config(path=None, overrides=None):
    given = {}
    if path is not None:
        given = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(given, dict):
            raise ValueError("Configuration must be a JSON object")
    config = _merge(DEFAULTS, given)
    if overrides:
        config = _merge(config, overrides)
    if config["epoch_filter"] not in ("epoch_ok", "all"):
        raise ValueError("epoch_filter must be 'epoch_ok' or 'all'")
    if not isinstance(config["window_epochs"], int) or config["window_epochs"] < 1:
        raise ValueError("window_epochs must be a positive integer")
    t = config["training"]
    if t["epochs"] < 1 or t["batch_size"] < 1 or not t["lr"] > 0 or not 0 <= t["ema_decay"] < 1:
        raise ValueError("training needs positive epochs/batch_size/lr and ema_decay in [0, 1)")
    v = config["validation"]
    if v["windows_per_subject"] < 1 or v["batch_size"] < 1 or (v["subjects"] is None and v["n_subjects"] < 1):
        raise ValueError("validation needs positive windows_per_subject, batch_size and n_subjects")
    return config


def channel_names(config):
    return BASE_CHANNELS + ((APNEA_CHANNEL,) if config["apnea_condition"] else ())


def window_samples(config):
    return config["window_epochs"] * EPOCH_SAMPLES


# --------------------------------------------------------------------------- data
def normalize_window(ppg, options):
    """Median-centred PPG divided by half its low-high percentile range, clipped to +-clip."""
    ppg = np.asarray(ppg, dtype=np.float64)
    low, high = np.percentile(ppg, [options["low_percentile"], options["high_percentile"]])
    scale = (high - low) / 2.0
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(ppg, dtype=np.float32)
    return np.clip((ppg - np.median(ppg)) / scale, -options["clip"], options["clip"]).astype(np.float32)


def read_participants(path):
    import pandas as pd

    frame = pd.read_csv(path, dtype={"SID": str})
    missing = {"SID", "AHI", "BMI", "AGE", "GENDER"} - set(frame.columns)
    if missing:
        raise ValueError(f"participant_info.csv lacks {sorted(missing)}")
    frame["SID"] = frame["SID"].str.strip()
    return frame.set_index("SID")


class PpgCohort:
    """All extracted participants in memory (PPG float32, stages, event flags, epoch_ok)."""

    def __init__(self, config):
        self.config = config
        self.data_dir = Path(config["data_dir"])
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"data_dir does not exist: {self.data_dir}")
        participants = read_participants(config["metadata_path"])
        files = sorted(p for p in self.data_dir.glob("S*.npz") if not p.name.startswith("."))
        excluded = set(config["exclude_subjects"])
        self.subjects, self.records, self.skipped = [], {}, {}
        for path in files:
            sid = path.stem
            if sid in excluded:
                self.skipped[sid] = "excluded by configuration"
                continue
            if sid not in participants.index:
                self.skipped[sid] = "not in participant_info.csv"
                continue
            with np.load(path) as z:
                schema = str(z["schema"]) if "schema" in z.files else ""
                if not schema.startswith("dreamt_ppg50_extraction_"):
                    self.skipped[sid] = f"not a PPG extraction file ({schema or 'no schema'})"
                    continue
                record = dict(ppg=z["ppg"].astype(np.float32), stages=z["stages"].astype(np.int64),
                              epoch_index=z["epoch_index"].astype(np.int64), epoch_ok=z["epoch_ok"].astype(bool),
                              apnea=z["events"].max(axis=2).astype(np.uint8), schema=schema,
                              epoch_start_s=z["epoch_start_s"].astype(np.float64) if "epoch_start_s" in z.files else None)
            row = participants.loc[sid]
            gender = str(row["GENDER"]).strip().upper()
            continuous = np.array([float(row[c]) for c in CONTINUOUS])
            if gender not in ("F", "M") or not np.isfinite(continuous).all():
                self.skipped[sid] = "incomplete participant conditions"
                continue
            record.update(continuous=continuous, gender=gender, sid=sid)
            self.records[sid] = record
            self.subjects.append(sid)
        if not self.subjects:
            raise ValueError(f"No usable PPG extraction files in {self.data_dir}")
        self.report = {}
        report_path = self.data_dir / "report.json"
        if report_path.is_file():
            self.report = json.loads(report_path.read_text(encoding="utf-8")).get("subjects", {})

    def windows(self, sids):
        """(sid, first epoch row) of every window of window_epochs consecutive, usable epochs."""
        count = self.config["window_epochs"]
        out = []
        for sid in sids:
            r = self.records[sid]
            usable = r["epoch_ok"] if self.config["epoch_filter"] == "epoch_ok" else np.ones(r["stages"].size, bool)
            n = r["stages"].size
            for first in range(0, n - count + 1, count):
                rows = slice(first, first + count)
                if usable[rows].all() and np.all(np.diff(r["epoch_index"][rows]) == 1):
                    out.append((sid, first))
        return out

    def fit_scaler(self, sids):
        values = np.stack([self.records[s]["continuous"] for s in sids])
        mean, std = values.mean(axis=0), values.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        self.scaler = dict(subjects=list(sids), continuous=list(CONTINUOUS), mean=mean.tolist(), std=std.tolist())
        return self.scaler

    def window_array(self, sid, first):
        r = self.records[sid]
        count = self.config["window_epochs"]
        length = count * EPOCH_SAMPLES
        channels = channel_names(self.config)
        array = np.empty((length, len(channels)), dtype=np.float32)
        array[:, 0] = normalize_window(r["ppg"][first:first + count].reshape(-1), self.config["normalization"])
        stage = STAGE_TO_CLASS[np.repeat(r["stages"][first:first + count], EPOCH_SAMPLES)]
        array[:, 1:5] = np.eye(4, dtype=np.float32)[stage]
        array[:, 5:8] = (r["continuous"] - np.asarray(self.scaler["mean"])) / np.asarray(self.scaler["std"])
        array[:, 8] = float(r["gender"] == "F")
        array[:, 9] = float(r["gender"] == "M")
        if self.config["apnea_condition"]:
            array[:, 10] = r["apnea"][first:first + count].reshape(-1)
        return array


def select_validation_subjects(cohort, config):
    """Explicit list, or n_subjects eligible participants spread evenly over AHI (never `never`)."""
    v = config["validation"]
    never = set(v["never"])
    if v["subjects"] is not None:
        chosen = [s for s in v["subjects"]]
        bad = [s for s in chosen if s in never or s not in cohort.records]
        if bad:
            raise ValueError(f"Validation subjects not usable or forbidden: {bad}")
        return chosen, dict(policy="explicit list")
    eligible, reasons = [], {}
    for sid in cohort.subjects:
        info = cohort.report.get(sid, {})
        alignment = info.get("alignment") or {}
        if sid in never:
            reasons[sid] = "never used for validation"
        elif alignment.get("ecg_irregular"):
            reasons[sid] = "irregular ECG rhythm"
        elif info and (info.get("median_sqi") or 0) < v["min_median_sqi"]:
            reasons[sid] = "low PPG quality"
        elif info and (info.get("epoch_ok_fraction") or 0) < v["min_epoch_ok_fraction"]:
            reasons[sid] = "too few usable epochs"
        elif alignment.get("lag_median_s") is not None and abs(alignment["lag_median_s"]) > v["max_abs_lag_s"]:
            reasons[sid] = "E4 clock far from PSG (stage labels shifted)"
        else:
            eligible.append(sid)
    if len(eligible) < v["n_subjects"]:
        raise ValueError(f"Only {len(eligible)} eligible validation participants; need {v['n_subjects']}")
    eligible.sort(key=lambda s: (cohort.records[s]["continuous"][0], s))
    picks = np.unique(np.round(np.linspace(0, len(eligible) - 1, v["n_subjects"])).astype(int))
    chosen = [eligible[i] for i in picks]
    return chosen, dict(policy=f"{v['n_subjects']} eligible participants evenly spaced by AHI",
                        eligible=len(eligible), not_eligible=reasons,
                        ahi={s: float(cohort.records[s]["continuous"][0]) for s in chosen})


def select_validation_windows(cohort, sids, per_subject):
    selected = []
    for sid in sids:
        candidates = cohort.windows([sid])
        if not candidates:
            continue
        index = np.unique(np.round(np.linspace(0, len(candidates) - 1, min(per_subject, len(candidates)))).astype(int))
        selected.extend(candidates[i] for i in index)
    return selected


# --------------------------------------------------------------------------- model and diffusion
def materialize_storage(model):
    """Give every parameter and buffer its own dense storage (same values).

    The S4 constructor registers B, P and w as einops-repeated views with stride 0 across the H
    features although tie_state=False: in-place optimizer updates then write many gradients into one
    memory cell, and checkpoints cannot be loaded back (resume). Dense copies make the H copies
    independent, as tie_state=False intends.
    """
    import torch

    changed = []
    for module_name, module in model.named_modules():
        for name, parameter in list(module._parameters.items()):
            if parameter is None or parameter.is_contiguous() and 0 not in parameter.stride():
                continue
            dense = torch.nn.Parameter(parameter.detach().clone(memory_format=torch.contiguous_format),
                                       requires_grad=parameter.requires_grad)
            dense.__dict__.update(parameter.__dict__)
            module._parameters[name] = dense
            changed.append(f"{module_name}.{name}")
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and (not buffer.is_contiguous() or 0 in buffer.stride()):
                module._buffers[name] = buffer.detach().clone(memory_format=torch.contiguous_format)
                changed.append(f"{module_name}.{name}")
    return changed


def build_model(config, in_channels):
    from TSImputers.SSSDS4Imputer import SSSDS4Imputer

    m = config["model"]
    model = SSSDS4Imputer(in_channels, m["res_channels"], m["skip_channels"], 1, m["num_res_layers"],
                          m["diff_step_embed_in"], m["diff_step_embed_mid"], m["diff_step_embed_out"],
                          window_samples(config), m["s4_dstate"], config["training"]["s4_dropout"], True, True)
    materialize_storage(model)
    return model


def build_diffusion(config, device):
    from diffusion_schedule import build_diffusion_config

    schedule = build_diffusion_config(dict(config["diffusion"]))
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in schedule.items()}


def diffusion_loss(model, batch, schedule):
    """Epsilon MSE on the PPG channel; the condition channels stay clean (upstream training step)."""
    import torch

    steps = torch.randint(schedule["T"], size=(batch.shape[0],), device=batch.device)
    noise = torch.randn(batch.shape[0], batch.shape[1], 1, device=batch.device)
    alpha_bar = schedule["alpha_bars"][steps].reshape(-1, 1, 1)
    noised = batch.clone()
    noised[:, :, :1] = torch.sqrt(alpha_bar) * batch[:, :, :1] + torch.sqrt(1 - alpha_bar) * noise
    predicted = model(noised, steps.reshape(-1, 1))                    # [B, 1, L]
    return torch.nn.functional.mse_loss(predicted, noise.permute(0, 2, 1))


def sample(model, conditions, schedule, *, generator=None):
    """DDPM reverse process for the PPG channel given clean conditions [B, L, C] (channel 0 ignored).

    x_{t-1} = (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_t) + sigma_t z, with
    sigma_t = sqrt(beta_t (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)) (posterior standard deviation)
    and no noise at the last step - the corrected upstream pipeline for all-unknown windows.
    """
    import torch

    x = conditions.clone()
    device = x.device
    x[:, :, 0] = torch.randn(x.shape[:2], generator=generator, device="cpu").to(device)
    with torch.no_grad():
        for step in range(schedule["T"] - 1, -1, -1):
            times = torch.full((x.shape[0], 1), step, device=device, dtype=torch.long)
            eps = model(x, times)[:, 0, :]
            alpha, alpha_bar, beta = (schedule[k][step, 0] for k in ("alphas", "alpha_bars", "betas"))
            mean = (x[:, :, 0] - beta / torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha)
            if step > 0:
                variance = beta * (1 - schedule["alpha_bars"][step - 1, 0]) / (1 - alpha_bar)
                z = torch.randn(x.shape[:2], generator=generator, device="cpu").to(device)
                mean = mean + torch.sqrt(variance) * z
            x[:, :, 0] = mean
    return x[:, :, 0]


class EMA:
    def __init__(self, model, decay):
        self.decay, self.updates = float(decay), 0
        self.model = copy.deepcopy(model).eval() if decay > 0 else None
        if self.model is not None:
            self.model.requires_grad_(False)

    def update(self, model):
        if self.model is None:
            return
        import torch

        self.updates += 1
        decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        with torch.no_grad():                # S4 buffers are fixed at construction; only parameters move
            for ema_p, p in zip(self.model.parameters(), model.parameters()):
                ema_p.lerp_(p.detach(), 1.0 - decay)

    def evaluation_model(self, model):
        return self.model if self.model is not None else model


# --------------------------------------------------------------------------- validation metrics
def derived_ibi(windows, config):
    """25 Hz IBI of independent windows [N, L] through ppg_extraction.ppg_to_ibi (one block per window)."""
    windows = np.asarray(windows, dtype=np.float64)
    count = config["window_epochs"]
    epochs = windows.reshape(-1, EPOCH_SAMPLES)
    starts = (np.arange(windows.shape[0])[:, None] * 1000.0 + np.arange(count)[None, :] * 30.0).reshape(-1)
    result = px.ppg_to_ibi(epochs, FS, starts)
    return result["ibi"].reshape(windows.shape[0], -1), result["beats_per_epoch"].reshape(windows.shape[0], -1).sum(1)


def log_spectrum(windows):
    from scipy.signal import welch

    freqs, power = welch(np.asarray(windows, dtype=np.float64), fs=FS, nperseg=256, axis=-1)
    band = (freqs >= 0.5) & (freqs <= 8.0)
    return np.log10(np.maximum(power[:, band].mean(axis=0), 1e-12))


def validation_metrics(generated, real, owners, config, *, missing_penalty_ms=2000.0):
    """Per-participant and mean metrics of generated vs real PPG windows (all normalized)."""
    from scipy.stats import wasserstein_distance

    ibi_gen, beats_gen = derived_ibi(generated, config)
    ibi_real, beats_real = derived_ibi(real, config)
    per_subject = {}
    for sid in sorted(set(owners)):
        rows = np.asarray([o == sid for o in owners])
        g, r = ibi_gen[rows], ibi_real[rows]
        g_values, r_values = g[g > 0], r[r > 0]
        w1 = (wasserstein_distance(g_values, r_values) * 1000.0 if g_values.size and r_values.size
              else missing_penalty_ms)
        per_subject[sid] = dict(
            ibi_w1_ms=float(w1),
            ibi_mean_gen_ms=float(g_values.mean() * 1000) if g_values.size else None,
            ibi_mean_real_ms=float(r_values.mean() * 1000) if r_values.size else None,
            ibi_sd_gen_ms=float(g_values.std() * 1000) if g_values.size else None,
            ibi_sd_real_ms=float(r_values.std() * 1000) if r_values.size else None,
            valid_fraction_gen=float((g > 0).mean()), valid_fraction_real=float((r > 0).mean()),
            beats_per_window_gen=float(beats_gen[rows].mean()), beats_per_window_real=float(beats_real[rows].mean()),
            log_psd_distance=float(np.abs(log_spectrum(generated[rows]) - log_spectrum(real[rows])).mean()),
        )
    keys = ("ibi_w1_ms", "valid_fraction_gen", "valid_fraction_real", "beats_per_window_gen",
            "beats_per_window_real", "log_psd_distance")
    summary = {k: float(np.mean([m[k] for m in per_subject.values()])) for k in keys}
    diffs = [m["ibi_mean_gen_ms"] - m["ibi_mean_real_ms"] for m in per_subject.values()
             if m["ibi_mean_gen_ms"] is not None and m["ibi_mean_real_ms"] is not None]
    summary["ibi_mean_bias_ms"] = float(np.mean(diffs)) if diffs else None
    # amplitude in normalized units: an untrained reverse process can grow far beyond the data scale
    summary["std_gen"] = float(np.std(generated))
    summary["std_real"] = float(np.std(real))
    summary["max_abs_gen"] = float(np.max(np.abs(generated)))
    return summary, per_subject


# --------------------------------------------------------------------------- training run
def _without_epochs(config):
    config = copy.deepcopy(config)
    config["training"].pop("epochs", None)
    return config


def _save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


class Trainer:
    def __init__(self, config, output_dir, *, device=None, resume=False, log=print):
        import torch

        self.config, self.output, self.log = config, Path(output_dir), log
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cohort = PpgCohort(config)
        self.validation_subjects, self.validation_selection = select_validation_subjects(self.cohort, config)
        held_out = config["validation"]["held_out"]
        self.training_subjects = [s for s in self.cohort.subjects
                                  if not (held_out and s in self.validation_subjects)]
        self.cohort.fit_scaler(self.training_subjects)
        self.training_windows = self.cohort.windows(self.training_subjects)
        self.validation_windows = select_validation_windows(self.cohort, self.validation_subjects,
                                                            config["validation"]["windows_per_subject"])
        if not self.training_windows or not self.validation_windows:
            raise ValueError("No training or validation windows")
        self.channels = channel_names(config)
        self.resume = resume

    def summary(self):
        from collections import Counter

        counts = Counter(s for s, _ in self.training_windows)
        return dict(schema=SCHEMA, channels=list(self.channels), window_samples=window_samples(self.config),
                    sample_rate_hz=FS, training_subjects=len(self.training_subjects),
                    training_windows=len(self.training_windows), validation_subjects=self.validation_subjects,
                    validation_windows=len(self.validation_windows), validation_selection=self.validation_selection,
                    validation_included_in_training=not self.config["validation"]["held_out"],
                    skipped_subjects=self.cohort.skipped, scaler=self.cohort.scaler,
                    windows_per_training_subject=dict(sorted(counts.items())))

    def _dataset(self):
        import torch
        from torch.utils.data import Dataset

        cohort, windows = self.cohort, self.training_windows

        class Windows(Dataset):
            def __len__(self):
                return len(windows)

            def __getitem__(self, index):
                return torch.from_numpy(cohort.window_array(*windows[index]))

        return Windows()

    def _validation_arrays(self):
        arrays = np.stack([self.cohort.window_array(*w) for w in self.validation_windows])
        owners = [s for s, _ in self.validation_windows]
        return arrays, owners

    def run(self, *, max_hours=None):
        import torch
        from torch.utils.data import DataLoader

        cfg, t = self.config, self.config["training"]
        started = time.time()
        self.output.mkdir(parents=True, exist_ok=True)
        state_path = self.output / "state.pt"
        torch.manual_seed(t["seed"])
        np.random.seed(t["seed"])
        model = build_model(cfg, len(self.channels)).to(self.device)
        ema = EMA(model, t["ema_decay"])
        optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"])
        schedule = build_diffusion(cfg, self.device)
        first_epoch, best = 1, dict(epoch=None, ibi_w1_ms=math.inf)
        if self.resume and state_path.is_file():
            # read on the CPU: RNG states must stay CPU ByteTensors; weights and optimizer state are
            # copied to the model's device by load_state_dict
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            if _without_epochs(state["config"]) != _without_epochs(cfg):
                raise ValueError("Resume configuration differs from the saved run (only training.epochs may change)")
            model.load_state_dict(state["model"])
            if ema.model is not None:
                ema.model.load_state_dict(state["ema"])
                ema.updates = state["ema_updates"]
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["torch_rng"].cpu())
            if self.device.type == "cuda" and state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(state["cuda_rng"].cpu())
            first_epoch, best = state["epoch"] + 1, state["best"]
            self.log(json.dumps(dict(event="resumed", from_epoch=state["epoch"])))
        else:
            _save_json(self.output / "config.json", cfg)
            _save_json(self.output / "run_summary.json", self.summary())
        loader = DataLoader(self._dataset(), batch_size=t["batch_size"], shuffle=True, drop_last=True,
                            num_workers=t["num_workers"], pin_memory=self.device.type == "cuda",
                            persistent_workers=t["num_workers"] > 0)
        validation_arrays, owners = self._validation_arrays()
        metrics_path = self.output / "metrics.csv"
        (self.output / "validation").mkdir(exist_ok=True)
        epoch_seconds = []
        for epoch in range(first_epoch, t["epochs"] + 1):
            epoch_start = time.time()
            model.train()
            total, count, steps_done = 0.0, 0, 0
            limit = t.get("max_steps_per_epoch")
            for step, batch in enumerate(loader, 1):
                if limit is not None and step > limit:              # quick tests only
                    break
                if max_hours is not None and (time.time() - started) / 3600.0 > max_hours:
                    # out of time inside an epoch: keep the last completed epoch's state.pt
                    status = dict(status="paused", epoch=epoch - 1, best=best,
                                  note="time ran out inside an epoch; that epoch restarts on resume")
                    if not epoch_seconds:
                        status["note"] = ("no epoch finished in this job: one epoch takes longer than --max-hours; "
                                          "give the job more time")
                    _save_json(self.output / "status.json", status)
                    self.log(json.dumps(dict(event="paused_for_time", **status, resume_with=str(self.output))))
                    return status
                batch = batch.to(self.device, non_blocking=True)
                loss = diffusion_loss(model, batch, schedule)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                ema.update(model)
                total += float(loss.detach()) * batch.shape[0]
                count += batch.shape[0]
                steps_done = step
                if step % t["log_every"] == 0:
                    self.log(json.dumps(dict(epoch=epoch, step=step, steps=len(loader), loss=total / count)))
            train_seconds = time.time() - epoch_start
            evaluation = ema.evaluation_model(model).eval()
            validation_start = time.time()
            generated = self.generate(evaluation, validation_arrays, schedule)
            summary, per_subject = validation_metrics(generated, validation_arrays[:, :, 0], owners, cfg)
            row = dict(epoch=epoch, train_loss=total / max(count, 1), **summary,
                       steps=steps_done, steps_per_full_epoch=len(loader),
                       seconds_per_step=round(train_seconds / max(steps_done, 1), 4),
                       validation_seconds=round(time.time() - validation_start, 1),
                       seconds=round(time.time() - epoch_start, 1))
            is_best = summary["ibi_w1_ms"] < best["ibi_w1_ms"]
            if is_best:
                best = dict(epoch=epoch, ibi_w1_ms=summary["ibi_w1_ms"])
            row["best_epoch"] = best["epoch"]
            new_file = not metrics_path.exists()
            with metrics_path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
            np.savez_compressed(self.output / "validation" / f"epoch_{epoch:03d}.npz", generated=generated,
                                real=validation_arrays[:, :, 0], owners=np.array(owners),
                                per_subject=np.array(json.dumps(per_subject)))
            weights = dict(model=model.state_dict(), ema=None if ema.model is None else ema.model.state_dict(),
                           config=cfg, channels=list(self.channels), scaler=self.cohort.scaler, epoch=epoch,
                           validation=summary, schema=SCHEMA)
            torch.save(weights, self.output / "last.pt")
            if is_best:
                torch.save(weights, self.output / "best.pt")
            if epoch % 10 == 0:                                   # kept, so epochs can be compared later
                (self.output / "checkpoints").mkdir(exist_ok=True)
                torch.save(weights, self.output / "checkpoints" / f"epoch_{epoch:03d}.pt")
            temporary = state_path.with_suffix(".pt.partial")              # never leave a half-written state
            torch.save(dict(model=model.state_dict(), ema=None if ema.model is None else ema.model.state_dict(),
                            ema_updates=ema.updates, optimizer=optimizer.state_dict(), epoch=epoch, best=best,
                            config=cfg, torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state() if self.device.type == "cuda" else None),
                       temporary)
            temporary.replace(state_path)
            self.log(json.dumps(dict(event="epoch", **row)))
            epoch_seconds.append(time.time() - epoch_start)
            if max_hours is not None and epoch < t["epochs"]:
                elapsed = (time.time() - started) / 3600.0
                if elapsed + 1.2 * max(epoch_seconds) / 3600.0 > max_hours:
                    _save_json(self.output / "status.json", dict(status="paused", epoch=epoch, best=best))
                    self.log(json.dumps(dict(event="paused_for_time", epoch=epoch, resume_with=str(self.output))))
                    return dict(status="paused", epoch=epoch, best=best)
        _save_json(self.output / "status.json", dict(status="completed", epoch=t["epochs"], best=best))
        return dict(status="completed", epoch=t["epochs"], best=best)

    def generate(self, model, arrays, schedule):
        import torch

        v = self.config["validation"]
        generator = torch.Generator().manual_seed(v["seed"])        # the same noise every epoch
        out = []
        for lo in range(0, len(arrays), v["batch_size"]):
            conditions = torch.from_numpy(arrays[lo:lo + v["batch_size"]]).to(self.device)
            out.append(sample(model, conditions, schedule, generator=generator).cpu().numpy())
        return np.concatenate(out).astype(np.float32)
