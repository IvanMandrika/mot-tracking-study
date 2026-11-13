import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mts import plotting
from mts.common import CACHE, RESULTS, SPLITS
from mts.data import dets_by_frame, dets_path, derive_subsplit_dets, group_by_frame, list_sequences, load_dets, \
    load_gt, pedestrian_mask, read_seqinfo, split_length
from mts.evaluate import COMBINED, metrics_long, run_trackeval, write_mot
from mts.tracker import cmc, reid
from mts.tracker.tracker import VARIANT_LABELS, VARIANTS, TrackerConfig, track_sequence

SOURCES = ["public_frcnn", "rtdetr_coco", "rtdetr_ft"]
SOURCE_LABELS = {"public_frcnn": "public FRCNN", "rtdetr_coco": "RT-DETR COCO", "rtdetr_ft": "RT-DETR fine-tuned"}
SELECT_SPLIT = "train_holdout"
RUN_SPLITS = ("train_half", SELECT_SPLIT, "val_half")
REID_VARIANT = "e_byte_cmc_gate_reid"
IOU_WEIGHTS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
TUNE_HIGH = [0.3, 0.4, 0.5, 0.6, 0.7]
DEFAULT_THRESH_VARIANT = "b_byte_default_thresh"
HIGH_THRESHES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MAX_AGES = [5, 10, 20, 30, 60, 90]
NOISE_SEEDS = 10
BOX_JITTER_PX = 0.5
SCORE_JITTER = 1e-3
NOISE_METRICS = ["HOTA", "IDF1"]
TRACKS = CACHE / "tracks"
HEADLINE = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"]
Job = tuple[str, TrackerConfig, str, str, int | None]


def jitter_dets(dets: np.ndarray, seed: int, seq: str) -> np.ndarray:
    rng = np.random.default_rng([seed, int(seq.split("-")[-1])])
    out = dets.copy()
    out[:, 1:5] += rng.uniform(-BOX_JITTER_PX, BOX_JITTER_PX, (len(out), 4)).astype(np.float32)
    out[:, 5] = np.clip(out[:, 5] + rng.uniform(-SCORE_JITTER, SCORE_JITTER, len(out)), 0.0, 1.0)
    return out


def load_inputs(source: str, split: str, seq: str, use_reid: bool, jitter: int | None = None) -> dict:
    info = read_seqinfo(seq)
    n = split_length(info.length, split)
    dets = load_dets(source, split, seq)
    if jitter is not None:
        dets = jitter_dets(dets, jitter, seq)
    feats = group_by_frame(reid.load_reid(source, split, seq), dets[:, 0].astype(int), n) if use_reid else None
    return {"frames": dets_by_frame(dets, n), "feats": feats, "affines": cmc.load_cmc(split, seq), "fps": info.fps}


def source_ready(source: str, split: str) -> bool:
    return all(dets_path(source, split, s).exists() for s in list_sequences())


def reid_ready(source: str, split: str) -> bool:
    return all(reid.is_cached(source, split, s) for s in list_sequences())


def prepare_selection_split(source: str) -> None:
    for seq in list_sequences():
        mask = derive_subsplit_dets(source, SELECT_SPLIT, seq)
        if reid.is_cached(source, "train_half", seq):
            reid.derive_subsplit(source, SELECT_SPLIT, seq, mask)
        cmc.derive_subsplit(SELECT_SPLIT, seq)


def run_config(cfg: TrackerConfig, source: str, split: str, out_dir: Path, parallel: bool = True,
               jitter: int | None = None) -> tuple[dict, float, int]:
    seconds, frames = 0.0, 0
    for seq in list_sequences():
        inp = load_inputs(source, split, seq, cfg.reid, jitter)
        tracks, s = track_sequence(cfg, inp["frames"], inp["fps"], inp["feats"], inp["affines"])
        write_mot(out_dir / f"{seq}.txt", tracks)
        seconds += s
        frames += len(inp["frames"])
    return run_trackeval(out_dir, split, parallel=parallel), seconds, frames


def _run_job(job: Job) -> tuple[str, dict, float, int]:
    name, cfg, source, split, jitter = job
    out_dir = TRACKS / source / split / (name if jitter is None else f"{name}_jitter{jitter}")
    res, seconds, frames = run_config(cfg, source, split, out_dir, parallel=False, jitter=jitter)
    return name, res, seconds, frames


def run_many(jobs: list[Job], workers: int) -> list[tuple[str, dict, float, int]]:
    if workers <= 1:
        return [_run_job(j) for j in jobs]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_run_job, jobs))


def tune(source: str, stage: str, grid: dict[str, TrackerConfig], workers: int) -> tuple[TrackerConfig, pd.DataFrame]:
    rows = []
    for name, res, _, _ in run_many([(n, c, source, SELECT_SPLIT, None) for n, c in grid.items()], workers):
        cfg = grid[name]
        rows.append({"source": source, "split": SELECT_SPLIT, "stage": stage, "config": name,
                     "high_thresh": cfg.high_thresh, "iou_weight": cfg.iou_weight, "motion_gate": cfg.motion_gate,
                     **{k: res[COMBINED][k] for k in HEADLINE}})
    df = pd.DataFrame(rows)
    best = df.loc[df["HOTA"].idxmax(), "config"]
    df["selected"] = df["config"] == best
    return grid[best], df


def with_thresh(cfg: TrackerConfig, high: float) -> TrackerConfig:
    return replace(cfg, high_thresh=high, new_track_thresh=round(high + 0.1, 2))


def tuned_configs(source: str, has_reid: bool, workers: int) -> tuple[dict[str, TrackerConfig], list[pd.DataFrame]]:
    grid = {f"tune_high{h}": with_thresh(VARIANTS["b_byte"], h) for h in TUNE_HIGH}
    best, thresh_df = tune(source, "score_threshold_on_b", grid, workers)
    configs = {k: with_thresh(v, best.high_thresh) for k, v in VARIANTS.items()}
    if not has_reid:
        del configs[REID_VARIANT]
        return configs, [thresh_df]
    grid = {f"tune_w{w}": replace(configs[REID_VARIANT], iou_weight=w) for w in IOU_WEIGHTS}
    configs[REID_VARIANT], reid_df = tune(source, "appearance_weight_on_e", grid, workers)
    return configs, [thresh_df, reid_df]


def ladder(source: str, split: str, configs: dict[str, TrackerConfig]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames, speed_rows = [], []
    for name, cfg in configs.items():
        res, seconds, frames = run_config(cfg, source, split, TRACKS / source / split / name)
        metric_frames.append(metrics_long(res, source=source, split=split, variant=name))
        speed_rows.append({"source": source, "split": split, "variant": name, "frames": frames,
                           "tracker_seconds": seconds, "tracker_fps": frames / seconds})
        c = res[COMBINED]
        print(f"{split} {source:13s} {name:22s} " + " ".join(f"{k} {c[k]:7.2f}" for k in HEADLINE)
              + f"  fps {frames / seconds:7.1f}", flush=True)
    return pd.concat(metric_frames, ignore_index=True), pd.DataFrame(speed_rows)


def noise_runs(source: str, configs: dict[str, TrackerConfig], seeds: int, workers: int) -> pd.DataFrame:
    jobs = [(name, cfg, source, "val_half", seed) for seed in range(seeds) for name, cfg in configs.items()]
    rows = []
    for (name, _, _, _, seed), (_, res, _, _) in zip(jobs, run_many(jobs, workers)):
        rows += [{"source": source, "variant": name, "seed": seed, "seq": seq, "metric": k, "value": r[k]}
                 for seq, r in res.items() for k in NOISE_METRICS]
    return pd.DataFrame(rows)


def step_table(val: pd.DataFrame, noise: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    rows = []
    for source in val.source.unique():
        present = [v for v in variants if ((val.source == source) & (val.variant == v)).any()]
        for prev, cur in zip(present, present[1:]):
            step = f"{cur[0]}-{prev[0]}"
            for metric in NOISE_METRICS:
                base = val[(val.source == source) & (val.metric == metric)].pivot(index="seq", columns="variant",
                                                                                 values="value")
                jit = noise[(noise.source == source) & (noise.metric == metric)].pivot_table(
                    index=["seq", "seed"], columns="variant", values="value")
                d_jit = (jit[cur] - jit[prev]).groupby(level="seq")
                stats = {"delta": base[cur] - base[prev], "delta_jitter_mean": d_jit.mean(),
                         "delta_jitter_std": d_jit.std(), "delta_jitter_min": d_jit.min(),
                         "delta_jitter_max": d_jit.max(), "prev_jitter_std": jit[prev].groupby(level="seq").std(),
                         "cur_jitter_std": jit[cur].groupby(level="seq").std()}
                for stat, series in stats.items():
                    rows += [{"source": source, "step": step, "from": prev, "to": cur, "seq": seq, "metric": metric,
                              "statistic": stat, "value": float(v)} for seq, v in series.items()]
    return pd.DataFrame(rows)


def sensitivity(source: str, cfg: TrackerConfig, workers: int) -> pd.DataFrame:
    params = {f"sens_high{h}": ("high_thresh", h, with_thresh(cfg, h)) for h in HIGH_THRESHES}
    params |= {f"sens_age{a}": ("max_age", a, replace(cfg, max_age=a)) for a in MAX_AGES}
    rows = []
    for name, res, _, _ in run_many([(n, c, source, "val_half", None) for n, (_, _, c) in params.items()], workers):
        param, value, _ = params[name]
        rows += [{"analysis": "sensitivity_val_half_not_used_for_selection", "source": source, "param": param,
                  "value": value, "metric": k, "score": res[COMBINED][k]} for k in HEADLINE]
    return pd.DataFrame(rows)


def oracle_sanity(split: str = "val_half") -> pd.DataFrame:
    frames = []
    for name in ("a_sort", "b_byte", "c_byte_cmc"):
        out_dir = TRACKS / "oracle_gt" / split / name
        for seq in list_sequences():
            info = read_seqinfo(seq)
            gt = load_gt(seq, split)
            gt = gt[pedestrian_mask(gt)]
            dets = np.column_stack([gt["frame"], gt["x1"], gt["y1"], gt["x2"], gt["y2"], np.ones(len(gt))])
            frames_in = dets_by_frame(dets, split_length(info.length, split))
            tracks, _ = track_sequence(VARIANTS[name], frames_in, info.fps, None, cmc.load_cmc(split, seq))
            write_mot(out_dir / f"{seq}.txt", tracks)
        frames.append(metrics_long(run_trackeval(out_dir, split), source="oracle_gt", split=split, variant=name))
    return pd.concat(frames, ignore_index=True)


def overhead_rows(source: str, split: str, has_reid: bool) -> list[dict]:
    seqs = list_sequences()
    rows = [{"source": "all", "variant": "c_byte_cmc", "seq": COMBINED, "metric": k, "value": v}
            for k, v in cmc.cmc_timing(split, seqs).items()]
    if has_reid:
        t = reid.reid_timing(source, split, seqs)
        rows += [{"source": source, "variant": REID_VARIANT, "seq": COMBINED, "metric": k, "value": t[k]}
                 for k in ("reid_ms_per_frame", "crops_per_frame")]
        rows.append({"source": source, "variant": REID_VARIANT, "seq": COMBINED,
                     "metric": f"reid_device_{t['device']}", "value": 1.0})
    return rows


def select_best(metrics: pd.DataFrame, chosen: dict[str, dict]) -> dict:
    sel = metrics[(metrics.split == SELECT_SPLIT) & (metrics.seq == COMBINED) & (metrics.metric == "HOTA")
                  & (metrics.variant != DEFAULT_THRESH_VARIANT)]
    per_source = {}
    for source, g in sel.groupby("source"):
        row = g.loc[g["value"].idxmax()]
        per_source[source] = {"variant": row["variant"], "selection_hota": float(row["value"]),
                              "config": chosen[source][row["variant"]]}
    best = max(per_source, key=lambda s: per_source[s]["selection_hota"])
    return {"source": best, **per_source[best], "selected_by": f"HOTA on {SELECT_SPLIT} (last 20% of each TRAIN-HALF, "
            "frames the detector was not trained on)", "per_source": per_source}


def plot_ablation(val: pd.DataFrame, noise: pd.DataFrame, sources: list[str]) -> None:
    plotting.setup()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    variants = list(VARIANTS)
    x = np.arange(len(variants))
    for ax, metric in zip(axes, NOISE_METRICS):
        for i, source in enumerate(SOURCES):
            if source not in sources:
                continue
            sub = val[(val.source == source) & (val.seq == COMBINED) & (val.metric == metric)].set_index("variant")
            jit = noise[(noise.source == source) & (noise.seq == COMBINED) & (noise.metric == metric)]
            std = jit.groupby("variant")["value"].std()
            ys = [sub["value"].get(v, np.nan) for v in variants]
            err = [std.get(v, np.nan) for v in variants]
            ax.errorbar(x + (i - 1) * 0.06, ys, yerr=err, marker="o", capsize=3, color=plotting.SERIES[i],
                        label=SOURCE_LABELS[source])
        ax.set_title(f"{metric} on MOT17 VAL-HALF")
        ax.set_ylabel(metric)
        ax.set_xticks(x, [VARIANT_LABELS[v] for v in variants], rotation=20, ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.06))
    fig.text(0.01, -0.02, f"error bars: std over {NOISE_SEEDS} runs with +/-{BOX_JITTER_PX} px box and "
             f"+/-{SCORE_JITTER:g} score jitter", fontsize=8.5, color=plotting.INK_SECONDARY)
    fig.tight_layout()
    plotting.save(fig, "tracking_ablation.png")


def plot_sensitivity(df: pd.DataFrame) -> None:
    plotting.setup()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, param, label in zip(axes, ["high_thresh", "max_age"], ["high-score threshold", "max age (frames at 30 fps)"]):
        for i, source in enumerate(SOURCES):
            sub = df[(df.source == source) & (df.param == param) & (df.metric == "HOTA")].sort_values("value")
            if len(sub):
                ax.plot(sub["value"], sub["score"], marker="o", color=plotting.SERIES[i], label=SOURCE_LABELS[source])
        ax.set_title(f"HOTA vs {label}")
        ax.set_xlabel(label)
        ax.set_ylabel("HOTA")
    axes[0].legend(loc="lower left")
    fig.suptitle("VAL-HALF sensitivity of the selected tracker (analysis only, not used for selection)", x=0.01,
                 ha="left", fontsize=10, color=plotting.INK_SECONDARY)
    fig.tight_layout()
    plotting.save(fig, "tracking_sensitivity.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tracker ablation ladder on cached detections, evaluated on VAL-HALF")
    parser.add_argument("--sources", nargs="+", default=SOURCES)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--noise-seeds", type=int, default=NOISE_SEEDS)
    parser.add_argument("--no-sensitivity", action="store_true")
    parser.add_argument("--oracle", action="store_true", help="only track GT boxes as detections (sanity check)")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    if args.oracle:
        df = oracle_sanity()
        df.to_csv(RESULTS / "tracking_oracle_gt_val_half.csv", index=False)
        print(df[(df.seq == COMBINED) & df.metric.isin(HEADLINE)].pivot(index="variant", columns="metric",
                                                                         values="value")[HEADLINE].round(2))
        return
    sources = [s for s in args.sources if all(source_ready(s, sp) for sp in SPLITS)]
    print("sources with detections:", sources, flush=True)
    metric_frames, speed_frames, tuning, overhead, noise_frames, chosen = [], [], [], [], [], {}
    for source in sources:
        prepare_selection_split(source)
        has_reid = all(reid_ready(source, sp) for sp in (SELECT_SPLIT, "val_half"))
        if not has_reid:
            print(f"{source}: ReID cache missing, variant e skipped", flush=True)
        configs, dfs = tuned_configs(source, has_reid, args.workers)
        tuning += dfs
        print(f"{source}: high_thresh tuned on {SELECT_SPLIT} = {configs['a_sort'].high_thresh}", flush=True)
        chosen[source] = {k: asdict(v) for k, v in configs.items()}
        for split in RUN_SPLITS:
            m, s = ladder(source, split, configs | {DEFAULT_THRESH_VARIANT: VARIANTS["b_byte"]})
            metric_frames.append(m)
            speed_frames.append(s)
        noise_frames.append(noise_runs(source, configs, args.noise_seeds, args.workers))
        overhead += overhead_rows(source, "val_half", has_reid)
    metrics = pd.concat(metric_frames, ignore_index=True)
    speed = pd.concat(speed_frames, ignore_index=True)
    noise = pd.concat(noise_frames, ignore_index=True)
    val = metrics[metrics.split == "val_half"].drop(columns="split")
    speed_long = speed[speed.split == "val_half"].melt(id_vars=["source", "variant"], value_vars=["tracker_fps"],
                                                       var_name="metric", value_name="value")
    speed_long.insert(2, "seq", COMBINED)
    table = pd.concat([val[["source", "variant", "seq", "metric", "value"]], speed_long,
                       pd.DataFrame(overhead).drop_duplicates(["source", "variant", "metric"])], ignore_index=True)
    table.to_csv(RESULTS / "tracking.csv", index=False)
    metrics[metrics.split != "val_half"].to_csv(RESULTS / "tracking_train_splits.csv", index=False)
    pd.concat(tuning, ignore_index=True).to_csv(RESULTS / f"tracking_tuning_{SELECT_SPLIT}.csv", index=False)
    noise.to_csv(RESULTS / "tracking_noise_val_half.csv", index=False)
    steps = step_table(val, noise, list(VARIANTS))
    steps.to_csv(RESULTS / "tracking_steps_val_half.csv", index=False)
    (RESULTS / "tracking_configs.json").write_text(json.dumps(chosen, indent=1))
    best = select_best(metrics, chosen)
    (RESULTS / "tracking_best.json").write_text(json.dumps(best, indent=1))
    plot_ablation(val, noise, sources)
    summary = val[(val.seq == COMBINED) & val.metric.isin(HEADLINE)].pivot_table(
        index=["source", "variant"], columns="metric", values="value")[HEADLINE]
    print(summary.round(2).to_string())
    print(steps[steps.seq == COMBINED].pivot_table(index=["source", "step", "metric"], columns="statistic",
                                                   values="value").round(2).to_string())
    print(speed.round(1).to_string(index=False))
    print(pd.DataFrame(overhead).to_string(index=False))
    print(f"best by {SELECT_SPLIT}:", best["source"], best["variant"],
          {s: v["variant"] for s, v in best["per_source"].items()})
    if not args.no_sensitivity:
        sens = pd.concat([sensitivity(s, TrackerConfig(**best["per_source"][s]["config"]), args.workers)
                          for s in sources], ignore_index=True)
        sens.to_csv(RESULTS / "tracking_sensitivity_val_half.csv", index=False)
        plot_sensitivity(sens)
        print(sens[sens.metric == "HOTA"].pivot_table(index=["source", "param", "value"], values="score").round(2)
              .to_string())


if __name__ == "__main__":
    main()
