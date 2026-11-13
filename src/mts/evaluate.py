import argparse
import configparser
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mts.common import CACHE, EVAL_GT, MOT17, RESULTS, SPLITS, TRACKEVAL
from mts.data import MOT_OFFSET, PEDESTRIAN, half_range, list_sequences, mot_xywh_to_xyxy, read_seqinfo, \
    select_split_frames, seq_dir

HOTA_FIELDS = ["HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA"]
CLEAR_FIELDS = ["MOTA", "MOTP", "IDSW", "CLR_TP", "CLR_FP", "CLR_FN", "MT", "PT", "ML", "Frag"]
IDENTITY_FIELDS = ["IDF1", "IDR", "IDP", "IDTP", "IDFP", "IDFN"]
COUNT_FIELDS = ["Dets", "GT_Dets", "IDs", "GT_IDs"]
PERCENT_FIELDS = set(HOTA_FIELDS) | {"MOTA", "MOTP", "IDF1", "IDR", "IDP"}
COMBINED = "COMBINED"
EVAL_SPLITS = SPLITS + ("train_holdout",)


def write_mot(path: Path, tracks: np.ndarray) -> None:
    tracks = np.asarray(tracks, dtype=np.float64)
    if tracks.shape[1] == 6:
        tracks = np.concatenate([tracks, np.ones((len(tracks), 1))], axis=1)
    tracks = tracks[np.lexsort((tracks[:, 1], tracks[:, 0]))]
    w = tracks[:, 4] - tracks[:, 2]
    h = tracks[:, 5] - tracks[:, 3]
    rows = np.stack([tracks[:, 0], tracks[:, 1], tracks[:, 2] + MOT_OFFSET, tracks[:, 3] + MOT_OFFSET, w, h,
                     tracks[:, 6]], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(f"{int(r[0])},{int(r[1])},{r[2]:.3f},{r[3]:.3f},{r[4]:.3f},{r[5]:.3f},{r[6]:.4f},-1,-1,-1\n")


def read_mot(path: Path) -> np.ndarray:
    if path.stat().st_size == 0:
        return np.zeros((0, 7))
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    return np.column_stack([raw[:, :2], mot_xywh_to_xyxy(raw), raw[:, 6]])


def crop_raw_gt(raw: np.ndarray, length: int, split: str) -> np.ndarray:
    mask, frames = select_split_frames(raw[:, 0].astype(int), length, split)
    out = raw[mask].copy()
    out[:, 0] = frames
    return out


def write_raw_gt(path: Path, raw: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, raw, delimiter=",", fmt=["%d", "%d", "%.6g", "%.6g", "%.6g", "%.6g", "%d", "%d", "%.6g"])


def write_split_seqinfo(src: Path, dst: Path, seq: str, length: int) -> None:
    ini = configparser.ConfigParser()
    ini.optionxform = str
    ini.read(src)
    ini["Sequence"]["name"] = seq
    ini["Sequence"]["seqLength"] = str(length)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        ini.write(f, space_around_delimiters=False)


def build_eval_gt(split: str, root: Path = MOT17, out: Path = EVAL_GT) -> Path:
    split_dir = out / split
    seqs = list_sequences(root)
    for seq in seqs:
        info = read_seqinfo(seq, root)
        first, last = half_range(info.length, split)
        raw = np.loadtxt(seq_dir(seq, root=root) / "gt" / "gt.txt", delimiter=",", ndmin=2)
        write_raw_gt(split_dir / seq / "gt" / "gt.txt", crop_raw_gt(raw, info.length, split))
        write_split_seqinfo(seq_dir(seq, root=root) / "seqinfo.ini", split_dir / seq / "seqinfo.ini", seq,
                            last - first + 1)
    (split_dir / "seqmap.txt").write_text("name\n" + "".join(f"{s}\n" for s in seqs))
    return split_dir


def import_trackeval():
    for alias, target in (("float", float), ("int", int), ("bool", bool)):
        if alias not in np.__dict__:
            setattr(np, alias, target)
    if str(TRACKEVAL) not in sys.path:
        sys.path.insert(0, str(TRACKEVAL))
    with contextlib.redirect_stdout(io.StringIO()):
        import trackeval
    return trackeval


def run_trackeval(tracker_dir: Path, split: str = "val_half", seqs: list[str] | None = None,
                  gt_root: Path = EVAL_GT, parallel: bool = False) -> dict[str, dict[str, float]]:
    trackeval = import_trackeval()
    tracker_dir = Path(tracker_dir).resolve()
    gt_dir = gt_root / split
    dataset_config = {
        "GT_FOLDER": str(gt_dir), "TRACKERS_FOLDER": str(tracker_dir.parent), "TRACKERS_TO_EVAL": [tracker_dir.name],
        "TRACKER_SUB_FOLDER": "", "OUTPUT_FOLDER": str(tracker_dir.parent), "BENCHMARK": "MOT17",
        "SKIP_SPLIT_FOL": True, "SEQMAP_FILE": str(gt_dir / "seqmap.txt"), "DO_PREPROC": True, "PRINT_CONFIG": False,
        "SEQ_INFO": {s: None for s in seqs} if seqs else None,
    }
    eval_config = {
        "USE_PARALLEL": parallel, "NUM_PARALLEL_CORES": 7, "PRINT_RESULTS": False, "PRINT_CONFIG": False,
        "TIME_PROGRESS": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False,
        "LOG_ON_ERROR": None,
    }
    metric_config = {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(metric_config),
               trackeval.metrics.Identity(metric_config)]
    with contextlib.redirect_stdout(io.StringIO()):
        dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
        res, msg = trackeval.Evaluator(eval_config).evaluate([dataset], metrics)
    status = msg["MotChallenge2DBox"][tracker_dir.name]
    if status != "Success":
        raise RuntimeError(f"TrackEval failed: {status}")
    per_seq = res["MotChallenge2DBox"][tracker_dir.name]
    return {(COMBINED if seq == "COMBINED_SEQ" else seq): flatten_metrics(r["pedestrian"]) for seq, r in per_seq.items()}


def flatten_metrics(r: dict) -> dict[str, float]:
    out = {f: float(np.mean(r["HOTA"][f])) for f in HOTA_FIELDS}
    out.update({f: float(r["CLEAR"][f]) for f in CLEAR_FIELDS})
    out.update({f: float(r["Identity"][f]) for f in IDENTITY_FIELDS})
    out.update({f: float(r["Count"][f]) for f in COUNT_FIELDS})
    return {k: v * 100 if k in PERCENT_FIELDS else v for k, v in out.items()}


def metrics_frame(res: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(res, orient="index").rename_axis("seq").reset_index()


def metrics_long(res: dict[str, dict[str, float]], **tags: object) -> pd.DataFrame:
    df = metrics_frame(res).melt(id_vars="seq", var_name="metric", value_name="value")
    for k, v in tags.items():
        df.insert(0, k, v)
    return df


def gt_as_tracks(split: str, root: Path = EVAL_GT) -> dict[str, np.ndarray]:
    out = {}
    for seq_path in sorted(p for p in (root / split).iterdir() if p.is_dir()):
        raw = np.loadtxt(seq_path / "gt" / "gt.txt", delimiter=",", ndmin=2)
        raw = raw[(raw[:, 7] == PEDESTRIAN) & (raw[:, 6] == 1)]
        out[seq_path.name] = np.column_stack([raw[:, :2], mot_xywh_to_xyxy(raw), np.ones(len(raw))])
    return out


def shuffle_ids_per_frame(tracks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = tracks.copy()
    for f in np.unique(out[:, 0]):
        idx = np.flatnonzero(out[:, 0] == f)
        out[idx, 1] = rng.permutation(out[idx, 1])
    return out


def sanity(split: str, seed: int, workdir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tracks = gt_as_tracks(split)
    variants = {"gt_as_tracker": tracks, "gt_shuffled_ids": {s: shuffle_ids_per_frame(t, rng) for s, t in tracks.items()}}
    frames = []
    for name, per_seq in variants.items():
        for seq, t in per_seq.items():
            write_mot(workdir / name / f"{seq}.txt", t)
        frames.append(metrics_long(run_trackeval(workdir / name, split), variant=name, split=split))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build half-split GT for TrackEval and evaluate MOT-format results")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-gt")
    ev = sub.add_parser("eval")
    ev.add_argument("tracker_dir", type=Path)
    ev.add_argument("--split", choices=EVAL_SPLITS, default="val_half")
    san = sub.add_parser("sanity")
    san.add_argument("--split", choices=EVAL_SPLITS, default="val_half")
    san.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    pd.set_option("display.width", 200)
    if args.cmd == "build-gt":
        for split in EVAL_SPLITS:
            print(build_eval_gt(split))
    elif args.cmd == "eval":
        print(metrics_frame(run_trackeval(args.tracker_dir, args.split)).round(2).to_string(index=False))
    else:
        df = sanity(args.split, args.seed, CACHE / "sanity")
        RESULTS.mkdir(parents=True, exist_ok=True)
        df.to_csv(RESULTS / f"sanity_eval_{args.split}.csv", index=False)
        keys = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"]
        table = df[(df.seq == COMBINED) & df.metric.isin(keys)].pivot(index="variant", columns="metric", values="value")
        print(table[keys].round(2).to_string())


if __name__ == "__main__":
    main()
