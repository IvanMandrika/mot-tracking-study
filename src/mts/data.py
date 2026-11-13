import argparse
import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mts.common import CACHE, MOT17, SPLITS, SUBSPLITS

DETECTOR_COPY = "FRCNN"
PEDESTRIAN = 1
MIN_VISIBILITY = 0.25
MIN_DET_SCORE = 0.05
HOLDOUT_FRACTION = 0.2
MOT_OFFSET = 1.0  # MOT files use 1-based pixel coordinates, arrays in this package are 0-based
GT_COLUMNS = ["frame", "id", "x1", "y1", "x2", "y2", "consider", "cls", "vis"]


@dataclass(frozen=True)
class SeqInfo:
    name: str
    length: int
    width: int
    height: int
    fps: float
    img_dir: Path
    img_ext: str


def seq_dir(seq: str, copy: str = DETECTOR_COPY, root: Path = MOT17) -> Path:
    return root / "train" / f"{seq}-{copy}"


def list_sequences(root: Path = MOT17) -> list[str]:
    suffix = f"-{DETECTOR_COPY}"
    return sorted(p.name[: -len(suffix)] for p in (root / "train").iterdir() if p.name.endswith(suffix))


def read_seqinfo(seq: str, root: Path = MOT17) -> SeqInfo:
    d = seq_dir(seq, root=root)
    ini = configparser.ConfigParser()
    ini.read(d / "seqinfo.ini")
    s = ini["Sequence"]
    return SeqInfo(
        name=seq, length=int(s["seqLength"]), width=int(s["imWidth"]), height=int(s["imHeight"]),
        fps=float(s["frameRate"]), img_dir=d / s["imDir"], img_ext=s["imExt"],
    )


def fit_length(n: int, holdout_fraction: float = HOLDOUT_FRACTION) -> int:
    return n - int(round(n * holdout_fraction))


def half_range(length: int, split: str) -> tuple[int, int]:
    mid = length // 2
    if split == "train_half":
        return 1, mid
    if split == "val_half":
        return mid + 1, length
    if split == "train_fit":
        return 1, fit_length(mid)
    if split == "train_holdout":
        return fit_length(mid) + 1, mid
    raise ValueError(f"unknown split {split!r}, expected one of {SPLITS + SUBSPLITS}")


def split_length(length: int, split: str) -> int:
    first, last = half_range(length, split)
    return last - first + 1


def frame_paths(seq: str, split: str | None = None, root: Path = MOT17) -> list[Path]:
    info = read_seqinfo(seq, root)
    first, last = (1, info.length) if split is None else half_range(info.length, split)
    return [info.img_dir / f"{f:06d}{info.img_ext}" for f in range(first, last + 1)]


def select_split_frames(frames: np.ndarray, length: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    first, last = half_range(length, split)
    mask = (frames >= first) & (frames <= last)
    return mask, frames[mask] - (first - 1)


def mot_xywh_to_xyxy(raw: np.ndarray) -> np.ndarray:
    x1 = raw[:, 2] - MOT_OFFSET
    y1 = raw[:, 3] - MOT_OFFSET
    return np.stack([x1, y1, x1 + raw[:, 4], y1 + raw[:, 5]], axis=1)


def parse_mot_gt(path: Path) -> pd.DataFrame:
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    b = mot_xywh_to_xyxy(raw)
    return pd.DataFrame({
        "frame": raw[:, 0].astype(int), "id": raw[:, 1].astype(int),
        "x1": b[:, 0], "y1": b[:, 1], "x2": b[:, 2], "y2": b[:, 3],
        "consider": raw[:, 6].astype(int), "cls": raw[:, 7].astype(int), "vis": raw[:, 8],
    })


def crop_gt_to_split(gt: pd.DataFrame, length: int, split: str) -> pd.DataFrame:
    mask, frames = select_split_frames(gt["frame"].to_numpy(), length, split)
    out = gt[mask].copy()
    out["frame"] = frames
    return out.reset_index(drop=True)


def pedestrian_mask(gt: pd.DataFrame, min_vis: float = 0.0) -> pd.Series:
    return (gt["cls"] == PEDESTRIAN) & (gt["consider"] == 1) & (gt["vis"] >= min_vis)


def filter_gt(gt: pd.DataFrame, min_vis: float = MIN_VISIBILITY) -> pd.DataFrame:
    return gt[pedestrian_mask(gt, min_vis)].reset_index(drop=True)


def load_gt(seq: str, split: str | None = None, root: Path = MOT17) -> pd.DataFrame:
    gt = parse_mot_gt(seq_dir(seq, root=root) / "gt" / "gt.txt")
    if split is None:
        return gt
    return crop_gt_to_split(gt, read_seqinfo(seq, root).length, split)


def load_training_gt(seq: str, split: str, min_vis: float = MIN_VISIBILITY, root: Path = MOT17) -> pd.DataFrame:
    return filter_gt(load_gt(seq, split, root), min_vis)


def parse_mot_dets(path: Path, min_score: float = MIN_DET_SCORE) -> np.ndarray:
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    raw = raw[raw[:, 6] >= min_score]
    return np.column_stack([raw[:, 0], mot_xywh_to_xyxy(raw), raw[:, 6]]).astype(np.float32)


def load_public_dets(seq: str, split: str, copy: str = DETECTOR_COPY, root: Path = MOT17) -> np.ndarray:
    dets = parse_mot_dets(seq_dir(seq, copy, root) / "det" / "det.txt")
    mask, frames = select_split_frames(dets[:, 0].astype(int), read_seqinfo(seq, root).length, split)
    dets = dets[mask]
    dets[:, 0] = frames
    return dets


def dets_path(source: str, split: str, seq: str, cache: Path = CACHE) -> Path:
    return cache / "dets" / source / split / f"{seq}.npy"


def save_dets(dets: np.ndarray, source: str, split: str, seq: str, cache: Path = CACHE) -> Path:
    if dets.ndim != 2 or dets.shape[1] != 6:
        raise ValueError(f"detections must have shape (M, 6), got {dets.shape}")
    path = dets_path(source, split, seq, cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, dets.astype(np.float32))
    return path


def load_dets(source: str, split: str, seq: str, cache: Path = CACHE) -> np.ndarray:
    return np.load(dets_path(source, split, seq, cache))


def group_by_frame(values: np.ndarray, frames: np.ndarray, n_frames: int) -> list[np.ndarray]:
    order = np.argsort(frames, kind="stable")
    bounds = np.searchsorted(frames[order], np.arange(1, n_frames + 2))
    return [values[order[bounds[t]: bounds[t + 1]]] for t in range(n_frames)]


def dets_by_frame(dets: np.ndarray, n_frames: int) -> list[np.ndarray]:
    return group_by_frame(dets[:, 1:], dets[:, 0].astype(int), n_frames)


def derive_subsplit_dets(source: str, split: str, seq: str, cache: Path = CACHE) -> np.ndarray:
    parent = load_dets(source, "train_half", seq, cache)
    mask, frames = select_split_frames(parent[:, 0].astype(int), read_seqinfo(seq).length, split)
    dets = parent[mask]
    dets[:, 0] = frames
    save_dets(dets, source, split, seq, cache)
    return mask


def split_summary(split: str, root: Path = MOT17) -> pd.DataFrame:
    rows = []
    for seq in list_sequences(root):
        info = read_seqinfo(seq, root)
        gt = load_gt(seq, split, root)
        ped = gt[pedestrian_mask(gt)]
        train = filter_gt(gt)
        rows.append({
            "split": split, "seq": seq, "frames": split_length(info.length, split),
            "gt_boxes_eval": len(ped), "ids_eval": ped["id"].nunique(),
            "gt_boxes_train_vis25": len(train), "ids_train_vis25": train["id"].nunique(),
        })
    return pd.DataFrame(rows)


def cache_public_dets(root: Path = MOT17, cache: Path = CACHE) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for seq in list_sequences(root):
            dets = load_public_dets(seq, split, root=root)
            save_dets(dets, "public_frcnn", split, seq, cache)
            rows.append({"split": split, "seq": seq, "dets": len(dets)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache public FRCNN detections and print MOT17 half-split statistics")
    parser.add_argument("--root", type=Path, default=MOT17)
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()
    dets = cache_public_dets(args.root, args.cache)
    stats = pd.concat([split_summary(s, args.root) for s in SPLITS], ignore_index=True)
    stats = stats.merge(dets, on=["split", "seq"])
    pd.set_option("display.width", 200)
    print(stats.to_string(index=False))
    totals = stats.groupby("split", sort=False).sum(numeric_only=True)
    print(totals.to_string())


if __name__ == "__main__":
    main()
