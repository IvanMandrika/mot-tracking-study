import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from mts.common import CACHE, SPLITS
from mts.data import dets_path, frame_paths, list_sequences, load_dets

MODEL = "facebook/dinov2-small"
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
CROP_HW = (224, 112)
MIN_SCORE = 0.3
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def reid_path(source: str, split: str, seq: str, cache: Path = CACHE) -> Path:
    return cache / "reid" / source / split / f"{seq}.npy"


def dets_digest(dets: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(dets).tobytes()).hexdigest()


def crop_batch(img: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    out = np.empty((len(boxes), CROP_HW[0], CROP_HW[1], 3), dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        xa, ya = int(np.clip(np.floor(x1), 0, w - 1)), int(np.clip(np.floor(y1), 0, h - 1))
        xb, yb = int(np.clip(np.ceil(x2), xa + 1, w)), int(np.clip(np.ceil(y2), ya + 1, h))
        crop = cv2.resize(img[ya:yb, xa:xb], (CROP_HW[1], CROP_HW[0]), interpolation=cv2.INTER_LINEAR)
        out[i] = (crop.astype(np.float32) / 255.0 - MEAN) / STD
    return out.transpose(0, 3, 1, 2)


class Embedder:
    def __init__(self, device: str = "cuda", batch: int = 64) -> None:
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = device
        self.batch = batch
        self.model = AutoModel.from_pretrained(MODEL, revision=MODEL_REVISION).to(device).eval()

    def _forward(self, x: np.ndarray) -> np.ndarray:
        with self.torch.inference_mode():
            feats = self.model(pixel_values=self.torch.from_numpy(x).to(self.device)).pooler_output
            feats = self.torch.nn.functional.normalize(feats, dim=1)
        return feats.float().cpu().numpy()

    def embed(self, x: np.ndarray) -> np.ndarray:
        out, i = [], 0
        while i < len(x):
            try:
                out.append(self._forward(x[i:i + self.batch]))
                i += self.batch
            except self.torch.cuda.OutOfMemoryError:
                self.torch.cuda.empty_cache()
                if self.batch == 1:
                    raise
                time.sleep(30)
                self.batch = max(1, self.batch // 2)
        return np.concatenate(out) if out else np.zeros((0, 384), dtype=np.float32)

    def sync(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()


def embed_sequence(emb: Embedder, dets: np.ndarray, paths: list[Path]) -> tuple[np.ndarray, float]:
    feats = np.zeros((len(dets), 384), dtype=np.float16)
    frames = dets[:, 0].astype(int)
    keep = dets[:, 5] >= MIN_SCORE
    seconds = 0.0
    for t, p in enumerate(paths, start=1):
        idx = np.flatnonzero((frames == t) & keep)
        if len(idx) == 0:
            continue
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        feats[idx] = emb.embed(crop_batch(img, dets[idx, 1:5]))
        emb.sync()
        seconds += time.perf_counter() - t0
    return feats, seconds


def is_cached(source: str, split: str, seq: str, cache: Path = CACHE) -> bool:
    meta_path = reid_path(source, split, seq, cache).with_suffix(".json")
    if not meta_path.exists():
        return False
    return json.loads(meta_path.read_text())["dets_md5"] == dets_digest(load_dets(source, split, seq, cache))


def cache_source(emb: Embedder, source: str, splits: tuple[str, ...] = SPLITS, cache: Path = CACHE) -> list[dict]:
    metas = []
    for split in splits:
        for seq in list_sequences():
            if not dets_path(source, split, seq, cache).exists() or is_cached(source, split, seq, cache):
                continue
            dets = load_dets(source, split, seq, cache)
            paths = frame_paths(seq, split)
            feats, seconds = embed_sequence(emb, dets, paths)
            path = reid_path(source, split, seq, cache)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, feats)
            meta = {"source": source, "split": split, "seq": seq, "frames": len(paths),
                    "crops": int((dets[:, 5] >= MIN_SCORE).sum()), "device": emb.device,
                    "reid_ms_per_frame": 1000 * seconds / len(paths), "dets_md5": dets_digest(dets)}
            path.with_suffix(".json").write_text(json.dumps(meta))
            metas.append(meta)
            print(meta, flush=True)
    return metas


def derive_subsplit(source: str, split: str, seq: str, mask: np.ndarray, cache: Path = CACHE) -> None:
    parent = reid_path(source, "train_half", seq, cache)
    feats = np.load(parent)[mask]
    path = reid_path(source, split, seq, cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, feats)
    dets = load_dets(source, split, seq, cache)
    meta = json.loads(parent.with_suffix(".json").read_text())
    meta |= {"split": split, "frames": len(frame_paths(seq, split)), "crops": int((dets[:, 5] >= MIN_SCORE).sum()),
             "dets_md5": dets_digest(dets)}
    path.with_suffix(".json").write_text(json.dumps(meta))


def load_reid(source: str, split: str, seq: str, cache: Path = CACHE) -> np.ndarray:
    return np.load(reid_path(source, split, seq, cache)).astype(np.float32)


def reid_timing(source: str, split: str, seqs: list[str], cache: Path = CACHE) -> dict[str, float | str]:
    metas = [json.loads(reid_path(source, split, s, cache).with_suffix(".json").read_text()) for s in seqs]
    frames = sum(m["frames"] for m in metas)
    return {"reid_ms_per_frame": sum(m["reid_ms_per_frame"] * m["frames"] for m in metas) / frames,
            "crops_per_frame": sum(m["crops"] for m in metas) / frames, "device": metas[0]["device"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache DINOv2-small embeddings of detection crops")
    parser.add_argument("--sources", nargs="+", default=["public_frcnn"])
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    emb = Embedder(args.device, args.batch)
    for source in args.sources:
        cache_source(emb, source, tuple(args.splits))


if __name__ == "__main__":
    main()
