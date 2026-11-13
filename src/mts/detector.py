from pathlib import Path

import cv2
import numpy as np
import torch

from mts.common import DATA

MODEL_ID = "PekingU/rtdetr_r18vd"
MODEL_REVISION = "ac77a11ff0170a41b771c03264987f8ce2b0d753"
COCO_WEIGHTS = DATA / "weights" / "rtdetr_r18vd"
CHECKPOINTS = DATA / "checkpoints"
FT_CHECKPOINT = CHECKPOINTS / "rtdetr_ft"
STRIDE = 32
LONG_SIDE = 960
MIN_SCORE = 0.05
TOP_K = 300


def person_label(id2label: dict[int, str]) -> int:
    matches = [int(k) for k, v in id2label.items() if v == "person"]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one 'person' label, found {matches}")
    return matches[0]


def input_size(width: int, height: int, long_side: int = LONG_SIDE) -> tuple[int, int]:
    scale = long_side / max(width, height)
    return (max(STRIDE, int(round(width * scale / STRIDE)) * STRIDE),
            max(STRIDE, int(round(height * scale / STRIDE)) * STRIDE))


def cxcywh_norm_to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    out = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return out * np.array([width, height, width, height], dtype=out.dtype)


def xyxy_to_cxcywh_norm(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    b = boxes / np.array([width, height, width, height], dtype=np.float64)
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)


def person_detections(logits: np.ndarray, boxes: np.ndarray, person: int, width: int, height: int,
                      min_score: float = MIN_SCORE, top_k: int = TOP_K) -> np.ndarray:
    scores = 1.0 / (1.0 + np.exp(-logits[:, person].astype(np.float64)))
    keep = np.flatnonzero(scores >= min_score)
    keep = keep[np.argsort(-scores[keep], kind="stable")][:top_k]
    xyxy = cxcywh_norm_to_xyxy(boxes[keep].astype(np.float64), width, height)
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)
    return np.concatenate([xyxy, scores[keep, None]], axis=1).astype(np.float32)


def worker_init(_: int) -> None:
    cv2.setNumThreads(1)


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_tensor(rgb: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    resized = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(resized).permute(2, 0, 1).float().div_(255.0)


def load_model(weights: Path | str, device: str = "cuda"):
    from transformers import RTDetrForObjectDetection

    revision = MODEL_REVISION if str(weights) == MODEL_ID else None
    return RTDetrForObjectDetection.from_pretrained(str(weights), revision=revision).to(device).eval()


def coco_weights() -> Path | str:
    return COCO_WEIGHTS if (COCO_WEIGHTS / "model.safetensors").exists() else MODEL_ID


class Detector:
    def __init__(self, model, device: str = "cuda", long_side: int = LONG_SIDE, min_score: float = MIN_SCORE,
                 top_k: int = TOP_K) -> None:
        self.model = model
        self.person = person_label(self.model.config.id2label)
        self.device = device
        self.long_side = long_side
        self.min_score = min_score
        self.top_k = top_k

    @classmethod
    def load(cls, weights: Path | str, device: str = "cuda", **kwargs) -> "Detector":
        return cls(load_model(weights, device), device, **kwargs)

    @torch.inference_mode()
    def forward(self, batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        out = self.model(pixel_values=batch.to(self.device, non_blocking=True))
        return out.logits.float().cpu().numpy(), out.pred_boxes.float().cpu().numpy()

    def detect_batch(self, batch: torch.Tensor, width: int, height: int) -> list[np.ndarray]:
        logits, boxes = self.forward(batch)
        return [person_detections(lg, bx, self.person, width, height, self.min_score, self.top_k)
                for lg, bx in zip(logits, boxes)]

    def __call__(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        height, width = frames[0].shape[:2]
        if any(f.shape[:2] != (height, width) for f in frames):
            raise ValueError("all frames in one call must share a size")
        size = input_size(width, height, self.long_side)
        return self.detect_batch(torch.stack([to_tensor(f, size) for f in frames]), width, height)

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()
