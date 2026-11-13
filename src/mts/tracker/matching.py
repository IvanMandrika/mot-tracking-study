import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def iou_cost(tracks: np.ndarray, dets: np.ndarray) -> np.ndarray:
    return 1.0 - iou_matrix(tracks, dets)


def fuse_score(cost: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - cost) * scores[None, :]


def cosine_cost(track_feats: np.ndarray, det_feats: np.ndarray) -> np.ndarray:
    if len(track_feats) == 0 or len(det_feats) == 0:
        return np.zeros((len(track_feats), len(det_feats)))
    return np.clip(1.0 - track_feats @ det_feats.T, 0.0, 2.0)


def fuse_appearance(iou_c: np.ndarray, emb_c: np.ndarray, iou_weight: float) -> np.ndarray:
    return iou_weight * iou_c + (1.0 - iou_weight) * emb_c


def linear_assignment(cost: np.ndarray, thresh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.zeros((0, 2), dtype=int), np.arange(n), np.arange(m)
    padded = np.where(np.isfinite(cost) & (cost <= thresh), cost, thresh + 1.0)
    rows, cols = linear_sum_assignment(padded)
    keep = padded[rows, cols] <= thresh
    matches = np.stack([rows[keep], cols[keep]], axis=1)
    unmatched_rows = np.setdiff1d(np.arange(n), matches[:, 0])
    unmatched_cols = np.setdiff1d(np.arange(m), matches[:, 1])
    return matches, unmatched_rows, unmatched_cols
