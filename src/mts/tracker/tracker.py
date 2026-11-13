import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from mts.tracker.cmc import warp_state
from mts.tracker.kalman import CHI2_95_4DOF, KalmanFilter, xyah_to_xyxy, xyxy_to_xyah
from mts.tracker.matching import cosine_cost, fuse_appearance, fuse_score, iou_cost, linear_assignment


@dataclass(frozen=True)
class TrackerConfig:
    # ByteTrack MOT17 defaults, except cmc, motion_gate, reid, iou_weight and ema_alpha, which are additions here
    high_thresh: float = 0.6
    low_thresh: float = 0.1
    new_track_thresh: float = 0.7
    match_thresh: float = 0.8
    second_match_thresh: float = 0.5
    unconfirmed_match_thresh: float = 0.7
    max_age: int = 30
    fps_scaled_max_age: bool = True
    fuse_score: bool = True
    min_box_area: float = 10.0
    max_aspect: float = 1.6
    second_stage: bool = False
    cmc: bool = False
    reid: bool = False
    iou_weight: float = 0.7
    ema_alpha: float = 0.9
    motion_gate: bool = False

    def max_age_frames(self, fps: float) -> int:
        return int(fps / 30.0 * self.max_age) if self.fps_scaled_max_age else self.max_age


VARIANTS = {
    "a_sort": TrackerConfig(),
    "b_byte": TrackerConfig(second_stage=True),
    "c_byte_cmc": TrackerConfig(second_stage=True, cmc=True),
    "d_byte_cmc_gate": TrackerConfig(second_stage=True, cmc=True, motion_gate=True),
    "e_byte_cmc_gate_reid": TrackerConfig(second_stage=True, cmc=True, motion_gate=True, reid=True),
}
VARIANT_LABELS = {"a_sort": "(a) SORT-like", "b_byte": "(b) + low-score stage", "c_byte_cmc": "(c) + CMC",
                  "d_byte_cmc_gate": "(d) + motion gate", "e_byte_cmc_gate_reid": "(e) + appearance"}


class State(Enum):
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2


class Track:
    def __init__(self, kf: KalmanFilter, box: np.ndarray, score: float, feat: np.ndarray | None, frame: int) -> None:
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(box))
        self.score = score
        self.feat = feat
        self.state = State.TENTATIVE
        self.track_id = -1
        self.last_frame = frame

    @property
    def box(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean[:4])

    def update(self, kf: KalmanFilter, box: np.ndarray, score: float, feat: np.ndarray | None, frame: int,
               alpha: float) -> None:
        self.mean, self.cov = kf.update(self.mean, self.cov, xyxy_to_xyah(box))
        self.score = score
        self.last_frame = frame
        if feat is not None:
            self.feat = feat if self.feat is None else alpha * self.feat + (1 - alpha) * feat
            self.feat = self.feat / max(np.linalg.norm(self.feat), 1e-12)


class Tracker:
    def __init__(self, cfg: TrackerConfig, fps: float = 30.0) -> None:
        self.cfg = cfg
        self.kf = KalmanFilter()
        self.max_age = cfg.max_age_frames(fps)
        self.tracks: list[Track] = []
        self.next_id = 1
        self.frame = 0

    def _confirm(self, t: Track) -> None:
        if t.track_id < 0:
            t.track_id = self.next_id
            self.next_id += 1
        t.state = State.CONFIRMED

    def _predict(self, affine: np.ndarray | None) -> None:
        for t in self.tracks:
            if t.state == State.LOST:
                t.mean[7] = 0.0
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)
            if affine is not None:
                t.mean, t.cov = warp_state(t.mean, t.cov, affine)

    def _first_cost(self, pool: list[Track], boxes: np.ndarray, scores: np.ndarray,
                    feats: np.ndarray | None) -> np.ndarray:
        cost = iou_cost(np.array([t.box for t in pool]), boxes)
        if self.cfg.fuse_score:
            cost = fuse_score(cost, scores)
        if len(pool) == 0 or len(boxes) == 0:
            return cost
        if self.cfg.reid and feats is not None:
            cost = fuse_appearance(cost, cosine_cost(np.stack([t.feat for t in pool]), feats), self.cfg.iou_weight)
        if self.cfg.motion_gate:
            means, covs = np.stack([t.mean for t in pool]), np.stack([t.cov for t in pool])
            cost = np.where(self.kf.gating_distance(means, covs, xyxy_to_xyah(boxes)) <= CHI2_95_4DOF, cost, np.inf)
        return cost

    def update(self, dets: np.ndarray, feats: np.ndarray | None = None,
               affine: np.ndarray | None = None) -> np.ndarray:
        cfg = self.cfg
        self.frame += 1
        frame = self.frame
        use_feats = feats if cfg.reid else None
        boxes, scores = dets[:, :4].astype(np.float64), dets[:, 4].astype(np.float64)
        high = np.flatnonzero(scores >= cfg.high_thresh)
        low = np.flatnonzero((scores >= cfg.low_thresh) & (scores < cfg.high_thresh))

        self._predict(affine if cfg.cmc else None)
        pool = [t for t in self.tracks if t.state != State.TENTATIVE]
        tentative = [t for t in self.tracks if t.state == State.TENTATIVE]

        hf = None if use_feats is None else use_feats[high]
        cost = self._first_cost(pool, boxes[high], scores[high], hf)
        matches, um_tracks, um_high = linear_assignment(cost, cfg.match_thresh)
        for ti, di in matches:
            d = high[di]
            pool[ti].update(self.kf, boxes[d], scores[d], None if hf is None else hf[di], frame, cfg.ema_alpha)
            self._confirm(pool[ti])

        remaining = [pool[i] for i in um_tracks]
        if cfg.second_stage:
            active = [t for t in remaining if t.state == State.CONFIRMED]
            rest = [t for t in remaining if t.state != State.CONFIRMED]
            cost = iou_cost(np.array([t.box for t in active]), boxes[low])
            matches, um2, _ = linear_assignment(cost, cfg.second_match_thresh)
            for ti, di in matches:
                d = low[di]
                active[ti].update(self.kf, boxes[d], scores[d], None, frame, cfg.ema_alpha)
            remaining = rest + [active[i] for i in um2]
        for t in remaining:
            t.state = State.LOST

        left = high[um_high]
        cost = iou_cost(np.array([t.box for t in tentative]), boxes[left])
        if cfg.fuse_score:
            cost = fuse_score(cost, scores[left])
        matches, um_tent, um_left = linear_assignment(cost, cfg.unconfirmed_match_thresh)
        for ti, di in matches:
            d = left[di]
            tentative[ti].update(self.kf, boxes[d], scores[d], None if use_feats is None else use_feats[d], frame,
                                 cfg.ema_alpha)
            self._confirm(tentative[ti])
        dead = {id(tentative[i]) for i in um_tent}

        for di in um_left:
            d = left[di]
            if scores[d] < cfg.new_track_thresh:
                continue
            t = Track(self.kf, boxes[d], scores[d], None if use_feats is None else use_feats[d], frame)
            if frame == 1:
                self._confirm(t)
            self.tracks.append(t)

        self.tracks = [t for t in self.tracks if id(t) not in dead
                       and not (t.state == State.LOST and frame - t.last_frame > self.max_age)]
        return self._output()

    def _output(self) -> np.ndarray:
        rows = []
        for t in self.tracks:
            if t.state != State.CONFIRMED or t.last_frame != self.frame:
                continue
            x1, y1, x2, y2 = t.box
            w, h = x2 - x1, y2 - y1
            if w * h <= self.cfg.min_box_area or w / h > self.cfg.max_aspect:
                continue
            rows.append([self.frame, t.track_id, x1, y1, x2, y2, t.score])
        return np.array(rows, dtype=np.float64).reshape(-1, 7)


def track_sequence(cfg: TrackerConfig, frames: list[np.ndarray], fps: float = 30.0,
                   feats: list[np.ndarray] | None = None,
                   affines: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    tracker = Tracker(cfg, fps)
    out = []
    t0 = time.perf_counter()
    for t, dets in enumerate(frames):
        out.append(tracker.update(dets, None if feats is None else feats[t], None if affines is None else affines[t]))
    return np.concatenate(out) if out else np.zeros((0, 7)), time.perf_counter() - t0
