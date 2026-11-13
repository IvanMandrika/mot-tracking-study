from dataclasses import replace

import cv2
import numpy as np
import pytest

from mts import exp_tracking
from mts.tracker.cmc import estimate_affine, warp_state
from mts.tracker.kalman import KalmanFilter, xyah_to_xyxy, xyxy_to_xyah
from mts.tracker.matching import iou_matrix, linear_assignment
from mts.tracker.reid import crop_batch
from mts.tracker.tracker import Tracker, TrackerConfig, track_sequence

W, H = 40.0, 100.0


def box(x: float, y: float = 100.0, score: float = 0.9) -> np.ndarray:
    return np.array([x, y, x + W, y + H, score])


def frame(*boxes: np.ndarray) -> np.ndarray:
    return np.array(boxes).reshape(-1, 5)


def test_box_conversion_roundtrip() -> None:
    b = np.array([[10.0, 20.0, 50.0, 120.0]])
    assert np.allclose(xyxy_to_xyah(b), [[30.0, 70.0, 0.4, 100.0]])
    assert np.allclose(xyah_to_xyxy(xyxy_to_xyah(b)), b)


def test_kalman_predict_is_exact_for_constant_velocity_state() -> None:
    kf = KalmanFilter()
    mean = np.array([100.0, 50.0, 0.4, 80.0, 3.0, -2.0, 0.0, 0.5])
    pred, _ = kf.predict(mean, np.eye(8))
    assert np.allclose(pred, [103.0, 48.0, 0.4, 80.5, 3.0, -2.0, 0.0, 0.5])


def test_kalman_converges_on_constant_velocity_box() -> None:
    kf = KalmanFilter()
    truth = [np.array([10.0 + 4 * t, 20.0 + 1.5 * t, 0.4, 100.0]) for t in range(80)]
    mean, cov = kf.initiate(truth[0])
    for z in truth[1:-1]:
        mean, cov = kf.update(*kf.predict(mean, cov), z)
    pred, _ = kf.predict(mean, cov)
    assert np.allclose(pred[:4], truth[-1], atol=1e-3)
    assert np.allclose(pred[4:6], [4.0, 1.5], atol=1e-3)


def test_gating_distance_grows_with_offset() -> None:
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([0.0, 0.0, 0.4, 100.0]))
    d = kf.gating_distance(mean[None], cov[None], np.array([[0.0, 0.0, 0.4, 100.0], [50.0, 0.0, 0.4, 100.0]]))
    assert d.shape == (1, 2) and d[0, 0] == pytest.approx(0.0) and d[0, 1] > 9.4877
    proj_mean, proj_cov = kf.project(mean, cov)
    z = np.array([3.0, -2.0, 0.41, 98.0])
    expected = (z - proj_mean) @ np.linalg.inv(proj_cov) @ (z - proj_mean)
    assert kf.gating_distance(mean[None], cov[None], z)[0, 0] == pytest.approx(expected)


def test_iou_and_thresholded_assignment() -> None:
    a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    b = np.array([[0, 0, 10, 5], [100, 100, 110, 110]], dtype=float)
    assert np.allclose(iou_matrix(a, b), [[0.5, 0.0], [0.0, 0.0]])
    matches, um_rows, um_cols = linear_assignment(1 - iou_matrix(a, b), thresh=0.6)
    assert matches.tolist() == [[0, 0]] and um_rows.tolist() == [1] and um_cols.tolist() == [1]


def test_warp_state_translation_and_scale() -> None:
    mean = np.array([100.0, 50.0, 0.4, 80.0, 2.0, 1.0, 0.0, 0.0])
    shifted, _ = warp_state(mean, np.eye(8), np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]]))
    assert np.allclose(shifted, [105.0, 47.0, 0.4, 80.0, 2.0, 1.0, 0.0, 0.0])
    zoomed, cov = warp_state(mean, np.eye(8), np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))
    assert np.allclose(zoomed[[0, 1, 2, 3, 4]], [200.0, 100.0, 0.4, 160.0, 4.0]) and cov[3, 3] == pytest.approx(4.0)


def ids_by_x(out: np.ndarray) -> list[int]:
    return out[np.argsort(out[:, 2]), 1].astype(int).tolist()


def bounce_scene() -> tuple[list[np.ndarray], list[np.ndarray]]:
    xs_a = [5.0 * t for t in range(20)] + [95.0 - 5.0 * t for t in range(20)]
    xs_b = [200.0 - 5.0 * t for t in range(20)] + [105.0 + 5.0 * t for t in range(20)]
    frames = [frame(box(a), box(b)) for a, b in zip(xs_a, xs_b)]
    feats = [np.array([[1.0, 0.0], [0.0, 1.0]]) for _ in frames]
    return frames, feats


def test_crossing_objects_keep_ids_with_appearance() -> None:
    frames, feats = bounce_scene()
    cfg = TrackerConfig(reid=True, iou_weight=0.5)
    out, _ = track_sequence(cfg, frames, feats=feats)
    first, last = out[out[:, 0] == 1], out[out[:, 0] == len(frames)]
    assert ids_by_x(first) == [1, 2] and ids_by_x(last) == [1, 2]
    assert len(np.unique(out[:, 1])) == 2


def test_crossing_objects_swap_without_appearance() -> None:
    frames, _ = bounce_scene()
    out, _ = track_sequence(TrackerConfig(), frames)
    assert ids_by_x(out[out[:, 0] == len(frames)]) == [2, 1]


def test_track_survives_short_occlusion_and_dies_after_max_age() -> None:
    cfg = TrackerConfig(max_age=10, fps_scaled_max_age=False)
    tracker = Tracker(cfg)
    for t in range(10):
        tracker.update(frame(box(10.0 + 3 * t)))
    for _ in range(5):
        assert len(tracker.update(frame())) == 0
    out = tracker.update(frame(box(10.0 + 3 * 15)))
    assert out[:, 1].tolist() == [1]
    for _ in range(10):
        tracker.update(frame())
    assert len(tracker.tracks) == 1
    tracker.update(frame())
    assert len(tracker.tracks) == 0


@pytest.mark.parametrize("second_stage,expected", [(True, [1]), (False, [])])
def test_second_stage_recovers_low_score_detection(second_stage: bool, expected: list[int]) -> None:
    tracker = Tracker(TrackerConfig(second_stage=second_stage))
    for t in range(5):
        tracker.update(frame(box(10.0 + 2 * t)))
    out = tracker.update(frame(box(20.0, score=0.3)))
    assert out[:, 1].tolist() == expected


def test_tentative_track_needs_second_hit_and_low_score_does_not_spawn() -> None:
    tracker = Tracker(TrackerConfig())
    tracker.update(frame())
    assert len(tracker.update(frame(box(10.0), box(300.0, score=0.65)))) == 0
    out = tracker.update(frame(box(11.0), box(301.0, score=0.65)))
    assert out[:, 1].tolist() == [1]


def test_motion_gate_blocks_far_iou_match_without_reid() -> None:
    far = TrackerConfig(match_thresh=1.0, unconfirmed_match_thresh=1.0)
    for gate, expected in ((False, [1]), (True, [])):
        tracker = Tracker(replace(far, motion_gate=gate))
        for _ in range(10):
            tracker.update(frame(box(10.0)))
        assert tracker.update(frame(box(40.0)))[:, 1].tolist() == expected


def test_estimate_affine_recovers_translation() -> None:
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur((rng.uniform(0, 255, (240, 320))).astype(np.uint8), (5, 5), 0)
    shifted = np.roll(img, (3, -5), axis=(0, 1))
    affine = estimate_affine(img[20:-20, 20:-20], shifted[20:-20, 20:-20], downscale=2)
    assert np.allclose(affine[:, :2], np.eye(2), atol=0.01)
    assert np.allclose(affine[:, 2], [-10.0, 6.0], atol=0.5)


def test_crop_batch_clips_boxes_outside_image() -> None:
    img = np.full((50, 80, 3), 200, np.uint8)
    boxes = np.array([[-20.0, -10, 30, 40], [70, 40, 120, 90], [10, 10, 10, 10]])
    out = crop_batch(img, boxes)
    assert out.shape == (3, 3, 224, 112) and np.isfinite(out).all()


def test_tuning_reads_only_the_selection_split(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def fake_run_many(jobs, workers):
        seen.extend(j[3] for j in jobs)
        return [(j[0], {"COMBINED": {k: float(i) for k in exp_tracking.HEADLINE}}, 0.0, 0) for i, j in enumerate(jobs)]

    monkeypatch.setattr(exp_tracking, "run_many", fake_run_many)
    configs, dfs = exp_tracking.tuned_configs("public_frcnn", True, workers=1)
    assert set(seen) == {"train_holdout"} and all((df["split"] == "train_holdout").all() for df in dfs)
    assert configs["a_sort"].high_thresh == max(exp_tracking.TUNE_HIGH)
    assert configs["e_byte_cmc_gate_reid"].iou_weight == 1.0 and configs["e_byte_cmc_gate_reid"].motion_gate
