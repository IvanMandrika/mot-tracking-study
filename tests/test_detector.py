import numpy as np
import pytest

from mts import data
from mts.detect import coco_metrics
from mts.detector import cxcywh_norm_to_xyxy, input_size, person_detections, person_label, xyxy_to_cxcywh_norm
from mts.train_detector import SizeGroupedBatches, augment, scale_jitter_affine, transform_boxes


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1 - p))


@pytest.mark.parametrize("width,height", [(1920, 1080), (640, 480)])
def test_box_round_trip(width: int, height: int) -> None:
    rng = np.random.default_rng(0)
    x1 = rng.uniform(0, width / 2, 50)
    y1 = rng.uniform(0, height / 2, 50)
    boxes = np.stack([x1, y1, x1 + rng.uniform(1, 200, 50), y1 + rng.uniform(1, 300, 50)], axis=1)
    norm = xyxy_to_cxcywh_norm(boxes, width, height)
    assert norm.min() >= 0 and norm[:, :2].max() <= 1
    np.testing.assert_allclose(cxcywh_norm_to_xyxy(norm, width, height), boxes, atol=1e-9)


def test_normalised_boxes_are_resolution_independent() -> None:
    norm = np.array([[0.5, 0.5, 0.1, 0.2]])
    np.testing.assert_allclose(cxcywh_norm_to_xyxy(norm, 1920, 1080), [[864, 432, 1056, 648]])
    np.testing.assert_allclose(cxcywh_norm_to_xyxy(norm, 640, 480), [[288, 192, 352, 288]])


@pytest.mark.parametrize("width,height,long_side,expected", [
    (1920, 1080, 960, (960, 544)), (640, 480, 960, (960, 704)), (1920, 1080, 640, (640, 352)), (10, 10, 20, (32, 32)),
])
def test_input_size_is_stride_multiple(width: int, height: int, long_side: int, expected: tuple[int, int]) -> None:
    assert input_size(width, height, long_side) == expected


def test_person_label() -> None:
    assert person_label({0: "person", 1: "bicycle"}) == 0
    assert person_label({"3": "person", "0": "car"}) == 3
    with pytest.raises(ValueError):
        person_label({0: "car"})


def test_person_filtering_uses_only_person_logit() -> None:
    probs = np.array([[0.9, 0.01], [0.02, 0.99], [0.3, 0.5], [0.04, 0.9]])
    boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.1, 0.4], [0.2, 0.2, 0.1, 0.1]])
    out = person_detections(logit(probs), boxes, person=0, width=100, height=200, min_score=0.05)
    assert out.shape == (2, 5) and out.dtype == np.float32
    np.testing.assert_allclose(out[:, 4], [0.9, 0.3], rtol=1e-6)
    np.testing.assert_allclose(out[0, :4], [40, 80, 60, 120], rtol=1e-6)


def test_person_filtering_top_k_and_clipping() -> None:
    probs = np.array([[0.6], [0.7], [0.8]])
    boxes = np.array([[0.0, 0.0, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2], [1.0, 1.0, 0.2, 0.2]])
    out = person_detections(logit(probs), boxes, person=0, width=10, height=10, top_k=2)
    np.testing.assert_allclose(out[:, 4], [0.8, 0.7], rtol=1e-6)
    np.testing.assert_allclose(out[0, :4], [9, 9, 10, 10], atol=1e-5)


def test_cache_file_format(tmp_path) -> None:
    dets = np.concatenate([np.full((3, 1), 2.0), person_detections(
        logit(np.array([[0.9], [0.5], [0.2]])), np.full((3, 4), 0.5), 0, 64, 32)], axis=1)
    path = data.save_dets(dets, "rtdetr_coco", "val_half", "MOT17-99", cache=tmp_path)
    assert path == tmp_path / "dets" / "rtdetr_coco" / "val_half" / "MOT17-99.npy"
    back = data.load_dets("rtdetr_coco", "val_half", "MOT17-99", cache=tmp_path)
    assert back.dtype == np.float32 and back.shape == (3, 6)
    assert (back[:, 5] >= 0.05).all() and np.all(np.diff(back[:, 5]) <= 0)
    per_frame = data.dets_by_frame(back, 3)
    assert [len(f) for f in per_frame] == [0, 3, 0]


def test_scale_jitter_identity_and_zoom() -> None:
    boxes = np.array([[100.0, 50.0, 300.0, 250.0]])
    m = scale_jitter_affine(400, 300, (400, 300), 1.0, 0.3, 0.7)
    np.testing.assert_allclose(transform_boxes(boxes, m, (400, 300)), boxes)
    m = scale_jitter_affine(400, 300, (800, 600), 2.0, 0.0, 0.0)
    np.testing.assert_allclose(transform_boxes(np.array([[20.0, 10, 60, 50]]), m, (800, 600)), [[80, 40, 240, 200]])


def test_transform_drops_mostly_cropped_boxes() -> None:
    m = np.array([[1.0, 0, -90], [0, 1.0, 0]])
    out = transform_boxes(np.array([[0.0, 0, 100, 100], [100, 0, 200, 100]]), m, (100, 100))
    np.testing.assert_allclose(out, [[10, 0, 100, 100]])


def test_augment_keeps_boxes_inside_output() -> None:
    rng = np.random.default_rng(1)
    img = np.full((120, 160, 3), 128, np.uint8)
    boxes = np.array([[10.0, 10, 40, 90], [100, 20, 150, 110]])
    for _ in range(20):
        out, b = augment(img, boxes, (192, 128), rng)
        assert out.shape == (128, 192, 3) and out.dtype == np.uint8
        assert (b[:, 0] >= 0).all() and (b[:, 2] <= 192).all() and (b[:, 3] <= 128).all()
        assert (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all()


def test_holdout_split_and_batches() -> None:
    assert data.fit_length(300) == 240 and data.fit_length(262) == 210
    assert data.half_range(525, "train_fit") == (1, 210) and data.half_range(525, "train_holdout") == (211, 262)
    sizes = [(64, 32)] * 5 + [(32, 32)] * 3
    batches = SizeGroupedBatches(sizes, 2, seed=0).batches()
    assert sorted(i for b in batches for i in b) == list(range(8))
    assert all(len({sizes[i] for i in b}) == 1 for b in batches)


def test_coco_metrics_perfect_and_ignore() -> None:
    gt = [np.array([[0.0, 0, 10, 20], [50, 50, 60, 70]])]
    ign = [np.array([[100.0, 100, 110, 120]])]
    perfect = [np.array([[0.0, 0, 10, 20, 0.9], [50, 50, 60, 70, 0.8], [100, 100, 110, 120, 0.95]])]
    m = coco_metrics(gt, ign, perfect)
    assert m["AP"] == pytest.approx(100) and m["recall50"] == pytest.approx(100)
    half = [np.array([[0.0, 0, 10, 20, 0.9], [200, 200, 210, 220, 0.95]])]
    m = coco_metrics(gt, [np.zeros((0, 4))], half)
    assert m["recall50"] == pytest.approx(50)
    assert m["AP50"] < 50
