import time

import numpy as np
import pytest
import torch

from mts.deploy.bench import time_calls
from mts.deploy.export import export_onnx, make_session, match_detections
from mts.tracker.matching import iou_matrix


class TwoHeads(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.cls = torch.nn.Linear(4, 2)
        self.box = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.conv(x)).flatten(2).transpose(1, 2)
        return self.cls(h), self.box(h).sigmoid()


def test_onnx_round_trip_matches_torch(tmp_path):
    torch.manual_seed(0)
    model = TwoHeads().eval()
    x = torch.rand(1, 3, 8, 12)
    path = export_onnx(model, x, tmp_path / "m.onnx")
    sess = make_session(path, "cpu", threads=1)
    got = sess.run(["logits", "pred_boxes"], {"pixel_values": x.numpy()})
    with torch.inference_mode():
        ref = model(x)
    assert got[0].shape == (1, 96, 2) and got[1].shape == (1, 96, 4)
    for g, r in zip(got, ref):
        np.testing.assert_allclose(g, r.numpy(), atol=1e-5)


def test_timer_statistics_on_sleep():
    calls = []
    stats = time_calls(lambda x: (calls.append(x), time.sleep(0.002)), list(range(30)), warmup=5)
    assert len(calls) == 35
    assert stats["latency_frames"] == 30
    assert 1.9 <= stats["latency_median_ms"] < 20
    assert stats["latency_p90_ms"] >= stats["latency_median_ms"]
    assert stats["latency_std_ms"] >= 0


def test_matching_pairs_shifted_boxes():
    ref = np.array([[0, 0, 10, 10], [50, 50, 70, 90]], dtype=float)
    got = np.array([[51, 50, 71, 90], [0, 1, 10, 11], [200, 200, 210, 210]], dtype=float)
    assert iou_matrix(ref[:1], ref[:1])[0, 0] == pytest.approx(1.0)
    r, g = match_detections(ref, got)
    assert dict(zip(r.tolist(), g.tolist())) == {0: 1, 1: 0}

