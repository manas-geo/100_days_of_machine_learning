"""Run with `pytest -q` or simply `python tests/test_core.py`."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import rs_utils as ru

def test_olofsson_perfect_map():
    rng = np.random.default_rng(0); m = rng.integers(0, 3, 300)
    s, _ = ru.olofsson_accuracy(m, m, {0: 5000, 1: 3000, 2: 2000}, 0.04)
    assert abs(s.attrs["OA"] - 1) < 1e-12 and np.allclose(s["adjusted_area_ha"], s["mapped_area_ha"])

def test_olofsson_area_unbiased():
    rng = np.random.default_rng(1); est = []
    truth = rng.integers(0, 2, 20000); mapped = np.where(rng.random(20000) < 0.1, 1 - truth, truth)
    counts = {0: int((mapped == 0).sum()), 1: int((mapped == 1).sum())}
    for s in range(200):
        idx = np.concatenate([rng.choice(np.flatnonzero(mapped == k), 100, replace=False) for k in (0, 1)])
        est.append(ru.olofsson_accuracy(mapped[idx], truth[idx], counts, 1.0)[0]["adjusted_area_ha"].iloc[1])
    assert abs(np.mean(est) - (truth == 1).sum()) / (truth == 1).sum() < 0.02

def test_stratified_sample_sizes():
    strata = np.repeat(np.arange(3), [100, 50, 10]).reshape(16, 10)
    r, c, lab = ru.stratified_random_sample(strata, {0: 20, 1: 20, 2: 20}, seed=0)
    assert np.bincount(lab).tolist() == [20, 20, 10]                  # capped at the stratum size

def test_spatial_blocks_group_neighbours():
    b = ru.spatial_blocks(np.array([0, 10, 2500]), np.array([0, 10, 0]), 2000)
    assert b[0] == b[1] != b[2]

def test_ndvi_zero_when_equal():
    b = {k: np.full((2, 2), 0.2) for k in ["B2", "B3", "B4", "B5", "B8", "B8A", "B11"]}
    assert np.allclose(ru.indices(b)["NDVI"], 0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
