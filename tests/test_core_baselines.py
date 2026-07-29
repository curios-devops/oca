import numpy as np
import pytest

from core.baselines import GRU, CopyLast, LinearAR
from core.data import rollout


def _tiny_gru_and_batch(seed=0):
    rng = np.random.default_rng(seed)
    g = GRU(n_in=5, hidden=4, horizons=(1, 2), seed=seed)
    xs = rng.normal(size=(6, 3, 5))  # (T, B, D)
    targets = {
        1: [(t, (t, rng.normal(size=(3, 5)))) for t in range(4)],
        2: [(t, (t, rng.normal(size=(3, 5)))) for t in range(4)],
    }
    return g, xs, targets


def test_bptt_matches_finite_differences():
    g, xs, targets = _tiny_gru_and_batch()
    h0 = np.zeros((3, 4))

    cache, _ = g._forward(xs, h0)
    _, grads = g._loss_and_grads(cache, targets)

    eps = 1e-6
    rng = np.random.default_rng(1)
    for name in ["Wz", "Ur", "bn", "Wo1", "bo2"]:
        param = g.p[name]
        flat = param.reshape(-1)
        for _ in range(4):
            k = int(rng.integers(flat.size))
            orig = flat[k]

            flat[k] = orig + eps
            loss_p, _ = g._loss_and_grads(g._forward(xs, h0)[0], targets)
            flat[k] = orig - eps
            loss_m, _ = g._loss_and_grads(g._forward(xs, h0)[0], targets)
            flat[k] = orig

            numeric = (loss_p - loss_m) / (2 * eps)
            analytic = grads[name].reshape(-1)[k]
            assert numeric == pytest.approx(analytic, rel=2e-4, abs=1e-8), (
                f"{name}[{k}]: analytic {analytic} vs numeric {numeric}"
            )


def test_gru_training_reduces_loss_on_a_learnable_stream():
    frames, _, _ = rollout(300, seed=11)
    g = GRU(n_in=frames[0].size, hidden=24, horizons=(1,), seed=0, lr=5e-3)
    before = g.evaluate(frames).cumulative()["mse_t1"]
    g.fit([frames], epochs=6, chunk=16)
    after = g.evaluate(frames).cumulative()["mse_t1"]
    assert after < before


def test_linear_ar_beats_copy_last_on_moving_objects():
    train, _, _ = rollout(600, seed=21)
    test, _, _ = rollout(300, seed=22)
    copy = CopyLast().evaluate(test).cumulative()
    lin = LinearAR().fit(train).evaluate(test).cumulative()
    # constant-velocity motion is exactly linear in two lags, so this must hold
    assert lin["mse_t1"] < copy["mse_t1"]


def test_copy_last_error_grows_with_horizon():
    frames, _, _ = rollout(400, seed=31)
    c = CopyLast().evaluate(frames).cumulative()
    assert c["mse_t1"] < c["mse_t4"] < c["mse_t16"]
