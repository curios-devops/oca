"""Local learning rules (SPEC_RPDU section 5).

Every rule here reads only quantities the unit already holds: its own state, its own
prediction error, its own precision. No error signal crosses a unit boundary and there
is no backward pass through the mesh. That restriction is the hypothesis under test, so
it is enforced structurally -- these functions are never given another unit's error.
"""

from __future__ import annotations

import numpy as np


def normaliser(h: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """1/||h||^2 per unit, for normalised LMS.

    The plain delta rule converges far too slowly here: a unit's state is strongly
    autocorrelated and its scale drifts as the landscape learns, so one fixed step size
    cannot serve both. Dividing by the unit's own squared state norm makes the step
    scale-free and is still entirely local -- it uses nothing but h_i.
    """
    return 1.0 / (np.einsum("nd,nd->n", h, h) + eps)


def delta_rule(theta: np.ndarray, err: np.ndarray, h_past: np.ndarray, lr: float,
               precision: np.ndarray, decay: float) -> None:
    """In-place normalised LMS: `theta += lr * precision * outer(err, h) / ||h||^2`.

    theta: (n, out, d)   err: (n, out)   h_past: (n, d)   precision: (n,)
    """
    gain = lr * precision * normaliser(h_past)
    theta *= 1.0 - decay
    theta += (gain[:, None] * err)[:, :, None] * h_past[:, None, :]


def update_landscape(state, err_self: np.ndarray, h_past: np.ndarray) -> None:
    """One-step gradient of 1/2||err||^2 through the unit's own dynamics.

    d/dU of h + eta*(U U' h + b - a||h||^2 h) gives err(h'U) + h(err'U); both terms are
    (d, r). This is one step through one tick of one unit -- not backprop through time,
    and not through the mesh.
    """
    cfg = state.cfg
    scale = cfg.eta_landscape * cfg.eta_h * state.conf * normaliser(h_past)
    hU = np.einsum("nd,ndr->nr", h_past, state.U)
    eU = np.einsum("nd,ndr->nr", err_self, state.U)
    dU = err_self[:, :, None] * hU[:, None, :] + h_past[:, :, None] * eU[:, None, :]

    state.U *= 1.0 - cfg.weight_decay
    state.U += scale[:, None, None] * dU
    state.b *= 1.0 - cfg.weight_decay
    state.b += (scale * 1.0)[:, None] * err_self

    fro = np.linalg.norm(state.U, axis=(1, 2), keepdims=True)
    over = fro > cfg.u_max_fro
    if over.any():
        state.U = np.where(over, state.U * (cfg.u_max_fro / np.maximum(fro, 1e-12)), state.U)


def update_links(state, err_self: np.ndarray, coeff: np.ndarray, gamma: np.ndarray,
                 msg_e: np.ndarray) -> None:
    """Read vectors and gains learn through the drive, which enters h' linearly."""
    cfg = state.cfg
    scale = cfg.eta_link * cfg.eta_h * state.conf                    # (N,)

    # a[i,k] += lr * c_i * (w*gamma*e)[i,k] * err_i
    state.a *= 1.0 - cfg.weight_decay
    state.a += (scale[:, None] * coeff)[:, :, None] * err_self[:, None, :]

    # w[i,k] += lr * c_i * (gamma*e)[i,k] * (a[i,k] . err_i)
    # Deliberately reads the freshly updated `a` rather than its pre-update value: a
    # Gauss-Seidel style sweep over the two coupled link parameters. Both orderings are
    # first-order correct; this one is marginally better conditioned because the gain
    # responds to the read vector the unit will actually use.
    a_dot_err = np.einsum("nkd,nd->nk", state.a, err_self)
    state.w *= 1.0 - cfg.weight_decay
    state.w += scale[:, None] * gamma * msg_e * a_dot_err
    np.clip(state.w, -3.0, 3.0, out=state.w)


def update_sensory_encoder(state, err_self: np.ndarray, s_now: np.ndarray) -> None:
    """The encoder that injects the retina patch into the drive."""
    cfg = state.cfg
    idx = state.sensory_idx
    scale = cfg.eta_link * cfg.eta_h * state.conf[idx]
    state.S_enc *= 1.0 - cfg.weight_decay
    state.S_enc += scale[:, None, None] * err_self[idx][:, :, None] * s_now[:, None, :]


def update_surprise_head(state, surprise: np.ndarray, h: np.ndarray) -> None:
    """The unit learns to predict the size of its own error."""
    cfg = state.cfg
    if not cfg.use_uncertainty:
        return
    err = surprise - state.sigma_hat
    state.v *= 1.0 - cfg.weight_decay
    state.v += cfg.eta_sigma * err[:, None] * h
    state.v0 += cfg.eta_sigma * err


def update_credit(state, msg_e: np.ndarray, pred_msg: np.ndarray,
                  d_surprise: np.ndarray) -> None:
    """Link credit: predictable traffic that coincides with falling surprise.

    Noise is unpredictable and earns nothing; perfectly predictable but uninformative
    traffic moves no surprise and also earns nothing. Both are pruned by one rule.
    """
    cfg = state.cfg
    resid = (msg_e - pred_msg) ** 2
    predictability = 1.0 / (1.0 + resid)
    contribution = predictability * d_surprise[:, None]
    state.credit *= 1.0 - cfg.credit_ema
    state.credit += cfg.credit_ema * contribution


def update_kappa(state) -> None:
    """Calibrate each unit's importance bid by the credit its out-links actually earned.

    This is the only quantity that crosses a unit boundary in the direction of learning:
    one scalar, every rewiring interval, about the unit's own broadcasting. Deliberately
    impoverished, and ablated in E3.
    """
    N = state.n_units
    out_credit = np.bincount(state.src.ravel(), weights=state.credit.ravel(), minlength=N)
    out_count = np.bincount(state.src.ravel(), minlength=N)
    mean_credit = out_credit / np.maximum(out_count, 1)
    norm = mean_credit / (np.abs(mean_credit).mean() + 1e-8)
    state.kappa = np.clip(0.9 * state.kappa + 0.1 * (1.0 + norm), 0.05, 5.0)
