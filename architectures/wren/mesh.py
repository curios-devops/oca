"""The mesh: construction and the tick loop (SPEC_RPDU section 6).

    1 receive   2 update state   3 predict   4 measure
    5 adjust    6 choose         7 broadcast

All seven steps are vectorized across units. Because every unit reads the message
buffer written on the previous tick, the update is order-independent and bit-identical
under a fixed seed -- there is no sweep order to bias the result.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from . import landscape as L
from . import plasticity as plast
from . import protocol as proto
from .state import Config, MeshState
from .topology import build_topology, rewire
from core.world.sensors import N_SENSORY, N_VISUAL, P, Sensors


def build_mesh(cfg: Config) -> MeshState:
    rng = np.random.default_rng(cfg.seed)
    N, d, r, q, D = cfg.n_units, cfg.d, cfg.r, cfg.q, cfg.degree
    H = cfg.n_horizons

    topo = build_topology(cfg, rng)
    sensory_idx = Sensors.lattice_slots(cfg.lattice_side)

    def n(*shape, scale=1.0):
        return rng.normal(0.0, scale, shape)

    state = MeshState(
        cfg=cfg,
        rng=rng,
        h=n(N, d, scale=0.1),
        U=n(N, d, r, scale=1.0 / np.sqrt(d)),
        b=np.zeros((N, d)),
        omega=n(N, d, scale=1.0 / np.sqrt(d)),
        src=topo["src"],
        w=np.full((N, D), 0.5),
        a=n(N, D, d, scale=1.0 / np.sqrt(d)),
        u_proj=n(N, D, d, scale=1.0 / np.sqrt(d)),
        credit=np.zeros((N, D)),
        is_long=topo["is_long"],
        P=n(N_SENSORY, H, P, d, scale=0.1 / np.sqrt(d)),
        M=n(N, H, D, d, scale=0.1 / np.sqrt(d)),
        C=n(N, H, q, d, scale=0.1 / np.sqrt(d)),
        v=np.zeros((N, d)),
        v0=np.zeros(N),
        kappa=np.ones(N),
        sensory_idx=sensory_idx,
        S_enc=n(N_SENSORY, d, P, scale=1.0 / np.sqrt(P)),
        R_proj=n(q, d, scale=1.0 / np.sqrt(d)),
        surprise=np.zeros(N),
        surprise_ema=np.ones(N),
        novelty=np.ones(N),
        sigma_hat=np.zeros(N),
        conf=np.ones(N),
        coalition=np.arange(N),
    )
    state.exec_ids = topo["exec_ids"]
    state.prev_surprise = np.zeros(N)
    state.msg_e = np.zeros((N, D))
    state.msg_c = np.ones((N, D))
    state.msg_n = np.zeros((N, D))
    state.msg_i = np.zeros((N, D))
    state.g = np.zeros(q)
    state.hist_h = deque(maxlen=max(cfg.max_horizon, cfg.coalition_window))
    state.hist_pred = {
        key: {tau: deque(maxlen=tau) for tau in cfg.horizons}
        for key in ("s", "m", "h", "g")
    }
    state.nov_hist = deque(maxlen=64)
    state.last_pred_s = {}
    state.learn = True   # runtime toggle: held-out evaluation freezes plasticity
    return state


def tick(state: MeshState, s_now: np.ndarray) -> dict:
    """Advance the mesh one step. `s_now` is (N_SENSORY, P) transduced input.

    Returns a diagnostics dict; the predicted sensory patches for every horizon are
    left in `state.last_pred_s` for the caller to score.
    """
    cfg, rng = state.cfg, state.rng
    N, d = state.n_units, cfg.d
    learn = cfg.use_plasticity and getattr(state, "learn", True)

    # -- 1. receive -------------------------------------------------------
    msg = {"e": state.msg_e, "c": state.msg_c, "n": state.msg_n, "i": state.msg_i}
    if cfg.instant_delivery:
        # E3 ablation: read the senders' *current* states instead of last tick's
        # broadcast, collapsing the one-tick propagation delay.
        msg = dict(msg, e=np.einsum("nkd,nkd->nk", state.u_proj, state.h[state.src]))
    gamma = proto.receiver_gate(state, msg)
    drive, coeff = proto.drive_from_messages(state, msg, gamma)

    drive[state.sensory_idx] += cfg.sensory_gain * np.einsum("sdp,sp->sd", state.S_enc, s_now)
    if cfg.use_oscillator:
        osc = cfg.osc_gain * np.sin(2 * np.pi * state.t / cfg.osc_period)
        drive += osc * state.omega

    # -- 2. update state --------------------------------------------------
    b_eff = state.b + drive
    h_prev = state.h
    h_new, n_clip = L.relax(
        state.h, state.U, b_eff,
        alpha=cfg.alpha, eta=cfg.eta_h, steps=cfg.relax_steps,
        noise=cfg.state_noise, rng=rng, h_max=cfg.h_max, kind=cfg.flow_kind,
    )
    state.h = h_new
    state.n_clipped += n_clip
    g_now = proto.global_context(state, h_new)

    # -- 3. predict -------------------------------------------------------
    self_pred = L.rollout(
        h_new, state.U, b_eff, cfg.horizons,
        alpha=cfg.alpha, eta=cfg.eta_h, steps=cfg.relax_steps,
        h_max=cfg.h_max, kind=cfg.flow_kind,
    )
    h_sens = h_new[state.sensory_idx]
    base_s = s_now if cfg.residual_sensory else 0.0
    pred_s = {tau: base_s + np.einsum("spd,sd->sp", state.P[:, i], h_sens)
              for i, tau in enumerate(cfg.horizons)}
    pred_m = {tau: np.einsum("nkd,nd->nk", state.M[:, i], h_new)
              for i, tau in enumerate(cfg.horizons)}
    pred_g = {tau: np.einsum("nqd,nd->nq", state.C[:, i], h_new)
              for i, tau in enumerate(cfg.horizons)}
    state.last_pred_s = pred_s

    if cfg.use_uncertainty:
        state.sigma_hat = proto.softplus(np.einsum("nd,nd->n", state.v, h_new) + state.v0)
        state.conf = 1.0 / (1.0 + state.sigma_hat)
    else:
        state.sigma_hat = np.zeros(N)
        state.conf = np.ones(N)

    # -- 4. measure, and 5. adjust (fused: each horizon is compared then learned from)
    surprise = np.zeros(N)
    err_self_1 = None
    for i, tau in enumerate(cfg.horizons):
        dq_s, dq_m, dq_h, dq_g = (state.hist_pred[k][tau] for k in ("s", "m", "h", "g"))
        ready = len(dq_h) == tau and len(state.hist_h) >= tau
        if ready:
            rho = 1.0 / tau
            h_past = state.hist_h[-tau]
            e_h = h_new - dq_h[0]
            e_m = msg["e"] - dq_m[0]
            e_g = g_now[None, :] - dq_g[0]
            e_s = s_now - dq_s[0]

            surprise += rho * cfg.beta[2] * np.einsum("nd,nd->n", e_h, e_h)
            surprise += rho * cfg.beta[1] * np.einsum("nk,nk->n", e_m, e_m)
            surprise += rho * cfg.beta[3] * np.einsum("nq,nq->n", e_g, e_g)
            np.add.at(surprise, state.sensory_idx,
                      rho * cfg.beta[0] * np.einsum("sp,sp->s", e_s, e_s))

            if learn:
                plast.delta_rule(state.M[:, i], e_m, h_past, cfg.eta_head,
                                 state.conf, cfg.weight_decay)
                plast.delta_rule(state.C[:, i], e_g, h_past, cfg.eta_head,
                                 state.conf, cfg.weight_decay)
                plast.delta_rule(state.P[:, i], e_s, h_past[state.sensory_idx],
                                 cfg.eta_head, state.conf[state.sensory_idx],
                                 cfg.weight_decay)
            if tau == 1:
                err_self_1 = e_h

        dq_s.append(pred_s[tau])
        dq_m.append(pred_m[tau])
        dq_h.append(self_pred[tau])
        dq_g.append(pred_g[tau])

    if learn and err_self_1 is not None:
        plast.update_landscape(state, err_self_1, h_prev)
        plast.update_links(state, err_self_1, coeff, gamma, msg["e"])
        plast.update_sensory_encoder(state, err_self_1, s_now)

    d_surprise = state.prev_surprise - surprise
    state.prev_surprise = surprise
    state.surprise = surprise
    state.surprise_ema = ((1 - cfg.novelty_ema) * state.surprise_ema
                          + cfg.novelty_ema * surprise)
    state.novelty = surprise / (state.surprise_ema + 1e-8)
    if learn:
        plast.update_surprise_head(state, surprise, h_new)
        plast.update_credit(state, msg["e"], pred_m[cfg.horizons[0]], d_surprise)

    # -- 6. choose neighbours --------------------------------------------
    state.hist_h.append(h_new.copy())
    state.nov_hist.append(state.novelty.copy())
    rewired = 0
    if (learn and state.t > 0 and state.t % cfg.rewire_every == 0
            and len(state.nov_hist) >= 16):
        plast.update_kappa(state)
        rewired = rewire(state, cfg, rng, np.stack(state.nov_hist))["n_rewired"]

    if state.t % cfg.coalition_every == 0 and len(state.hist_h) >= cfg.coalition_window:
        window = np.stack(list(state.hist_h)[-cfg.coalition_window:])
        state.coalition = proto.detect_coalitions(state, window, gamma)

    # -- 7. broadcast ------------------------------------------------------
    out = proto.assemble_messages(state, self_pred[1])
    state.msg_e, state.msg_c = out["e"], out["c"]
    state.msg_n, state.msg_i = out["n"], out["i"]

    state.g = g_now
    state.t += 1

    return {
        "t": state.t,
        "surprise": float(surprise.mean()),
        "novelty": float(state.novelty.mean()),
        "conf": float(state.conf.mean()),
        "h_norm": float(np.linalg.norm(h_new, axis=1).mean()),
        "n_clipped": n_clip,
        "rewired": rewired,
    }


def predicted_retina(state: MeshState, tau: int, sensors: Sensors) -> np.ndarray:
    """Reassemble the visual units' patch predictions into a full frame."""
    return sensors.from_patches(state.last_pred_s[tau][:N_VISUAL])
