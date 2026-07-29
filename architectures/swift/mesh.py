"""The v2 mesh: construction and tick loop.

Same seven-step tick as v1 (receive, update, predict, measure, adjust, choose,
broadcast) and the same locality guarantee -- no error signal crosses a unit boundary.
What changed is the unit's dynamics (limit cycles rather than fixed points), the gate
(phase agreement rather than a 1.2x nudge), the timescales (a spread rather than one),
and the objective (a masked unit must reconstruct its own input from its neighbours).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from architectures.wren.plasticity import delta_rule, normaliser
from architectures.wren.topology import build_topology, rewire
from core.world.sensors import N_SENSORY, N_VISUAL, P, Sensors

from . import dynamics as dyn
from . import protocol as proto
from .state import Config2, MeshState2


def build_mesh2(cfg: Config2) -> MeshState2:
    rng = np.random.default_rng(cfg.seed)
    N, d, m, q, D, H = cfg.n_units, cfg.d, cfg.m, cfg.q, cfg.degree, cfg.n_horizons
    ro, k = cfg.readout, cfg.vote_dim
    if d % 2:
        raise ValueError("d must be even: state dimensions are paired into rotors")

    topo = build_topology(cfg, rng)

    def logu(lo, hi, shape):
        return np.exp(rng.uniform(np.log(lo), np.log(hi), shape))

    eta = (logu(*cfg.eta_range, (N,)) if cfg.use_timescales
           else np.full(N, float(np.sqrt(cfg.eta_range[0] * cfg.eta_range[1]))))
    mu = np.full((N, m), cfg.mu_init if cfg.use_oscillation else -0.3)

    def n(*shape, scale=1.0):
        return rng.normal(0.0, scale, shape)

    state = MeshState2(
        cfg=cfg, rng=rng,
        z=n(N, m, 2, scale=0.3),
        omega=logu(*cfg.omega_range, (N, m)) * rng.choice([-1.0, 1.0], (N, m)),
        mu=mu, eta=eta,
        src=topo["src"], w=np.full((N, D), 0.5),
        a=n(N, D, d, scale=1.0 / np.sqrt(d)),
        u_proj=n(N, D, m, scale=1.0 / np.sqrt(m)),
        credit=np.zeros((N, D)), is_long=topo["is_long"],
        P=n(N_SENSORY, H, P, ro, scale=0.1 / np.sqrt(ro)),
        M=n(N, H, D, ro, scale=0.1 / np.sqrt(ro)),
        C=n(N, H, q, ro, scale=0.1 / np.sqrt(ro)),
        v=np.zeros((N, ro)), v0=np.zeros(N), kappa=np.ones(N),
        S_enc=n(N_SENSORY, d, P, scale=1.0 / np.sqrt(P)),
        sensory_idx=Sensors.lattice_slots(cfg.lattice_side),
        R_proj=n(q, ro, scale=1.0 / np.sqrt(ro)),
        y=rng.normal(0, 1.0, (N, k)),
        V=n(N, k, m, scale=1.0 / np.sqrt(m)),
        surprise=np.zeros(N), surprise_ema=np.ones(N), novelty=np.ones(N),
        sigma_hat=np.zeros(N), conf=np.ones(N), coalition=np.arange(N),
    )
    state.y /= np.linalg.norm(state.y, axis=1, keepdims=True)
    state.exec_ids = topo["exec_ids"]
    state.prev_surprise = np.zeros(N)
    state.msg_e = np.zeros((N, D))
    state.msg_c = np.ones((N, D))
    state.msg_n = np.zeros((N, D))
    state.msg_i = np.zeros((N, D))
    state.g = np.zeros(q)
    state.hist_z = deque(maxlen=cfg.coalition_window)
    state.hist_h = deque(maxlen=max(cfg.max_horizon, cfg.coalition_window))
    state.hist_pred = {k: {tau: deque(maxlen=tau) for tau in cfg.horizons}
                       for k in ("s", "m", "h", "g")}
    state.nov_hist = deque(maxlen=64)
    state.last_pred_s = {}
    state.last_mask = np.zeros(N_SENSORY, dtype=bool)
    state.vote_coalition = np.arange(N)
    state.learn = True
    return state


def tick2(state: MeshState2, s_true: np.ndarray) -> dict:
    """Advance one step. `s_true` is the (N_SENSORY, P) ground-truth sensory input.

    The unit is scored against `s_true` but may not get to *see* it: with probability
    `mask_prob` its input is zeroed. That is the whole non-local objective -- a masked
    unit can only predict its own patch by using what its neighbours tell it.
    """
    cfg, rng = state.cfg, state.rng
    N, d = state.n_units, cfg.d
    learn = cfg.use_plasticity and getattr(state, "learn", True)

    # -- 1. receive --------------------------------------------------------
    msg = {"e": state.msg_e, "c": state.msg_c, "n": state.msg_n, "i": state.msg_i}
    gamma = proto.receiver_gate(state, msg)
    drive_flat, coeff = proto.drive_from_messages(state, msg, gamma)
    if cfg.drive_norm:
        drive_flat = drive_flat / np.sqrt(cfg.degree)
    drive_flat = drive_flat * cfg.coupling

    # sensory drive, masked
    if cfg.use_masking and cfg.mask_prob > 0:
        mask = rng.random(N_SENSORY) < cfg.mask_prob
    else:
        mask = np.zeros(N_SENSORY, dtype=bool)
    s_seen = np.where(mask[:, None], 0.0, s_true)
    state.last_mask = mask
    drive_flat[state.sensory_idx] += cfg.sensory_gain * np.einsum(
        "sdp,sp->sd", state.S_enc, s_seen)

    # votes are reconciled before the state moves, so this tick's prediction can
    # use the hypothesis the neighbourhood currently holds
    if cfg.use_voting:
        vote_err = proto.vote_consensus_error(state, gamma)
        state.y = proto.vote_update(state, gamma)
    else:
        vote_err = None

    # -- 2. update state ---------------------------------------------------
    drive = drive_flat.reshape(N, cfg.m, 2)
    z_prev = state.z
    h_prev = state.h.copy()          # readout: amplitudes (+ vote)
    h_amp_prev = state.h_amp.copy()
    # novelty-modulated frequency: phase advances faster where the world is
    # changing faster, making phase a running integral of local motion
    omega_eff = state.omega
    if cfg.use_freq_modulation:
        omega_eff = state.omega * (1.0 + cfg.freq_mod
                                   * np.tanh(state.novelty - 1.0)[:, None])
    z_new, n_clip = dyn.step(state.z, omega_eff, state.mu, drive,
                             eta=state.eta, sub_steps=cfg.sub_steps,
                             noise=cfg.state_noise, rng=rng, r_max=cfg.r_max)
    state.z = z_new
    state.n_clipped += n_clip
    h_new = state.h
    h_amp_new = state.h_amp
    g_now = proto.global_context(state, h_new)

    # -- 3. predict --------------------------------------------------------
    self_pred = dyn.rollout(z_new, omega_eff, state.mu, drive, cfg.horizons,
                            eta=state.eta, sub_steps=cfg.sub_steps, r_max=cfg.r_max)
    rest = np.sqrt(np.maximum(state.mu, 0.0))
    self_pred = {tau: dyn.radius(z) - rest for tau, z in self_pred.items()}

    h_sens = h_new[state.sensory_idx]
    # a masked unit has no input to fall back on, so its base is zero and the head has
    # to produce the whole patch from state built out of its neighbours' messages
    base_s = np.where(mask[:, None], 0.0, s_seen)
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

    # -- 4/5. measure and adjust ------------------------------------------
    surprise = np.zeros(N)
    err_self_1 = None
    vote_task = None
    for i, tau in enumerate(cfg.horizons):
        dq_s, dq_m, dq_h, dq_g = (state.hist_pred[k][tau] for k in ("s", "m", "h", "g"))
        if len(dq_h) == tau and len(state.hist_h) >= tau:
            rho = 1.0 / tau
            h_past = state.hist_h[-tau]
            e_h = h_amp_new - dq_h[0]
            e_m = msg["e"] - dq_m[0]
            e_g = g_now[None, :] - dq_g[0]
            e_s = s_true - dq_s[0]           # scored against truth, not what was seen

            surprise += rho * cfg.beta[2] * np.einsum("nm,nm->n", e_h, e_h)
            surprise += rho * cfg.beta[1] * np.einsum("nk,nk->n", e_m, e_m)
            surprise += rho * cfg.beta[3] * np.einsum("nq,nq->n", e_g, e_g)
            np.add.at(surprise, state.sensory_idx,
                      rho * cfg.beta[0] * np.einsum("sp,sp->s", e_s, e_s))

            if learn:
                delta_rule(state.M[:, i], e_m, h_past, cfg.eta_head,
                           state.conf, cfg.weight_decay)
                delta_rule(state.C[:, i], e_g, h_past, cfg.eta_head,
                           state.conf, cfg.weight_decay)
                delta_rule(state.P[:, i], e_s, h_past[state.sensory_idx],
                           cfg.eta_head, state.conf[state.sensory_idx],
                           cfg.weight_decay)
            if tau == 1:
                err_self_1 = e_h
                if cfg.use_voting:
                    # how the heads wanted the vote to change: the only
                    # pressure that makes votes *differ*, since pure
                    # agreement has one stable outcome (everyone identical)
                    k = cfg.vote_dim
                    vote_task = np.einsum('nok,no->nk',
                                          state.M[:, i][:, :, -k:], e_m)
                    np.add.at(vote_task, state.sensory_idx,
                              np.einsum('sok,so->sk',
                                        state.P[:, i][:, :, -k:], e_s))

        dq_s.append(pred_s[tau])
        dq_m.append(pred_m[tau])
        dq_h.append(self_pred[tau])
        dq_g.append(pred_g[tau])

    if learn and err_self_1 is not None:
        _update_links(state, err_self_1, coeff, gamma, msg["e"], s_seen, z_prev)
        if cfg.use_voting and vote_err is not None:
            # two pressures on the proposal map, and both are needed:
            # agreement alone collapses the population onto one vote, task
            # alone never agrees with anybody.
            drive_y = cfg.vote_agree * vote_err
            if vote_task is not None:
                drive_y = drive_y + cfg.vote_task * vote_task
            state.V *= 1.0 - cfg.weight_decay
            state.V += (cfg.eta_vote * normaliser(h_amp_prev))[:, None, None] \
                       * drive_y[:, :, None] * h_amp_prev[:, None, :]

    d_surprise = state.prev_surprise - surprise
    state.prev_surprise = surprise
    state.surprise = surprise
    state.surprise_ema = ((1 - cfg.novelty_ema) * state.surprise_ema
                          + cfg.novelty_ema * surprise)
    state.novelty = surprise / (state.surprise_ema + 1e-8)
    if learn:
        if cfg.use_uncertainty:
            err = surprise - state.sigma_hat
            state.v *= 1.0 - cfg.weight_decay
            state.v += cfg.eta_sigma * err[:, None] * h_new
            state.v0 += cfg.eta_sigma * err
        resid = (msg["e"] - pred_m[cfg.horizons[0]]) ** 2
        state.credit *= 1.0 - cfg.credit_ema
        state.credit += cfg.credit_ema * (1.0 / (1.0 + resid)) * d_surprise[:, None]

    # -- 6. choose neighbours ---------------------------------------------
    state.hist_h.append(h_new.copy())
    state.hist_z.append(z_new.copy())
    state.nov_hist.append(state.novelty.copy())
    rewired = 0
    if (learn and cfg.use_rewiring and state.t > 0
            and state.t % cfg.rewire_every == 0 and len(state.nov_hist) >= 16):
        rewired = rewire(state, cfg, rng, np.stack(state.nov_hist))["n_rewired"]

    if state.t % cfg.coalition_every == 0 and len(state.hist_z) == cfg.coalition_window:
        state.coalition = proto.detect_coalitions(state, list(state.hist_z))
        if cfg.use_voting:
            state.vote_coalition = proto.vote_coalitions(state)

    # -- 7. broadcast -------------------------------------------------------
    out = proto.assemble_messages(state, self_pred[1])
    state.msg_e, state.msg_c = out["e"], out["c"]
    state.msg_n, state.msg_i = out["n"], out["i"]
    state.g = g_now
    state.t += 1

    return {
        "t": state.t,
        "surprise": float(surprise.mean()),
        "conf": float(state.conf.mean()),
        "radius": float(dyn.radius(z_new).mean()),
        "n_clipped": n_clip,
        "rewired": rewired,
        "masked": int(mask.sum()),
    }


def _update_links(state, err_self, coeff, gamma, msg_e, s_seen, z_prev) -> None:
    """Local rules for read vectors, gains and the sensory encoder.

    The error lives in amplitude space (m) but the read vectors live in rotor space (d =
    2m), so the two have to be related. Drive enters a rotor as a 2-vector and only its
    *radial* component changes that rotor's amplitude, to first order:
    `dr_j = eta * (zhat_j . drive_j)`. So an amplitude error pushes each read vector
    along the rotor's current radial direction, and the tangential component -- which
    moves phase, not content -- is deliberately left alone by this rule. Phase is shaped
    by the coupling, not by the amplitude error.
    """
    cfg = state.cfg
    N, D, m = state.n_units, cfg.degree, cfg.m
    r = np.sqrt(np.einsum("nmk,nmk->nm", z_prev, z_prev))
    zhat = z_prev / (r[..., None] + 1e-6)                     # (N, m, 2)

    scale = cfg.eta_link * state.conf * normaliser(r)         # (N,)
    gain = scale[:, None] * coeff                             # (N, D)

    a = state.a.reshape(N, D, m, 2)
    a *= 1.0 - cfg.weight_decay
    a += gain[:, :, None, None] * err_self[:, None, :, None] * zhat[:, None, :, :]

    # effective read vector in amplitude space, for the gain update
    a_eff = np.einsum("nkmj,nmj->nkm", a, zhat)               # (N, D, m)
    a_dot_err = np.einsum("nkm,nm->nk", a_eff, err_self)
    state.w *= 1.0 - cfg.weight_decay
    state.w += scale[:, None] * gamma * msg_e * a_dot_err
    np.clip(state.w, -3.0, 3.0, out=state.w)

    idx = state.sensory_idx
    s_scale = cfg.eta_link * state.conf[idx] * normaliser(r[idx])
    S = state.S_enc.reshape(len(idx), m, 2, -1)
    S *= 1.0 - cfg.weight_decay
    S += (s_scale[:, None, None, None]
          * err_self[idx][:, :, None, None]
          * zhat[idx][:, :, :, None]
          * s_seen[:, None, None, :])


def predicted_retina2(state, tau: int, sensors: Sensors) -> np.ndarray:
    return sensors.from_patches(state.last_pred_s[tau][:N_VISUAL])
