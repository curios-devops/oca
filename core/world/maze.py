"""World v4 — a maze with a moving agent and a local view.

Every world so far has been passive: the mesh watches, and nothing it does changes what
it sees next. The assessment's strongest hypothesis is that the missing ingredient is an
active control loop -- choose an action, predict its consequence, compare, update -- and
that object identity is learned through action rather than through vision.

A maze is the cheapest honest version of that loop, and it is worth building for a reason
beyond being watchable. The agent sees only a small window around itself, so the walls it
must navigate are mostly *not visible*. Representing structure that is currently out of
view is object permanence in spatial form, and unlike a ball behind an occluder it is
unambiguously required: you cannot navigate a maze you cannot remember.

Two deliberate limits keep this an experiment rather than a demo. The agent moves under a
fixed exploration policy, so this remains a *prediction* problem -- action and efference
copy, without action selection, which is the cheap majority of the sensorimotor loop. And
success is measured by probing whether the mesh state encodes the surrounding walls, not
by whether the agent reaches the exit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

WALL, FREE, EXIT, TUNNEL = 0, 1, 2, 3
ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))     # up, down, left, right
# paired so that `a ^ 1` is the reverse of action `a`


def _to_square(img: np.ndarray, px: int) -> np.ndarray:
    """Nearest-neighbour upsample to exactly px x px, centre-cropping any excess."""
    scale = int(np.ceil(px / img.shape[0]))
    big = np.kron(img, np.ones((scale, scale), dtype=np.float32))
    off = (big.shape[0] - px) // 2
    return big[off:off + px, off:off + px]


@dataclass
class MazeConfig:
    size: int = 21                    # odd, so the carve algorithm works
    view: int = 5                     # odd; the agent sees a view x view window
    seed: int = 0
    render_px: int = 64               # rendered frame is render_px square, to match Sensors
    wall_value: float = 0.15
    free_value: float = 0.55
    exit_value: float = 1.0
    agent_value: float = 0.85
    braid: float = 0.35
    """Fraction of interior walls knocked out after carving.

    Not decoration. A DFS-carved maze puts walls on strictly even coordinates, so
    once the agent's parity is visible the *entire* out-of-view neighbourhood is
    deducible without remembering anything -- measured: a raw-pixel probe scored
    94% on cells it could not see. Braiding destroys that regularity (and adds
    loops, so the maze stops being a tree), which is what makes out-of-view
    structure genuinely require memory."""

    p_random: float = 0.25
    """Exploration policy: mostly keep going, sometimes turn. A pure random walk
    re-treads the same few cells and the agent never sees enough maze to model."""

    tunnels: bool = False
    tunnel_frac: float = 0.30
    tunnel_min_len: int = 4
    tunnel_value: float = 0.05
    """Covered corridors: inside one the agent moves normally and still receives its
    efference copy, but every frame it sees is identical.

    This is a better spatial-memory test than the open maze, for a reason beyond being
    harder. In the open maze a 5x5 view largely identifies *where you are*, so a raw-pixel
    probe scores 77% on cells it cannot see and the control is weak. Inside a tunnel every
    view is the same by construction, so pixels carry exactly zero positional information
    and the control drops to chance. What is left is dead reckoning: position can only come
    from integrating your own moves since you went in."""


@dataclass
class MazeWorld:
    cfg: MazeConfig = field(default_factory=MazeConfig)

    def __post_init__(self) -> None:
        cfg = self.cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.grid = self._carve(cfg.size)
        self._braid(cfg.braid)
        free = np.argwhere(self.grid == FREE)
        start = free[self.rng.integers(len(free))]
        self.pos = np.array(start, dtype=int)      # (row, col)
        far = free[np.argmax(np.abs(free - start).sum(axis=1))]
        self.grid[far[0], far[1]] = EXIT
        self.exit_pos = np.array(far, dtype=int)
        # tunnels are laid after the exit exists, so the exit can be excluded from them
        self.tunnel = (self._lay_tunnels() if cfg.tunnels
                       else np.zeros_like(self.grid, dtype=bool))
        self.heading = int(self.rng.integers(4))
        self.t = 0
        self.last_action = self.heading
        self.last_blocked = False
        self.n_reached = 0
        self._tunnel_steps = 0
        self._tunnel_entry = None
        # the somatic channel Sensors expects: here it carries the efference copy
        # (which way the agent just tried to move, and whether it was blocked)
        self.contact = np.zeros(16, dtype=np.float32)
        self._set_contact()
        self._update_tunnel()

    # ------------------------------------------------------------- generation

    def _carve(self, n: int) -> np.ndarray:
        """Randomised depth-first maze carving on an odd grid."""
        g = np.full((n, n), WALL, dtype=int)
        stack = [(1, 1)]
        g[1, 1] = FREE
        while stack:
            r, c = stack[-1]
            nb = []
            for dr, dc in ((0, 2), (0, -2), (2, 0), (-2, 0)):
                rr, cc = r + dr, c + dc
                if 1 <= rr < n - 1 and 1 <= cc < n - 1 and g[rr, cc] == WALL:
                    nb.append((rr, cc, dr, dc))
            if not nb:
                stack.pop()
                continue
            rr, cc, dr, dc = nb[self.rng.integers(len(nb))]
            g[r + dr // 2, c + dc // 2] = FREE
            g[rr, cc] = FREE
            stack.append((rr, cc))
        return g

    def _braid(self, frac: float) -> None:
        if frac <= 0:
            return
        n = self.cfg.size
        walls = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)
                 if self.grid[r, c] == WALL]
        k = int(len(walls) * frac)
        for idx in self.rng.choice(len(walls), size=k, replace=False):
            r, c = walls[idx]
            self.grid[r, c] = FREE

    def _lay_tunnels(self) -> np.ndarray:
        """Cover runs of corridor, so a tunnel is a stretch rather than a single cell.

        Length matters: a one-cell tunnel is bridged by any model that can remember one
        step, and the question is how far dead reckoning survives. Runs are grown along
        whichever axis is locally open.
        """
        cfg = self.cfg
        n = cfg.size
        tun = np.zeros((n, n), dtype=bool)
        free = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)
                if self.grid[r, c] == FREE]
        target = int(len(free) * cfg.tunnel_frac)
        self.rng.shuffle(free)

        for (r, c) in free:
            if tun.sum() >= target:
                break
            if tun[r, c] or self.grid[r, c] != FREE:
                continue
            for dr, dc in ((0, 1), (1, 0)):
                run = []
                rr, cc = r, c
                while (0 <= rr < n and 0 <= cc < n and self.grid[rr, cc] == FREE
                       and not tun[rr, cc] and len(run) < cfg.tunnel_min_len * 2):
                    run.append((rr, cc))
                    rr, cc = rr + dr, cc + dc
                if len(run) >= cfg.tunnel_min_len:
                    for (tr, tc) in run:
                        tun[tr, tc] = True
                    break
        # the exit is never covered: the goal has to stay recognisable when reached
        tun[self.exit_pos[0], self.exit_pos[1]] = False
        return tun

    # ---------------------------------------------------------------- dynamics

    def _free(self, r, c) -> bool:
        n = self.cfg.size
        return 0 <= r < n and 0 <= c < n and self.grid[r, c] != WALL

    def choose_action(self) -> int:
        """Fixed exploration policy: go straight, else turn, and never double back
        unless the corridor is a dead end.

        The no-backtracking rule is doing real work. A plain random walk oscillates in
        one corridor -- measured: 27 of 199 cells visited in 4000 steps and the exit
        never reached, which would leave the mesh modelling a tiny fragment of the maze.
        """
        reverse = self.heading ^ 1          # actions are paired N/S, W/E
        if self.rng.random() > self.cfg.p_random:
            dr, dc = ACTIONS[self.heading]
            if self._free(self.pos[0] + dr, self.pos[1] + dc):
                return self.heading
        opts = [a for a, (dr, dc) in enumerate(ACTIONS)
                if self._free(self.pos[0] + dr, self.pos[1] + dc)]
        forward = [a for a in opts if a != reverse]
        if forward:
            return int(self.rng.choice(forward))
        return int(opts[0]) if opts else self.heading

    def step(self, action: int | None = None) -> None:
        if action is None:
            action = self.choose_action()
        dr, dc = ACTIONS[action]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        self.last_action = action
        if self._free(r, c):
            self.pos = np.array([r, c])
            self.heading = action
            self.last_blocked = False
        else:
            self.last_blocked = True
        self._set_contact()
        if tuple(self.pos) == tuple(self.exit_pos):
            self.n_reached += 1
            free = np.argwhere(self.grid == FREE)
            self.pos = np.array(free[self.rng.integers(len(free))], dtype=int)
        self._update_tunnel()
        self.t += 1

    def _update_tunnel(self) -> None:
        """Track the mouth the agent walked in through, and how long it has been blind.

        Runs after any exit teleport: landing straight inside a tunnel still needs an
        entry point recorded, or the frozen-at-entry baseline has nothing to freeze.
        """
        if self.in_tunnel():
            if self._tunnel_entry is None:
                self._tunnel_entry = tuple(self.pos)
            self._tunnel_steps += 1
        else:
            self._tunnel_entry = None
            self._tunnel_steps = 0

    def _set_contact(self) -> None:
        self.contact[:] = 0.0
        self.contact[:5] = self.efference()

    # ------------------------------------------------------------------ senses

    def in_tunnel(self) -> bool:
        return bool(self.cfg.tunnels and self.tunnel[self.pos[0], self.pos[1]])

    def steps_in_tunnel(self) -> int:
        """How many moves since the agent lost sight of where it is. 0 when in the open."""
        return self._tunnel_steps

    def local_view(self) -> np.ndarray:
        """(view, view) window centred on the agent. Out-of-bounds reads as wall.

        Inside a tunnel this returns a constant, identical on every tick and at every
        tunnel cell, so nothing about position survives into the image.
        """
        v = self.cfg.view
        h = v // 2
        if self.in_tunnel():
            return np.full((v, v), TUNNEL, dtype=int)
        out = np.full((v, v), WALL, dtype=int)
        for i, dr in enumerate(range(-h, h + 1)):
            for j, dc in enumerate(range(-h, h + 1)):
                r, c = self.pos[0] + dr, self.pos[1] + dc
                if 0 <= r < self.cfg.size and 0 <= c < self.cfg.size:
                    cell = self.grid[r, c]
                    # a covered corridor looks like a dark mouth from outside; the agent
                    # can see that a tunnel is there, just not into it
                    out[i, j] = TUNNEL if self.tunnel[r, c] else cell
        return out

    def efference(self) -> np.ndarray:
        """One-hot action plus a blocked flag -- the copy of what the agent just did.

        Without this the mesh cannot tell "the world moved" from "I moved", which is the
        distinction the whole sensorimotor story rests on.
        """
        e = np.zeros(len(ACTIONS) + 1, dtype=np.float32)
        e[self.last_action] = 1.0
        e[-1] = float(self.last_blocked)
        return e

    def render(self, egocentric: bool = True) -> np.ndarray:
        """Square float image for the retina: the local view, scaled up."""
        cfg = self.cfg
        cell = self.local_view() if egocentric else self.grid
        lut = {WALL: cfg.wall_value, FREE: cfg.free_value, EXIT: cfg.exit_value,
               TUNNEL: cfg.tunnel_value}
        img = np.vectorize(lut.get)(cell).astype(np.float32)
        if egocentric and not self.in_tunnel():
            h = cfg.view // 2
            img[h, h] = cfg.agent_value
        return _to_square(img, cfg.render_px)

    def full_render(self) -> np.ndarray:
        """God's-eye view, for the visualisation only -- never fed to the mesh."""
        cfg = self.cfg
        lut = {WALL: cfg.wall_value, FREE: cfg.free_value, EXIT: cfg.exit_value,
               TUNNEL: cfg.tunnel_value}
        img = np.vectorize(lut.get)(self.grid).astype(np.float32)
        img[self.tunnel] = cfg.tunnel_value
        img[self.pos[0], self.pos[1]] = cfg.agent_value
        return img

    # ------------------------------------------------------------------ probes

    def surrounding_walls(self, radius: int = 3) -> np.ndarray:
        """Ground truth for the probe: walls in a neighbourhood wider than the view.

        Deliberately wider than `view`, so the outer ring is *not currently visible* and
        can only be answered from memory of where the agent has been.
        """
        out = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = self.pos[0] + dr, self.pos[1] + dc
                inside = 0 <= r < self.cfg.size and 0 <= c < self.cfg.size
                out.append(0.0 if (inside and self.grid[r, c] != WALL) else 1.0)
        return np.array(out, dtype=np.float32)

    def exit_direction(self) -> np.ndarray:
        """Unit vector from the agent toward the exit -- never visible locally."""
        d = (self.exit_pos - self.pos).astype(float)
        n = np.linalg.norm(d)
        return d / n if n > 0 else d

    def state_snapshot(self) -> dict:
        return {
            "t": self.t,
            "pos": self.pos.copy(),
            "heading": self.heading,
            "action": self.last_action,
            "blocked": self.last_blocked,
            "reached": self.n_reached,
            "in_tunnel": self.in_tunnel(),
            "tunnel_steps": self._tunnel_steps,
            "tunnel_entry": self._tunnel_entry,
        }


def make_maze_world(seed: int = 0, cfg: MazeConfig | None = None) -> MazeWorld:
    base = cfg if cfg is not None else MazeConfig()
    return MazeWorld(MazeConfig(**{**base.__dict__, "seed": seed}))
