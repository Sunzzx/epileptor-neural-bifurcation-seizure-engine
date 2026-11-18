"""Epileptor neural mass dynamics (Jirsa et al. 2014) with bifurcation analysis.

Implements the canonical 5D Epileptor ODEs used in The Virtual Brain (TVB):
  - fast subsystem (x1, y1): spike-wave / fast oscillations
  - slow permittivity variable z: seizure onset-offset bistability
  - spike-and-wave subsystem (x2, y2) + low-pass g

Plus RK4 integration, coupled multi-node networks, Lyapunov/bifurcation
utilities, and seizure-onset detection from the slow manifold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np


@dataclass
class EpileptorParams:
    """Canonical parameters for Jirsa et al. (2014) 5-variable Epileptor."""

    a: float = 1.0
    b: float = 3.0
    c: float = 1.0
    d: float = 5.0
    r: float = 0.00035
    s: float = 4.0
    x0: float = -1.6
    Irest1: float = 3.1
    Irest2: float = 0.45
    tau2: float = 10.0
    gamma: float = 0.01


@dataclass
class EpileptorState:
    """State vector (x1, y1, z, x2, y2, g)."""

    x1: float = 0.0
    y1: float = -5.0
    z: float = 3.0
    x2: float = 0.0
    y2: float = 0.0
    g: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.x1, self.y1, self.z, self.x2, self.y2, self.g], dtype=float
        )

    @classmethod
    def from_array(cls, v: np.ndarray) -> "EpileptorState":
        v = np.asarray(v, dtype=float).ravel()
        g = float(v[5]) if v.size > 5 else 0.0
        return cls(float(v[0]), float(v[1]), float(v[2]), float(v[3]), float(v[4]), g)


def epileptor_rhs(
    t: float,
    y: np.ndarray,
    p: EpileptorParams,
    coupling: float = 0.0,
) -> np.ndarray:
    """TVB-aligned Epileptor RHS (Jirsa 2014 / Proix 2014)."""
    del t  # autonomous
    x1, y1, z, x2, y2 = float(y[0]), float(y[1]), float(y[2]), float(y[3]), float(y[4])
    g = float(y[5]) if y.size > 5 else 0.0

    # Soft clamps keep the stiff polynomial terms in a physical band (TVB-safe).
    x1c = float(np.clip(x1, -5.0, 5.0))
    x2c = float(np.clip(x2, -5.0, 5.0))
    zc = float(np.clip(z, -2.0, 10.0))
    gc = float(np.clip(g, -5.0, 5.0))

    # Fast subsystem
    if x1c < 0.0:
        dx1 = y1 - (x1c**3 + 2.0 * x1c**2) - zc + p.Irest1 + coupling
    else:
        dx1 = y1 + (0.6 * (zc - 4.0) ** 2 - x1c) * x1c - zc + p.Irest1 + coupling
    dy1 = p.c - p.d * x1c**2 - y1

    # Slow permittivity (seizure energy). z^7 is extremely stiff — clip base.
    z7 = zc**7 if abs(zc) < 5.0 else np.sign(zc) * (5.0**7)
    dz = p.r * (p.s * (x1c - p.x0) - zc - 0.1 * z7)

    # Spike-and-wave subsystem
    dx2 = -y2 + x2c - x2c**3 + p.Irest2 + 2.0 * gc - 0.3 * (zc - 3.5)
    dy2 = (1.0 / p.tau2) * (-y2 + p.gamma * (x2c + 0.25))

    # Low-pass coupling filter
    dg = -0.01 * (gc - 0.1 * x1c)

    out = np.array([dx1, dy1, dz, dx2, dy2, dg], dtype=float)
    return out[: y.size]


def rk4_step(
    fun: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    dt: float,
) -> np.ndarray:
    k1 = fun(t, y)
    k2 = fun(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = fun(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = fun(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class EpileptorSimulator:
    """Single-node Epileptor integrator with optional external current."""

    def __init__(
        self,
        params: Optional[EpileptorParams] = None,
        state: Optional[EpileptorState] = None,
    ):
        self.params = params or EpileptorParams()
        self.state = state or EpileptorState()
        self.time = 0.0

    def rhs(self, t: float, y: np.ndarray, coupling: float = 0.0) -> np.ndarray:
        return epileptor_rhs(t, y, self.params, coupling=coupling)

    def step(self, dt: float = 0.05, coupling: float = 0.0) -> np.ndarray:
        y = self.state.as_array()
        y_new = rk4_step(lambda t, yy: self.rhs(t, yy, coupling), self.time, y, dt)
        # Hard state clamps after each step (prevents runaway RK4 stages).
        y_new = np.asarray(y_new, dtype=float)
        y_new[0] = float(np.clip(y_new[0], -5.0, 5.0))
        y_new[1] = float(np.clip(y_new[1], -50.0, 50.0))
        y_new[2] = float(np.clip(y_new[2], -2.0, 10.0))
        y_new[3] = float(np.clip(y_new[3], -5.0, 5.0))
        y_new[4] = float(np.clip(y_new[4], -50.0, 50.0))
        if y_new.size > 5:
            y_new[5] = float(np.clip(y_new[5], -5.0, 5.0))
        self.state = EpileptorState.from_array(y_new)
        self.time += dt
        return y_new

    def integrate(
        self,
        t_max: float,
        dt: float = 0.05,
        coupling_fn: Optional[Callable[[float], float]] = None,
        store_every: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_steps = int(np.ceil(t_max / dt))
        times: list[float] = []
        states: list[np.ndarray] = []
        for i in range(n_steps):
            c = float(coupling_fn(self.time)) if coupling_fn else 0.0
            y = self.step(dt, coupling=c)
            if i % store_every == 0:
                times.append(self.time)
                states.append(y.copy())
        return np.asarray(times), np.asarray(states)

    def lfp(self, states: np.ndarray) -> np.ndarray:
        """Local field potential proxy: -x1 + x2 (TVB convention)."""
        states = np.asarray(states)
        return -states[:, 0] + states[:, 3]


@dataclass
class NetworkConfig:
    n_nodes: int = 3
    coupling_strength: float = 0.1
    weights: Optional[np.ndarray] = None


class CoupledEpileptorNetwork:
    """Diffusively coupled Epileptor nodes (linear coupling on x1)."""

    def __init__(
        self,
        config: Optional[NetworkConfig] = None,
        params: Optional[EpileptorParams] = None,
        node_x0: Optional[Sequence[float]] = None,
    ):
        self.config = config or NetworkConfig()
        n = self.config.n_nodes
        base = params or EpileptorParams()
        x0s = list(node_x0) if node_x0 is not None else [base.x0] * n
        self.params_list = [
            EpileptorParams(**{**base.__dict__, "x0": float(x0s[i])}) for i in range(n)
        ]
        self.states = np.zeros((n, 6), dtype=float)
        for i in range(n):
            self.states[i] = EpileptorState(z=3.0 + 0.1 * i, y1=-5.0).as_array()
        if self.config.weights is None:
            W = np.zeros((n, n))
            for i in range(n):
                W[i, (i - 1) % n] = 1.0
                W[i, (i + 1) % n] = 1.0
            rs = W.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1.0
            self.W = W / rs
        else:
            self.W = np.asarray(self.config.weights, dtype=float)
        self.time = 0.0
        self.K = float(self.config.coupling_strength)

    def step(self, dt: float = 0.05) -> np.ndarray:
        x1 = self.states[:, 0]
        coupling = self.K * ((self.W @ x1) - x1)
        new_states = np.zeros_like(self.states)
        for i in range(self.config.n_nodes):
            p = self.params_list[i]
            y = self.states[i]

            def fun(t, yy, _i=i):
                return epileptor_rhs(t, yy, p, coupling=float(coupling[_i]))

            yi = rk4_step(fun, self.time, y, dt)
            yi[0] = float(np.clip(yi[0], -5.0, 5.0))
            yi[1] = float(np.clip(yi[1], -50.0, 50.0))
            yi[2] = float(np.clip(yi[2], -2.0, 10.0))
            yi[3] = float(np.clip(yi[3], -5.0, 5.0))
            yi[4] = float(np.clip(yi[4], -50.0, 50.0))
            yi[5] = float(np.clip(yi[5], -5.0, 5.0))
            new_states[i] = yi
        self.states = new_states
        self.time += dt
        return self.states.copy()

    def integrate(
        self, t_max: float, dt: float = 0.05, store_every: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_steps = int(np.ceil(t_max / dt))
        times, traj = [], []
        for i in range(n_steps):
            s = self.step(dt)
            if i % store_every == 0:
                times.append(self.time)
                traj.append(s.copy())
        return np.asarray(times), np.asarray(traj)


def detect_seizure_onsets(
    times: np.ndarray,
    states: np.ndarray,
    z_threshold: float = 2.5,
    min_duration: float = 5.0,
) -> list[dict]:
    """Detect ictal-like intervals via LFP envelope + low z."""
    st = states[:, 0, :] if states.ndim == 3 else states
    lfp = -st[:, 0] + st[:, 3]
    z = st[:, 2]
    win = max(5, int(len(lfp) * 0.01))
    kernel = np.ones(win) / win
    env = np.convolve(np.abs(lfp), kernel, mode="same")
    thr = np.median(env) + 2.5 * (np.median(np.abs(env - np.median(env))) + 1e-9)
    ictal = (env > thr) & (z < z_threshold)
    events: list[dict] = []
    i = 0
    while i < len(ictal):
        if not ictal[i]:
            i += 1
            continue
        j = i
        while j < len(ictal) and ictal[j]:
            j += 1
        t0, t1 = float(times[i]), float(times[min(j - 1, len(times) - 1)])
        if t1 - t0 >= min_duration:
            events.append({"onset": t0, "offset": t1, "duration": t1 - t0})
        i = j
    return events


def estimate_lyapunov_exponent(
    params: Optional[EpileptorParams] = None,
    dt: float = 0.05,
    t_transient: float = 200.0,
    t_sample: float = 800.0,
    delta0: float = 1e-8,
) -> float:
    """Benettin algorithm: largest Lyapunov exponent."""
    p = params or EpileptorParams()
    sim = EpileptorSimulator(params=p)
    sim.integrate(t_transient, dt=dt, store_every=10**9)
    y = sim.state.as_array()
    rng = np.random.default_rng(0)
    d = rng.normal(size=y.size)
    d *= delta0 / (np.linalg.norm(d) + 1e-30)
    n_steps = int(t_sample / dt)
    s = 0.0
    t = sim.time
    for _ in range(n_steps):
        y1 = rk4_step(lambda tt, yy: epileptor_rhs(tt, yy, p), t, y, dt)
        y2 = rk4_step(lambda tt, yy: epileptor_rhs(tt, yy, p), t, y + d, dt)
        d = y2 - y1
        dist = np.linalg.norm(d) + 1e-30
        s += np.log(dist / delta0)
        d *= delta0 / dist
        y = y1
        t += dt
    return float(s / (n_steps * dt))


def bifurcation_x0_scan(
    x0_values: Optional[Sequence[float]] = None,
    t_transient: float = 150.0,
    t_sample: float = 250.0,
    dt: float = 0.1,
) -> dict:
    """Scan excitability x0 → max |LFP| and mean z."""
    if x0_values is None:
        x0_values = np.linspace(-2.2, -1.2, 11)
    max_amp, mean_z = [], []
    for x0 in x0_values:
        p = EpileptorParams(x0=float(x0))
        sim = EpileptorSimulator(params=p)
        sim.integrate(t_transient, dt=dt, store_every=10**9)
        _, states = sim.integrate(t_sample, dt=dt, store_every=5)
        lfp = sim.lfp(states)
        max_amp.append(float(np.max(np.abs(lfp))))
        mean_z.append(float(np.mean(states[:, 2])))
    return {
        "x0": np.asarray(list(x0_values), dtype=float),
        "max_lfp_amp": np.asarray(max_amp),
        "mean_z": np.asarray(mean_z),
    }


def critical_x0_estimate(scan: Optional[dict] = None) -> float:
    """Heuristic critical excitability where LFP amp jumps."""
    scan = scan or bifurcation_x0_scan()
    amp = scan["max_lfp_amp"]
    x0 = scan["x0"]
    d = np.diff(amp)
    idx = int(np.argmax(d))
    return float(0.5 * (x0[idx] + x0[idx + 1]))
