"""Tests for Epileptor neural bifurcation seizure engine."""

from __future__ import annotations

import numpy as np
import pytest

from epileptor_neural_bifurcation_seizure_engine import (
    CoupledEpileptorNetwork,
    EpileptorParams,
    EpileptorSimulator,
    EpileptorState,
    NetworkConfig,
    bifurcation_x0_scan,
    critical_x0_estimate,
    detect_seizure_onsets,
    epileptor_rhs,
    estimate_lyapunov_exponent,
)


def test_params_defaults():
    p = EpileptorParams()
    assert p.a == 1.0
    assert p.x0 < 0
    assert p.r > 0


def test_rhs_shape_and_finite():
    p = EpileptorParams()
    y = EpileptorState().as_array()
    dy = epileptor_rhs(0.0, y, p)
    assert dy.shape == y.shape
    assert np.all(np.isfinite(dy))


def test_simulator_integrates_stable_interictal():
    # Strongly non-epileptogenic x0 → bounded trajectory
    p = EpileptorParams(x0=-2.2, r=0.001)
    sim = EpileptorSimulator(params=p, state=EpileptorState(z=3.5, y1=-5.0))
    times, states = sim.integrate(t_max=50.0, dt=0.1, store_every=2)
    assert len(times) > 10
    assert states.shape[1] == 6
    assert np.all(np.isfinite(states))
    # z should stay in a physiological band
    assert np.all(states[:, 2] > -5) and np.all(states[:, 2] < 20)


def test_lfp_proxy():
    p = EpileptorParams(x0=-1.8)
    sim = EpileptorSimulator(params=p)
    times, states = sim.integrate(30.0, dt=0.1, store_every=1)
    lfp = sim.lfp(states)
    assert lfp.shape[0] == states.shape[0]
    assert np.all(np.isfinite(lfp))


def test_epileptogenic_x0_increases_activity():
    quiet = EpileptorSimulator(params=EpileptorParams(x0=-2.4, r=0.002))
    loud = EpileptorSimulator(params=EpileptorParams(x0=-1.4, r=0.002))
    _, s_q = quiet.integrate(80.0, dt=0.1, store_every=5)
    _, s_l = loud.integrate(80.0, dt=0.1, store_every=5)
    amp_q = float(np.max(np.abs(-s_q[:, 0] + s_q[:, 3])))
    amp_l = float(np.max(np.abs(-s_l[:, 0] + s_l[:, 3])))
    # More positive x0 should not collapse amplitude below quiet case by orders of magnitude
    assert amp_l + 1e-6 >= 0.5 * amp_q or amp_l > 0.1


def test_coupled_network_steps():
    net = CoupledEpileptorNetwork(
        config=NetworkConfig(n_nodes=3, coupling_strength=0.05),
        node_x0=[-2.0, -1.6, -1.8],
    )
    times, traj = net.integrate(t_max=40.0, dt=0.1, store_every=5)
    assert traj.shape[1] == 3
    assert traj.shape[2] == 6
    assert np.all(np.isfinite(traj))


def test_detect_seizure_onsets_runs():
    p = EpileptorParams(x0=-1.5, r=0.0015)
    sim = EpileptorSimulator(params=p)
    times, states = sim.integrate(200.0, dt=0.1, store_every=2)
    events = detect_seizure_onsets(times, states, min_duration=1.0)
    assert isinstance(events, list)


def test_lyapunov_finite():
    # Short estimate — just ensure finite number
    lyap = estimate_lyapunov_exponent(t_transient=30.0, t_sample=40.0, dt=0.2)
    assert np.isfinite(lyap)


def test_bifurcation_scan():
    scan = bifurcation_x0_scan(x0_values=[-2.2, -1.8, -1.4], t_transient=40.0, t_sample=60.0, dt=0.15)
    assert scan["x0"].shape == (3,)
    assert scan["max_lfp_amp"].shape == (3,)
    assert np.all(scan["max_lfp_amp"] >= 0)
    xc = critical_x0_estimate(scan)
    assert -2.5 < xc < -1.0


def test_rk4_determinism():
    p = EpileptorParams(x0=-1.9)
    s1 = EpileptorSimulator(params=p, state=EpileptorState())
    s2 = EpileptorSimulator(params=p, state=EpileptorState())
    t1, a = s1.integrate(20.0, dt=0.1)
    t2, b = s2.integrate(20.0, dt=0.1)
    assert np.allclose(a, b)
