"""Epileptor Neural Bifurcation Seizure Engine.

Jirsa et al. (2014) 5D Epileptor mass model with RK4 integration,
coupled networks, Lyapunov exponents, and x0 bifurcation scans.
"""

from .models import (
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

__all__ = [
    "EpileptorParams",
    "EpileptorState",
    "EpileptorSimulator",
    "CoupledEpileptorNetwork",
    "NetworkConfig",
    "epileptor_rhs",
    "detect_seizure_onsets",
    "estimate_lyapunov_exponent",
    "bifurcation_x0_scan",
    "critical_x0_estimate",
]

__version__ = "0.1.0"
