# Epileptor Neural Bifurcation Seizure Engine

Production-grade implementation of the **Jirsa et al. (2014) Epileptor** neural mass model used in The Virtual Brain (TVB).

## Features

- Canonical 5D (+ optional g) Epileptor ODEs with RK4 integration
- LFP proxy (`-x1 + x2`)
- Diffusively coupled multi-node networks
- Seizure onset detection from envelope + slow variable `z`
- Largest Lyapunov exponent (Benettin)
- Excitability `x0` bifurcation scan and critical-point heuristic

## Install

```bash
cd epileptor-neural-bifurcation-seizure-engine
pip install -e ".[dev]"
# or
PYTHONPATH=src pytest -q
```

## Quick start

```python
from epileptor_neural_bifurcation_seizure_engine import EpileptorParams, EpileptorSimulator

sim = EpileptorSimulator(params=EpileptorParams(x0=-1.6))
times, states = sim.integrate(t_max=500.0, dt=0.05)
lfp = sim.lfp(states)
```

## References

- Jirsa, V. K., Stacey, W. C., Quilichini, P. P., Ivanov, A. I., & Bernard, C. (2014).
  On the nature of seizure dynamics. *Brain*.
- Proix, T., et al. (2014). Permittivity coupling across brain regions determines seizure spreading.
