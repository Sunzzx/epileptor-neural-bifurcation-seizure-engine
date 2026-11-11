"""Epileptor 5D & 6D dynamical systems and coupled multi-region network models."""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class EpileptorParams:
    """Canonical parameters for Jirsa et al. (2014) 5-variable Epileptor."""
    a: float = 1.0
    b: float = 3.0
    c: float = 1.0
    d: float = 5.0
    r: float …
