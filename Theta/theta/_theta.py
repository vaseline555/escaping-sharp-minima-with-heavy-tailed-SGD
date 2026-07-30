"""Theta: Truncated heavy-tailed noise injection for SGD optimizers.

Wraps any PyTorch optimizer with Lomax (Type-II Pareto) noise sampling
and optional gradient clipping. The user controls the training loop;
Theta handles noise sampling, tail-gating, clipping, and stepping.
"""

from dataclasses import dataclass
from typing import Optional, Type

import torch
from torch.optim import Optimizer


def sample_lomax(alpha: float, scale: float) -> float:
    """Sample from Lomax(alpha, scale), i.e. scale * (Pareto(alpha) - 1)."""
    u = torch.rand(()).item()
    return scale * ((1.0 - u) ** (-1.0 / alpha) - 1.0)


def lomax_cdf(value: float, alpha: float, scale: float) -> float:
    """Evaluate the Lomax CDF at *value*."""
    return 1.0 - (1.0 + value / scale) ** (-alpha)


@dataclass(frozen=True)
class NoiseStep:
    """Result of a single noise sample: magnitude, CDF percentile, and
    whether the tail-gating threshold was exceeded."""

    value: float
    cdf: float
    active: bool


class Theta:
    """Truncated heavy-tailed optimizer wrapper.

    Parameters
    ----------
    params : iterable
        Model parameters (same as any ``torch.optim`` constructor).
    base : type
        Optimizer class to wrap, e.g. ``torch.optim.SGD``.
    lr : float
        Learning rate (forwarded to *base*).
    alpha : float
        Lomax shape parameter (tail index).
    scale : float
        Lomax scale parameter.
    clip : float or None
        Max gradient norm.  ``None`` disables clipping.
    tail_prob : float
        CDF threshold for tail-gating.  Noise is injected only when the
        sampled CDF value ≥ *tail_prob*.  Set to ``0.0`` to always inject.
    **base_kwargs
        Extra keyword arguments forwarded to *base*
        (e.g. ``momentum``, ``weight_decay``).

    Example
    -------
    >>> optimizer = Theta(model.parameters(), base=SGD, lr=0.1,
    ...                  alpha=1.4, scale=0.5, clip=1.0, tail_prob=0.95)
    >>> noise = optimizer.sample_noise()
    >>> optimizer.zero_grad()
    >>> if noise.active:
    ...     ((-noise.value) * loss_minus).backward()
    ...     (noise.value * loss_plus).backward()
    >>> loss.backward()
    >>> optimizer.step()
    """

    def __init__(
        self,
        params,
        base: Type[Optimizer],
        lr: float,
        alpha: float,
        scale: float,
        clip: Optional[float] = None,
        tail_prob: float = 0.0,
        **base_kwargs,
    ):
        if alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if scale <= 0.0:
            raise ValueError(f"scale must be positive, got {scale}")
        if clip is not None and clip <= 0.0:
            raise ValueError(f"clip must be positive, got {clip}")
        if not 0.0 <= tail_prob <= 1.0:
            raise ValueError(f"tail_prob must be in [0, 1], got {tail_prob}")

        self.base = base(params, lr=lr, **base_kwargs)
        self.alpha = float(alpha)
        self.scale = float(scale)
        self.clip = float(clip) if clip is not None else None
        self.tail_prob = float(tail_prob)

    # ------------------------------------------------------------------
    # Delegated optimizer interface
    # ------------------------------------------------------------------

    @property
    def param_groups(self):
        return self.base.param_groups

    @property
    def _params(self):
        return [p for g in self.param_groups for p in g["params"]]

    def zero_grad(self, set_to_none: bool = True):
        self.base.zero_grad(set_to_none=set_to_none)

    # ------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------

    def sample_noise(self, use_noise: bool = True) -> NoiseStep:
        """Sample Lomax noise and decide whether to inject it.

        Returns a :class:`NoiseStep` whose *active* flag reflects both
        *use_noise* and the tail-gating threshold.
        """
        if not use_noise:
            return NoiseStep(value=0.0, cdf=0.0, active=False)
        z = sample_lomax(self.alpha, self.scale)
        cdf = lomax_cdf(z, self.alpha, self.scale)
        return NoiseStep(value=z, cdf=cdf, active=(cdf >= self.tail_prob))

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self):
        """Clip gradients (if configured) then step the base optimizer."""
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self._params, self.clip)
        self.base.step()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self):
        return {
            "base": self.base.state_dict(),
            "alpha": self.alpha,
            "scale": self.scale,
            "clip": self.clip,
            "tail_prob": self.tail_prob,
        }

    def load_state_dict(self, state_dict):
        self.base.load_state_dict(state_dict["base"])
        self.alpha = float(state_dict["alpha"])
        self.scale = float(state_dict["scale"])
        self.clip = state_dict["clip"]
        self.tail_prob = float(state_dict["tail_prob"])
