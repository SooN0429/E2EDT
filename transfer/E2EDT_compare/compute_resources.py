#!/usr/bin/env python3
"""Wall-clock + FLOPs accounting helpers for distill vs from-scratch."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn


def require_thop():
    try:
        from thop import profile  # type: ignore

        return profile
    except ImportError as e:
        raise ImportError(
            "FLOPs accounting requires the 'thop' package. "
            "Install with: pip install thop"
        ) from e


@torch.no_grad()
def measure_forward_flops(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    example_input: torch.Tensor,
    model_for_profile: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """
    Measure FLOPs for one forward pass.

    When ``model_for_profile`` is set, register it as a child module so thop
    can hook its layers; ``model_fn`` still defines the call path (e.g. predict).
    """
    profile = require_thop()
    device = example_input.device
    x = example_input.detach()

    class _Wrap(nn.Module):
        def __init__(self, core: Optional[nn.Module], fn: Callable):
            super().__init__()
            if core is not None:
                self.core = core
            self._fn = fn

        def forward(self, inp: torch.Tensor) -> torch.Tensor:
            return self._fn(inp)

    core = model_for_profile
    was_training = None
    if core is not None:
        was_training = core.training
        core.eval()

    wrap = _Wrap(core, model_fn).to(device)
    flops, params = profile(wrap, inputs=(x,), verbose=False)

    if core is not None and was_training is not None:
        core.train(was_training)

    return {
        "forward_flops": float(flops),
        "params": float(params),
        "input_shape": list(example_input.shape),
    }


def estimate_train_flops(
    forward_flops: float,
    n_steps: int,
    backward_multiplier: float = 2.0,
) -> Dict[str, Any]:
    """
    Training FLOPs ≈ forward_flops * n_steps * (1 + backward_multiplier).

    Default backward_multiplier=2 → total ≈ 3× forward per step (common convention).
    """
    per_step = float(forward_flops) * (1.0 + float(backward_multiplier))
    total = per_step * int(n_steps)
    return {
        "forward_flops_per_step": float(forward_flops),
        "backward_multiplier": float(backward_multiplier),
        "flops_per_train_step": per_step,
        "n_train_steps": int(n_steps),
        "total_train_flops": total,
    }


class WallClock:
    def __init__(self) -> None:
        self.t0: Optional[float] = None
        self.elapsed_sec: float = 0.0

    def __enter__(self) -> "WallClock":
        self.t0 = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        assert self.t0 is not None
        self.elapsed_sec = time.time() - self.t0


def pack_resource_report(
    *,
    wall_clock_sec: float,
    n_train_steps: int,
    forward_flops: Optional[float],
    epoch_times: Optional[list] = None,
    extra: Optional[Dict[str, Any]] = None,
    backward_multiplier: float = 2.0,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "wall_clock_sec": float(wall_clock_sec),
        "n_train_steps": int(n_train_steps),
        "epoch_times_sec": list(epoch_times or []),
    }
    if forward_flops is not None:
        out["flops"] = estimate_train_flops(
            forward_flops, n_train_steps, backward_multiplier=backward_multiplier
        )
    else:
        out["flops"] = None
    if extra:
        out.update(extra)
    return out
