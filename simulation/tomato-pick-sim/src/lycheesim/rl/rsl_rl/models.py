"""Custom RSL-RL model classes for LycheeSim.

Referenced as ``"lycheesim.rl.rsl_rl.models:ZeroInitMLPModel"``;
rsl_rl resolves the dotted path at construction time.
"""

from __future__ import annotations

import torch.nn as nn

from rsl_rl.models import MLPModel


class ZeroInitMLPModel(MLPModel):
    """MLPModel whose final linear layer can be zero-initialized.

    Ported from regrind: with a scalar/log Gaussian distribution the MLP emits the
    action mean directly, so zeroing the last layer makes the initial action mean
    exactly 0 for any observation — the policy starts as pure demo playback in
    residual-action tracking tasks.
    """

    def __init__(self, *args, zero_init_last_layer: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Must run after super().__init__, which may re-initialize MLP weights.
        if zero_init_last_layer:
            self._zero_init_last_layer()

    def _zero_init_last_layer(self) -> None:
        linears = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)]
        if not linears:
            raise ValueError(f"Expected an MLP with at least one nn.Linear, got {type(self.mlp)}")
        last = linears[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
