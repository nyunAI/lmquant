# -*- coding: utf-8 -*-
"""Phi3 model patcher."""

from typing import Any
import torch.nn as nn
from torch import Tensor, cat

__all__ = ["QKVProj"]


class QKVProj(nn.Module):
    """QKV projection layer.
    This class is used to represent the qkv_proj layer in the attention block.
    """
    # TODO: add conversion to and from torch.nn.Linear

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
    ):
        """Initialize the QKVProj."""
        super(QKVProj, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
    
    @classmethod
    def from_q_k_v(cls, q: nn.Linear, k: nn.Linear, v: nn.Linear):
        """Create a QKVProj from q, k, v Linear layers."""
        return cls(q.in_features, q.out_features + k.out_features + v.out_features, False, q, k, v)

    def forward(self, x: Tensor) -> Tensor:
        """Forward method."""
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        print("Dimension of fwd:", cat([q, k, v], dim=-1).shape)
        return cat([q, k, v], dim=-1)
