from __future__ import annotations

import torch

from common_efficiency import parameter_count
from fusion import build_fusion


def run():
    for name in ("mean", "concat", "gate", "interaction"):
        module = build_fusion(name)
        a = torch.randn(8, 64, requires_grad=True)
        b = torch.randn(8, 64, requires_grad=True)
        z = module(a, b)
        assert z.shape == (8, 64)
        z.square().mean().backward()
        assert a.grad is not None and b.grad is not None
        assert parameter_count(module)["TOTAL_PARAMS"] >= 0
    print("FUSION_STATIC_TEST=PASS")


if __name__ == "__main__":
    run()
