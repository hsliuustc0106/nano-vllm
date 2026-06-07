import torch
from torch import nn
import torch.nn.functional as F

try:
    import flashinfer
except ImportError:  # pragma: no cover - optional kernel acceleration
    flashinfer = None


class SiluAndMul(nn.Module):

    @torch.compile
    def _compiled_forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if flashinfer is not None and x.is_cuda:
            return flashinfer.silu_and_mul(x)
        return self._compiled_forward(x)
