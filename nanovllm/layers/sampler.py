import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        noise = torch.empty_like(logits).exponential_(1).log_().neg_()
        sample_tokens = logits.add_(noise).argmax(dim=-1)
        return sample_tokens
