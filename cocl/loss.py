import torch
import torch.nn as nn
import torch.nn.functional as F


class COCLLoss(nn.Module):
    """Crack-oriented contrastive loss (Eq. 5-6 of the paper).

    Input features are ordered as [z^r_1, ..., z^r_B, z^a_1, ..., z^a_B].
    For each anchor (raw or crack-aware), the positive is the other view of
    the same instance, and the negatives are all views of samples from other
    classes. Same-class samples are excluded from the denominator.
    Similarity is the dot product scaled by the temperature, and the loss is
    averaged over all 2B anchors.
    """

    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        n = feats.size(0)
        b = n // 2
        idx = torch.arange(n, device=feats.device)
        pos_idx = (idx + b) % n

        sim = feats @ feats.T / self.temperature

        negative_mask = labels.view(-1, 1) != labels.view(1, -1)
        negative_mask[idx, pos_idx] = True  # keep the positive in the denominator
        logits = sim.masked_fill(~negative_mask, float('-inf'))

        return F.cross_entropy(logits, pos_idx)
