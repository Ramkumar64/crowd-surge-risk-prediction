import torch, torch.nn as nn, torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha  # tensor (C,) or None
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        loss = (1-pt)**self.gamma * ce
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum": return loss.sum()
        return loss