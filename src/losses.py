"""
Funcoes de perda para segmentacao com desbalanceamento de classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import TverskyLoss


class TverskyFocalLoss(nn.Module):
    """
    Combinacao de Tversky Loss + Focal Loss para dados desbalanceados.

    Tversky com alpha=0.3, beta=0.7 penaliza mais os falsos negativos,
    favorecendo recall — importante quando a classe positiva e rara (~2-3%).
    Focal Loss concentra o gradiente em pixels dificeis.

    Loss = 0.7 * Tversky + 0.3 * Focal
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 2.0):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, sigmoid=True)
        self.gamma   = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        tversky_loss = self.tversky(logits, targets)

        probs  = torch.sigmoid(logits)
        pt     = targets * probs + (1 - targets) * (1 - probs)
        focal  = ((1 - pt) ** self.gamma * F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )).mean()

        return 0.7 * tversky_loss + 0.3 * focal
