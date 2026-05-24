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





class WeightedTverskyFocalLoss(nn.Module):
    """
    Variante de TverskyFocalLoss que incorpora pixel-wise weight maps
    gerados pelo esquema da U-Net (pixelWeights.py).
 
    O termo Focal recebe os pesos diretamente — cada pixel tem sua
    contribuicao na BCE escalada por w(x).  O termo Tversky opera sobre
    toda a predicao normalmente (ele ja e espacialmente agregado por
    natureza e nao tem uma formulacao pixel-wise direta).
 
    Loss = lambda_t * Tversky + lambda_f * mean( w(x) * FocalBCE(x) )
 
    Parametros
    ----------
    alpha, beta  : parametros do Tversky (beta > 0.5 penaliza FN)
    gamma        : expoente focal
    lambda_t     : peso do termo Tversky  (default 0.7)
    lambda_f     : peso do termo Focal    (default 0.3)
    normalize_w  : se True, normaliza os pesos para media=1 dentro do
                   batch antes de aplicar, evitando que a escala absoluta
                   dos weight maps altere a magnitude total da loss
 
    Uso
    ---
    criterion = WeightedTverskyFocalLoss()
 
    # weight_map vem do DataLoader como tensor float32 (B, 1, H, W)
    # gerado por pixelWeights.compute_weight_map()
    loss = criterion(logits, targets, weight_map)
    """
 
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 2.0,
        lambda_t: float = 0.7,
        lambda_f: float = 0.3,
        normalize_w: bool = True,
    ):
        super().__init__()
        self.tversky    = TverskyLoss(alpha=alpha, beta=beta, sigmoid=True)
        self.gamma      = gamma
        self.lambda_t   = lambda_t
        self.lambda_f   = lambda_f
        self.normalize_w = normalize_w
 
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weight_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parametros
        ----------
        logits     : (B, 1, H, W)  — saida bruta da rede
        targets    : (B, 1, H, W)  — ground truth binario
        weight_map : (B, 1, H, W)  — peso por pixel (float32, >= 0)
        """
        # --- Tversky (agregado espacialmente, sem pesos) ------------------
        tversky_loss = self.tversky(logits, targets)
 
        # --- Focal ponderado por pixel ------------------------------------
        probs = torch.sigmoid(logits)
        pt    = targets * probs + (1 - targets) * (1 - probs)
 
        # BCE pixel-wise sem reducao: shape (B, 1, H, W)
        bce_map = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
 
        # Modulacao focal: (1 - pt)^gamma * BCE
        focal_map = (1 - pt) ** self.gamma * bce_map  # (B, 1, H, W)
 
        # Normaliza os pesos para media=1 dentro do batch (opcional)
        w = weight_map
        if self.normalize_w:
            w = w / (w.mean() + 1e-8)
 
        # Media ponderada: sum(w * focal) / sum(w)
        weighted_focal = (w * focal_map).sum() / (w.sum() + 1e-8)
 
        return self.lambda_t * tversky_loss + self.lambda_f * weighted_focal