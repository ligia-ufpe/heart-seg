"""
Attention UNet (MONAI) — configuracao, build e espaco Optuna.
"""

from monai.networks.nets import AttentionUnet

NAME = "attentionunet"

DEFAULT_CONFIG = {
    "channels":     (32, 64, 128, 256),
    "strides":      (2, 2, 2),
    "dropout":      0.2,
}


def build(in_ch: int, config: dict, device):
    cfg = {**DEFAULT_CONFIG, **config}
    return AttentionUnet(
        spatial_dims=2,
        in_channels=in_ch,
        out_channels=1,
        channels=cfg["channels"],
        strides=cfg["strides"],
        dropout=cfg["dropout"],
    ).to(device)


def optuna_space(trial) -> dict:
    base_filters = trial.suggest_categorical("base_filters", [16, 32, 64])
    return {
        "channels": (base_filters, base_filters * 2, base_filters * 4, base_filters * 8),
        "strides":  (2, 2, 2),
        "dropout":  trial.suggest_float("dropout_prob", 0.0, 0.5),
    }
