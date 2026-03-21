"""
SegResNet (MONAI) — configuracao, build e espaco Optuna.
"""

from monai.networks.nets import SegResNet

NAME = "segresnet"

DEFAULT_CONFIG = {
    "init_filters": 32,
    "blocks_down":  [1, 2, 2, 4],
    "blocks_up":    [1, 1, 1],
    "dropout_prob": 0.2,
}


def build(in_ch: int, config: dict, device):
    cfg = {**DEFAULT_CONFIG, **config}
    return SegResNet(
        spatial_dims=2,
        in_channels=in_ch,
        out_channels=1,
        init_filters=cfg["init_filters"],
        blocks_down=cfg["blocks_down"],
        blocks_up=cfg["blocks_up"],
        dropout_prob=cfg["dropout_prob"],
    ).to(device)


def optuna_space(trial) -> dict:
    return {
        "init_filters": trial.suggest_categorical("init_filters", [16, 32, 64]),
        "dropout_prob": trial.suggest_float("dropout_prob", 0.0, 0.5),
        "blocks_down":  [1, 2, 2, 4],
        "blocks_up":    [1, 1, 1],
    }
