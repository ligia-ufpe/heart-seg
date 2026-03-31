"""
SegResNet (MONAI) — configuracao, build e espaco Optuna.
hyperparameters:
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE
"""

from monai.networks.nets import SegResNet
from monai.utils.enums import UpsampleMode

NAME = "segresnet"

DEFAULT_CONFIG = {
    "init_filters": 32,
    "in_channels": None,
    "out_channels": 1,
    "dropout_prob": 0.2,
    "act": ("RELU", {"inplace": True}),
    "norm": ("GROUP", {"num_groups": 8}),
    "norm_name": "",
    "num_groups": 8,
    "use_conv_final": True,
    "blocks_down": (1, 2, 2, 4),
    "blocks_up": (1, 1, 1),
    "upsample_mode": UpsampleMode.NONTRAINABLE,
}


def build(in_ch: int, config: dict, device):
    cfg = {**DEFAULT_CONFIG, **config}
    return SegResNet(
        spatial_dims=2,
        in_channels=in_ch,
        out_channels=cfg["out_channels"],
        init_filters=cfg["init_filters"],
        dropout_prob=cfg["dropout_prob"],
        act=cfg["act"],
        norm=cfg["norm"],
        norm_name=cfg["norm_name"],
        num_groups=cfg["num_groups"],
        use_conv_final=cfg["use_conv_final"],
        blocks_down=tuple(cfg["blocks_down"]),
        blocks_up=tuple(cfg["blocks_up"]),
        upsample_mode=cfg["upsample_mode"],
    ).to(device)


def optuna_space(trial) -> dict:
    return {
        "init_filters": trial.suggest_categorical("init_filters", [16, 32, 64]),
        "dropout_prob": trial.suggest_float("dropout_prob", 0.0, 0.5),
        "num_groups": trial.suggest_categorical("num_groups", [4, 8, 16]),
        "use_conv_final": trial.suggest_categorical("use_conv_final", [True, False]),
        "blocks_down": (1, 2, 2, 4),
        "blocks_up": (1, 1, 1),
        "upsample_mode": trial.suggest_categorical("upsample_mode", [
            UpsampleMode.NONTRAINABLE,
            UpsampleMode.DECONV,
        ]),
    }
