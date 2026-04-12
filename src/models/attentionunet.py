"""
Attention UNet (MONAI) — configuracao, build e espaco Optuna.

Hyperparameters aceitos pela MONAI 1.x:
        spatial_dims - number of spatial dimensions of the input image.
        in_channels - number of the input channel.
        out_channels - number of the output classes.
        channels (Sequence[int]) - sequence of channels. Top block first. The length of channels should be no less than 2.
        strides (Sequence[int]) - stride to use for convolutions.
        kernel_size - convolution kernel size.
        up_kernel_size - convolution kernel size for transposed convolution layers.
        dropout - dropout ratio. Defaults to no dropout.
"""

from monai.networks.nets import AttentionUnet

NAME = "attentionunet"

DEFAULT_CONFIG = {
    "channels":       (32, 64, 128, 256),
    "strides":        (2, 2, 2),
    "kernel_size":    3,
    "up_kernel_size": 3,
    "dropout":        0.2,
}


def build(in_ch: int, config: dict, device):
    cfg = {**DEFAULT_CONFIG, **config}
    return AttentionUnet(
        spatial_dims=2,
        in_channels=in_ch,
        out_channels=1,
        channels=cfg["channels"],
        strides=cfg["strides"],
        kernel_size=cfg["kernel_size"],
        up_kernel_size=cfg["up_kernel_size"],
        dropout=cfg["dropout"],
    ).to(device)


def optuna_space(trial) -> dict:
    base_filters = trial.suggest_categorical("attn_base_filters", [16, 32, 64])
    return {
        "channels":       (base_filters, base_filters * 2, base_filters * 4, base_filters * 8),
        "strides":        (2, 2, 2),
        "kernel_size":    trial.suggest_categorical("attn_kernel_size",    [3, 5]),
        "up_kernel_size": trial.suggest_categorical("attn_up_kernel_size", [3, 5]),
        "dropout":        trial.suggest_float("attn_dropout", 0.0, 0.5),
    }
