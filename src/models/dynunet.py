"""Dynamic U-Net (MONAI).

Hyperparameters:
- spatial_dims
- kernel_size
- strides
- upsample_kernel_size
- filters
- dropout
- norm_name
- act_name
- deep_supervision
- deep_supervision_num
- res_block
- trans_bias
"""

from monai.networks.nets import DynUNet

NAME = "dynunet"

DEFAULT_CONFIG = {
    "kernel_size":       (3, 3, 3),
    "strides":           (1, 2, 2),
    "upsample_kernel_size": (1, 2, 2),
    "filters":           (16, 32, 64, 128),
    "dropout":           0.2,
    "norm_name":         ("INSTANCE", {"affine": True}),
    "act_name":          ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
    "res_block":         False,
    "trans_bias":        False,
}


def build(in_ch: int, config: dict, device):
    cfg = {**DEFAULT_CONFIG, **config}
    return DynUNet(
        spatial_dims=2,
        in_channels=in_ch,
        out_channels=1,
        kernel_size=cfg["kernel_size"],
        strides=cfg["strides"],
        upsample_kernel_size=cfg["upsample_kernel_size"],
        filters=cfg["filters"],
        dropout=cfg["dropout"],
        norm_name=cfg["norm_name"],
        act_name=cfg["act_name"],
        deep_supervision=False,
        res_block=cfg["res_block"],
        trans_bias=cfg["trans_bias"],
    ).to(device)


def optuna_space(trial) -> dict:
    return {
        "kernel_size":       (3, 3, 3),
        "strides":           (1, 2, 2),
        "upsample_kernel_size": (1, 2, 2),
        "filters":           tuple(trial.suggest_categorical("dynunet_filters", [ (8, 16, 32, 64), (16, 32, 64, 128), (32, 64, 128, 256) ])),
        "dropout":           trial.suggest_float("dynunet_dropout", 0.0, 0.5),
        "res_block":         trial.suggest_categorical("dynunet_res_block", [False, True]),
        "trans_bias":        trial.suggest_categorical("dynunet_trans_bias", [False, True]),
    }
