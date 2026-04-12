"""
SwinUNETR (MONAI 1.x) — configuracao, build e espaco Optuna.

Hyperparameters:
    in_channels: int
    out_channels: int
    patch_size: int = 2
    depths: Sequence[int] = (2, 2, 2, 2)
    num_heads: Sequence[int] = (3, 6, 12, 24)
    window_size: int = 7  -- janela de atencao local do Swin Transformer
    feature_size: int = 24
    norm_name: str = "instance"
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    dropout_path_rate: float = 0.0
    normalize: bool = True
    patch_norm: bool = False
    use_checkpoint: bool = False
    spatial_dims: int = 2
    downsample: str = "merging"
    use_v2: bool = False

Nota: num_heads[i] deve dividir feature_size * 2^i.
      Com feature_size=24: (3,6,12,24) e valido.
      Com feature_size=48: (3,6,12,24) tambem e valido (48/3=16 etc.).
"""

from monai.networks.nets import SwinUNETR

NAME = "swinunetr"

DEFAULT_CONFIG = {
    "out_channels":       1,
    "patch_size":         2,
    "depths":             (2, 2, 2, 2),
    "num_heads":          (3, 6, 12, 24),
    "window_size":        7,
    "feature_size":       24,
    "norm_name":          "instance",
    "drop_rate":          0.0,
    "attn_drop_rate":     0.0,
    "dropout_path_rate":  0.0,
    "normalize":          True,
    "patch_norm":         False,
    "use_checkpoint":     False,
    "downsample":         "merging",
    "use_v2":             False,
}


def build(in_ch: int, config: dict, device):
    cfg = {**DEFAULT_CONFIG, **config}
    return SwinUNETR(
        in_channels=in_ch,
        out_channels=cfg["out_channels"],
        patch_size=cfg["patch_size"],
        depths=tuple(cfg["depths"]),
        num_heads=tuple(cfg["num_heads"]),
        window_size=cfg["window_size"],
        feature_size=cfg["feature_size"],
        norm_name=cfg["norm_name"],
        drop_rate=cfg["drop_rate"],
        attn_drop_rate=cfg["attn_drop_rate"],
        dropout_path_rate=cfg["dropout_path_rate"],
        normalize=cfg["normalize"],
        patch_norm=cfg["patch_norm"],
        use_checkpoint=cfg["use_checkpoint"],
        spatial_dims=2,
        downsample=cfg["downsample"],
        use_v2=cfg["use_v2"],
    ).to(device)


def optuna_space(trial) -> dict:
    return {
        "feature_size":      trial.suggest_categorical("swin_feature_size",      [24, 48]),
        "drop_rate":         trial.suggest_float("swin_drop_rate",         0.0, 0.4),
        "attn_drop_rate":    trial.suggest_float("swin_attn_drop_rate",    0.0, 0.3),
        "dropout_path_rate": trial.suggest_float("swin_dropout_path_rate", 0.0, 0.3),
        "use_v2":            trial.suggest_categorical("swin_use_v2", [False, True]),
    }
