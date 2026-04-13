"""
UNETR (MONAI 1.x) — configuracao, build e espaco Optuna.

Hyperparameters:
    in_channels: int
    out_channels: int
    img_size: Sequence[int] -- tamanho espacial da entrada (H, W) para 2D
                               DEVE ser divisivel por 16 (patch_size padrao do ViT).
                               Ajuste para o tamanho real das imagens do dataset.
    feature_size: int = 16
    hidden_size: int = 768   -- dimensao de embedding do ViT
    mlp_dim: int = 3072      -- dimensao interna do MLP do Transformer
    num_heads: int = 12      -- cabecas de atencao (deve dividir hidden_size)
    proj_type: str = "conv"  -- tipo de projecao: "conv" ou "perceptron"
    norm_name: str = "instance"
    conv_block: bool = True
    res_block: bool = True
    dropout_rate: float = 0.0
    spatial_dims: int = 2
    qkv_bias: bool = False

Nota: hidden_size deve ser divisivel por num_heads.
      hidden_size=768, num_heads=12 → 64 por cabeca.
      hidden_size=384, num_heads=6  → 64 por cabeca.
"""

from monai.networks.nets import UNETR

NAME = "unetr"

DEFAULT_CONFIG = {
    "img_size":     (512, 512),   # tamanho real das imagens do dataset
    "out_channels": 1,
    "feature_size": 16,
    "hidden_size":  768,
    "mlp_dim":      3072,
    "num_heads":    12,
    "proj_type":    "conv",
    "norm_name":    "instance",
    "conv_block":   True,
    "res_block":    True,
    "dropout_rate": 0.0,
    "qkv_bias":     False,
}


def build(in_ch: int, config: dict, device):
    cfg = {**DEFAULT_CONFIG, **config}
    return UNETR(
        in_channels=in_ch,
        out_channels=cfg["out_channels"],
        img_size=tuple(cfg["img_size"]),
        feature_size=cfg["feature_size"],
        hidden_size=cfg["hidden_size"],
        mlp_dim=cfg["mlp_dim"],
        num_heads=cfg["num_heads"],
        proj_type=cfg["proj_type"],
        norm_name=cfg["norm_name"],
        conv_block=cfg["conv_block"],
        res_block=cfg["res_block"],
        dropout_rate=cfg["dropout_rate"],
        spatial_dims=2,
        qkv_bias=cfg["qkv_bias"],
    ).to(device)


def optuna_space(trial) -> dict:
    hidden_size = trial.suggest_categorical("unetr_hidden_size", [384, 768])
    num_heads   = 6 if hidden_size == 384 else 12
    return {
        "feature_size": trial.suggest_categorical("unetr_feature_size", [16, 32]),
        "hidden_size":  hidden_size,
        "mlp_dim":      hidden_size * 4,
        "num_heads":    num_heads,
        "dropout_rate": trial.suggest_float("unetr_dropout_rate", 0.0, 0.5),
        "res_block":    trial.suggest_categorical("unetr_res_block", [True, False]),
        "proj_type":    trial.suggest_categorical("unetr_proj_type", ["conv", "perceptron"]),
    }
