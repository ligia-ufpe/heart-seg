"""
Registry de modelos disponíveis para segmentacao epicardica.

Para adicionar um novo modelo:
  1. Crie src/models/<nome>.py com NAME, DEFAULT_CONFIG, build() e optuna_space()
  2. Importe e registre aqui
"""

from . import attentionunet, dynunet, segresnet, unet
from . import SwinUNETR as swinunetr_module
from . import UNETR as unetr_module

_REGISTRY = {m.NAME: m for m in [segresnet, unet, attentionunet, dynunet, swinunetr_module, unetr_module]}

AVAILABLE = list(_REGISTRY)


def get(name: str):
    if name not in _REGISTRY:
        raise ValueError(f"Modelo desconhecido: '{name}'. Opcoes: {AVAILABLE}")
    return _REGISTRY[name]
