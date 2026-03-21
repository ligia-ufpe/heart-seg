# Heart-Seg — Segmentação de Gordura Epicardial em CT

Paper científico: **benchmark de modelos de segmentação de gordura epicardial em CT + Conformal Prediction.**

---

## Dataset

- **Fonte:** [Cardiac Fat Database](http://www.invisible-details.com/cardiac-fat-database.html) — 20 pacientes, CT não-contrastada (Cálcio Score)
- `data/fat_images/` — BMP 8-bit, janelamento HU [-200, -30] → [0, 255]
- `data/ground_truth/` — BMP RGB: **R=epicardial**, G=mediastinal, B=pericárdio
- `data/dicom/` — DICOMs originais (não usados no treino)
- Splits por paciente (seed=42): **Train=14, Val=3, Test=3**
- Gordura epicardial: ~0.7% dos pixels → classe muito rara

---

## Setup

**Requisitos:** Python ≥ 3.10, CUDA (recomendado), [uv](https://github.com/astral-sh/uv)

```bash
# Instalar dependências
uv sync

# Verificar GPU antes de treinar
nvidia-smi
```

Dependências principais: `torch`, `monai`, `optuna`, `numpy`, `pillow`, `scipy`, `tqdm`.

---

## Estrutura do Projeto

```
heart-seg/
├── data/
│   ├── fat_images/          # imagens de entrada por paciente
│   └── ground_truth/        # máscaras RGB por paciente
├── checkpoints/             # modelos salvos (gerado pelo treino)
├── results/                 # métricas e saídas da inferência (gerado pelo evaluate)
└── src/
    ├── models/
    │   ├── __init__.py      # registry: models.get("unet")
    │   ├── segresnet.py     # SegResNet (MONAI)
    │   ├── unet.py          # UNet (MONAI)
    │   └── attentionunet.py # AttentionUnet (MONAI)
    ├── dataset.py           # EpicardialDataset 2.5D + get_patient_splits
    ├── losses.py            # TverskyFocalLoss
    ├── metrics.py           # Dice, IoU, Sensitivity, Specificity, Precision, HD95
    ├── train.py             # treino simples ou busca Optuna
    └── evaluate.py          # inferência de um ou múltiplos modelos
```

---

## Treino — `src/train.py`

### Treino simples

```bash
# SegResNet com defaults (model=segresnet, epochs=100, lr=1e-3, batch=8, patience=20)
uv run python src/train.py

# Escolher modelo
uv run python src/train.py --model unet
uv run python src/train.py --model attentionunet

# Customizar hiperparâmetros
uv run python src/train.py --model segresnet --epochs 150 --lr 5e-4 --batch_size 16 --patience 30

# Contexto 2.5D: número de slices adjacentes empilhados (in_channels = 2*context+1)
uv run python src/train.py --context_slices 2

# Retomar de checkpoint
uv run python src/train.py --resume checkpoints/segresnet_20260313_131412/best_model.pth
```

**Argumentos disponíveis:**

| Argumento | Padrão | Descrição |
|---|---|---|
| `--model` | `segresnet` | Arquitetura: `segresnet`, `unet`, `attentionunet` |
| `--epochs` | `100` | Número máximo de épocas |
| `--batch_size` | `8` | Tamanho do batch |
| `--lr` | `1e-3` | Learning rate inicial (AdamW + CosineAnnealing) |
| `--patience` | `20` | Épocas sem melhora para early stopping |
| `--context_slices` | `1` | Slices adjacentes (in\_channels = 2\*context+1) |
| `--tversky_alpha` | `0.3` | Alpha da TverskyFocalLoss (beta = 1 - alpha) |
| `--seed` | `42` | Seed de reproducibilidade |
| `--workers` | `4` | Workers do DataLoader |
| `--resume` | — | Caminho para `.pth` para continuar treino |

**Saídas** em `checkpoints/<model>_<timestamp>/`:
```
checkpoints/segresnet_20260321_120000/
├── best_model.pth    # melhor val Dice (contém model_config + args + val_metrics)
├── epoch_0010.pth    # checkpoint periódico a cada 10 épocas
├── splits.json       # train/val/test IDs dos pacientes
├── history.json      # curvas de treino (loss, dice, iou, lr por época)
└── test_metrics.json # métricas no test set ao final
```

### Busca de hiperparâmetros com Optuna

```bash
# 30 trials com SegResNet (padrão)
uv run python src/train.py --optuna --n_trials 30

# 50 trials com UNet, 60 épocas por trial
uv run python src/train.py --optuna --n_trials 50 --model unet --epochs 60
```

**Hiperparâmetros buscados pelo Optuna:**

| Parâmetro | Espaço |
|---|---|
| `lr` | log-uniform [1e-4, 1e-2] |
| `batch_size` | {4, 8, 16} |
| `context_slices` | {1, 2, 3} |
| `tversky_alpha` | uniform [0.2, 0.5] |
| `init_filters` (segresnet) | {16, 32, 64} |
| `dropout_prob` | uniform [0.0, 0.5] |
| `base_filters` (unet/attentionunet) | {16, 32, 64} |

Pruning automático com `MedianPruner` (começa após 5 trials, ignora primeiras 10 épocas).

**Saídas** em `checkpoints/<model>_<timestamp>/optuna/`:
```
checkpoints/unet_20260321_120000/
└── optuna/
    ├── trial_000/
    │   ├── best_model.pth
    │   ├── splits.json
    │   └── history.json
    ├── trial_001/
    │   └── ...
    ├── best_trial/          # cópia do trial com maior val Dice
    │   └── best_model.pth
    └── study_results.json   # best_trial, best_dice, best_params, todos os trials
```

**Argumentos adicionais do Optuna:**

| Argumento | Padrão | Descrição |
|---|---|---|
| `--optuna` | — | Ativa busca de hiperparâmetros |
| `--n_trials` | `30` | Número de trials |
| `--study_name` | `epicardial_seg` | Nome do estudo Optuna |

---

## Inferência — `src/evaluate.py`

Avalia um ou múltiplos modelos sobre o mesmo conjunto de pacientes. Por padrão salva as probabilidades e ground-truth como `.npy` para uso em conformal prediction.

### Uso básico

```bash
# Um modelo — test set do splits.json do checkpoint
uv run python src/evaluate.py \
  --checkpoints checkpoints/segresnet_20260313_131412/best_model.pth

# Múltiplos modelos comparados no mesmo conjunto
uv run python src/evaluate.py \
  --checkpoints checkpoints/segresnet_.../best_model.pth \
               checkpoints/unet_.../best_model.pth \
               checkpoints/unet_.../optuna/best_trial/best_model.pth
```

### Opções

```bash
# Pacientes específicos (necessário quando os checkpoints têm splits diferentes)
uv run python src/evaluate.py --checkpoints ckpt.pth --patients DSan ACel

# Threshold customizado (ex: pós-calibração conformal)
uv run python src/evaluate.py --checkpoints ckpt.pth --threshold 0.35

# Salvar máscaras binarizadas como PNG
uv run python src/evaluate.py --checkpoints ckpt.pth --save_masks

# Desativar salvamento de prob.npy + gt.npy (ativado por padrão)
uv run python src/evaluate.py --checkpoints ckpt.pth --no_save_logits
```

**Argumentos disponíveis:**

| Argumento | Padrão | Descrição |
|---|---|---|
| `--checkpoints` | — | Um ou mais caminhos `.pth` (obrigatório) |
| `--patients` | test set do 1º checkpoint | IDs dos pacientes a avaliar |
| `--threshold` | `0.5` | Threshold de binarização |
| `--batch_size` | `8` | Tamanho do batch |
| `--workers` | `4` | Workers do DataLoader |
| `--save_masks` | off | Salva máscaras PNG em `results/masks/<ckpt>/` |
| `--no_save_logits` | off | Desativa salvamento de `prob.npy` + `gt.npy` |
| `--model` | do checkpoint | Sobrescreve a arquitetura (raramente necessário) |

**Saídas** em `results/`:
```
results/
├── eval_metrics.json              # métricas de todos os modelos avaliados
├── logits/
│   └── <ckpt_id>/                 # ex: segresnet_20260313_131412
│       └── <patient_id>/
│           ├── <slice>_prob.npy   # probabilidades float32 — para conformal prediction
│           └── <slice>_gt.npy     # ground-truth float32
└── masks/                         # apenas com --save_masks
    └── <ckpt_id>/
        └── <patient_id>/
            └── <slice>_mask.png
```

O `<ckpt_id>` é derivado do caminho relativo a `checkpoints/`:
- Single run: `segresnet_20260313_131412`
- Optuna: `unet_20260321_120000_optuna_best_trial`

Quando múltiplos modelos são avaliados, um resumo comparativo é impresso no terminal.

---

## Adicionar um Novo Modelo

1. Crie `src/models/<nome>.py` seguindo o padrão:

```python
from monai.networks.nets import MinhaArquitetura

NAME = "meumodelo"

DEFAULT_CONFIG = {
    "param1": valor1,
    ...
}

def build(in_ch: int, config: dict, device):
    cfg = {**DEFAULT_CONFIG, **config}
    return MinhaArquitetura(in_channels=in_ch, out_channels=1, **cfg).to(device)

def optuna_space(trial) -> dict:
    return {
        "param1": trial.suggest_categorical("param1", [opA, opB]),
        ...
    }
```

2. Registre em `src/models/__init__.py`:

```python
from . import attentionunet, segresnet, unet, meumodelo  # adicionar aqui

_REGISTRY = {m.NAME: m for m in [segresnet, unet, attentionunet, meumodelo]}
```

---

## Métricas (padrão TMI / MedIA / MICCAI)

Implementadas em `src/metrics.py`:

| Métrica | Implementada |
|---|---|
| DSC (Dice) | ✓ |
| IoU (Jaccard) | ✓ |
| Sensitivity (Recall) | ✓ |
| Specificity | ✓ |
| Precision | ✓ |
| HD95 (Hausdorff 95%) | ✓ |
| Volume Error (mL) | a implementar |

**Protocolo do paper:** reportar **mean ± std por paciente**, nunca pixels agregados.
Volume Error em mL requer pixel spacing: 0.74mm × 0.74mm × 2.5mm (dos DICOMs).

---

## Resultados Atuais

### SegResNet — Baseline

**Checkpoint:** `checkpoints/segresnet_20260313_131412/best_model.pth`

| Dice   | IoU    | Sensitivity | Specificity | Precision | HD95  |
|--------|--------|-------------|-------------|-----------|-------|
| 0.7536 | 0.6539 | 0.8747      | 0.9935      | 0.7353    | 13.85 |

---

## Próximos Passos

1. Treinar UNet e AttentionUnet com Optuna e comparar com SegResNet
2. Implementar SwinUNETR / UNETR em `src/models/`
3. Volume Error (mL) em `src/metrics.py`
4. Conformal Prediction — calibrar threshold usando `results/logits/`
5. Análise comparativa final — tabela de benchmark + Bland-Altman + curvas de calibração

---

## Notas Técnicas

- Split é **por paciente**, não por slice — sem data leakage
- Abordagem **2.5D**: `2*context_slices+1` slices empilhados como canais; bordas replicam o slice mais próximo
- Critério de máscara epicardial: `R > 10 AND R > G AND R > B`
- `model_config` é salvo em todo checkpoint — `evaluate.py` reconstrói a arquitetura automaticamente
- 1 paciente do train pode ser ignorado por pasta ausente (aviso esperado no log)
