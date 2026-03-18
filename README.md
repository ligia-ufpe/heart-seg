# Heart-Seg — Segmentação de Gordura Epicardial em CT

## Objetivo
Paper científico: **benchmark de modelos de segmentação de gordura epicardial em CT** + Conformal Prediction.

Pipeline completo:
1. **MONAI SegResNet** (baseline — treinado)
2. **UNet++** (a implementar)
3. **UNETR** (a implementar)
4. **SwinUNETR** (a implementar)
5. **Conformal Prediction** sobre as saídas dos modelos (a implementar)

---

## Dataset
- **Fonte:** [Cardiac Fat Database](http://www.invisible-details.com/cardiac-fat-database.html) — 20 pacientes, CT não-contrastada (Cálcio Score)
- `data/fat_images/` — BMP 8-bit, janelamento HU [-200, -30] → [0, 255]
- `data/ground_truth/` — BMP RGB: **R=epicardial**, G=mediastinal, B=pericárdio
- `data/dicom/` — DICOMs originais (não usados no treino atual)
- Splits fixos (seed=42): **Train=14, Val=3, Test=3** pacientes
- Cobertura epicardial: ~0.7% dos pixels (classe muito rara → uso de TverskyFocalLoss)

---

## Ambiente

- **Gestor de pacotes:** `uv` — sempre usar `uv run python ...`
- PyTorch 2.10.0+cu128, MONAI 1.5.2
- GPU: RTX 5060 Ti 16GB, CUDA 13.1

### Instalação

```bash
uv sync
```

---

## Estrutura do Código

```
src/
├── dataset.py    — EpicardialDataset (2.5D, context_slices=1) + get_patient_splits
├── losses.py     — TverskyFocalLoss (alpha=0.3, beta=0.7, gamma=2.0)
├── metrics.py    — Dice, IoU, Sensitivity, Specificity, Precision, HD95
├── train.py      — script principal de treino
└── evaluate.py   — script de inferência e avaliação
```

---

## Como Rodar

### Treinar do zero

```bash
uv run python src/train.py
```

### Retomar de um checkpoint

```bash
uv run python src/train.py --resume checkpoints/segresnet_20260313_131412/best_model.pth
```

### Inferência / Avaliação

```bash
# Test set padrão (usa splits.json do checkpoint)
uv run python src/evaluate.py --checkpoint checkpoints/segresnet_20260313_131412/best_model.pth

# Pacientes específicos
uv run python src/evaluate.py --checkpoint checkpoints/segresnet_20260313_131412/best_model.pth --patients DSan ACel

# Salvar máscaras preditas em results/masks/
uv run python src/evaluate.py --checkpoint checkpoints/segresnet_20260313_131412/best_model.pth --save_masks

# Threshold customizado (ex: conformal prediction)
uv run python src/evaluate.py --checkpoint checkpoints/segresnet_20260313_131412/best_model.pth --threshold 0.35
```

As métricas são salvas em `results/eval_metrics.json`.

> Antes de treinar, verifique que a GPU está livre: `nvidia-smi`

---

## Status dos Modelos

### SegResNet (MONAI) — Baseline

**Melhor checkpoint:** `checkpoints/segresnet_20260313_131412/best_model.pth`

| Epoch | Dice   | IoU    | Sensitivity | Specificity | Precision | HD95  |
|-------|--------|--------|-------------|-------------|-----------|-------|
| best  | 0.7536 | 0.6539 | 0.8747      | 0.9935      | 0.7353    | 13.85 |

### UNet++ — a implementar

Variante da U-Net com skip connections densas (nested). Disponível via MONAI (`BasicUNetPlusPlus`) ou segmentation-models-pytorch.

### UNETR — a implementar

Transformer puro como encoder, decoder CNN. Disponível no MONAI (`UNETR`). Requer imagens de tamanho fixo (múltiplo do patch size).

### SwinUNETR — a implementar

Swin Transformer como encoder, decoder hierárquico. Disponível no MONAI (`SwinUNETR`). Estado da arte em vários benchmarks de segmentação médica.

---

## Métricas para o Paper (padrão TMI / MedIA / MICCAI)

**Obrigatórias:**
- DSC (Dice), HD95, Sensitivity, Precision, **Volume Error (mL)**

**Recomendadas:**
- IoU, Pearson r / ICC para volume, Bland-Altman plot

**Protocolo:** reportar **mean ± std por paciente**, nunca pixels agregados.

`metrics.py` já tem Dice, IoU, Sensitivity, Specificity, Precision, HD95.
**Falta implementar:** Volume Error em mL (pixel spacing: 0.74mm × 0.74mm × 2.5mm).

---

## Próximos Passos (em ordem)

1. **Implementar UNet++** — adaptar `train.py` e `evaluate.py` para suportar múltiplos modelos
2. **Implementar UNETR** — atenção ao tamanho de entrada (patch size)
3. **Implementar SwinUNETR** — modelo mais pesado, verificar VRAM
4. **Adicionar Volume Error (mL)** ao `metrics.py` usando pixel spacing dos DICOMs
5. **Conformal Prediction** — aplicar RAPS ou adaptação para segmentação sobre as probabilidades de saída
6. **Análise comparativa** — tabela de benchmark + curvas de calibração + visualizações

---

## Observações Técnicas Importantes

- Split é **por paciente** (não por slice) — sem data leakage
- Abordagem **2.5D**: cada sample = 3 slices adjacentes empilhados como canais
- Máscara epicardial: critério `R > 10 AND R > G AND R > B` (alinhado com documentação do dataset)
- 1 paciente no train é ignorado por pasta ausente (aviso esperado no log)
- Sempre verificar que não há outro processo Python na GPU antes de treinar: `nvidia-smi`
