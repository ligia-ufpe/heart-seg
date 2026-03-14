# Heart-Seg — Contexto do Projeto

## Objetivo
Paper científico: **segmentação de gordura epicardial em CT** com benchmark entre dois modelos + Conformal Prediction.

Pipeline completo:
1. **MONAI SegResNet** (baseline — em andamento)
2. **SAM3** (a implementar)
3. **Conformal Prediction** sobre as saídas dos dois modelos (a implementar)

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
- Para instalar deps: `uv sync`

---

## Estrutura do Código
```
src/
├── dataset.py   — EpicardialDataset (2.5D, context_slices=1) + get_patient_splits
├── losses.py    — TverskyFocalLoss (alpha=0.3, beta=0.7, gamma=2.0)
├── metrics.py   — Dice, IoU, Sensitivity, Specificity, Precision, HD95
└── train.py     — script principal de treino (SegResNet MONAI)
```

### Como treinar do zero
```bash
uv run python src/train.py
```

### Como retomar do checkpoint
```bash
uv run python src/train.py --resume checkpoints/segresnet_20260313_131412/best_model.pth
```

---

## Status do Treino MONAI (Baseline)

**Melhor checkpoint salvo:** `checkpoints/segresnet_20260313_131412/best_model.pth`

| Epoch | Dice   | IoU    | Sensitivity | Specificity | Precision | HD95  |
|-------|--------|--------|-------------|-------------|-----------|-------|
| 9     | 0.5875 | 0.4522 | 0.8736      | 0.9749      | 0.5005    | 44.46 |

Treino foi interrompido na epoch ~16 e ainda estava melhorando — continuar do checkpoint acima.

---

## Métricas para o Paper (padrão TMI / MedIA / MICCAI)

**Obrigatórias:**
- DSC (Dice), HD95, Sensitivity, Precision, **Volume Error (mL)**

**Recomendadas:**
- IoU, Pearson r / ICC para volume, Bland-Altman plot

**Protocolo:** reportar **mean ± std por paciente**, nunca pixels agregados.

`metrics.py` já tem Dice, IoU, Sensitivity, Specificity, Precision, HD95.
**Falta implementar:** Volume Error em mL (precisa do pixel spacing do DICOM).

---

## Próximos Passos (em ordem)

1. **Terminar treino MONAI** — retomar do checkpoint, deixar convergir até early stopping
2. **Avaliação completa no test set** — já automática ao final do `train.py`
3. **Adicionar Volume Error (mL)** ao `metrics.py` usando pixel spacing dos DICOMs (0.74mm × 0.74mm × 2.5mm)
4. **Implementar SAM3** — adaptar SAM2/SAM3 para CT, prompts automáticos por slice
5. **Conformal Prediction** — aplicar RAPS ou adaptação para segmentação sobre as probabilidades de saída
6. **Análise comparativa** — tabela de benchmark + curvas de calibração + visualizações

---

## Observações Técnicas Importantes

- Split é **por paciente** (não por slice) — sem data leakage
- Abordagem **2.5D**: cada sample = 3 slices adjacentes empilhados como canais
- Máscara epicardial: critério `R > 10 AND R > G AND R > B` (alinhado com documentação do dataset)
- 1 paciente no train é ignorado por pasta ausente (aviso esperado no log)
- Sempre verificar que não há outro processo Python na GPU antes de treinar: `nvidia-smi`
