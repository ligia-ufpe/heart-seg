
# # Heart Fat Detection - Infraestrutura de Desenvolvimento

Pipeline em Python para (1) converter séries DICOM para NIfTI (`.nii.gz`) e (2) treinar/usar um modelo (Somente **U-Net 2D** por enquanto) para segmentação cardíaca em imagens RGB.

> Nota: o repositório também inclui o diretório `sam2/` (código do Segment Anything Model 2) como dependência/experimento, mas o código em `src/` atualmente implementa de forma funcional apenas a etapa de conversão DICOM→NIfTI e o modelo U-Net.

## O que existe hoje no código (fonte da verdade)

- `src/convert.py`: lê uma série DICOM a partir de uma pasta e grava um arquivo NIfTI (`.nii.gz`). Também suporta modo *batch* processando múltiplas subpastas dentro de `data/raw/`.
- `src/models/unet.py`: treinamento e inferência de uma U-Net 2D (usa **MONAI** se disponível; caso contrário, usa uma U-Net básica em PyTorch). Espera dataset em PNG (RGB + máscara).
- `tests/test_dicon_ingest.py`: teste simples que valida se um NIfTI existente em `data/processed/` pode ser carregado e é 3D.

Arquivos como `src/preprocess.py`, `src/train_model.py` e `src/utils/*` estão presentes, mas hoje ainda são placeholders (sem lógica implementada).

## Estrutura de pastas

```text
src/
	convert.py              # DICOM -> NIfTI (.nii.gz)
	run_model.py            # Wrapper CLI (atenção: contém referências a SAM2 não implementadas em src/)
	models/
		unet.py                # Treino e inferência U-Net
		trained/               # Checkpoints salvos (ex.: unet_heart.pt)
data/
	raw/                     # Pastas com séries DICOM (uma pasta por exame)
	processed/               # NIfTIs gerados (.nii.gz)
tests/
	test_dicon_ingest.py
```

## Requisitos

- Python 3.9+ (o Dockerfile usa 3.9)
- Dependências do arquivo `requirements-segmentation.txt`

Observação importante: os testes e o conversor dependem de **SimpleITK**.

## Clonar

- Para Clonar o projeto precisa-se adicionar com submodulos

```bash
git clone [link do repo] --recursive
```

## Instalação

### Via pip (recomendado)

Crie um venv/conda e instale as dependências.

Se você quiser usar o script automático:

```bash
./setup.sh
```

> `setup.sh` instala `-r requirements.txt` e `pytest`.

### Via Docker (Work in progress)

O projeto tem um `Dockerfile` simples para ambiente de execução.

## Conversão DICOM → NIfTI

### Processar uma única série DICOM

Entrada: uma pasta contendo arquivos DICOM.
Saída: um arquivo `.nii.gz` em `data/processed/` (ou caminho informado).

Exemplo:

```bash
python src/convert.py --input data/raw/CINE_EC_12 --output data/processed
```

### Processar várias séries (modo batch)

Se `--input` apontar para uma pasta que contém várias subpastas (cada uma com uma série DICOM), o script processa todas:

```bash
python src/convert.py --input data/raw --output data/processed
```

## U-Net (treino e inferência)

O script `src/models/unet.py` assume um dataset no formato:

```text
data/test_dataset_rgb/
	images/   # *.png RGB
	labels/   # *.png grayscale (máscara binária)
```

### Treinar

```bash
python src/models/unet.py --train --dataset data/test_dataset_rgb --epochs 50
```

O checkpoint padrão é salvo em `src/models/trained/unet_heart.pt`.

### Predizer

```bash
python src/models/unet.py --predict --input path/para/imagem.png --output src/output
```

Saída: uma máscara `*_mask.png`.

## Testes

O teste atual valida carregamento de um NIfTI existente em `data/processed/CINE_EC_12.nii.gz`.

```bash
pytest -q
```

Se aparecer `ModuleNotFoundError: No module named 'SimpleITK'`, significa que o ambiente Python em uso não tem as dependências instaladas (instale via `requirements-segmentation.txt`).

## Notas / limitações atuais

- `src/run_model.py` expõe uma CLI que menciona SAM2, mas **não existe** um `src/models/sam2.py` no estado atual do `src/` — então essa parte não roda sem adaptação.
- `src/preprocess.py` e `src/utils/*` estão vazios por enquanto.
- O pipeline principal “de ponta a ponta” (DICOM → corte/crop → segmentação final) ainda não está totalmente implementado em `src/`.


