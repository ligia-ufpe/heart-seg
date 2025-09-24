
# Heart Fat Detection - Infraestrutura de Desenvolvimento

## Ambiente Virtual

1. Crie o ambiente Conda:
	```sh
	conda env create -f environment.yml
	```
2. Ative o ambiente:
	```sh
	conda activate heartenv
	```
3. (Opcional) Instale dependências extras via pip, se necessário:
	```sh
	pip install pytest
	```

## Preparando os dados

Para preparar os dados, deve-se primeiro transformar os dados DICOM em .nii.gz para tratar no MONAI:

```sh
python src/dicon_ingest.py -i data/Cardio_Prj1802_Cardio_Onco/CINE_EC_12 -o data/processed
```

## Segmentação Manual (Infraestrutura para Anotação)

### Workflow Completo
```sh
# Workflow completo automatizado
python scripts/run_segmentation_workflow.py full --data-dir data/processed

# Ou executar passos individuais:

# 1. Preparar dados para segmentação
python scripts/run_segmentation_workflow.py prepare --data-dir data/processed

# 2. Ver instruções do 3D Slicer
python scripts/run_segmentation_workflow.py instructions

# 3. Validar segmentações após anotar
python scripts/run_segmentation_workflow.py validate --annotation-dir annotations
```

### Manual do 3D Slicer
- **Guia completo**: `docs/manual_segmentation_guide.md`
- **Script automatizado**: `annotations/slicer_annotation_script.py`
- **Tipos de gordura**: Epicardial, Pericardial, Myocardial

## Rodando os Testes

Para rodar os testes automatizados:
```sh
pytest tests/
```

## Estrutura
- `src/`: código principal
- `tests/`: testes para ver se a conversão deu certo
- `data/`: dados DICOM e convertidos
- `environment.yml`: dependências do projeto

## Dependências
Principais bibliotecas: SimpleITK, pydicom, numpy, matplotlib, nibabel, pytorch, monai, opencv, tqdm, pandas, scikit-image, pytest.

## Observações
- Se algum pacote não for encontrado via conda, instale via pip.
- Para desenvolvimento/testes, recomenda-se instalar também `pytest` e garantir que `pip` está atualizado no ambiente.


