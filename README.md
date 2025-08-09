
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

Para prepara os dados, deve-se primeiro transformar os dados DICON em .nii.gz para tratar no MONAI

'''sh
python src/dicon_ingest.py -i path_to_data -o data/processed
'''


## Rodando os Testes

Para rodar os testes automatizados:
```sh
pytest tests/
```

## Estrutura Recomendada
- `src/`: código principal
- `tests/`: testes automatizados
- `data/`: dados DICOM e convertidos
- `environment.yml`: dependências do projeto

## Dependências
Principais bibliotecas: SimpleITK, pydicom, numpy, matplotlib, nibabel, pytorch, monai, opencv, tqdm, pandas, scikit-image, pytest.

## Observações
- Se algum pacote não for encontrado via conda, instale via pip.
- Para desenvolvimento/testes, recomenda-se instalar também `pytest` e garantir que `pip` está atualizado no ambiente.

---
Atualize este README conforme novas funcionalidades forem adicionadas.
