# Cardiac Fat Database - Computed Tomography

## Visao Geral

Este dataset contem imagens de tomografia computadorizada (CT) cardiaca nao-contrastada de **20 pacientes**, com segmentacao manual de gordura epicardica e mediastinal.

---

## ⚠️ Configuracao Inicial (Obrigatorio)

Os dados **nao estao incluidos no repositorio** por questoes de tamanho e privacidade. Voce deve baixar e configurar manualmente.

### 1. Baixar o Dataset

Fonte oficial: [Cardiac Fat Database](http://www.invisible-details.com/cardiac-fat-database.html)

### 2. Criar as Pastas

Dentro de `ct_data/`, crie as seguintes pastas:

```
ct_data/
├── Dicom_original/          # Imagens DICOM originais
├── Fat Images/              # Imagens BMP pre-processadas
├── Ground Truth - Fat Range/        # Mascaras (principal)
├── Ground Truth - Higher Range/     # Mascaras (opcional)
└── Ground Truth - Combined Range/   # Mascaras (opcional)
```

### 3. Organizar por Paciente

Cada pasta acima deve conter subpastas com os IDs dos pacientes:

```
Dicom_original/
├── ACel/
│   ├── IM-0003-0001_an.dcm
│   └── ...
├── AEdu/
├── AFre/
└── ... (20 pacientes)
```

### 4. Verificar Instalacao

```bash
python -c "from src.train_simple import get_patient_splits; print(get_patient_splits('ct_data'))"
```

---

## Estrutura do Dataset

```
ct_data/
├── Dicom_original/                     # Imagens CT originais (16-bit, HU completo)
│   ├── ACel/
│   │   ├── IM-0003-0001_an.dcm
│   │   └── ...
│   └── [outros pacientes]/
│
├── Fat Images/                         # Imagens BMP pre-processadas (8-bit)
│   ├── ACel/                           # Derivadas do DICOM, filtradas para gordura
│   │   ├── IM-0003-0001.BMP
│   │   └── ...
│   └── [outros pacientes]/
│
├── Ground Truth - Fat Range/           # Mascaras RGB segmentadas manualmente
│   ├── ACel/                           # Anotacao na faixa [-200, -30] HU
│   │   ├── 001.bmp
│   │   └── ...
│   └── [outros pacientes]/
│
├── Ground Truth - Higher Range/        # Mascaras RGB [-200, 500] HU
│   └── [mesma estrutura]/
│
└── Ground Truth - Combined Range/      # Combinacao dos dois ranges
    └── [mesma estrutura]/
```

### Pacientes (20 total)
`ACel`, `AEdu`, `AFre`, `AMar`, `AXav`, `CFer`, `CLis`, `DLag`, `DSan`, `DSil`, `EGra`, `FGas`, `FPiq`, `ISou`, `JFul`, `JMir`, `MPai`, `MSil`, `TJes`, `VMar`

---

## Descricao dos Arquivos

### 1. Dicom_original (Imagens CT Originais)

Imagens de tomografia computadorizada em formato DICOM, com toda informacao original.

| Propriedade | Valor |
|-------------|-------|
| Formato | DICOM (.dcm) |
| Profundidade | 16-bit unsigned (12 bits stored) |
| Dimensoes | 512 x 512 pixels |
| Range HU | -1024 a ~643 |
| Pixel Spacing | ~0.74mm x 0.74mm |
| Slice Thickness | 2.5mm |
| Modalidade | CT nao-contrastada (Calcio Score) |
| Fabricante | GE Medical Systems |
| ImageType | ORIGINAL, PRIMARY, AXIAL |

Uso: 
- Entrada para modelos de segmentacao 3D (volumes)
- Janelamento customizado para diferentes tecidos
- Analise volumetrica com espacamento real

### 2. Fat Images (Imagens de Gordura)

| Propriedade | Valor |
|-------------|-------|
| Formato | BMP |
| Profundidade | 8-bit (0-255) |
| Dimensoes | 512 x 512 pixels |
| Conversao | Linear de [-200, -30] HU para [0, 255] |

### 3. Ground Truth (Mascaras de Segmentacao)

As mascaras foram **segmentadas manualmente** 

#### Codificacao RGB (3 classes):

| Canal | Cor | Estrutura Anatomica |
|-------|-----|---------------------|
| R (Red) | Vermelho | Gordura Epicardica |
| G (Green) | Verde | Gordura Mediastinal |
| B (Blue) | Azul | Pericardio |

#### Variantes de Ground Truth:

| Pasta | Range HU | Descricao |
|-------|----------|-----------|
| Ground Truth - Fat Range | [-200, -30] | Segmentacao na faixa de gordura |
| Ground Truth - Higher Range | [-200, 500] | Segmentacao em faixa expandida |
| Ground Truth - Combined Range | Combinado | Partes de ambos os ranges |

---

## Anatomia Cardiaca

```
                    ┌─────────────────────────────────┐
                    │      MEDIASTINO (verde)         │
                    │  ┌─────────────────────────┐    │
                    │  │    PERICARDIO (azul)    │    │
                    │  │  ┌───────────────────┐  │    │
                    │  │  │  EPICARDICA (verm)│  │    │
                    │  │  │  ┌─────────────┐  │  │    │
                    │  │  │  │  MIOCARDIO  │  │  │    │
                    │  │  │  │  (musculo)  │  │  │    │
                    │  │  │  └─────────────┘  │  │    │
                    │  │  └───────────────────┘  │    │
                    │  └─────────────────────────┘    │
                    └─────────────────────────────────┘
```

- Gordura Epicardica: Entre o miocardio e o pericardio visceral
- Gordura Mediastinal: Externa ao pericardio parietal
- Pericardio: Membrana que separa as duas gorduras

---

## Estatisticas do Dataset

| Metrica | Valor |
|---------|-------|
| Total de pacientes | 20 |
| Total de slices | ~844-926 |
| Slices por paciente | ~40-50 |
| Dimensao das imagens | 512 x 512 |

### Cobertura media por classe (Ground Truth - Fat Range):

| Classe | Cobertura Media | Desvio Padrao |
|--------|-----------------|---------------|
| Gordura Epicardica (R) | ~11.25% | +/- 4.78% |
| Gordura Mediastinal (G) | ~11.34% | +/- 4.79% |
| Pericardio (B) | ~9.77% | +/- 4.52% |

### Gordura Epicardica (R dominante - recomendado):

| Metrica | Valor |
|---------|-------|
| Cobertura media | ~2-3% |
| Criterio | R > 10 AND R > G AND R > B |

---

## Como Usar

### Para Segmentacao de Gordura Epicardica (Recomendado)

```python
import numpy as np
from PIL import Image

def load_epicardial_mask(gt_path, threshold=10):
    """
    Carrega mascara de gordura EPICARDICA usando R dominante.
    
    Criterio: R > threshold AND R > G AND R > B
    Isso captura apenas onde epicardica e a classe principal.
    """
    gt_img = np.array(Image.open(gt_path))
    
    R = gt_img[:, :, 0]  # Epicardica
    G = gt_img[:, :, 1]  # Mediastinal
    B = gt_img[:, :, 2]  # Pericardio
    
    # R dominante: epicardica e a classe principal
    mask = (R > threshold) & (R > G) & (R > B)
    
    return mask.astype(np.float32)
```

### Para Segmentacao Multi-classe

```python
def load_multiclass_mask(gt_path, threshold=10):
    """
    Carrega mascara com 3 classes + fundo.
    
    Classes:
        0 = Fundo
        1 = Gordura Epicardica
        2 = Gordura Mediastinal
        3 = Pericardio
    """
    gt_img = np.array(Image.open(gt_path))
    
    R = gt_img[:, :, 0]
    G = gt_img[:, :, 1]
    B = gt_img[:, :, 2]
    
    mask = np.zeros(R.shape, dtype=np.uint8)
    
    # Classe dominante por pixel
    r_dom = (R > threshold) & (R >= G) & (R >= B)
    g_dom = (G > threshold) & (G > R) & (G >= B)
    b_dom = (B > threshold) & (B > R) & (B > G)
    
    mask[r_dom] = 1  # Epicardica
    mask[g_dom] = 2  # Mediastinal
    mask[b_dom] = 3  # Pericardio
    
    return mask
```

---

## Observacoes Importantes

1. **Pixel preto (0) = Fundo**
   - Nao deve ser contado como gordura

2. **Valores de intensidade**
   - Representam confianca/probabilidade da anotacao (0-255)
   - Use threshold > 10 para remover ruido

3. **Relacao Fat Image <-> Ground Truth**
   - Fat Image = max(R, G, B) do Ground Truth
   - Contem toda gordura, sem distincao de classe

4. **Regioes de Overlap**
   - Algumas regioes tem valores em multiplos canais
   - Representam zonas de transicao/incerteza anatomica

---

## Referencia

Se utilizar este dataset, cite o trabalho publicado pelos autores originais.

Fonte: Cardiac Fat Database - Computed Tomography (http://www.invisible-details.com/cardiac-fat-database.html)

---

## Relacao entre os arquivos

```
Dicom_original (CT completo, 16-bit)
       │
       ├──> Janela [-200, -30] HU ──> Fat Images (8-bit, apenas gordura)
       │
       └──> Segmentacao Manual ──> Ground Truth (RGB, 3 classes)
                                        │
                                        ├── R: Gordura Epicardica
                                        ├── G: Gordura Mediastinal
                                        └── B: Pericardio
```
