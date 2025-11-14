import glob
import os
import torch
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord, DivisiblePadd
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet
from utils import visualize_sample, save_pred_label

# Configurações iniciais
SAVE_OUTPUTS_N_EPOCHS = 5 # Salvar saídas a cada n épocas
observed_data = None # Tensor para armazenar as imagens e rótulos observados ao longo do treinamento
SHOW_SAMPLE = True # Mostrar uma amostra do dataset

# Diretórios e padrões
IMAGE_DIR = 'data/processed'
LABEL_DIR = 'data/segmented'
OUTPUT_DIR = 'intermediate_outputs'
IMAGE_PATTERN = 'CINE_EC_*.nii.gz'
LABEL_SUFFIX = '_SEG.nii'

# Coletar imagens e rótulos
images = sorted(glob.glob(os.path.join(IMAGE_DIR, IMAGE_PATTERN)))
data_dicts = []
for img in images:
    base = os.path.basename(img).replace('.nii.gz', '')
    label = os.path.join(LABEL_DIR, f'{base}{LABEL_SUFFIX}')
    if os.path.exists(label):
        data_dicts.append({'image': img, 'label': label})
    else:
        print(f'Label não encontrada para: {img}')

# Transforms com padding divisível por 16
transforms = Compose([
    LoadImaged(keys=['image', 'label']),
    EnsureChannelFirstd(keys=['image', 'label']),
    ScaleIntensityd(keys=['image']),
    ToTensord(keys=['image', 'label']),
    DivisiblePadd(keys=["image", "label"], k=16)
])

# Dataset e DataLoader
train_ds = Dataset(data=data_dicts, transform=transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

# Visualização de exemplo
batch = next(iter(train_loader))

if SHOW_SAMPLE:
    visualize_sample(batch['image'], batch['label'], slice=6)

# Configurar imagens a serem observadas
observed_data = batch

# Definir modelo UNet
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3, # Ajuste conforme o número de classes
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)

# Dispositivo: GPU se disponível
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Usando dispositivo: {device}')
model = model.to(device)

# Otimizador e função de perda
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_function = torch.nn.CrossEntropyLoss()

# Treinamento
model.train()
n = 100
for epoch in range(n):  # ajuste epochs conforme necessário
    for batch in train_loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        outputs = model(images)

        if labels.shape[1] == 1:
            labels = labels.squeeze(1)

        loss = loss_function(outputs, labels.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    if SAVE_OUTPUTS_N_EPOCHS and epoch % SAVE_OUTPUTS_N_EPOCHS == 0:
        images = observed_data['image'].to(device)
        labels = observed_data['label'].to(device)
        outputs = model(images)
        save_pred_label(outputs, labels, OUTPUT_DIR, slice=6, epoch=epoch)

# Salvar modelo
model_path = 'src/models'

torch.save(model.state_dict(), model_path + '/unet_heart_seg.pth')
print('Modelo salvo')

