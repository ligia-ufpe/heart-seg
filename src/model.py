import glob
import os
import torch
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord, DivisiblePadd
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet

# Diretórios e padrões
IMAGE_DIR = 'data/processed'
LABEL_DIR = 'data/segmented'
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

# Definir modelo UNet
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)

# Dispositivo: GPU se disponível
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

# Salvar modelo
torch.save(model.state_dict(), 'unet_heart_seg.pth')
print('Modelo salvo')

# Função para extrair bounding box da máscara segmentada
def get_bounding_box(mask):
    coords = np.argwhere(mask == 1)
    if coords.size == 0:
        return None, None
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    return min_coords, max_coords
