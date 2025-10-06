import matplotlib.pyplot as plt
import torch
import os
from PIL import Image
import numpy as np

def visualize_sample(image_tensor: torch.Tensor, 
                     label_tensor: torch.Tensor, 
                     slice:int =6):
    """
    Visualiza uma fatia específica da imagem e do rótulo lado a lado.
    Usa como padrão a primeira amostra do batch e a fatia de índice 6.

    Args:
        image_tensor (torch.Tensor): Tensor da imagem com shape (C, H, W, D).
        label_tensor (torch.Tensor): Tensor do rótulo com shape (C, H, W, D).
        slice (int): Índice da fatia a ser visualizada.
    """
    # Selecionar a fatia específica
    image_slice = image_tensor[0, 0, :, :, slice].cpu().numpy()
    label_slice = label_tensor[0, 0, :, :, slice].cpu().numpy()

    # Criar figura com subplots lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot da imagem
    ax1.imshow(image_slice, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Plot da máscara/label
    ax2.imshow(label_slice, cmap='gray')
    ax2.set_title('Label Mask')
    ax2.axis('off')

    plt.show()

def save_pred_label(output_tensor: torch.Tensor, 
                    label_tensor: torch.Tensor, 
                    output_dir: str, 
                    epoch: int,
                    slice:int = 6):
    """
    Salva a imagem, o rótulo verdadeiro e a predição do modelo como arquivos PNG.

    Args:
        image_tensor (torch.Tensor): Tensor da imagem com shape (C, H, W, D).
        label_tensor (torch.Tensor): Tensor do rótulo com shape (C, H, W, D).
        output_dir (str): Diretório onde os arquivos serão salvos.
        slice (int): Índice da fatia a ser salva.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (label, pred) in enumerate(zip(label_tensor, output_tensor)):
        # Selecionar a fatia específica
        label_slice = label[0, :, :, slice].detach().cpu().numpy()
        output_slice = np.argmax(pred[:, :, :, slice].detach().cpu().numpy(), axis=0)

        # Criar figura com subplots lado a lado
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot da imagem
        ax1.imshow(output_slice, cmap='gray')
        ax1.set_title('Prediction')
        ax1.axis('off')

        # Plot da máscara/label
        ax2.imshow(label_slice, cmap='gray')
        ax2.set_title('Label Mask')
        ax2.axis('off')

        # Salvar figura
        plt.savefig(os.path.join(output_dir, f'image_label_i={i}_epoch={epoch}.png'))
        plt.close()