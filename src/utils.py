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

# ==================== Utils from the SAM2 notebook ====================
def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
# ==================== Utils from the SAM2 notebook ====================