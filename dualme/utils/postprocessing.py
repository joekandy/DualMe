import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Tuple

def postprocess_result(
    tensor: torch.Tensor,
    original_size: Tuple[int, int]
) -> Image.Image:
    """Postprocessa il tensore risultante in un'immagine."""
    # Rimuovi la dimensione del batch
    tensor = tensor.squeeze(0)
    
    # Denormalizza
    tensor = tensor * 0.5 + 0.5
    
    # Converti in numpy array
    array = tensor.cpu().numpy()
    
    # Trasponi da (C, H, W) a (H, W, C)
    array = np.transpose(array, (1, 2, 0))
    
    # Converti in uint8
    array = (array * 255).astype(np.uint8)
    
    # Crea l'immagine PIL
    image = Image.fromarray(array)
    
    # Ridimensiona alle dimensioni originali
    image = image.resize(original_size, Image.LANCZOS)
    
    return image 