import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Tuple

def preprocess_person(
    image_path: str,
    target_size: int,
    device: torch.device
) -> torch.Tensor:
    """Preprocessa l'immagine della persona."""
    # Carica l'immagine
    image = Image.open(image_path).convert('RGB')
    
    # Definisci le trasformazioni
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Applica le trasformazioni
    tensor = transform(image).unsqueeze(0).to(device)
    
    return tensor

def preprocess_cloth(
    image_path: str,
    target_size: int,
    device: torch.device
) -> torch.Tensor:
    """Preprocessa l'immagine del vestito."""
    # Carica l'immagine
    image = Image.open(image_path).convert('RGB')
    
    # Definisci le trasformazioni
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Applica le trasformazioni
    tensor = transform(image).unsqueeze(0).to(device)
    
    return tensor

def preprocess_image(image, target_size=(768, 1024)):
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    w, h = image.size
    ratio = min(target_size[0]/w, target_size[1]/h)
    new_size = (int(w*ratio), int(h*ratio))
    image = image.resize(new_size, Image.LANCZOS)
    
    new_image = Image.new("RGB", target_size, (255, 255, 255))
    new_image.paste(image, ((target_size[0]-new_size[0])//2, 
                           (target_size[1]-new_size[1])//2))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(new_image).unsqueeze(0) 