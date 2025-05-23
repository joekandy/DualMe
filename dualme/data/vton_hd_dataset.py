import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path

class VTONHDDataset(Dataset):
    def __init__(self, root_dir, phase='train', image_size=512):
        """
        Dataset per HD-VTON
        
        Args:
            root_dir (str): Directory principale del dataset
            phase (str): 'train' o 'test'
            image_size (int): Dimensione delle immagini
        """
        self.root_dir = Path(root_dir)
        self.phase = phase
        self.image_size = image_size
        
        # Directory per le immagini
        self.person_dir = self.root_dir / phase / 'person'
        self.clothing_dir = self.root_dir / phase / 'clothing'
        
        # Lista dei file
        self.person_files = sorted(list(self.person_dir.glob('*.jpg')))
        self.clothing_files = sorted(list(self.clothing_dir.glob('*.jpg')))
        
        # Trasformazioni
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.person_files)
    
    def __getitem__(self, idx):
        # Carica le immagini
        person_img = Image.open(self.person_files[idx]).convert('RGB')
        clothing_img = Image.open(self.clothing_files[idx]).convert('RGB')
        
        # Applica le trasformazioni
        person_tensor = self.transform(person_img)
        clothing_tensor = self.transform(clothing_img)
        
        return {
            'person': person_tensor,
            'clothing': clothing_tensor,
            'person_path': str(self.person_files[idx]),
            'clothing_path': str(self.clothing_files[idx])
        }

def get_dataloader(root_dir, batch_size=8, num_workers=4, phase='train'):
    """
    Crea il DataLoader per il dataset
    
    Args:
        root_dir (str): Directory principale del dataset
        batch_size (int): Dimensione del batch
        num_workers (int): Numero di worker per il caricamento
        phase (str): 'train' o 'test'
    """
    dataset = VTONHDDataset(root_dir, phase=phase)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(phase == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader 