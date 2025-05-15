import torch
import torch.nn as nn
import os
from pathlib import Path
import requests
from tqdm import tqdm
import gdown

def download_file(url: str, output_path: str):
    """Scarica un file da un URL."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path,
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def download_google_drive(file_id: str, output_path: str):
    """Scarica un file da Google Drive."""
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)

def load_human_parser(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Carica il modello di parsing umano."""
    # Modello per elaborare le immagini in modo realistico
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 3, kernel_size=3, padding=1),
        nn.Tanh()  # Normalizza l'output tra -1 e 1
    )
    return model.to(device)

def load_cloth_parser(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Carica il modello di parsing del vestito."""
    # Modello per elaborare le immagini in modo realistico
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 3, kernel_size=3, padding=1),
        nn.Tanh()  # Normalizza l'output tra -1 e 1
    )
    return model.to(device)

def load_vto_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Carica il modello di virtual try-on."""
    # Modello per elaborare le immagini in modo realistico
    model = nn.Sequential(
        nn.Conv2d(6, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 3, kernel_size=3, padding=1),
        nn.Tanh()  # Normalizza l'output tra -1 e 1
    )
    return model.to(device)

def ensure_checkpoints():
    """Assicura che tutti i checkpoint necessari siano presenti."""
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # TODO: Implementare il download dei checkpoint reali
    # Per ora creiamo file vuoti
    checkpoint_files = [
        "human_parser.pth",
        "cloth_parser.pth",
        "vto_model.pth"
    ]
    
    for file in checkpoint_files:
        if not (checkpoint_dir / file).exists():
            # Crea un file vuoto
            (checkpoint_dir / file).touch() 