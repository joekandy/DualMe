import os
import gdown
import torch
from pathlib import Path

def download_models():
    """Scarica i modelli necessari per il VTO"""
    # Crea la directory dei modelli se non esiste
    model_dir = Path("/workspace/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # URL dei modelli
    model_urls = {
        "vto_upper": "https://huggingface.co/franciszzj/Leffa/resolve/main/virtual_tryon.pth",
        "vto_dressed": "https://huggingface.co/franciszzj/Leffa/resolve/main/virtual_tryon_dc.pth",
        "human_parsing": "https://huggingface.co/franciszzj/Leffa/resolve/main/humanparsing/human_parsing_model.pth",
        "pose_estimation": "https://huggingface.co/franciszzj/Leffa/resolve/main/openpose/body_pose_model.pth"
    }
    
    # Nomi dei file di output
    output_names = {
        "vto_upper": "upper_model.pth",
        "vto_dressed": "dressed_model.pth",
        "human_parsing": "human_parsing_model.pth",
        "pose_estimation": "pose_model.pth"
    }
    
    for model_type, url in model_urls.items():
        output_path = model_dir / output_names[model_type]
        if not output_path.exists():
            print(f"Scaricamento modello {model_type}...")
            try:
                gdown.download(url, str(output_path), quiet=False)
                print(f"Modello {model_type} scaricato con successo")
            except Exception as e:
                print(f"Errore durante il download del modello {model_type}: {str(e)}")
        else:
            print(f"Modello {model_type} già presente")

if __name__ == "__main__":
    download_models() 