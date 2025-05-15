from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    # Configurazioni del modello
    model_name: str = "dualme"
    image_size: int = 512
    batch_size: int = 1
    
    # Tipi di abiti supportati
    CLOTH_TYPES = ["dressed", "bottom", "upper"]
    
    # Percorsi dei checkpoint
    checkpoint_dir: str = "checkpoints"
    
    # Configurazioni per l'inferenza
    device: str = "cpu"
    precision: str = "float32"  # float32 per evitare errori di tipo
    
    # Configurazioni per il preprocessing
    human_parser_model: str = "human_parser"
    cloth_parser_model: str = "cloth_parser"

@dataclass
class AppConfig:
    # Configurazioni per l'app Gradio
    app_name: str = "DualMe Virtual Try-On"
    app_description: str = "Prova virtualmente i tuoi vestiti"
    app_port: int = 7860
    app_share: bool = True
    
    # Configurazioni per RunPod
    runpod_api_key: Optional[str] = None
    runpod_endpoint: Optional[str] = None 