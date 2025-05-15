import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import torch.nn.functional as F

from ..configs.config import ModelConfig
from ..utils.preprocessing import preprocess_person, preprocess_cloth
from ..utils.postprocessing import postprocess_result
from ..utils.model_loader import (
    load_human_parser,
    load_cloth_parser,
    load_vto_model,
    ensure_checkpoints
)

class DualMeModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Assicura che i checkpoint siano presenti
        ensure_checkpoints()
        
        # Carica i modelli necessari
        self._load_models()
        
    def _load_models(self):
        """Carica i modelli necessari per il virtual try-on."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        
        # Carica il modello di parsing umano
        self.human_parser = load_human_parser(
            checkpoint_dir / "human_parser.pth",
            self.device
        )
        
        # Carica il modello di parsing del vestito
        self.cloth_parser = load_cloth_parser(
            checkpoint_dir / "cloth_parser.pth",
            self.device
        )
        
        # Carica il modello di virtual try-on
        self.vto_model = load_vto_model(
            checkpoint_dir / "vto_model.pth",
            self.device
        )
        
        # Imposta i modelli in modalità eval
        self.human_parser.eval()
        self.cloth_parser.eval()
        self.vto_model.eval()
        
        # Imposta la precisione se necessario
        if self.config.precision == "float16":
            self.human_parser.half()
            self.cloth_parser.half()
            self.vto_model.half()
    
    def preprocess(
        self,
        person_path: str,
        cloth_path: str,
        cloth_type: str
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Preprocessa le immagini di input."""
        # Preprocessa l'immagine della persona
        person_tensor = preprocess_person(
            person_path,
            self.config.image_size,
            self.device
        )
        
        # Preprocessa l'immagine del vestito
        cloth_tensor = preprocess_cloth(
            cloth_path,
            self.config.image_size,
            self.device
        )
        
        # Ottieni le maschere di parsing
        with torch.no_grad():
            person_mask = self.human_parser(person_tensor)
            cloth_mask = self.cloth_parser(cloth_tensor)
        
        # Prepara i metadati
        metadata = {
            "cloth_type": cloth_type,
            "original_size": Image.open(person_path).size,
            "person_mask": person_mask,
            "cloth_mask": cloth_mask
        }
        
        return person_tensor, cloth_tensor, metadata
        
    def forward(
        self,
        person_tensor: torch.Tensor,
        cloth_tensor: torch.Tensor,
        metadata: dict
    ) -> torch.Tensor:
        """Esegue il forward pass del modello."""
        # Combina i tensori di input
        combined_input = torch.cat([person_tensor, cloth_tensor], dim=1)
        
        # Applica le maschere in base al tipo di vestito
        if metadata["cloth_type"] == "dressed":
            mask = metadata["cloth_mask"]
        elif metadata["cloth_type"] == "upper":
            mask = metadata["cloth_mask"] * (metadata["person_mask"] > 0.5)
        else:  # bottom
            mask = metadata["cloth_mask"] * (metadata["person_mask"] < 0.5)
        
        # Esegui il forward pass del modello VTO
        output = self.vto_model(combined_input)
        
        # Applica la maschera
        result = output * mask + person_tensor * (1 - mask)
        
        return result
        
    def inference(
        self,
        person_path: str,
        cloth_path: str,
        cloth_type: str
    ) -> Optional[Image.Image]:
        """Esegue l'inferenza completa."""
        try:
            # Preprocessa le immagini
            person_tensor, cloth_tensor, metadata = self.preprocess(
                person_path,
                cloth_path,
                cloth_type
            )
            
            # Esegui il forward pass
            with torch.no_grad():
                result_tensor = self.forward(
                    person_tensor,
                    cloth_tensor,
                    metadata
                )
            
            # Postprocessa il risultato
            result_image = postprocess_result(
                result_tensor,
                metadata["original_size"]
            )
            
            return result_image
            
        except Exception as e:
            print(f"Errore durante l'inferenza: {str(e)}")
            return None 