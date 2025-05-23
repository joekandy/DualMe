import os
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_models():
    """Configura i modelli nella posizione corretta"""
    workspace_dir = Path("/workspace")
    models_dir = workspace_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Sposta i modelli nella directory corretta
    model_files = {
        "virtual_tryon.pth": "upper_model.pth",
        "virtual_tryon_dc.pth": "dressed_model.pth"
    }
    
    for src_name, dst_name in model_files.items():
        src_path = workspace_dir / src_name
        dst_path = models_dir / dst_name
        
        if src_path.exists():
            logger.info(f"Sposto {src_name} in {dst_path}")
            shutil.move(str(src_path), str(dst_path))
        else:
            logger.error(f"Modello {src_name} non trovato!")
            raise FileNotFoundError(f"Modello {src_name} non trovato!")

if __name__ == "__main__":
    setup_models() 