import os
import torch
import gradio as gr
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import logging
from ..utils.monitoring import get_performance_monitor
from ..utils.setup_models import setup_models
from ..utils.theme import get_theme
from ..utils.custom_css import CUSTOM_CSS

# Configurazione del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/logs/app.log'),
        logging.StreamHandler()
    ]
)

# Configurazioni
MODEL_PATH = "/workspace/models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGO_PATH = "/workspace/logo-DualMe.png"

# Trasformazioni per le immagini
transform = transforms.Compose([
    transforms.Resize((512, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Trasformazione inversa per convertire il tensore in immagine
inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
    transforms.ToPILImage()
])

def load_model(model_type="upper"):
    """Carica il modello appropriato"""
    model_path = os.path.join(MODEL_PATH, f"{model_type}_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modello {model_type} non trovato: {model_path}")
    
    model = torch.load(model_path, map_location=DEVICE)
    model.eval()
    return model

def process_images(person_image, clothing_image, model_type):
    """Processa le immagini e genera il risultato del VTO"""
    try:
        # Inizia il monitoraggio
        monitor = get_performance_monitor()
        request_data = monitor.start_request()
        
        # Carica il modello
        model = load_model(model_type)
        
        # Converti le immagini in tensori
        person_tensor = transform(Image.fromarray(person_image))
        clothing_tensor = transform(Image.fromarray(clothing_image))
        
        # Aggiungi dimensione batch
        person_tensor = person_tensor.unsqueeze(0).to(DEVICE)
        clothing_tensor = clothing_tensor.unsqueeze(0).to(DEVICE)
        
        # Esegui l'inferenza
        with torch.no_grad():
            output = model(person_tensor, clothing_tensor)
        
        # Converti l'output in immagine
        output = output.squeeze(0).cpu()
        output_image = inverse_transform(output)
        
        # Registra le metriche
        phase_times = {
            'model_loading': 0.0,  # Il modello è già caricato
            'keypoint_mask_generation': 0.0,  # Non necessario per questo modello
            'densepose_inference': 0.0,  # Non necessario per questo modello
            'image_generation': 0.0  # Non necessario per questo modello
        }
        monitor.end_request(request_data, phase_times)
        
        return np.array(output_image)
        
    except Exception as e:
        logging.error(f"Errore durante il processing: {str(e)}")
        return None

def create_interface():
    """Crea l'interfaccia Gradio"""
    with gr.Blocks(theme=get_theme(), title="DualMe Virtual Try-On", css=CUSTOM_CSS) as demo:
        # Header con logo
        with gr.Row():
            gr.Image(LOGO_PATH, show_label=False, elem_id="logo")
        
        gr.Markdown(
            """
            # Benvenuto in DualMe Virtual Try-On
            Prova virtualmente i tuoi vestiti preferiti in pochi secondi!
            """
        )
        
        with gr.Row():
            with gr.Column():
                person_input = gr.Image(
                    label="Immagine Persona",
                    type="numpy",
                    elem_id="person_input"
                )
                clothing_input = gr.Image(
                    label="Immagine Vestito",
                    type="numpy",
                    elem_id="clothing_input"
                )
                model_type = gr.Radio(
                    choices=["upper", "dressed"],
                    label="Tipo di Vestito",
                    value="upper",
                    elem_id="model_type"
                )
                process_btn = gr.Button(
                    "Prova il Vestito",
                    variant="primary",
                    elem_id="process_btn"
                )
            
            with gr.Column():
                output_image = gr.Image(
                    label="Risultato",
                    elem_id="output_image"
                )
        
        # Footer
        gr.Markdown(
            """
            ---
            *DualMe Virtual Try-On - Powered by AI*
            """
        )
        
        process_btn.click(
            fn=process_images,
            inputs=[person_input, clothing_input, model_type],
            outputs=output_image
        )
    
    return demo

def main():
    """Funzione principale"""
    try:
        # Configura i modelli
        setup_models()
        
        # Crea e avvia l'interfaccia
        demo = create_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True
        )
    except Exception as e:
        logging.error(f"Errore durante l'avvio dell'applicazione: {str(e)}")
        raise

if __name__ == "__main__":
    main() 