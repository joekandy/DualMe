import os
import gradio as gr
from gradio_client import Client, handle_file
from PIL import Image
import logging
import traceback
import numpy as np
import cv2
import io
import tempfile

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Percorsi
LOGO_PATH = os.path.join(os.path.dirname(__file__), "logo-DualMe.png")

# Colori personalizzati
PRIMARY_COLOR = "#1E88E5"  # Blu
SECONDARY_COLOR = "#43A047"  # Verde
BACKGROUND_COLOR = "#FFFFFF"  # Bianco
TEXT_COLOR = "#333333"  # Grigio scuro

def save_image(image_array):
    """Salva l'immagine in un file temporaneo e restituisce il percorso"""
    try:
        # Converti l'array numpy in immagine PIL
        image = Image.fromarray(image_array)
        
        # Crea un file temporaneo
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_path = temp_file.name
        
        # Salva l'immagine
        image.save(temp_path, 'PNG')
        return temp_path
    except Exception as e:
        logger.error(f"Errore durante il salvataggio dell'immagine: {str(e)}")
        raise

def process_image(human_img_path, garm_img_path, model_type, garment_type, accelerate=False, repaint=False):
    """Processa le immagini usando il modello LeFFA"""
    try:
        logger.info(f"Caricamento immagine persona: {human_img_path}")
        logger.info(f"Caricamento immagine vestito: {garm_img_path}")
        logger.info(f"Tipo modello: {model_type}")
        logger.info(f"Tipo vestito: {garment_type}")
        
        logger.info("Inizializzazione client LeFFA...")
        client = Client("franciszzj/Leffa")
        logger.info("Client LeFFA inizializzato")
        
        logger.info("Preparazione file per la predizione...")
        src_file = handle_file(human_img_path)
        ref_file = handle_file(garm_img_path)
        logger.info("File preparati")
        
        logger.info("Avvio predizione...")
        result = client.predict(
            src_image_path=src_file,
            ref_image_path=ref_file,
            ref_acceleration=accelerate,
            step=30,
            scale=2.5,
            seed=42,
            vt_model_type=model_type,
            vt_garment_type=garment_type,
            vt_repaint=repaint,
            api_name="/leffa_predict_vt"
        )
        logger.info(f"Predizione completata: {result}")
        
        generated_image_path = result[0]
        logger.info(f"Apertura immagine generata: {generated_image_path}")
        
        # Leggi l'immagine con PIL
        generated_image = Image.open(generated_image_path)
        logger.info(f"Immagine letta con PIL, formato: {generated_image.format}, dimensione: {generated_image.size}")
        
        # Converti in array numpy
        generated_image = np.array(generated_image)
        logger.info(f"Immagine convertita in array numpy con shape: {generated_image.shape}")
        
        return generated_image
        
    except Exception as e:
        logger.error(f"Errore durante il processing: {str(e)}")
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        raise

def clear_all():
    """Pulisce tutti gli input e output"""
    return None, None, None

def create_interface():
    """Crea l'interfaccia Gradio"""
    custom_css = f"""
        .gradio-container {{
            background-color: {BACKGROUND_COLOR} !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        .primary-button {{
            background-color: {PRIMARY_COLOR} !important;
            border-color: {PRIMARY_COLOR} !important;
            color: white !important;
            font-weight: bold !important;
            padding: 10px 20px !important;
            border-radius: 5px !important;
        }}
        .secondary-button {{
            background-color: {SECONDARY_COLOR} !important;
            border-color: {SECONDARY_COLOR} !important;
            color: white !important;
            font-weight: bold !important;
            padding: 10px 20px !important;
            border-radius: 5px !important;
        }}
        .gradio-interface {{
            max-width: 1200px !important;
            margin: 0 auto !important;
            padding: 20px !important;
        }}
        .gradio-image {{
            border-radius: 10px !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }}
        .gradio-radio {{
            background-color: #f5f5f5 !important;
            padding: 15px !important;
            border-radius: 8px !important;
            margin: 10px 0 !important;
        }}
        .gradio-accordion {{
            background-color: #f5f5f5 !important;
            border-radius: 8px !important;
            margin: 10px 0 !important;
        }}
        #logo {{
            max-width: 200px !important;
            height: auto !important;
            margin: 0 auto !important;
            display: block !important;
        }}
    """
    
    with gr.Blocks(title="DualMe Virtual Try-On", css=custom_css) as demo:
        # Header con logo
        with gr.Row():
            gr.Image(LOGO_PATH, show_label=False, elem_id="logo")
        
        gr.HTML(f"<center><h1 style='color: {TEXT_COLOR}; font-size: 2.5em; margin: 20px 0;'>DualMe Virtual Try-On</h1></center>")
        gr.HTML(f"<center><p style='color: {TEXT_COLOR}; font-size: 1.2em; margin-bottom: 30px;'>Carica un'immagine di una persona e un'immagine di un capo d'abbigliamento ✨</p></center>")
        
        with gr.Row():
            with gr.Column():
                human_img = gr.Image(type="filepath", label='Persona', interactive=True)
            
            with gr.Column():
                garm_img = gr.Image(label="Vestito", type="filepath", interactive=True)
            
            with gr.Column():
                image_out = gr.Image(label="Risultato", type="numpy", format="png")
        
        with gr.Row():
            with gr.Column():
                model_type = gr.Radio(
                    choices=["viton_hd", "dress_code"],
                    value="viton_hd",
                    label="Tipo Modello",
                    info="VITON-HD (Consigliato) o DressCode (Sperimentale)"
                )
                
                garment_type = gr.Radio(
                    choices=["upper_body", "lower_body", "dress"],
                    value="upper_body",
                    label="Tipo Vestito",
                    info="Upper (maglie), Lower (pantaloni), Dress (vestiti)"
                )
                
                with gr.Accordion("Opzioni Avanzate", open=False):
                    accelerate = gr.Checkbox(
                        label="Accelera Reference UNet",
                        value=False,
                        info="Può ridurre leggermente le prestazioni"
                    )
                    
                    repaint = gr.Checkbox(
                        label="Modalità Repaint",
                        value=False
                    )
        
        with gr.Row():
            try_button = gr.Button(value="Prova", variant='primary', elem_classes=["primary-button"])
            clear_button = gr.Button(value="Pulisci", variant='secondary', elem_classes=["secondary-button"])
            download_button = gr.Button(value="Scarica", variant='secondary', elem_classes=["secondary-button"])
        
        # Collegamenti dei pulsanti
        try_button.click(
            fn=process_image, 
            inputs=[
                human_img, 
                garm_img, 
                model_type, 
                garment_type, 
                accelerate, 
                repaint
            ], 
            outputs=image_out
        )
        
        clear_button.click(
            fn=clear_all,
            inputs=[],
            outputs=[human_img, garm_img, image_out]
        )
        
        download_button.click(
            fn=save_image,
            inputs=[image_out],
            outputs=[gr.File(label="Scarica immagine")]
        )
    
    return demo

def main():
    """Funzione principale"""
    try:
        logger.info("Avvio applicazione...")
        demo = create_interface()
        logger.info("Interfaccia creata, avvio server...")
        demo.launch(share=True)
    except Exception as e:
        logger.error(f"Errore durante l'avvio dell'applicazione: {str(e)}")
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main() 