import os
import gradio as gr
from gradio_client import Client, handle_file
from PIL import Image
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_image(human_img_path, garm_img_path):
    """Processa le immagini usando il modello DualMe"""
    try:
        client = Client("dualme/DualMe")  # Aggiorna qui se hai un repo HuggingFace personale
        
        result = client.predict(
            src_image_path=handle_file(human_img_path),
            ref_image_path=handle_file(garm_img_path),
            ref_acceleration=False,
            step=30,
            scale=2.5,
            seed=42,
            vt_model_type="viton_hd",
            vt_garment_type="upper_body",
            vt_repaint=False,
            api_name="/dualme_predict_vt"
        )
        
        generated_image_path = result[0]
        generated_image = Image.open(generated_image_path)
        return generated_image
        
    except Exception as e:
        logger.error(f"Errore durante il processing: {str(e)}")
        raise

def create_interface():
    """Crea l'interfaccia Gradio"""
    with gr.Blocks(title="DualMe Test") as demo:
        gr.HTML("<center><h1>DualMe Virtual Try-On Test</h1></center>")
        gr.HTML("<center><p>Carica un'immagine di una persona e un'immagine di un capo d'abbigliamento ✨</p></center>")
        
        with gr.Row():
            with gr.Column():
                human_img = gr.Image(type="filepath", label='Persona', interactive=True)
            
            with gr.Column():
                garm_img = gr.Image(label="Vestito", type="filepath", interactive=True)
            
            with gr.Column():
                image_out = gr.Image(label="Risultato", type="pil")
        
        with gr.Row():
            try_button = gr.Button(value="Prova", variant='primary')
        
        try_button.click(fn=process_image, inputs=[human_img, garm_img], outputs=image_out)
    
    return demo

def main():
    """Funzione principale"""
    try:
        demo = create_interface()
        demo.launch(share=True)
    except Exception as e:
        logger.error(f"Errore durante l'avvio dell'applicazione: {str(e)}")
        raise

if __name__ == "__main__":
    main()
