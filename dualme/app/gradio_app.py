import os
import torch
import gradio as gr
import yaml
from PIL import Image
import torchvision.transforms as transforms
import logging

# Neptune (disabilitato di default)
try:
    import neptune
    NEPTUNE_ENABLED = False  # Cambia a True per abilitare
except ImportError:
    NEPTUNE_ENABLED = False

from dualme.models.virtual_tryon import DualMeModel
from dualme.utils.preprocessing import preprocess_image

class DualMeApp:
    def __init__(self):
        with open('dualme/configs/config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DualMeModel(self.config).to(self.device)
        
        if os.path.exists(self.config['paths']['checkpoints'] + 'model.pth'):
            self.model.load_state_dict(torch.load(
                self.config['paths']['checkpoints'] + 'model.pth',
                map_location=self.device
            ))
    
    def predict(self, person_image, garment_image, garment_type):
        person_tensor = preprocess_image(person_image).to(self.device)
        garment_tensor = preprocess_image(garment_image).to(self.device)
        
        with torch.no_grad():
            output = self.model(person_tensor, garment_tensor, garment_type)
        
        output = output.cpu().squeeze(0)
        output = (output + 1) / 2
        output = output.clamp(0, 1)
        output = transforms.ToPILImage()(output)
        
        return output
    
    def create_interface(self):
        custom_css = """
            .gradio-container {
                background-color: #FFFFFF !important;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .primary-button {
                background-color: #1E88E5 !important;
                border-color: #1E88E5 !important;
                color: white !important;
                font-weight: bold !important;
                padding: 10px 20px !important;
                border-radius: 5px !important;
            }
            .secondary-button {
                background-color: #43A047 !important;
                border-color: #43A047 !important;
                color: white !important;
                font-weight: bold !important;
                padding: 10px 20px !important;
                border-radius: 5px !important;
            }
            .gradio-interface {
                max-width: 1200px !important;
                margin: 0 auto !important;
                padding: 20px !important;
            }
            .gradio-image {
                border-radius: 10px !important;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
            }
            .gradio-radio {
                background-color: #f5f5f5 !important;
                padding: 15px !important;
                border-radius: 8px !important;
                margin: 10px 0 !important;
            }
            .gradio-accordion {
                background-color: #f5f5f5 !important;
                border-radius: 8px !important;
                margin: 10px 0 !important;
            }
            #logo {
                max-width: 200px !important;
                height: auto !important;
                margin: 0 auto !important;
                display: block !important;
            }
        """
        with gr.Blocks(title="DualMe Virtual Try-On", css=custom_css) as demo:
            gr.Markdown("# 👔 DualMe Virtual Try-On")
            gr.Markdown("Carica un'immagine della persona e un capo d'abbigliamento per vedere il risultato.")
            # Logo opzionale
            logo_path = os.path.join(os.path.dirname(__file__), "../../logo-DualMe.png")
            if os.path.exists(logo_path):
                gr.Image(logo_path, show_label=False, elem_id="logo")
            with gr.Row():
                with gr.Column():
                    person_input = gr.Image(label="Immagine Persona", type="pil")
                    garment_input = gr.Image(label="Immagine Capo", type="pil")
                    garment_type = gr.Radio(
                        choices=self.config['garment_types'],
                        label="Tipo di Capo",
                        value="dressed"
                    )
                    generate_btn = gr.Button("Genera", variant="primary", elem_classes=["primary-button"])
                    clear_btn = gr.Button("Pulisci", variant="secondary", elem_classes=["secondary-button"])
                with gr.Column():
                    output_image = gr.Image(label="Risultato", type="pil")
            generate_btn.click(
                fn=self.predict,
                inputs=[person_input, garment_input, garment_type],
                outputs=output_image
            )
            clear_btn.click(
                fn=lambda: (None, None, None),
                inputs=[],
                outputs=[person_input, garment_input, output_image]
            )
        return demo

def main():
    app = DualMeApp()
    demo = app.create_interface()
    demo.launch(
        server_name=app.config['app']['host'],
        server_port=app.config['app']['port'],
        share=app.config['app']['share'],
        enable_queue=app.config['app']['enable_queue']
    )

if __name__ == "__main__":
    main() 