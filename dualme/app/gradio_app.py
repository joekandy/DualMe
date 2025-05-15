import os
import torch
import gradio as gr
import yaml
from PIL import Image
import torchvision.transforms as transforms

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
        with gr.Blocks(title="DualMe Virtual Try-On") as demo:
            gr.Markdown("# 👔 DualMe Virtual Try-On")
            gr.Markdown("Carica un'immagine della persona e un capo d'abbigliamento per vedere il risultato.")
            
            with gr.Row():
                with gr.Column():
                    person_input = gr.Image(label="Immagine Persona", type="pil")
                    garment_input = gr.Image(label="Immagine Capo", type="pil")
                    garment_type = gr.Radio(
                        choices=self.config['garment_types'],
                        label="Tipo di Capo",
                        value="dressed"
                    )
                    generate_btn = gr.Button("Genera", variant="primary")
                
                with gr.Column():
                    output_image = gr.Image(label="Risultato", type="pil")
            
            generate_btn.click(
                fn=self.predict,
                inputs=[person_input, garment_input, garment_type],
                outputs=output_image
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