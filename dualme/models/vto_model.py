import torch
import torch.nn as nn
import torch.nn.functional as F

class VTOModel(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(VTOModel, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels * 2, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_channels, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, person, clothing):
        # Concatena le immagini
        x = torch.cat([person, clothing], dim=1)
        
        # Encoder
        features = self.encoder(x)
        
        # Decoder
        output = self.decoder(features)
        
        return output

def load_vto_model(model_path, device):
    """Carica il modello VTO"""
    model = VTOModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model 