import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

class DualMeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config['model']['image_size']
        
        # Inizializzazione dei componenti del modello
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.attention = self._build_attention()
        
    def _build_encoder(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
    def _build_decoder(self):
        return nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
    
    def _build_attention(self):
        return nn.MultiheadAttention(128, 4)
    
    def forward(self, person_image, garment_image, garment_type):
        person_features = self.encoder(person_image)
        garment_features = self.encoder(garment_image)
        
        attn_output, _ = self.attention(
            person_features.flatten(2).permute(2, 0, 1),
            garment_features.flatten(2).permute(2, 0, 1),
            garment_features.flatten(2).permute(2, 0, 1)
        )
        
        output = self.decoder(attn_output.permute(1, 2, 0).view_as(person_features))
        
        return output 