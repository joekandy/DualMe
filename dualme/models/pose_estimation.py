import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseEstimationModel(nn.Module):
    def __init__(self, num_keypoints=18):
        super(PoseEstimationModel, self).__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Keypoint detection
        self.keypoints = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_keypoints, 1)
        )
        
    def forward(self, x):
        features = self.features(x)
        keypoints = self.keypoints(features)
        return keypoints

def load_pose_model(model_path, device):
    """Carica il modello di stima della posa"""
    model = PoseEstimationModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model 