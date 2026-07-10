# sensors/multimodal_sensor.py
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class VisionEncoder(nn.Module):
    """Кодирует визуальную информацию в вектор."""

    def __init__(self, input_dim: int = 64, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.encoder(x)


class SoundEncoder(nn.Module):
    """Кодирует звуковую информацию в вектор."""

    def __init__(self, input_dim: int = 32, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.encoder(x)


class SmellEncoder(nn.Module):
    """Кодирует информацию о запахе в вектор."""

    def __init__(self, input_dim: int = 16, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.encoder(x)


class MultimodalEncoder(nn.Module):
    """
    Объединяет vision, sound, smell в единый вектор.
    """

    def __init__(self,
                 vision_dim: int = 128,
                 sound_dim: int = 128,
                 smell_dim: int = 128,
                 combined_dim: int = 256):
        super().__init__()

        self.vision_encoder = VisionEncoder(latent_dim=vision_dim)
        self.sound_encoder = SoundEncoder(latent_dim=sound_dim)
        self.smell_encoder = SmellEncoder(latent_dim=smell_dim)

        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + sound_dim + smell_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, combined_dim),
            nn.Tanh()
        )

    def forward(self, vision, sound, smell):
        vision_encoded = self.vision_encoder(vision)
        sound_encoded = self.sound_encoder(sound)
        smell_encoded = self.smell_encoder(smell)

        combined = torch.cat([vision_encoded, sound_encoded, smell_encoded], dim=1)
        return self.fusion(combined)

    def encode_sensory(self, vision: np.ndarray, sound: np.ndarray, smell: np.ndarray) -> np.ndarray:
        """
        Удобный метод для кодирования сенсорных данных.
        """
        vision_tensor = torch.FloatTensor(vision).unsqueeze(0)
        sound_tensor = torch.FloatTensor(sound).unsqueeze(0)
        smell_tensor = torch.FloatTensor(smell).unsqueeze(0)

        with torch.no_grad():
            combined = self.forward(vision_tensor, sound_tensor, smell_tensor)

        return combined.numpy().flatten()