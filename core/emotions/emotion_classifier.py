# core/emotions/emotion_classifier.py
import torch
import torch.nn as nn

class EmotionClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, num_classes=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)