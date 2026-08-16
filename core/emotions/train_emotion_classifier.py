import torch

import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

from core.emotions.emotion_classifier import EmotionClassifier


def train_emotion_classifier(event_embs, emotion_labels, num_epochs=200):
    # Преобразуем метки в числа
    le = LabelEncoder()
    y_encoded = le.fit_transform(emotion_labels)
    num_classes = len(le.classes_)

    X = torch.FloatTensor(event_embs)
    y = torch.LongTensor(y_encoded)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = EmotionClassifier(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    return model, le