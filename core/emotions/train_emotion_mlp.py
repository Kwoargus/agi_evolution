import json
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
from collections import Counter

DB_CONFIG = {
    'host': 'localhost',
    'database': 'postgres',
    'user': 'postgres',
    'password': 'postgres'
}

def load_embeddings_and_labels():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT 
            e.embedding AS event_embedding,
            r.type AS emotion_type
        FROM agi_evolution.event_emotion_link l
        JOIN agi_evolution.trigger_event e ON l.event_id = e.id
        JOIN agi_evolution.emotion_respons r ON l.emotion_id = r.id
        WHERE e.embedding IS NOT NULL AND r.embedding IS NOT NULL
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    X = []
    y = []
    for row in rows:
        # psycopg2 возвращает list для jsonb
        emb = np.array(row['event_embedding'])
        X.append(emb)
        y.append(row['emotion_type'])
    return np.array(X), y

class EmotionMLP(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, num_classes=10):
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

def main():
    X, y = load_embeddings_and_labels()
    print(f"Загружено {len(X)} примеров.")

    if len(X) < 10:
        print("Слишком мало данных для обучения.")
        return

    # Проверяем распределение классов
    counter = Counter(y)
    print(f"Распределение классов: {dict(counter)}")

    # Оставляем только классы с >= 2 примерами
    min_samples_per_class = 2
    valid_classes = [cls for cls, count in counter.items() if count >= min_samples_per_class]
    if len(valid_classes) < 2:
        print("⚠️ Слишком мало классов для обучения.")
        return

    filtered_indices = [i for i, cls in enumerate(y) if cls in valid_classes]
    X_filtered = X[filtered_indices]
    y_filtered = [y[i] for i in filtered_indices]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_filtered)
    num_classes = len(le.classes_)
    print(f"Количество классов после фильтрации: {num_classes}")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
    except ValueError:
        print("⚠️ Стратификация невозможна, используем обычное разбиение.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_encoded, test_size=0.2, random_state=42
        )

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = EmotionMLP(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 200
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        preds = []
        for batch_x, batch_y in test_loader:
            pred = model(batch_x)
            _, pred_idx = torch.max(pred, 1)
            preds.extend(pred_idx.numpy())
        accuracy = accuracy_score(y_test, preds)
        print(f"Точность на тесте: {accuracy:.3f}")

    torch.save(model.state_dict(), "emotion_mlp.pth")
    with open("emotion_label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print("✅ Модель и энкодер сохранены.")

if __name__ == '__main__':
    main()