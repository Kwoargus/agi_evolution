# core/thinking/training_thinking_gan.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from core.thinking.create_embeddings_kg import load_all_node_embeddings
from core.thinking.train_massive_for_thinking_gan import load_dataset_with_embeddings


# --- Архитектуры ---

class Discriminator(nn.Module):
    def __init__(self, input_dim=384 * 2 + 1):  # task_emb + combo_emb + dot_product
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, task_emb, combo_emb):
        # Дополнительный признак: скалярное произведение
        dot = (task_emb * combo_emb).sum(dim=1, keepdim=True)
        combined = torch.cat([task_emb, combo_emb, dot], dim=1)
        return self.net(combined)


class Generator(nn.Module):
    def __init__(self, task_dim=384, noise_dim=64, output_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(task_dim + noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()  # нормализуем выход в [-1, 1]
        )

    def forward(self, task_emb, noise):
        x = torch.cat([task_emb, noise], dim=1)
        return self.net(x)


# --- Обучение ---

def train_gan(X_task, X_combo, y, node_embeddings, epochs=1000, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Преобразуем в тензоры
    X_task_t = torch.FloatTensor(X_task).to(device)
    X_combo_t = torch.FloatTensor(X_combo).to(device)
    y_t = torch.FloatTensor(y).unsqueeze(1).to(device)

    dataset = TensorDataset(X_task_t, X_combo_t, y_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Инициализация моделей
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)

    # Оптимизаторы
    d_optim = optim.Adam(discriminator.parameters(), lr=0.001)
    g_optim = optim.Adam(generator.parameters(), lr=0.001)

    # Функции потерь
    d_loss_fn = nn.MSELoss()
    g_loss_fn = nn.MSELoss()

    # Кэш эмбеддингов узлов для генератора (чтобы он мог выбирать реальные узлы)
    all_node_embs = torch.FloatTensor(list(node_embeddings.values())).to(device)

    for epoch in range(epochs):
        for task_emb, real_combo_emb, real_score in dataloader:
            batch_size_actual = task_emb.size(0)

            # --- Обучение дискриминатора ---
            # 1. Реальные комбинации
            pred_real = discriminator(task_emb, real_combo_emb)
            d_loss_real = d_loss_fn(pred_real, real_score)

            # 2. Фейковые комбинации (от генератора)
            noise = torch.randn(batch_size_actual, 64).to(device)
            fake_combo_emb = generator(task_emb, noise)
            pred_fake = discriminator(task_emb, fake_combo_emb)
            # Для фейковых целей – низкие оценки (0.1–0.3)
            fake_target = torch.full((batch_size_actual, 1), 0.2).to(device)
            d_loss_fake = d_loss_fn(pred_fake, fake_target)

            d_loss = d_loss_real + d_loss_fake
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            # --- Обучение генератора ---
            # Генератор пытается создать комбинацию, которую дискриминатор оценит высоко
            noise = torch.randn(batch_size_actual, 64).to(device)
            fake_combo_emb = generator(task_emb, noise)
            pred_fake = discriminator(task_emb, fake_combo_emb)
            # Цель: получить оценку 0.95
            g_target = torch.full((batch_size_actual, 1), 0.95).to(device)
            g_loss = g_loss_fn(pred_fake, g_target)

            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")

    return discriminator, generator


# --- Запуск ---
if __name__ == "__main__":
    # Загружаем эмбеддинги узлов
    node_embeddings = load_all_node_embeddings()
    print(f"Загружено {len(node_embeddings)} узлов.")

    # Загружаем датасет
    X_task, X_combo, y = load_dataset_with_embeddings("training_data_inventions.json", node_embeddings)
    print(f"Загружено {len(X_task)} примеров.")

    # Обучаем GAN
    discriminator, generator = train_gan(X_task, X_combo, y, node_embeddings, epochs=500)

    # Сохраняем модели
    torch.save(discriminator.state_dict(), "discriminator.pth")
    torch.save(generator.state_dict(), "generator.pth")
    print("Модели сохранены.")