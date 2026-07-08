# models/instinct_gan.py
"""
GAN для генерации паттернов инстинктов.
Работает напрямую с 256-мерными паттернами.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Any, Optional, Tuple
import random


class InstinctGenerator(nn.Module):
    """Генератор для паттернов инстинктов."""

    def __init__(self, latent_dim: int = 64, output_dim: int = 256,
                 hidden_dims: List[int] = [512, 512, 256]):
        super(InstinctGenerator, self).__init__()

        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class InstinctDiscriminator(nn.Module):
    """Дискриминатор для оценки паттернов инстинктов."""

    def __init__(self, input_dim: int = 256, hidden_dims: List[int] = [512, 256, 128]):
        super(InstinctDiscriminator, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2, True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def evaluate_pattern(self, pattern: np.ndarray, device: torch.device = None) -> float:
        """Оценивает один паттерн."""
        if device is None:
            device = next(self.parameters()).device

        if pattern.ndim == 1:
            pattern = pattern.reshape(1, -1)

        pattern_tensor = torch.FloatTensor(pattern).to(device)
        with torch.no_grad():
            score = self.forward(pattern_tensor).cpu().item()
        return score


class InstinctGAN:
    """
    GAN для генерации паттернов инстинктов.
    Работает напрямую с 256-мерными паттернами.
    """

    def __init__(self,
                 latent_dim: int = 64,
                 pattern_dim: int = 256,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 lr_generator: float = 0.0002,
                 lr_discriminator: float = 0.0002,
                 beta1: float = 0.5,
                 beta2: float = 0.999,
                 batch_size: int = 8):

        self.device = torch.device(device)
        self.latent_dim = latent_dim
        self.pattern_dim = pattern_dim
        self.batch_size = batch_size

        # Инициализируем модели
        self.generator = InstinctGenerator(latent_dim, pattern_dim).to(self.device)
        self.discriminator = InstinctDiscriminator(pattern_dim).to(self.device)

        # Оптимизаторы
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=lr_generator,
            betas=(beta1, beta2)
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=lr_discriminator,
            betas=(beta1, beta2)
        )

        # Функция потерь
        self.criterion = nn.BCELoss()

        # Буфер для хранения паттернов
        self.pattern_buffer = []

        # Счётчики для статистики
        self.generator_losses = []
        self.discriminator_losses = []
        self.generation_count = 0

        print(f"InstinctGAN инициализирован на устройстве: {self.device}")
        print(f"  Pattern dim: {self.pattern_dim}")
        print(f"  Latent dim: {self.latent_dim}")
        print(f"  Batch size: {self.batch_size}")

    def add_patterns(self, patterns: List[np.ndarray]):
        """Добавляет паттерны в буфер."""
        self.pattern_buffer.extend(patterns)
        # Ограничиваем размер буфера
        if len(self.pattern_buffer) > 10000:
            self.pattern_buffer = self.pattern_buffer[-10000:]

    def prepare_training_data(self, patterns: Optional[List[np.ndarray]] = None) -> torch.Tensor:
        """Подготавливает данные для тренировки."""
        if patterns is None:
            patterns = self.pattern_buffer

        if not patterns:
            return None

        # Преобразуем в тензор
        patterns_tensor = torch.FloatTensor(np.array(patterns)).to(self.device)
        return patterns_tensor

    def train_step(self, real_patterns: torch.Tensor) -> Tuple[float, float]:
        """Выполняет один шаг тренировки GAN."""
        batch_size = min(self.batch_size, real_patterns.size(0))

        if batch_size == 0:
            return 0.0, 0.0

        if batch_size < 2:
            batch_size = 2
            if real_patterns.size(0) < batch_size:
                indices = torch.randint(0, real_patterns.size(0), (batch_size,))
                real_patterns = real_patterns[indices]

        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        # --- Тренируем дискриминатор ---
        self.optimizer_d.zero_grad()

        real_output = self.discriminator(real_patterns[:batch_size])
        d_loss_real = self.criterion(real_output, real_labels)

        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_patterns = self.generator(z)
        fake_output = self.discriminator(fake_patterns.detach())
        d_loss_fake = self.criterion(fake_output, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.optimizer_d.step()

        # --- Тренируем генератор ---
        self.optimizer_g.zero_grad()

        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_patterns = self.generator(z)
        fake_output = self.discriminator(fake_patterns)

        g_loss = self.criterion(fake_output, real_labels)
        g_loss.backward()
        self.optimizer_g.step()

        # Сохраняем потери
        g_loss_val = g_loss.item()
        d_loss_val = d_loss.item()
        self.generator_losses.append(g_loss_val)
        self.discriminator_losses.append(d_loss_val)

        return g_loss_val, d_loss_val

    def train(self, patterns: Optional[List[np.ndarray]] = None,
              epochs: int = 10, verbose: bool = False) -> Dict[str, List[float]]:
        """Тренирует GAN на паттернах."""
        if patterns is None:
            patterns = self.pattern_buffer

        if not patterns:
            print("Нет данных для тренировки GAN.")
            return {'g_loss': [], 'd_loss': []}

        patterns_tensor = self.prepare_training_data(patterns)
        if patterns_tensor is None:
            return {'g_loss': [], 'd_loss': []}

        dataset = TensorDataset(patterns_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        g_losses = []
        d_losses = []

        for epoch in range(epochs):
            epoch_g_losses = []
            epoch_d_losses = []

            for batch in dataloader:
                real_batch = batch[0]
                g_loss, d_loss = self.train_step(real_batch)
                g_losses.append(g_loss)
                d_losses.append(d_loss)
                epoch_g_losses.append(g_loss)
                epoch_d_losses.append(d_loss)

            if verbose and (epoch + 1) % 5 == 0:
                avg_g_loss = np.mean(epoch_g_losses)
                avg_d_loss = np.mean(epoch_d_losses)
                print(f"InstinctGAN Epoch {epoch + 1}/{epochs}: G_loss={avg_g_loss:.4f}, D_loss={avg_d_loss:.4f}")

        return {'g_loss': g_losses, 'd_loss': d_losses}

    def generate_pattern(self) -> np.ndarray:
        """Генерирует новый паттерн инстинкта."""
        self.generation_count += 1
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(1, self.latent_dim, device=self.device)
            pattern = self.generator(z).cpu().numpy().flatten()
        self.generator.train()
        return pattern

    def generate_batch(self, n: int = 10) -> np.ndarray:
        """Генерирует батч паттернов."""
        self.generation_count += n
        if n < 2:
            n = 2
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=self.device)
            patterns = self.generator(z).cpu().numpy()
        self.generator.train()
        return patterns

    # def generate_pattern(self) -> np.ndarray:
    #     """Генерирует новый паттерн инстинкта."""
    #     self.generation_count += 1
    #     self.generator.eval()
    #     with torch.no_grad():
    #         z = torch.randn(1, self.latent_dim, device=self.device)
    #         pattern = self.generator(z).cpu().numpy().flatten()
    #     self.generator.train()
    #     return pattern
    #
    # def generate_batch(self, n: int = 10) -> np.ndarray:
    #     """Генерирует батч паттернов."""
    #     self.generation_count += n
    #     if n < 2:
    #         n = 2
    #     self.generator.eval()
    #     with torch.no_grad():
    #         z = torch.randn(n, self.latent_dim, device=self.device)
    #         patterns = self.generator(z).cpu().numpy()
    #     self.generator.train()
    #     return patterns

    def get_training_stats(self) -> Dict[str, Any]:
        """Возвращает статистику тренировки."""
        return {
            'generator_losses': self.generator_losses[-100:] if self.generator_losses else [],
            'discriminator_losses': self.discriminator_losses[-100:] if self.discriminator_losses else [],
            'total_generations': self.generation_count,
            'buffer_size': len(self.pattern_buffer),
            'avg_g_loss': np.mean(self.generator_losses[-50:]) if self.generator_losses else 0,
            'avg_d_loss': np.mean(self.discriminator_losses[-50:]) if self.discriminator_losses else 0,
            'pattern_dim': self.pattern_dim,
            'latent_dim': self.latent_dim
        }

    def save_models(self, path_prefix: str = 'models/instinct_gan'):
        """Сохраняет модели на диск."""
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'generator_losses': self.generator_losses,
            'discriminator_losses': self.discriminator_losses,
            'generation_count': self.generation_count,
            'latent_dim': self.latent_dim,
            'pattern_dim': self.pattern_dim
        }, f'{path_prefix}_checkpoint.pth')
        print(f"Модели сохранены в {path_prefix}_checkpoint.pth")

    def load_models(self, path_prefix: str = 'models/instinct_gan'):
        """Загружает модели с диска."""
        checkpoint = torch.load(f'{path_prefix}_checkpoint.pth', map_location=self.device)

        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.generator_losses = checkpoint['generator_losses']
        self.discriminator_losses = checkpoint['discriminator_losses']
        self.generation_count = checkpoint['generation_count']

        print(f"Модели загружены из {path_prefix}_checkpoint.pth")



# # models/instinct_gan.py
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from typing import List, Dict, Tuple, Optional
#
#
# class InstinctGenerator(nn.Module):
#     """Генератор для создания паттернов инстинктов."""
#
#     def __init__(self, latent_dim: int = 128, output_dim: int = 256,
#                  hidden_dims: List[int] = [512, 512, 256]):
#         super().__init__()
#
#         layers = []
#         prev_dim = latent_dim
#
#         for hidden_dim in hidden_dims:
#             layers.extend([
#                 nn.Linear(prev_dim, hidden_dim),
#                 nn.BatchNorm1d(hidden_dim),
#                 nn.ReLU(True),
#                 nn.Dropout(0.3)
#             ])
#             prev_dim = hidden_dim
#
#         layers.append(nn.Linear(prev_dim, output_dim))
#         layers.append(nn.Tanh())
#
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, z):
#         return self.model(z)
#
#
# class InstinctDiscriminator(nn.Module):
#     """Дискриминатор для оценки паттернов инстинктов."""
#
#     def __init__(self, input_dim: int = 256, hidden_dims: List[int] = [512, 256, 128]):
#         super().__init__()
#
#         layers = []
#         prev_dim = input_dim
#
#         for hidden_dim in hidden_dims:
#             layers.extend([
#                 nn.Linear(prev_dim, hidden_dim),
#                 nn.LeakyReLU(0.2, True),
#                 nn.Dropout(0.3)
#             ])
#             prev_dim = hidden_dim
#
#         layers.append(nn.Linear(prev_dim, 1))
#         layers.append(nn.Sigmoid())
#
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.model(x)
#
#
# class InstinctGAN:
#     """
#     GAN для генерации паттернов инстинктов.
#     """
#
#     def __init__(self, latent_dim: int = 128, pattern_dim: int = 256,
#                  batch_size: int = 16, device: str = 'cuda'):
#
#         self.latent_dim = latent_dim
#         self.pattern_dim = pattern_dim
#         self.batch_size = batch_size
#         self.device = torch.device(device)
#
#         self.generator = InstinctGenerator(latent_dim, pattern_dim).to(device)
#         self.discriminator = InstinctDiscriminator(pattern_dim).to(device)
#
#         self.optimizer_g = optim.Adam(self.generator.parameters(), lr=0.0002)
#         self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.0002)
#
#         self.criterion = nn.BCELoss()
#         self.pattern_buffer = []
#
#     def generate_pattern(self) -> np.ndarray:
#         """Генерирует новый паттерн инстинкта."""
#         self.generator.eval()
#         with torch.no_grad():
#             z = torch.randn(1, self.latent_dim, device=self.device)
#             pattern = self.generator(z).cpu().numpy().flatten()
#         self.generator.train()
#         return pattern
#
#     def train_step(self, real_patterns: torch.Tensor) -> Tuple[float, float]:
#         """Один шаг обучения."""
#         batch_size = min(self.batch_size, real_patterns.size(0))
#
#         real_labels = torch.ones(batch_size, 1, device=self.device)
#         fake_labels = torch.zeros(batch_size, 1, device=self.device)
#
#         # Тренируем дискриминатор
#         self.optimizer_d.zero_grad()
#
#         real_output = self.discriminator(real_patterns[:batch_size])
#         d_loss_real = self.criterion(real_output, real_labels)
#
#         z = torch.randn(batch_size, self.latent_dim, device=self.device)
#         fake_patterns = self.generator(z)
#         fake_output = self.discriminator(fake_patterns.detach())
#         d_loss_fake = self.criterion(fake_output, fake_labels)
#
#         d_loss = d_loss_real + d_loss_fake
#         d_loss.backward()
#         self.optimizer_d.step()
#
#         # Тренируем генератор
#         self.optimizer_g.zero_grad()
#
#         z = torch.randn(batch_size, self.latent_dim, device=self.device)
#         fake_patterns = self.generator(z)
#         fake_output = self.discriminator(fake_patterns)
#
#         g_loss = self.criterion(fake_output, real_labels)
#         g_loss.backward()
#         self.optimizer_g.step()
#
#         return g_loss.item(), d_loss.item()
#
#     def train(self, patterns: np.ndarray, epochs: int = 10) -> Dict:
#         """Обучает GAN на паттернах."""
#         patterns_tensor = torch.FloatTensor(patterns).to(self.device)
#
#         g_losses, d_losses = [], []
#
#         for epoch in range(epochs):
#             # Перемешиваем данные
#             indices = torch.randperm(patterns_tensor.size(0))
#             shuffled = patterns_tensor[indices]
#
#             for i in range(0, len(shuffled), self.batch_size):
#                 batch = shuffled[i:i + self.batch_size]
#                 if batch.size(0) < 2:
#                     continue
#                 g_loss, d_loss = self.train_step(batch)
#                 g_losses.append(g_loss)
#                 d_losses.append(d_loss)
#
#         return {'g_loss': g_losses, 'd_loss': d_losses}