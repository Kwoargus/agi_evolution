# models/gan.py - ПОЛНОСТЬЮ ИСПРАВЛЕННЫЙ ФАЙЛ

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Any, Optional


class Generator(nn.Module):
    """Генератор паттернов для GAN."""

    def __init__(self, latent_dim: int = 128, pattern_dim: int = 47):
        super().__init__()
        self.latent_dim = latent_dim
        self.pattern_dim = pattern_dim

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, pattern_dim),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class Discriminator(nn.Module):
    """Дискриминатор паттернов для GAN."""

    def __init__(self, pattern_dim: int = 47):
        super().__init__()
        self.pattern_dim = pattern_dim

        self.model = nn.Sequential(
            nn.Linear(pattern_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Encoder(nn.Module):
    """Энкодер для сжатия паттернов в латентное пространство."""

    def __init__(self, pattern_dim: int = 47, latent_dim: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(pattern_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class GAN:
    """
    Generative Adversarial Network для генерации паттернов поведения.
    """

    def __init__(self, latent_dim: int = 128, pattern_dim: int = 47,
                 batch_size: int = 16, state_dim: int = 21,
                 action_dim: int = 4, device: Optional[str] = None):

        self.latent_dim = latent_dim
        self.pattern_dim = pattern_dim
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Инициализируем модели
        self.generator = Generator(latent_dim, pattern_dim).to(self.device)
        self.discriminator = Discriminator(pattern_dim).to(self.device)
        self.encoder = Encoder(pattern_dim, latent_dim).to(self.device)

        # Оптимизаторы
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Критерий
        self.criterion = nn.BCELoss()

        # Буфер опыта
        self.experience_buffer = deque(maxlen=10000)

        print(f"GAN инициализирован на устройстве: {self.device}")
        print(f"  Pattern dim: {pattern_dim}")
        print(f"  Latent dim: {latent_dim}")
        print(f"  Batch size: {batch_size}")

    def add_experiences(self, experiences: List[Tuple]) -> None:
        """Добавляет опыт в буфер."""
        for exp in experiences:
            self.experience_buffer.append(exp)

    def _prepare_patterns(self, experiences: List[Tuple]) -> np.ndarray:
        """Преобразует опыт в паттерны размерности 47."""
        patterns = []

        for state, action, reward, next_state in experiences:
            # state (21)
            if hasattr(state, '__len__'):
                state_list = list(state)[:21]
            else:
                state_list = [state]
            while len(state_list) < 21:
                state_list.append(0.0)
            state_list = state_list[:21]

            # action one-hot (4)
            action_one_hot = [0.0, 0.0, 0.0, 0.0]
            if isinstance(action, int) and 0 <= action < 4:
                action_one_hot[action] = 1.0

            # reward (1)
            if isinstance(reward, (int, float)):
                reward_norm = max(-1.0, min(1.0, reward / 10.0))
            else:
                reward_norm = 0.0

            # next_state (21)
            if hasattr(next_state, '__len__'):
                next_state_list = list(next_state)[:21]
            else:
                next_state_list = [next_state]
            while len(next_state_list) < 21:
                next_state_list.append(0.0)
            next_state_list = next_state_list[:21]

            # Собираем паттерн: state(21) + action(4) + reward(1) + next_state(21) = 47
            pattern = state_list + action_one_hot + [reward_norm] + next_state_list

            # Гарантируем ровно 47
            while len(pattern) < 47:
                pattern.append(0.0)
            pattern = pattern[:47]

            patterns.append(pattern)

        return np.array(patterns, dtype=np.float32)

    def train_step(self, real_patterns: np.ndarray) -> Tuple[float, float]:
        """Один шаг обучения GAN."""
        batch_size = min(len(real_patterns), self.batch_size)

        # Гарантируем размерность 47
        if real_patterns.shape[1] != self.pattern_dim:
            if real_patterns.shape[1] > self.pattern_dim:
                real_patterns = real_patterns[:, :self.pattern_dim]
            else:
                padding = np.zeros((real_patterns.shape[0], self.pattern_dim - real_patterns.shape[1]))
                real_patterns = np.concatenate([real_patterns, padding], axis=1)

        real_patterns = torch.FloatTensor(real_patterns).to(self.device)

        # ============================================================
        # 1. ОБУЧАЕМ ДИСКРИМИНАТОР
        # ============================================================
        self.d_optimizer.zero_grad()

        # Реальные паттерны
        real_output = self.discriminator(real_patterns[:batch_size])
        real_labels = torch.ones(batch_size, 1, device=self.device)
        d_real_loss = self.criterion(real_output, real_labels)

        # Фейковые паттерны
        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_patterns = self.generator(noise)
        fake_output = self.discriminator(fake_patterns.detach())
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        d_fake_loss = self.criterion(fake_output, fake_labels)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_optimizer.step()

        # ============================================================
        # 2. ОБУЧАЕМ ГЕНЕРАТОР
        # ============================================================
        self.g_optimizer.zero_grad()

        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_patterns = self.generator(noise)
        fake_output = self.discriminator(fake_patterns)
        g_loss = self.criterion(fake_output, real_labels)

        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.item(), d_loss.item()

    def train(self, epochs: int = 5, verbose: bool = True) -> Dict[str, List[float]]:
        """Обучает GAN."""
        if len(self.experience_buffer) < self.batch_size:
            if verbose:
                print(f"⚠️ Недостаточно данных: {len(self.experience_buffer)}/{self.batch_size}")
            return {'g_loss': [], 'd_loss': []}

        # Подготавливаем паттерны
        patterns = self._prepare_patterns(list(self.experience_buffer))

        # Перемешиваем
        np.random.shuffle(patterns)

        g_losses = []
        d_losses = []

        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            steps = 0

            # Итерации по батчам
            for i in range(0, len(patterns), self.batch_size):
                batch = patterns[i:i + self.batch_size]
                if len(batch) < 2:
                    continue

                g_loss, d_loss = self.train_step(batch)
                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
                steps += 1

            if steps > 0:
                avg_g_loss = epoch_g_loss / steps
                avg_d_loss = epoch_d_loss / steps
                g_losses.append(avg_g_loss)
                d_losses.append(avg_d_loss)

                if verbose:
                    print(f"  Epoch {epoch + 1}/{epochs}: G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")

        return {'g_loss': g_losses, 'd_loss': d_losses}

    def generate_batch(self, n: int = 20) -> np.ndarray:
        """Генерирует batch паттернов."""
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(n, self.latent_dim, device=self.device)
            patterns = self.generator(noise).cpu().numpy()

            # Гарантируем размерность 47
            if patterns.shape[1] != self.pattern_dim:
                if patterns.shape[1] > self.pattern_dim:
                    patterns = patterns[:, :self.pattern_dim]
                else:
                    padding = np.zeros((patterns.shape[0], self.pattern_dim - patterns.shape[1]))
                    patterns = np.concatenate([patterns, padding], axis=1)

        self.generator.train()
        return patterns

    def generate_pattern(self) -> np.ndarray:
        """Генерирует один паттерн."""
        return self.generate_batch(1)[0]

    def generate_rule(self) -> Dict[str, Any]:
        """Генерирует правило поведения из паттерна."""
        pattern = self.generate_pattern()

        # Преобразуем паттерн в правило
        # state (первые 21) → sense_type
        # action (следующие 4) → action
        # reward → threshold

        state_part = pattern[:21]
        action_part = pattern[21:25]
        reward_part = pattern[25]

        # Определяем сенсор
        max_state_idx = np.argmax(np.abs(state_part))
        sense_types = ['smell', 'sound', 'vision', 'touch', 'position']
        sense_type = sense_types[max_state_idx % len(sense_types)]

        # Определяем действие
        action_idx = np.argmax(action_part)
        actions = ['grab', 'move_on', 'avoid', 'investigate']
        action = actions[action_idx % len(actions)]

        # Порог
        threshold = 0.3 + 0.4 * (reward_part + 1) / 2

        return {
            'sense_type': sense_type,
            'action': action,
            'threshold': float(np.clip(threshold, 0.1, 0.9)),
            'priority': float(0.5 + 0.5 * np.abs(state_part[max_state_idx])),
            'confidence': float(np.abs(reward_part))
        }


class PatternRepository:
    """Хранилище сгенерированных паттернов."""

    def __init__(self, max_size: int = 1000):
        self.patterns: List[np.ndarray] = []
        self.scores: List[float] = []
        self.max_size = max_size

    def add_pattern(self, pattern: np.ndarray, score: float) -> None:
        """Добавляет паттерн с оценкой."""
        if len(self.patterns) >= self.max_size:
            # Удаляем самый низкий score
            min_idx = np.argmin(self.scores)
            self.patterns.pop(min_idx)
            self.scores.pop(min_idx)

        self.patterns.append(pattern)
        self.scores.append(score)

    def get_best_patterns(self, n: int = 5) -> List[np.ndarray]:
        """Возвращает n лучших паттернов."""
        if not self.patterns:
            return []

        sorted_indices = np.argsort(self.scores)[::-1]
        return [self.patterns[i] for i in sorted_indices[:n]]

    def get_all_patterns(self) -> List[np.ndarray]:
        """Возвращает все паттерны."""
        return self.patterns.copy()


class PatternEvaluator:
    """Оценщик качества паттернов."""

    def __init__(self, discriminator: Discriminator, generator: Generator):
        self.discriminator = discriminator
        self.generator = generator
        self.device = next(discriminator.parameters()).device

    def evaluate_pattern(self, pattern: np.ndarray) -> float:
        """Оценивает один паттерн."""
        if pattern.shape[0] != 47:
            if pattern.shape[0] > 47:
                pattern = pattern[:47]
            else:
                padded = np.zeros(47)
                padded[:len(pattern)] = pattern
                pattern = padded

        pattern_tensor = torch.FloatTensor(pattern).unsqueeze(0).to(self.device)

        with torch.no_grad():
            score = self.discriminator(pattern_tensor).item()

        return score

    def evaluate_batch(self, patterns: List[np.ndarray]) -> List[float]:
        """Оценивает batch паттернов."""
        scores = []
        for pattern in patterns:
            scores.append(self.evaluate_pattern(pattern))
        return scores




# # models/gan.py (исправленная версия)
#
# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from typing import List, Dict, Optional, Tuple, Any, Union
# import random
# import json
# from collections import deque
#
#
# class Generator(nn.Module):
#     def __init__(self, latent_dim=128, pattern_dim=47):  # ← pattern_dim = 47
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.pattern_dim = pattern_dim
#
#         self.model = nn.Sequential(
#             nn.Linear(latent_dim, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, pattern_dim),  # ← pattern_dim = 47
#             nn.Tanh()
#         )
#         self.generator = Generator(latent_dim, pattern_dim).to(self.device)
#         self.discriminator = Discriminator(pattern_dim).to(self.device)
#         self.encoder = Encoder(pattern_dim, latent_dim).to(self.device)
#
# def forward(self, z):
#         return self.model(z)
#
# # class Generator(nn.Module):
# #     """Генератор создаёт новые модели поведения."""
# #
# #     def __init__(self, latent_dim=128, pattern_dim=None, batch_size=16, state_dim=8, action_dim=4, device=None, output_dim: int = 64,
# #                   hidden_dims: List[int] = [256, 512, 256], dropout_rate: float = 0.3):
# #
# #         # Если pattern_dim не указан, вычисляем автоматически
# #         if pattern_dim is None:
# #             pattern_dim = state_dim + action_dim  # 8 + 4 = 12
# #             # Но чтобы было 21, добавим запас
# #             pattern_dim = max(pattern_dim, 21)
# #
# #         self.latent_dim = latent_dim
# #         self.pattern_dim = pattern_dim
# #         self.batch_size = batch_size
# #         self.state_dim = state_dim
# #         self.action_dim = action_dim
# #
# #     # def __init__(self, latent_dim: int = 100, output_dim: int = 64,
# #     #              hidden_dims: List[int] = [256, 512, 256], dropout_rate: float = 0.3):
# #     #
# #
# #
# #         super(Generator, self).__init__()
# #
# #         self.latent_dim = latent_dim # Размер входного шума (например, 128)
# #         self.output_dim = output_dim # Размер выходного паттерна (например, 21)
# #
# #         layers = []
# #         prev_dim = latent_dim
# #
# #         # Создаем скрытые слои
# #         for hidden_dim in hidden_dims:
# #             layers.extend([
# #                 nn.Linear(prev_dim, hidden_dim), # Полносвязный слой
# #                 nn.BatchNorm1d(hidden_dim),      # Нормализация
# #                 nn.ReLU(True),                   # Активация
# #                 nn.Dropout(dropout_rate)         # Регуляризация
# #             ])
# #             prev_dim = hidden_dim
# #
# #         # Выходной слой
# #         layers.append(nn.Linear(prev_dim, output_dim))
# #         layers.append(nn.Tanh()) # Ограничиваем значения в [-1, 1]
# #
# #         self.model = nn.Sequential(*layers)
# #
# #     def forward(self, z: torch.Tensor) -> torch.Tensor:
# #         return self.model(z)
#
#
# class Discriminator(nn.Module):
#     def __init__(self, pattern_dim=47):  # ← pattern_dim = 47
#         super().__init__()
#         self.pattern_dim = pattern_dim
#
#         self.model = nn.Sequential(
#             nn.Linear(pattern_dim, 256),  # ← pattern_dim = 47
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.2),
#             nn.Linear(256, 128),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.2),
#             nn.Linear(128, 64),
#             nn.LeakyReLU(0.2),
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.model(x)
#
# # class Discriminator(nn.Module):
# #     """Дискриминатор оценивает качество паттернов."""
# #
# #     def __init__(self, input_dim: int = 64, hidden_dims: List[int] = [256, 128, 64],
# #                  dropout_rate: float = 0.3):
# #         super(Discriminator, self).__init__()
# #
# #         self.input_dim = input_dim # Размер входного паттерна (например, 21)
# #
# #         layers = []
# #         prev_dim = input_dim
# #
# #         # Создаем скрытые слои
# #         for hidden_dim in hidden_dims:
# #             layers.extend([
# #                 nn.Linear(prev_dim, hidden_dim),
# #                 nn.LeakyReLU(0.2, True), # LeakyReLU для лучшего градиента
# #                 nn.Dropout(dropout_rate)
# #             ])
# #             prev_dim = hidden_dim
# #
# #         # Выходной слой - сигмоид для вероятности [0,1]
# #         layers.append(nn.Linear(prev_dim, 1))
# #         layers.append(nn.Sigmoid())
# #
# #         self.model = nn.Sequential(*layers)
# #
# #     def forward(self, pattern: torch.Tensor) -> torch.Tensor:
# #         return self.model(pattern)
# #
# #     def evaluate_pattern(self, pattern: np.ndarray, device: torch.device = None) -> float:
# #         """Оценивает один паттерн."""
# #         if device is None:
# #             device = next(self.parameters()).device
# #
# #         if pattern.ndim == 1:
# #             pattern = pattern.reshape(1, -1)
# #
# #         pattern_tensor = torch.FloatTensor(pattern).to(device)
# #         with torch.no_grad():
# #             score = self.forward(pattern_tensor).cpu().item()
# #         return score
#
#
# class ExperienceEncoder:
#     """Кодирует опыт в векторное представление."""
#
#     def __init__(self, state_dim: int = 8, action_dim: int = 4, max_pattern_len: int = 100):
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.max_pattern_len = max_pattern_len
#         self.pattern_dim = state_dim + action_dim + 1 + state_dim
#
#     def encode_experience(self, experience: Tuple) -> np.ndarray:
#         """Кодирует один переход в паттерн."""
#         state, action, reward, next_state = experience
#
#         # Преобразуем в векторы
#         state_vec = np.array(state[:self.state_dim], dtype=np.float32)
#         next_state_vec = np.array(next_state[:self.state_dim], dtype=np.float32)
#
#         action_vec = np.zeros(self.action_dim, dtype=np.float32)
#         if isinstance(action, int) and 0 <= action < self.action_dim:
#             action_vec[action] = 1.0
#
#         # Конкатенируем
#         pattern = np.concatenate([
#             state_vec,
#             action_vec,
#             np.array([reward], dtype=np.float32),
#             next_state_vec
#         ])
#
#         norm = np.linalg.norm(pattern) + 1e-8
#
#         # Нормализуем
#         pattern = pattern / norm
#
#         return pattern # Вектор размером 21
#
#     def encode_experiences(self, experiences: List[Tuple]) -> np.ndarray:
#         """Кодирует список переходов."""
#         patterns = []
#         for exp in experiences:
#             pattern = self.encode_experience(exp)
#             patterns.append(pattern)
#
#         if len(patterns) > self.max_pattern_len:
#             indices = random.sample(range(len(patterns)), self.max_pattern_len)
#             patterns = [patterns[i] for i in indices]
#
#         return np.array(patterns, dtype=np.float32)
#
#     def decode_pattern(self, pattern: np.ndarray) -> Dict[str, Any]:
#         """Декодирует паттерн обратно в читаемый формат."""
#         state_dim = self.state_dim
#         action_dim = self.action_dim
#
#         state = pattern[:state_dim]
#         action_vec = pattern[state_dim:state_dim + action_dim]
#         reward = pattern[state_dim + action_dim]
#         next_state = pattern[state_dim + action_dim + 1:state_dim + action_dim + 1 + state_dim]
#
#         # Более надежное определение действия
#         # Находим индекс с максимальным значением
#         max_val = np.max(action_vec)
#         # Если максимальное значение > 0.3, считаем что действие определено
#         if max_val > 0.3:
#             action = int(np.argmax(action_vec))
#         else:
#             # Если все значения маленькие, пробуем найти ближайшее к 1
#             # или используем сумму для определения
#             action = int(np.argmax(np.abs(action_vec)))
#             # Проверяем, что действие в допустимом диапазоне
#             if action >= action_dim or action < 0:
#                 action = 0  # Действие по умолчанию
#
#         return {
#             'state': state.tolist(),
#             'action': action,
#             'reward': float(reward),
#             'next_state': next_state.tolist(),
#             'action_vec': action_vec.tolist()
#         }
#
#     # def decode_pattern(self, pattern: np.ndarray) -> Dict[str, Any]:
#     #     """Декодирует паттерн обратно в читаемый формат."""
#     #     state_dim = self.state_dim
#     #     action_dim = self.action_dim
#     #
#     #     state = pattern[:state_dim]
#     #     action_vec = pattern[state_dim:state_dim + action_dim]
#     #     reward = pattern[state_dim + action_dim]
#     #     next_state = pattern[state_dim + action_dim + 1:state_dim + action_dim + 1 + state_dim]
#     #
#     #     action = int(np.argmax(action_vec)) if np.max(action_vec) > 0.5 else -1
#     #
#     #     return {
#     #         'state': state.tolist(),
#     #         'action': action,
#     #         'reward': float(reward),
#     #         'next_state': next_state.tolist(),
#     #         'action_vec': action_vec.tolist()
#     #     }
#
#
# class PatternRepository:
#     """Хранит и управляет сгенерированными паттернами."""
#
#     def __init__(self, max_stored: int = 1000):
#         self.patterns = []
#         self.scores = []
#         self.max_stored = max_stored
#
#     def add_pattern(self, pattern: np.ndarray, score: float):
#         """Добавляет паттерн с оценкой."""
#         self.patterns.append(pattern)
#         self.scores.append(score)
#
#         combined = sorted(zip(self.patterns, self.scores), key=lambda x: x[1], reverse=True)
#         self.patterns = [p for p, _ in combined[:self.max_stored]]
#         self.scores = [s for _, s in combined[:self.max_stored]]
#
#     def get_best_patterns(self, n: int = 10) -> List[np.ndarray]:
#         """Возвращает n лучших паттернов."""
#         return self.patterns[:n]
#
#     def get_patterns_with_scores(self, min_score: float = 0.5) -> List[Tuple[np.ndarray, float]]:
#         """Возвращает паттерны с оценками выше порога."""
#         result = []
#         for pattern, score in zip(self.patterns, self.scores):
#             if score >= min_score:
#                 result.append((pattern, score))
#         return result
#
#     def get_average_score(self) -> float:
#         if not self.scores:
#             return 0.0
#         return np.mean(self.scores)
#
#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             'patterns': [p.tolist() for p in self.patterns],
#             'scores': self.scores
#         }
#
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> 'PatternRepository':
#         repo = cls()
#         repo.patterns = [np.array(p) for p in data['patterns']]
#         repo.scores = data['scores']
#         return repo
#
#
# class PatternEvaluator:
#     """
#     Оценивает качество сгенерированных паттернов.
#     """
#
#     def __init__(self, discriminator, generator):
#         self.discriminator = discriminator
#         self.generator = generator
#         self.device = next(discriminator.parameters()).device
#
#     def evaluate_pattern(self, pattern):
#         """Оценивает один паттерн."""
#         if isinstance(pattern, np.ndarray):
#             pattern = torch.FloatTensor(pattern).unsqueeze(0).to(self.device)
#
#         with torch.no_grad():
#             # Оценка дискриминатором (чем ближе к 1, тем лучше)
#             score = self.discriminator(pattern).item()
#
#         return score
#
#     def evaluate_batch(self, patterns):
#         """Оценивает batch паттернов."""
#         if isinstance(patterns, np.ndarray):
#             patterns = torch.FloatTensor(patterns).to(self.device)
#
#         with torch.no_grad():
#             scores = self.discriminator(patterns).squeeze().cpu().numpy()
#
#         return scores
#
#
# # class PatternEvaluator:
# #     """Оценивает качество сгенерированных паттернов."""
# #
# #     def __init__(self, discriminator: Discriminator, encoder: ExperienceEncoder):
# #         self.discriminator = discriminator
# #         self.encoder = encoder
# #         self.device = next(discriminator.parameters()).device
# #
# #     def evaluate_pattern(self, pattern: np.ndarray) -> float:
# #         """Оценивает качество одного паттерна."""
# #         if pattern.ndim == 1:
# #             pattern = pattern.reshape(1, -1)
# #
# #         pattern_tensor = torch.FloatTensor(pattern).to(self.device)
# #         with torch.no_grad():
# #             score = self.discriminator(pattern_tensor).cpu().item()
# #         return score
# #
# #     def select_best_patterns(self, patterns: np.ndarray, n: int = 5) -> List[np.ndarray]:
# #         """Выбирает n лучших паттернов."""
# #         scores = [self.evaluate_pattern(p) for p in patterns]
# #         best_indices = np.argsort(scores)[-n:]
# #         return [patterns[i] for i in best_indices]
# #
# #     def validate_pattern(self, pattern: np.ndarray, min_score: float = 0.7) -> bool:
# #         """Проверяет, достаточно ли хорош паттерн."""
# #         score = self.evaluate_pattern(pattern)
# #         return score >= min_score
#
#
# class GAN:
#
#     """Полноценный GAN для генерации новых моделей поведения."""
#
#     def __init__(self, latent_dim=128, pattern_dim=47, batch_size=16,
#                  state_dim=21, action_dim=4, device=None):  # ← pattern_dim по умолчанию 47
#         self.latent_dim = latent_dim
#         self.pattern_dim = pattern_dim
#         self.batch_size = batch_size
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#
#         if device is None:
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         else:
#             self.device = device
#
#         self.generator = Generator(latent_dim, pattern_dim).to(self.device)
#         self.discriminator = Discriminator(pattern_dim).to(self.device)
#
#         # Оптимизаторы
#         self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#         self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#
#         # Критерий
#         self.criterion = nn.BCELoss()
#
#         # Буфер опыта
#         self.experience_buffer = deque(maxlen=10000)
#
#         print(f"GAN инициализирован на устройстве: {self.device}")
#         print(f"  Pattern dim: {pattern_dim}")
#         print(f"  Latent dim: {latent_dim}")
#         print(f"  Batch size: {batch_size}")
#
#
#     # def __init__(self,
#     #              latent_dim: int = 100,
#     #              pattern_dim: int = None,
#     #              device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
#     #              lr_generator: float = 0.0002,
#     #              lr_discriminator: float = 0.0002,
#     #              beta1: float = 0.5,
#     #              beta2: float = 0.999,
#     #              batch_size: int = 32,
#     #              experience_buffer_size: int = 10000,
#     #              state_dim: int = 8,
#     #              action_dim: int = 4):
#     #
#     #     self.device = torch.device(device)
#     #     self.latent_dim = latent_dim
#     #     self.state_dim = state_dim
#     #     self.action_dim = action_dim
#     #
#     #     # 1. Определяем размерности
#     #     # Если pattern_dim не указан, вычисляем автоматически
#     #     if pattern_dim is None:
#     #         self.pattern_dim = state_dim + action_dim + 1 + state_dim
#     #     else:
#     #         self.pattern_dim = pattern_dim
#     #
#     #     self.batch_size = batch_size
#     #
#     #     # 2. Создаем модели
#     #     # Инициализируем модели с правильной размерностью
#     #     self.generator = Generator(latent_dim, self.pattern_dim).to(self.device)
#     #     self.discriminator = Discriminator(self.pattern_dim).to(self.device)
#     #
#     #     # 3. Создаем оптимизаторы
#     #     # Оптимизаторы
#     #     self.optimizer_g = optim.Adam(
#     #         self.generator.parameters(),
#     #         lr=lr_generator,
#     #         betas=(beta1, beta2)
#     #     )
#     #     self.optimizer_d = optim.Adam(
#     #         self.discriminator.parameters(),
#     #         lr=lr_discriminator,
#     #         betas=(beta1, beta2)
#     #     )
#     #
#     #     # 4. Функция потерь
#     #     self.criterion = nn.BCELoss()
#     #
#     #     # 5. Буфер для хранения опыта
#     #     self.experience_buffer = deque(maxlen=experience_buffer_size)
#     #     self.encoder = ExperienceEncoder(state_dim=state_dim, action_dim=action_dim)
#     #
#     #     # Репозиторий и оценщик
#     #     self.pattern_repository = PatternRepository()
#     #     self.pattern_evaluator = PatternEvaluator(self.discriminator, self.encoder)
#     #
#     #     # Счётчики для статистики
#     #     self.generator_losses = []
#     #     self.discriminator_losses = []
#     #     self.generation_count = 0
#     #
#     #     print(f"GAN инициализирован на устройстве: {self.device}")
#     #     print(f"  Pattern dim: {self.pattern_dim}")
#     #     print(f"  Latent dim: {self.latent_dim}")
#     #     print(f"  Batch size: {self.batch_size}")
#
#     def add_experience(self, experience: Tuple):
#         """Добавляет один переход в буфер."""
#         self.experience_buffer.append(experience)
#
#     def add_experiences(self, experiences: List[Tuple]):
#         """Добавляет список переходов в буфер."""
#         self.experience_buffer.extend(experiences)
#
#     def prepare_training_data(self, experiences: Optional[List[Tuple]] = None) -> torch.Tensor:
#         """Подготавливает данные для тренировки."""
#         if experiences is None:
#             experiences = list(self.experience_buffer)
#
#         if not experiences:
#             return None
#
#         # 1. Кодируем опыт в паттерны
#         patterns = self.encoder.encode_experiences(experiences)
#         # 2. Преобразуем в тензор PyTorch
#         patterns_tensor = torch.FloatTensor(patterns).to(self.device)
#
#         return patterns_tensor
#
#     def train_step(self, real_patterns):
#         """Один шаг обучения GAN."""
#         batch_size = min(len(real_patterns), self.batch_size)
#
#         # Гарантируем размерность 47
#         if real_patterns.shape[1] != self.pattern_dim:
#             if real_patterns.shape[1] > self.pattern_dim:
#                 real_patterns = real_patterns[:, :self.pattern_dim]
#             else:
#                 padding = np.zeros((real_patterns.shape[0], self.pattern_dim - real_patterns.shape[1]))
#                 real_patterns = np.concatenate([real_patterns, padding], axis=1)
#
#         real_patterns = torch.FloatTensor(real_patterns).to(self.device)
#
#         # ============================================================
#         # 1. ОБУЧАЕМ ДИСКРИМИНАТОР
#         # ============================================================
#         self.d_optimizer.zero_grad()
#
#         # Реальные паттерны
#         real_output = self.discriminator(real_patterns[:batch_size])
#         real_labels = torch.ones(batch_size, 1, device=self.device)
#         d_real_loss = self.criterion(real_output, real_labels)
#
#         # Фейковые паттерны
#         noise = torch.randn(batch_size, self.latent_dim, device=self.device)
#         fake_patterns = self.generator(noise)
#         fake_output = self.discriminator(fake_patterns.detach())
#         fake_labels = torch.zeros(batch_size, 1, device=self.device)
#         d_fake_loss = self.criterion(fake_output, fake_labels)
#
#         d_loss = d_real_loss + d_fake_loss
#         d_loss.backward()
#         self.d_optimizer.step()
#
#         # ============================================================
#         # 2. ОБУЧАЕМ ГЕНЕРАТОР
#         # ============================================================
#         self.g_optimizer.zero_grad()
#
#         noise = torch.randn(batch_size, self.latent_dim, device=self.device)
#         fake_patterns = self.generator(noise)
#         fake_output = self.discriminator(fake_patterns)
#         g_loss = self.criterion(fake_output, real_labels)
#
#         g_loss.backward()
#         self.g_optimizer.step()
#
#         return g_loss.item(), d_loss.item()
#
#
#     # def train_step(self, real_patterns: torch.Tensor) -> Tuple[float, float]:
#     #     """Выполняет один шаг тренировки GAN."""
#     #     batch_size = min(self.batch_size, real_patterns.size(0))
#     #
#     #     if batch_size == 0:
#     #         return 0.0, 0.0
#     #
#     #     if batch_size < 2:
#     #         batch_size = 2
#     #         if real_patterns.size(0) < batch_size:
#     #             indices = torch.randint(0, real_patterns.size(0), (batch_size,))
#     #             real_patterns = real_patterns[indices]
#     #
#     #     real_labels = torch.ones(batch_size, 1, device=self.device)
#     #     fake_labels = torch.zeros(batch_size, 1, device=self.device)
#     #
#     #     # --- Тренируем дискриминатор ---
#     #     self.optimizer_d.zero_grad()
#     #
#     #     real_output = self.discriminator(real_patterns[:batch_size])
#     #     d_loss_real = self.criterion(real_output, real_labels)
#     #
#     #     z = torch.randn(batch_size, self.latent_dim, device=self.device)
#     #     fake_patterns = self.generator(z)
#     #     fake_output = self.discriminator(fake_patterns.detach())
#     #     d_loss_fake = self.criterion(fake_output, fake_labels)
#     #
#     #     d_loss = d_loss_real + d_loss_fake
#     #     d_loss.backward()
#     #     self.optimizer_d.step()
#     #
#     #     # --- Тренируем генератор ---
#     #     self.optimizer_g.zero_grad()
#     #
#     #     z = torch.randn(batch_size, self.latent_dim, device=self.device)
#     #     fake_patterns = self.generator(z)
#     #     fake_output = self.discriminator(fake_patterns)
#     #
#     #     g_loss = self.criterion(fake_output, real_labels)
#     #     g_loss.backward()
#     #     self.optimizer_g.step()
#     #
#     #     # <-- ВАЖНО: СОХРАНЯЕМ ПОТЕРИ В СПИСКИ -->
#     #     g_loss_val = g_loss.item()
#     #     d_loss_val = d_loss.item()
#     #     self.generator_losses.append(g_loss_val)
#     #     self.discriminator_losses.append(d_loss_val)
#     #
#     #     return g_loss_val, d_loss_val
#
#
#     # def train_step(self, real_patterns: torch.Tensor) -> Tuple[float, float]:
#     #     """Выполняет один шаг тренировки GAN."""
#     #     batch_size = min(self.batch_size, real_patterns.size(0))
#     #
#     #     # Если batch_size == 0, пропускаем
#     #     if batch_size == 0:
#     #         return 0.0, 0.0
#     #
#     #     # Если batch_size == 1, увеличиваем его
#     #     if batch_size < 2:
#     #         batch_size = 2
#     #         if real_patterns.size(0) < batch_size:
#     #             # Если данных слишком мало, дублируем
#     #             indices = torch.randint(0, real_patterns.size(0), (batch_size,))
#     #             real_patterns = real_patterns[indices]
#     #
#     #     # Создаем метки: 1 - реальные, 0 - фейковые
#     #     real_labels = torch.ones(batch_size, 1, device=self.device)
#     #     fake_labels = torch.zeros(batch_size, 1, device=self.device)
#     #
#     #     # --- ШАГ 1: Тренируем дискриминатор ---
#     #     self.optimizer_d.zero_grad()
#     #
#     #     # a) Обучаем на реальных данных
#     #     real_output = self.discriminator(real_patterns[:batch_size])
#     #     d_loss_real = self.criterion(real_output, real_labels)
#     #
#     #     # b) Обучаем на сгенерированных данных
#     #     z = torch.randn(batch_size, self.latent_dim, device=self.device)
#     #     fake_patterns = self.generator(z)
#     #     fake_output = self.discriminator(fake_patterns.detach())
#     #     d_loss_fake = self.criterion(fake_output, fake_labels)
#     #
#     #     # Суммарная потеря дискриминатора
#     #     d_loss = d_loss_real + d_loss_fake
#     #     d_loss.backward()
#     #     self.optimizer_d.step()
#     #
#     #     # --- ШАГ 2: Тренируем генератор ---
#     #     self.optimizer_g.zero_grad()
#     #
#     #     # Генерируем новые паттерны
#     #     z = torch.randn(batch_size, self.latent_dim, device=self.device)
#     #     fake_patterns = self.generator(z)
#     #     fake_output = self.discriminator(fake_patterns)
#     #
#     #     # Цель генератора - обмануть дискриминатор
#     #     g_loss = self.criterion(fake_output, real_labels)
#     #     g_loss.backward()
#     #     self.optimizer_g.step()
#     #
#     #     # ДОБАВЛЯЕМ ЛОГИРОВАНИЕ КАЖДОГО ШАГА
#     #     print(f"  Шаг: G_loss={g_loss.item():.4f}, D_loss={d_loss.item():.4f}")
#     #     return g_loss.item(), d_loss.item()
#
#     def train(self, experiences: Optional[List[Tuple]] = None,
#               epochs: int = 10, verbose: bool = False) -> Dict[str, List[float]]:
#
#         """Обучает GAN."""
#         if len(self.experience_buffer) < self.batch_size:
#             if verbose:
#                 print(f"⚠️ Недостаточно данных: {len(self.experience_buffer)}/{self.batch_size}")
#             return {'g_loss': [], 'd_loss': []}
#
#         # ПОДГОТАВЛИВАЕМ ПАТТЕРНЫ
#         patterns = self._prepare_patterns(list(self.experience_buffer))
#
#         # Гарантируем размерность 21
#         if patterns.shape[1] != 21:
#             if patterns.shape[1] > 21:
#                 patterns = patterns[:, :21]
#             else:
#                 padding = np.zeros((patterns.shape[0], 21 - patterns.shape[1]))
#                 patterns = np.concatenate([patterns, padding], axis=1)
#
#         """Тренирует GAN на накопленном опыте."""
#         if experiences is None:
#             experiences = list(self.experience_buffer)
#
#         if not experiences:
#             print("Нет данных для тренировки GAN.")
#             return {'g_loss': [], 'd_loss': []}
#
#         # 1. Подготавливаем данные
#         patterns = self.prepare_training_data(experiences)
#         if patterns is None:
#             return {'g_loss': [], 'd_loss': []}
#
#         # 2. Создаем DataLoader для батчей
#         dataset = TensorDataset(patterns)
#         dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
#
#         g_losses = []
#         d_losses = []
#
#         # 3. Цикл обучения
#         for epoch in range(epochs):
#             for batch in dataloader:
#                 real_batch = batch[0]
#                 g_loss, d_loss = self.train_step(real_batch)
#                 g_losses.append(g_loss)
#                 d_losses.append(d_loss)
#
#             # 4. Логирование
#             if verbose and (epoch + 1) % 5 == 0:
#                 avg_g_loss = np.mean(g_losses[-len(dataloader):])
#                 avg_d_loss = np.mean(d_losses[-len(dataloader):])
#                 print(f"GAN Epoch {epoch + 1}/{epochs}: G_loss={avg_g_loss:.4f}, D_loss={avg_d_loss:.4f}")
#
#             # ЛОГИРУЕМ ПОСЛЕ КАЖДОЙ ЭПОХИ
#             avg_g_loss = np.mean(g_losses)
#             avg_d_loss = np.mean(d_losses)
#             print(f"  Эпоха {epoch + 1}/{epochs}: G_loss={avg_g_loss:.4f}, D_loss={avg_d_loss:.4f}")
#
#             if verbose and (epoch + 1) % 5 == 0:
#                 print(f"GAN Epoch {epoch + 1}/{epochs}: G_loss={avg_g_loss:.4f}, D_loss={avg_d_loss:.4f}")
#
#         self.generator_losses.extend(g_losses)
#         self.discriminator_losses.extend(d_losses)
#
#         return {'g_loss': g_losses, 'd_loss': d_losses}
#
#     def generate_pattern(self) -> np.ndarray:
#         """Генерирует новый паттерн поведения."""
#         self.generation_count += 1
#         self.generator.eval() # Переключаем в режим оценки
#         with torch.no_grad(): # Отключаем вычисление градиентов
#             z = torch.randn(1, self.latent_dim, device=self.device)
#             pattern = self.generator(z).cpu().numpy().flatten()
#         self.generator.train() # Возвращаем в режим обучения
#         return pattern
#
#     def generate_batch(self, n=20):
#         """Генерирует batch паттернов."""
#         self.generator.eval()
#         with torch.no_grad():
#             noise = torch.randn(n, self.latent_dim, device=self.device)
#             patterns = self.generator(noise).cpu().numpy()
#
#             # Гарантируем размерность 47
#             if patterns.shape[1] != self.pattern_dim:
#                 if patterns.shape[1] > self.pattern_dim:
#                     patterns = patterns[:, :self.pattern_dim]
#                 else:
#                     padding = np.zeros((patterns.shape[0], self.pattern_dim - patterns.shape[1]))
#                     patterns = np.concatenate([patterns, padding], axis=1)
#
#         self.generator.train()
#         return patterns
#
#     # def generate_batch(self, n: int = 10) -> np.ndarray:
#     #     """Генерирует батч новых паттернов."""
#     #     self.generation_count += n
#     #     if n < 2:
#     #         n = 2
#     #     self.generator.eval()
#     #     with torch.no_grad():
#     #         z = torch.randn(n, self.latent_dim, device=self.device)
#     #         patterns = self.generator(z).cpu().numpy()
#     #     self.generator.train()
#     #     return patterns
#
#     def generate_rule(self) -> Dict[str, Any]:
#         """Генерирует новое правило рефлекса."""
#         pattern = self.generate_pattern()
#         decoded = self.encoder.decode_pattern(pattern)
#
#         rule = {
#             'sense_type': self._interpret_sense_type(decoded['state']),
#             'signal_type': self._interpret_signal_type(decoded['state']),
#             'action': self._interpret_action(decoded['action']),
#             'threshold': float(abs(decoded['reward'])),
#             'confidence': 0.7
#         }
#         return rule
#
#     def generate_instinct_pattern(self) -> Dict[str, Any]:
#         """Генерирует новый паттерн инстинкта."""
#         pattern = self.generate_pattern()
#         decoded = self.encoder.decode_pattern(pattern)
#
#         instinct_pattern = {
#             'pattern': {
#                 'signals': {
#                     'sound': self._interpret_signal_type(decoded['state']),
#                     'vision': self._interpret_vision_signal(decoded['state'])
#                 }
#             },
#             'action': {
#                 'action': self._interpret_action(decoded['action'])
#             }
#         }
#         return instinct_pattern
#
#     def get_training_stats(self) -> Dict[str, Any]:
#         """Возвращает статистику тренировки."""
#         return {
#             'generator_losses': self.generator_losses[-100:] if self.generator_losses else [],
#             'discriminator_losses': self.discriminator_losses[-100:] if self.discriminator_losses else [],
#             'total_generations': self.generation_count,
#             'buffer_size': len(self.experience_buffer),
#             'avg_g_loss': np.mean(self.generator_losses[-50:]) if self.generator_losses else 0,
#             'avg_d_loss': np.mean(self.discriminator_losses[-50:]) if self.discriminator_losses else 0,
#             'pattern_dim': self.pattern_dim,
#             'latent_dim': self.latent_dim
#         }
#
#     def save_models(self, path_prefix: str = 'models/gan'):
#         """Сохраняет модели на диск."""
#         # Создаем директорию если её нет
#         os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
#
#         # Сохраняем все компоненты
#         torch.save({
#             'generator_state_dict': self.generator.state_dict(), # Веса генератора - в self.generator # Словарь всех весов
#             'discriminator_state_dict': self.discriminator.state_dict(), # Веса дискриминатора - в self.discriminator # Словарь всех весов
#             'optimizer_g_state_dict': self.optimizer_g.state_dict(), # Оптимизаторы тоже сохраняются
#             'optimizer_d_state_dict': self.optimizer_d.state_dict(),
#             'generator_losses': self.generator_losses,
#             'discriminator_losses': self.discriminator_losses,
#             'generation_count': self.generation_count,
#             'latent_dim': self.latent_dim,
#             'pattern_dim': self.pattern_dim
#         }, f'{path_prefix}_checkpoint.pth')
#         print(f"Модели сохранены в {path_prefix}_checkpoint.pth")
#
#     def load_models(self, path_prefix: str = 'models/gan'):
#         """Загружает модели с диска."""
#         # Загружаем чекпоинт
#         checkpoint = torch.load(f'{path_prefix}_checkpoint.pth', map_location=self.device)
#
#         # Восстанавливаем состояния
#         self.generator.load_state_dict(checkpoint['generator_state_dict'])
#         self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
#         self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
#         self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
#
#         # Восстанавливаем метаданные
#         self.generator_losses = checkpoint['generator_losses']
#         self.discriminator_losses = checkpoint['discriminator_losses']
#         self.generation_count = checkpoint['generation_count']
#
#         print(f"Модели загружены из {path_prefix}_checkpoint.pth")
#
#     # --- Вспомогательные методы для интерпретации ---
#
#     def _interpret_sense_type(self, state: np.ndarray) -> str:
#         sense_types = ['smell', 'sound', 'vision', 'touch', 'temperature']
#         idx = int(abs(state[0]) * len(sense_types)) % len(sense_types) if len(state) > 0 else 0
#         return sense_types[idx]
#
#     def _interpret_signal_type(self, state: np.ndarray) -> str:
#         signal_types = ['food_smell', 'predator_smell', 'fire_smell',
#                         'loud_crash', 'bright_flash', 'danger_signal']
#         idx = int(abs(state[1]) * len(signal_types)) % len(signal_types) if len(state) > 1 else 0
#         return signal_types[idx]
#
#     def _interpret_vision_signal(self, state: np.ndarray) -> str:
#         vision_signals = ['bright_flash', 'darkness', 'motion', 'color_change']
#         idx = int(abs(state[2]) * len(vision_signals)) % len(vision_signals) if len(state) > 2 else 0
#         return vision_signals[idx]
#
#     def _interpret_action(self, action: int) -> str:
#         actions = ['grab', 'avoid', 'run_away', 'move_on', 'explore']
#         if 0 <= action < len(actions):
#             return actions[action]
#         return actions[abs(action) % len(actions)]
#
#     def _prepare_patterns(self, experiences):
#         """Преобразует опыт в паттерны размерности 47."""
#         patterns = []
#
#         for state, action, reward, next_state in experiences:
#             # state (21)
#             if hasattr(state, '__len__'):
#                 state_list = list(state)[:21]
#             else:
#                 state_list = [state]
#             while len(state_list) < 21:
#                 state_list.append(0.0)
#             state_list = state_list[:21]
#
#             # action one-hot (4)
#             action_one_hot = [0.0, 0.0, 0.0, 0.0]
#             if isinstance(action, int) and 0 <= action < 4:
#                 action_one_hot[action] = 1.0
#
#             # reward (1)
#             if isinstance(reward, (int, float)):
#                 reward_norm = max(-1.0, min(1.0, reward / 10.0))
#             else:
#                 reward_norm = 0.0
#
#             # next_state (21)
#             if hasattr(next_state, '__len__'):
#                 next_state_list = list(next_state)[:21]
#             else:
#                 next_state_list = [next_state]
#             while len(next_state_list) < 21:
#                 next_state_list.append(0.0)
#             next_state_list = next_state_list[:21]
#
#             # Собираем паттерн: state(21) + action(4) + reward(1) + next_state(21) = 47
#             pattern = state_list + action_one_hot + [reward_norm] + next_state_list
#
#             # Гарантируем ровно 47
#             while len(pattern) < 47:
#                 pattern.append(0.0)
#             pattern = pattern[:47]
#
#             patterns.append(pattern)
#
#         return np.array(patterns, dtype=np.float32)
#
#
#
#     # def _prepare_patterns(self, experiences):
#     #     """Преобразует опыт в паттерны."""
#     #     patterns = []
#     #
#     #     for state, action, reward, next_state in experiences:
#     #         # Приводим state к фиксированной размерности
#     #         if hasattr(state, '__len__'):
#     #             state_list = list(state)
#     #         else:
#     #             state_list = [state]
#     #
#     #         # Обрезаем или дополняем до state_dim
#     #         if len(state_list) > self.state_dim:
#     #             state_list = state_list[:self.state_dim]
#     #         else:
#     #             while len(state_list) < self.state_dim:
#     #                 state_list.append(0.0)
#     #
#     #         # action one-hot
#     #         action_one_hot = [0.0] * self.action_dim
#     #         if 0 <= action < self.action_dim:
#     #             action_one_hot[action] = 1.0
#     #
#     #         # reward (нормализуем)
#     #         reward_norm = max(-1.0, min(1.0, reward / 10.0))
#     #
#     #         # next_state (приводим к размерности)
#     #         if hasattr(next_state, '__len__'):
#     #             next_state_list = list(next_state)
#     #         else:
#     #             next_state_list = [next_state]
#     #
#     #         if len(next_state_list) > self.state_dim:
#     #             next_state_list = next_state_list[:self.state_dim]
#     #         else:
#     #             while len(next_state_list) < self.state_dim:
#     #                 next_state_list.append(0.0)
#     #
#     #         # Собираем паттерн
#     #         pattern = state_list + action_one_hot + [reward_norm] + next_state_list
#     #
#     #         # Обрезаем до pattern_dim
#     #         if len(pattern) > self.pattern_dim:
#     #             pattern = pattern[:self.pattern_dim]
#     #         else:
#     #             while len(pattern) < self.pattern_dim:
#     #                 pattern.append(0.0)
#     #
#     #         patterns.append(pattern)
#     #
#     #     return np.array(patterns, dtype=np.float32)
#
#
# class Encoder(nn.Module):
#     def __init__(self, pattern_dim=47, latent_dim=128):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(pattern_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, latent_dim),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         return self.model(x)
#
#
#
#
#
#
# # # models/gan.py (полная исправленная версия)
# #
# # import os
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.utils.data import DataLoader, TensorDataset
# # from typing import List, Tuple, Dict, Any, Optional
# # import random
# # import json
# # from collections import deque
# #
# #
# # class Generator(nn.Module):
# #     """Генератор создаёт новые модели поведения."""
# #
# #     def __init__(self, latent_dim: int = 100, output_dim: int = 64,
# #                  hidden_dims: List[int] = [256, 512, 256], dropout_rate: float = 0.3):
# #         super(Generator, self).__init__()
# #
# #         self.latent_dim = latent_dim
# #         self.output_dim = output_dim
# #
# #         layers = []
# #         prev_dim = latent_dim
# #
# #         for hidden_dim in hidden_dims:
# #             layers.extend([
# #                 nn.Linear(prev_dim, hidden_dim),
# #                 nn.BatchNorm1d(hidden_dim),
# #                 nn.ReLU(True),
# #                 nn.Dropout(dropout_rate)
# #             ])
# #             prev_dim = hidden_dim
# #
# #         layers.append(nn.Linear(prev_dim, output_dim))
# #         layers.append(nn.Tanh())
# #
# #         self.model = nn.Sequential(*layers)
# #
# #     def forward(self, z: torch.Tensor) -> torch.Tensor:
# #         return self.model(z)
# #
# #
# # class Discriminator(nn.Module):
# #     """Дискриминатор оценивает качество паттернов."""
# #
# #     def __init__(self, input_dim: int = 64, hidden_dims: List[int] = [256, 128, 64],
# #                  dropout_rate: float = 0.3):
# #         super(Discriminator, self).__init__()
# #
# #         self.input_dim = input_dim
# #
# #         layers = []
# #         prev_dim = input_dim
# #
# #         for hidden_dim in hidden_dims:
# #             layers.extend([
# #                 nn.Linear(prev_dim, hidden_dim),
# #                 nn.LeakyReLU(0.2, True),
# #                 nn.Dropout(dropout_rate)
# #             ])
# #             prev_dim = hidden_dim
# #
# #         layers.append(nn.Linear(prev_dim, 1))
# #         layers.append(nn.Sigmoid())
# #
# #         self.model = nn.Sequential(*layers)
# #
# #     def forward(self, pattern: torch.Tensor) -> torch.Tensor:
# #         return self.model(pattern)
# #
# #     def evaluate_pattern(self, pattern: np.ndarray, device: torch.device = None) -> float:
# #         """Оценивает один паттерн."""
# #         if device is None:
# #             device = next(self.parameters()).device
# #
# #         if pattern.ndim == 1:
# #             pattern = pattern.reshape(1, -1)
# #
# #         pattern_tensor = torch.FloatTensor(pattern).to(device)
# #         with torch.no_grad():
# #             score = self.forward(pattern_tensor).cpu().item()
# #         return score
# #
# #
# # class ExperienceEncoder:
# #     """Кодирует опыт в векторное представление."""
# #
# #     def __init__(self, state_dim: int = 8, action_dim: int = 4, max_pattern_len: int = 100):
# #         self.state_dim = state_dim
# #         self.action_dim = action_dim
# #         self.max_pattern_len = max_pattern_len
# #         # Важно: pattern_dim вычисляется автоматически
# #         self.pattern_dim = state_dim + action_dim + 1 + state_dim  # 8+4+1+8 = 21
# #
# #     def encode_experience(self, experience: Tuple) -> np.ndarray:
# #         """Кодирует один переход в паттерн."""
# #         state, action, reward, next_state = experience
# #
# #         state_vec = np.array(state[:self.state_dim], dtype=np.float32)
# #         next_state_vec = np.array(next_state[:self.state_dim], dtype=np.float32)
# #
# #         action_vec = np.zeros(self.action_dim, dtype=np.float32)
# #         if isinstance(action, int) and 0 <= action < self.action_dim:
# #             action_vec[action] = 1.0
# #
# #         pattern = np.concatenate([
# #             state_vec,
# #             action_vec,
# #             np.array([reward], dtype=np.float32),
# #             next_state_vec
# #         ])
# #
# #         norm = np.linalg.norm(pattern) + 1e-8
# #         pattern = pattern / norm
# #
# #         return pattern
# #
# #     def encode_experiences(self, experiences: List[Tuple]) -> np.ndarray:
# #         """Кодирует список переходов."""
# #         patterns = []
# #         for exp in experiences:
# #             pattern = self.encode_experience(exp)
# #             patterns.append(pattern)
# #
# #         if len(patterns) > self.max_pattern_len:
# #             indices = random.sample(range(len(patterns)), self.max_pattern_len)
# #             patterns = [patterns[i] for i in indices]
# #
# #         return np.array(patterns, dtype=np.float32)
# #
# #     def decode_pattern(self, pattern: np.ndarray) -> Dict[str, Any]:
# #         """Декодирует паттерн обратно в читаемый формат."""
# #         state_dim = self.state_dim
# #         action_dim = self.action_dim
# #
# #         state = pattern[:state_dim]
# #         action_vec = pattern[state_dim:state_dim + action_dim]
# #         reward = pattern[state_dim + action_dim]
# #         next_state = pattern[state_dim + action_dim + 1:state_dim + action_dim + 1 + state_dim]
# #
# #         action = int(np.argmax(action_vec)) if np.max(action_vec) > 0.5 else -1
# #
# #         return {
# #             'state': state.tolist(),
# #             'action': action,
# #             'reward': float(reward),
# #             'next_state': next_state.tolist(),
# #             'action_vec': action_vec.tolist()
# #         }
# #
# #
# # class PatternRepository:
# #     """Хранит и управляет сгенерированными паттернами."""
# #
# #     def __init__(self, max_stored: int = 1000):
# #         self.patterns = []
# #         self.scores = []
# #         self.max_stored = max_stored
# #
# #     def add_pattern(self, pattern: np.ndarray, score: float):
# #         """Добавляет паттерн с оценкой."""
# #         self.patterns.append(pattern)
# #         self.scores.append(score)
# #
# #         combined = sorted(zip(self.patterns, self.scores), key=lambda x: x[1], reverse=True)
# #         self.patterns = [p for p, _ in combined[:self.max_stored]]
# #         self.scores = [s for _, s in combined[:self.max_stored]]
# #
# #     def get_best_patterns(self, n: int = 10) -> List[np.ndarray]:
# #         """Возвращает n лучших паттернов."""
# #         return self.patterns[:n]
# #
# #     def get_patterns_with_scores(self, min_score: float = 0.5) -> List[Tuple[np.ndarray, float]]:
# #         """Возвращает паттерны с оценками выше порога."""
# #         result = []
# #         for pattern, score in zip(self.patterns, self.scores):
# #             if score >= min_score:
# #                 result.append((pattern, score))
# #         return result
# #
# #     def get_average_score(self) -> float:
# #         if not self.scores:
# #             return 0.0
# #         return np.mean(self.scores)
# #
# #     def to_dict(self) -> Dict[str, Any]:
# #         return {
# #             'patterns': [p.tolist() for p in self.patterns],
# #             'scores': self.scores
# #         }
# #
# #     @classmethod
# #     def from_dict(cls, data: Dict[str, Any]) -> 'PatternRepository':
# #         repo = cls()
# #         repo.patterns = [np.array(p) for p in data['patterns']]
# #         repo.scores = data['scores']
# #         return repo
# #
# #
# # class PatternEvaluator:
# #     """Оценивает качество сгенерированных паттернов."""
# #
# #     def __init__(self, discriminator: Discriminator, encoder: ExperienceEncoder):
# #         self.discriminator = discriminator
# #         self.encoder = encoder
# #         self.device = next(discriminator.parameters()).device
# #
# #     def evaluate_pattern(self, pattern: np.ndarray) -> float:
# #         """Оценивает качество одного паттерна."""
# #         if pattern.ndim == 1:
# #             pattern = pattern.reshape(1, -1)
# #
# #         pattern_tensor = torch.FloatTensor(pattern).to(self.device)
# #         with torch.no_grad():
# #             score = self.discriminator(pattern_tensor).cpu().item()
# #         return score
# #
# #     def select_best_patterns(self, patterns: np.ndarray, n: int = 5) -> List[np.ndarray]:
# #         """Выбирает n лучших паттернов."""
# #         scores = [self.evaluate_pattern(p) for p in patterns]
# #         best_indices = np.argsort(scores)[-n:]
# #         return [patterns[i] for i in best_indices]
# #
# #     def validate_pattern(self, pattern: np.ndarray, min_score: float = 0.7) -> bool:
# #         """Проверяет, достаточно ли хорош паттерн."""
# #         score = self.evaluate_pattern(pattern)
# #         return score >= min_score
# #
# #
# # # models/gan.py (исправленный GAN)
# #
# # class GAN:
# #     """Полноценный GAN для генерации новых моделей поведения."""
# #
# #     def __init__(self,
# #                  latent_dim: int = 100,
# #                  pattern_dim: int = None,  # Теперь опционально, будет вычислен автоматически
# #                  device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
# #                  lr_generator: float = 0.0002,
# #                  lr_discriminator: float = 0.0002,
# #                  beta1: float = 0.5,
# #                  beta2: float = 0.999,
# #                  batch_size: int = 32,
# #                  experience_buffer_size: int = 10000,
# #                  state_dim: int = 8,
# #                  action_dim: int = 4):
# #
# #         self.device = torch.device(device)
# #         self.latent_dim = latent_dim
# #         self.state_dim = state_dim
# #         self.action_dim = action_dim
# #
# #         # Если pattern_dim не указан, вычисляем автоматически
# #         if pattern_dim is None:
# #             self.pattern_dim = state_dim + action_dim + 1 + state_dim
# #         else:
# #             self.pattern_dim = pattern_dim
# #
# #         self.batch_size = batch_size
# #
# #         # Инициализируем модели с правильной размерностью
# #         self.generator = Generator(latent_dim, self.pattern_dim).to(self.device)
# #         self.discriminator = Discriminator(self.pattern_dim).to(self.device)
# #
# #         # Оптимизаторы
# #         self.optimizer_g = optim.Adam(
# #             self.generator.parameters(),
# #             lr=lr_generator,
# #             betas=(beta1, beta2)
# #         )
# #         self.optimizer_d = optim.Adam(
# #             self.discriminator.parameters(),
# #             lr=lr_discriminator,
# #             betas=(beta1, beta2)
# #         )
# #
# #         # Функция потерь
# #         self.criterion = nn.BCELoss()
# #
# #         # Буфер для хранения опыта
# #         self.experience_buffer = deque(maxlen=experience_buffer_size)
# #
# #         # Encoder должен использовать те же размерности
# #         self.encoder = ExperienceEncoder(state_dim=state_dim, action_dim=action_dim)
# #
# #         # Репозиторий и оценщик
# #         self.pattern_repository = PatternRepository()
# #         self.pattern_evaluator = PatternEvaluator(self.discriminator, self.encoder)
# #
# #         # Счётчики для статистики
# #         self.generator_losses = []
# #         self.discriminator_losses = []
# #         self.generation_count = 0
# #
# #         print(f"GAN инициализирован на устройстве: {self.device}")
# #
# #     def add_experience(self, experience: Tuple):
# #         """Добавляет один переход в буфер."""
# #         self.experience_buffer.append(experience)
# #
# #     def add_experiences(self, experiences: List[Tuple]):
# #         """Добавляет список переходов в буфер."""
# #         self.experience_buffer.extend(experiences)
# #
# #     def prepare_training_data(self, experiences: Optional[List[Tuple]] = None) -> torch.Tensor:
# #         """Подготавливает данные для тренировки."""
# #         if experiences is None:
# #             experiences = list(self.experience_buffer)
# #
# #         if not experiences:
# #             return None
# #
# #         patterns = self.encoder.encode_experiences(experiences)
# #         patterns_tensor = torch.FloatTensor(patterns).to(self.device)
# #
# #         return patterns_tensor
# #
# #     def train_step(self, real_patterns: torch.Tensor) -> Tuple[float, float]:
# #         """Выполняет один шаг тренировки GAN."""
# #         batch_size = min(self.batch_size, real_patterns.size(0))
# #
# #         # Если batch_size == 1, увеличиваем его
# #         if batch_size < 2:
# #             batch_size = 2
# #             indices = torch.randperm(real_patterns.size(0))[:batch_size]
# #             real_patterns = real_patterns[indices]
# #
# #         real_labels = torch.ones(batch_size, 1, device=self.device)
# #         fake_labels = torch.zeros(batch_size, 1, device=self.device)
# #
# #         # --- Тренируем дискриминатор ---
# #         self.optimizer_d.zero_grad()
# #
# #         real_output = self.discriminator(real_patterns[:batch_size])
# #         d_loss_real = self.criterion(real_output, real_labels)
# #
# #         z = torch.randn(batch_size, self.latent_dim, device=self.device)
# #         fake_patterns = self.generator(z)
# #         fake_output = self.discriminator(fake_patterns.detach())
# #         d_loss_fake = self.criterion(fake_output, fake_labels)
# #
# #         d_loss = d_loss_real + d_loss_fake
# #         d_loss.backward()
# #         self.optimizer_d.step()
# #
# #         # --- Тренируем генератор ---
# #         self.optimizer_g.zero_grad()
# #
# #         z = torch.randn(batch_size, self.latent_dim, device=self.device)
# #         fake_patterns = self.generator(z)
# #         fake_output = self.discriminator(fake_patterns)
# #
# #         g_loss = self.criterion(fake_output, real_labels)
# #         g_loss.backward()
# #         self.optimizer_g.step()
# #
# #         return g_loss.item(), d_loss.item()
# #
# #     def train(self, experiences: Optional[List[Tuple]] = None,
# #               epochs: int = 10, verbose: bool = False) -> Dict[str, List[float]]:
# #         """Тренирует GAN на накопленном опыте."""
# #         if experiences is None:
# #             experiences = list(self.experience_buffer)
# #
# #         if not experiences:
# #             print("Нет данных для тренировки GAN.")
# #             return {'g_loss': [], 'd_loss': []}
# #
# #         patterns = self.prepare_training_data(experiences)
# #         if patterns is None:
# #             return {'g_loss': [], 'd_loss': []}
# #
# #         dataset = TensorDataset(patterns)
# #         dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
# #
# #         g_losses = []
# #         d_losses = []
# #
# #         for epoch in range(epochs):
# #             for batch in dataloader:
# #                 real_batch = batch[0]
# #                 g_loss, d_loss = self.train_step(real_batch)
# #                 g_losses.append(g_loss)
# #                 d_losses.append(d_loss)
# #
# #             if verbose and (epoch + 1) % 5 == 0:
# #                 avg_g_loss = np.mean(g_losses[-len(dataloader):])
# #                 avg_d_loss = np.mean(d_losses[-len(dataloader):])
# #                 print(f"GAN Epoch {epoch + 1}/{epochs}: G_loss={avg_g_loss:.4f}, D_loss={avg_d_loss:.4f}")
# #
# #         self.generator_losses.extend(g_losses)
# #         self.discriminator_losses.extend(d_losses)
# #
# #         return {'g_loss': g_losses, 'd_loss': d_losses}
# #
# #     def generate_pattern(self) -> np.ndarray:
# #         """Генерирует новый паттерн поведения."""
# #         self.generation_count += 1
# #         # Генерируем с batch_size=1, но в режиме оценки
# #         self.generator.eval()
# #         with torch.no_grad():
# #             z = torch.randn(1, self.latent_dim, device=self.device)
# #             pattern = self.generator(z).cpu().numpy().flatten()
# #         self.generator.train()
# #         return pattern
# #
# #     def generate_batch(self, n: int = 10) -> np.ndarray:
# #         """Генерирует батч новых паттернов."""
# #         self.generation_count += n
# #         # Для батча используем batch_size > 1
# #         if n < 2:
# #             n = 2
# #         self.generator.eval()
# #         with torch.no_grad():
# #             z = torch.randn(n, self.latent_dim, device=self.device)
# #             patterns = self.generator(z).cpu().numpy()
# #         self.generator.train()
# #         return patterns
# #
# #     def generate_rule(self) -> Dict[str, Any]:
# #         """Генерирует новое правило рефлекса."""
# #         pattern = self.generate_pattern()
# #         decoded = self.encoder.decode_pattern(pattern)
# #
# #         rule = {
# #             'sense_type': self._interpret_sense_type(decoded['state']),
# #             'signal_type': self._interpret_signal_type(decoded['state']),
# #             'action': self._interpret_action(decoded['action']),
# #             'threshold': float(abs(decoded['reward'])),
# #             'confidence': 0.7
# #         }
# #         return rule
# #
# #     def generate_instinct_pattern(self) -> Dict[str, Any]:
# #         """Генерирует новый паттерн инстинкта."""
# #         pattern = self.generate_pattern()
# #         decoded = self.encoder.decode_pattern(pattern)
# #
# #         instinct_pattern = {
# #             'pattern': {
# #                 'signals': {
# #                     'sound': self._interpret_signal_type(decoded['state']),
# #                     'vision': self._interpret_vision_signal(decoded['state'])
# #                 }
# #             },
# #             'action': {
# #                 'action': self._interpret_action(decoded['action'])
# #             }
# #         }
# #         return instinct_pattern
# #
# #     def get_training_stats(self) -> Dict[str, Any]:
# #         """Возвращает статистику тренировки."""
# #         return {
# #             'generator_losses': self.generator_losses[-100:] if self.generator_losses else [],
# #             'discriminator_losses': self.discriminator_losses[-100:] if self.discriminator_losses else [],
# #             'total_generations': self.generation_count,
# #             'buffer_size': len(self.experience_buffer),
# #             'avg_g_loss': np.mean(self.generator_losses[-50:]) if self.generator_losses else 0,
# #             'avg_d_loss': np.mean(self.discriminator_losses[-50:]) if self.discriminator_losses else 0,
# #             'pattern_dim': self.pattern_dim,
# #             'latent_dim': self.latent_dim
# #         }
# #
# #     def save_models(self, path_prefix: str = 'models/gan'):
# #         """Сохраняет модели на диск."""
# #         os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
# #         torch.save({
# #             'generator_state_dict': self.generator.state_dict(),
# #             'discriminator_state_dict': self.discriminator.state_dict(),
# #             'optimizer_g_state_dict': self.optimizer_g.state_dict(),
# #             'optimizer_d_state_dict': self.optimizer_d.state_dict(),
# #             'generator_losses': self.generator_losses,
# #             'discriminator_losses': self.discriminator_losses,
# #             'generation_count': self.generation_count,
# #             'latent_dim': self.latent_dim,
# #             'pattern_dim': self.pattern_dim
# #         }, f'{path_prefix}_checkpoint.pth')
# #         print(f"Модели сохранены в {path_prefix}_checkpoint.pth")
# #
# #     def load_models(self, path_prefix: str = 'models/gan'):
# #         """Загружает модели с диска."""
# #         checkpoint = torch.load(f'{path_prefix}_checkpoint.pth', map_location=self.device)
# #
# #         self.generator.load_state_dict(checkpoint['generator_state_dict'])
# #         self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
# #         self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
# #         self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
# #         self.generator_losses = checkpoint['generator_losses']
# #         self.discriminator_losses = checkpoint['discriminator_losses']
# #         self.generation_count = checkpoint['generation_count']
# #
# #         print(f"Модели загружены из {path_prefix}_checkpoint.pth")
# #
# #     # --- Вспомогательные методы для интерпретации ---
# #
# #     def _interpret_sense_type(self, state: np.ndarray) -> str:
# #         sense_types = ['smell', 'sound', 'vision', 'touch', 'temperature']
# #         idx = int(abs(state[0]) * len(sense_types)) % len(sense_types) if len(state) > 0 else 0
# #         return sense_types[idx]
# #
# #     def _interpret_signal_type(self, state: np.ndarray) -> str:
# #         signal_types = ['food_smell', 'predator_smell', 'fire_smell',
# #                         'loud_crash', 'bright_flash', 'danger_signal']
# #         idx = int(abs(state[1]) * len(signal_types)) % len(signal_types) if len(state) > 1 else 0
# #         return signal_types[idx]
# #
# #     def _interpret_vision_signal(self, state: np.ndarray) -> str:
# #         vision_signals = ['bright_flash', 'darkness', 'motion', 'color_change']
# #         idx = int(abs(state[2]) * len(vision_signals)) % len(vision_signals) if len(state) > 2 else 0
# #         return vision_signals[idx]
# #
# #     def _interpret_action(self, action: int) -> str:
# #         actions = ['grab', 'avoid', 'run_away', 'move_on', 'explore']
# #         if 0 <= action < len(actions):
# #             return actions[action]
# #         return actions[abs(action) % len(actions)]
# #
# # # class GAN:
# # #     """Полноценный GAN для генерации новых моделей поведения."""
# # #
# # #     def __init__(self,
# # #                  latent_dim: int = 100,
# # #                  pattern_dim: int = 64,
# # #                  device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
# # #                  lr_generator: float = 0.0002,
# # #                  lr_discriminator: float = 0.0002,
# # #                  beta1: float = 0.5,
# # #                  beta2: float = 0.999,
# # #                  batch_size: int = 32,
# # #                  experience_buffer_size: int = 10000):
# # #
# # #         self.device = torch.device(device)
# # #         self.latent_dim = latent_dim
# # #         self.pattern_dim = pattern_dim
# # #         self.batch_size = batch_size
# # #
# # #         # Инициализируем модели
# # #         self.generator = Generator(latent_dim, pattern_dim).to(self.device)
# # #         self.discriminator = Discriminator(pattern_dim).to(self.device)
# # #
# # #         # Оптимизаторы
# # #         self.optimizer_g = optim.Adam(
# # #             self.generator.parameters(),
# # #             lr=lr_generator,
# # #             betas=(beta1, beta2)
# # #         )
# # #         self.optimizer_d = optim.Adam(
# # #             self.discriminator.parameters(),
# # #             lr=lr_discriminator,
# # #             betas=(beta1, beta2)
# # #         )
# # #
# # #         # Функция потерь
# # #         self.criterion = nn.BCELoss()
# # #
# # #         # Буфер для хранения опыта
# # #         self.experience_buffer = deque(maxlen=experience_buffer_size)
# # #         self.encoder = ExperienceEncoder()
# # #
# # #         # Репозиторий и оценщик
# # #         self.pattern_repository = PatternRepository()
# # #         self.pattern_evaluator = PatternEvaluator(self.discriminator, self.encoder)
# # #
# # #         # Счётчики для статистики
# # #         self.generator_losses = []
# # #         self.discriminator_losses = []
# # #         self.generation_count = 0
# # #
# # #         print(f"GAN инициализирован на устройстве: {self.device}")
# # #
# # #     def add_experience(self, experience: Tuple):
# # #         """Добавляет один переход в буфер."""
# # #         self.experience_buffer.append(experience)
# # #
# # #     def add_experiences(self, experiences: List[Tuple]):
# # #         """Добавляет список переходов в буфер."""
# # #         self.experience_buffer.extend(experiences)
# # #
# # #     def prepare_training_data(self, experiences: Optional[List[Tuple]] = None) -> torch.Tensor:
# # #         """Подготавливает данные для тренировки."""
# # #         if experiences is None:
# # #             experiences = list(self.experience_buffer)
# # #
# # #         if not experiences:
# # #             return None
# # #
# # #         patterns = self.encoder.encode_experiences(experiences)
# # #         patterns_tensor = torch.FloatTensor(patterns).to(self.device)
# # #
# # #         return patterns_tensor
# # #
# # #     def train_step(self, real_patterns: torch.Tensor) -> Tuple[float, float]:
# # #         """Выполняет один шаг тренировки GAN."""
# # #         batch_size = min(self.batch_size, real_patterns.size(0))
# # #
# # #         real_labels = torch.ones(batch_size, 1, device=self.device)
# # #         fake_labels = torch.zeros(batch_size, 1, device=self.device)
# # #
# # #         # --- Тренируем дискриминатор ---
# # #         self.optimizer_d.zero_grad()
# # #
# # #         real_output = self.discriminator(real_patterns[:batch_size])
# # #         d_loss_real = self.criterion(real_output, real_labels)
# # #
# # #         z = torch.randn(batch_size, self.latent_dim, device=self.device)
# # #         fake_patterns = self.generator(z)
# # #         fake_output = self.discriminator(fake_patterns.detach())
# # #         d_loss_fake = self.criterion(fake_output, fake_labels)
# # #
# # #         d_loss = d_loss_real + d_loss_fake
# # #         d_loss.backward()
# # #         self.optimizer_d.step()
# # #
# # #         # --- Тренируем генератор ---
# # #         self.optimizer_g.zero_grad()
# # #
# # #         z = torch.randn(batch_size, self.latent_dim, device=self.device)
# # #         fake_patterns = self.generator(z)
# # #         fake_output = self.discriminator(fake_patterns)
# # #
# # #         g_loss = self.criterion(fake_output, real_labels)
# # #         g_loss.backward()
# # #         self.optimizer_g.step()
# # #
# # #         return g_loss.item(), d_loss.item()
# # #
# # #     def train(self, experiences: Optional[List[Tuple]] = None,
# # #               epochs: int = 10, verbose: bool = False) -> Dict[str, List[float]]:
# # #         """Тренирует GAN на накопленном опыте."""
# # #         if experiences is None:
# # #             experiences = list(self.experience_buffer)
# # #
# # #         if not experiences:
# # #             print("Нет данных для тренировки GAN.")
# # #             return {'g_loss': [], 'd_loss': []}
# # #
# # #         patterns = self.prepare_training_data(experiences)
# # #         if patterns is None:
# # #             return {'g_loss': [], 'd_loss': []}
# # #
# # #         dataset = TensorDataset(patterns)
# # #         dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
# # #
# # #         g_losses = []
# # #         d_losses = []
# # #
# # #         for epoch in range(epochs):
# # #             for batch in dataloader:
# # #                 real_batch = batch[0]
# # #                 g_loss, d_loss = self.train_step(real_batch)
# # #                 g_losses.append(g_loss)
# # #                 d_losses.append(d_loss)
# # #
# # #             if verbose and (epoch + 1) % 5 == 0:
# # #                 avg_g_loss = np.mean(g_losses[-len(dataloader):])
# # #                 avg_d_loss = np.mean(d_losses[-len(dataloader):])
# # #                 print(f"GAN Epoch {epoch + 1}/{epochs}: G_loss={avg_g_loss:.4f}, D_loss={avg_d_loss:.4f}")
# # #
# # #         self.generator_losses.extend(g_losses)
# # #         self.discriminator_losses.extend(d_losses)
# # #
# # #         return {'g_loss': g_losses, 'd_loss': d_losses}
# # #
# # #     def generate_pattern(self) -> np.ndarray:
# # #         """Генерирует новый паттерн поведения."""
# # #         self.generation_count += 1
# # #         z = torch.randn(1, self.latent_dim, device=self.device)
# # #         with torch.no_grad():
# # #             pattern = self.generator(z).cpu().numpy().flatten()
# # #         return pattern
# # #
# # #     def generate_batch(self, n: int = 10) -> np.ndarray:
# # #         """Генерирует батч новых паттернов."""
# # #         self.generation_count += n
# # #         z = torch.randn(n, self.latent_dim, device=self.device)
# # #         with torch.no_grad():
# # #             patterns = self.generator(z).cpu().numpy()
# # #         return patterns
# # #
# # #     def generate_rule(self) -> Dict[str, Any]:
# # #         """Генерирует новое правило рефлекса."""
# # #         pattern = self.generate_pattern()
# # #         decoded = self.encoder.decode_pattern(pattern)
# # #
# # #         rule = {
# # #             'sense_type': self._interpret_sense_type(decoded['state']),
# # #             'signal_type': self._interpret_signal_type(decoded['state']),
# # #             'action': self._interpret_action(decoded['action']),
# # #             'threshold': float(abs(decoded['reward'])),
# # #             'confidence': 0.7
# # #         }
# # #         return rule
# # #
# # #     def generate_instinct_pattern(self) -> Dict[str, Any]:
# # #         """Генерирует новый паттерн инстинкта."""
# # #         pattern = self.generate_pattern()
# # #         decoded = self.encoder.decode_pattern(pattern)
# # #
# # #         instinct_pattern = {
# # #             'pattern': {
# # #                 'signals': {
# # #                     'sound': self._interpret_signal_type(decoded['state']),
# # #                     'vision': self._interpret_vision_signal(decoded['state'])
# # #                 }
# # #             },
# # #             'action': {
# # #                 'action': self._interpret_action(decoded['action'])
# # #             }
# # #         }
# # #         return instinct_pattern
# # #
# # #     def get_training_stats(self) -> Dict[str, Any]:
# # #         """Возвращает статистику тренировки."""
# # #         return {
# # #             'generator_losses': self.generator_losses[-100:] if self.generator_losses else [],
# # #             'discriminator_losses': self.discriminator_losses[-100:] if self.discriminator_losses else [],
# # #             'total_generations': self.generation_count,
# # #             'buffer_size': len(self.experience_buffer),
# # #             'avg_g_loss': np.mean(self.generator_losses[-50:]) if self.generator_losses else 0,
# # #             'avg_d_loss': np.mean(self.discriminator_losses[-50:]) if self.discriminator_losses else 0,
# # #             'pattern_dim': self.pattern_dim,
# # #             'latent_dim': self.latent_dim
# # #         }
# # #
# # #     def save_models(self, path_prefix: str = 'models/gan'):
# # #         """Сохраняет модели на диск."""
# # #         torch.save({
# # #             'generator_state_dict': self.generator.state_dict(),
# # #             'discriminator_state_dict': self.discriminator.state_dict(),
# # #             'optimizer_g_state_dict': self.optimizer_g.state_dict(),
# # #             'optimizer_d_state_dict': self.optimizer_d.state_dict(),
# # #             'generator_losses': self.generator_losses,
# # #             'discriminator_losses': self.discriminator_losses,
# # #             'generation_count': self.generation_count,
# # #             'latent_dim': self.latent_dim,
# # #             'pattern_dim': self.pattern_dim
# # #         }, f'{path_prefix}_checkpoint.pth')
# # #         print(f"Модели сохранены в {path_prefix}_checkpoint.pth")
# # #
# # #     def load_models(self, path_prefix: str = 'models/gan'):
# # #         """Загружает модели с диска."""
# # #         checkpoint = torch.load(f'{path_prefix}_checkpoint.pth', map_location=self.device)
# # #
# # #         self.generator.load_state_dict(checkpoint['generator_state_dict'])
# # #         self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
# # #         self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
# # #         self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
# # #         self.generator_losses = checkpoint['generator_losses']
# # #         self.discriminator_losses = checkpoint['discriminator_losses']
# # #         self.generation_count = checkpoint['generation_count']
# # #
# # #         print(f"Модели загружены из {path_prefix}_checkpoint.pth")
# # #
# # #     # --- Вспомогательные методы для интерпретации ---
# # #
# # #     def _interpret_sense_type(self, state: np.ndarray) -> str:
# # #         sense_types = ['smell', 'sound', 'vision', 'touch', 'temperature']
# # #         idx = int(abs(state[0]) * len(sense_types)) % len(sense_types)
# # #         return sense_types[idx]
# # #
# # #     def _interpret_signal_type(self, state: np.ndarray) -> str:
# # #         signal_types = ['food_smell', 'predator_smell', 'fire_smell',
# # #                         'loud_crash', 'bright_flash', 'danger_signal']
# # #         idx = int(abs(state[1]) * len(signal_types)) % len(signal_types)
# # #         return signal_types[idx]
# # #
# # #     def _interpret_vision_signal(self, state: np.ndarray) -> str:
# # #         vision_signals = ['bright_flash', 'darkness', 'motion', 'color_change']
# # #         idx = int(abs(state[2]) * len(vision_signals)) % len(vision_signals)
# # #         return vision_signals[idx]
# # #
# # #     def _interpret_action(self, action: int) -> str:
# # #         actions = ['grab', 'avoid', 'run_away', 'move_on', 'explore']
# # #         if 0 <= action < len(actions):
# # #             return actions[action]
# # #         return actions[abs(action) % len(actions)]
# #
# #
# #
# #
# # # # models/gan.py
# # # import numpy as np
# # # import torch
# # # import torch.nn as nn
# # # import torch.optim as optim
# # # from torch.utils.data import DataLoader, TensorDataset
# # # from typing import List, Tuple, Dict, Any, Optional
# # # import random
# # # import json
# # # from collections import deque
# # #
# # #
# # # class Generator(nn.Module):
# # #     """
# # #     Генератор создаёт новые модели поведения (паттерны) на основе случайного шума
# # #     и исторических данных.
# # #     """
# # #
# # #     def __init__(self,
# # #                  latent_dim: int = 100,
# # #                  output_dim: int = 64,  # Размер выходного вектора (кодировка паттерна)
# # #                  hidden_dims: List[int] = [256, 512, 256],
# # #                  dropout_rate: float = 0.3):
# # #         super(Generator, self).__init__()
# # #
# # #         self.latent_dim = latent_dim
# # #         self.output_dim = output_dim
# # #
# # #         # Строим слои генератора
# # #         layers = []
# # #         prev_dim = latent_dim
# # #
# # #         for hidden_dim in hidden_dims:
# # #             layers.extend([
# # #                 nn.Linear(prev_dim, hidden_dim),
# # #                 nn.BatchNorm1d(hidden_dim),
# # #                 nn.ReLU(True),
# # #                 nn.Dropout(dropout_rate)
# # #             ])
# # #             prev_dim = hidden_dim
# # #
# # #         # Выходной слой
# # #         layers.append(nn.Linear(prev_dim, output_dim))
# # #         layers.append(nn.Tanh())  # Выход в диапазоне [-1, 1]
# # #
# # #         self.model = nn.Sequential(*layers)
# # #
# # #     def forward(self, z: torch.Tensor) -> torch.Tensor:
# # #         """Генерирует паттерн из вектора шума."""
# # #         return self.model(z)
# # #
# # #     def generate_pattern(self, device: torch.device = None) -> np.ndarray:
# # #         """Генерирует один паттерн."""
# # #         if device is None:
# # #             device = next(self.parameters()).device
# # #
# # #         z = torch.randn(1, self.latent_dim, device=device)
# # #         with torch.no_grad():
# # #             pattern = self.forward(z).cpu().numpy().flatten()
# # #         return pattern
# # #
# # #     def generate_batch(self, batch_size: int, device: torch.device = None) -> np.ndarray:
# # #         """Генерирует батч паттернов."""
# # #         if device is None:
# # #             device = next(self.parameters()).device
# # #
# # #         z = torch.randn(batch_size, self.latent_dim, device=device)
# # #         with torch.no_grad():
# # #             patterns = self.forward(z).cpu().numpy()
# # #         return patterns
# # #
# # #
# # # class Discriminator(nn.Module):
# # #     """
# # #     Дискриминатор оценивает, насколько сгенерированный паттерн похож на
# # #     реальные модели поведения из опыта бота.
# # #     """
# # #
# # #     def __init__(self,
# # #                  input_dim: int = 64,
# # #                  hidden_dims: List[int] = [256, 128, 64],
# # #                  dropout_rate: float = 0.3):
# # #         super(Discriminator, self).__init__()
# # #
# # #         self.input_dim = input_dim
# # #
# # #         # Строим слои дискриминатора
# # #         layers = []
# # #         prev_dim = input_dim
# # #
# # #         for hidden_dim in hidden_dims:
# # #             layers.extend([
# # #                 nn.Linear(prev_dim, hidden_dim),
# # #                 nn.LeakyReLU(0.2, True),
# # #                 nn.Dropout(dropout_rate)
# # #             ])
# # #             prev_dim = hidden_dim
# # #
# # #         # Выходной слой - вероятность того, что паттерн реальный
# # #         layers.append(nn.Linear(prev_dim, 1))
# # #         layers.append(nn.Sigmoid())
# # #
# # #         self.model = nn.Sequential(*layers)
# # #
# # #     def forward(self, pattern: torch.Tensor) -> torch.Tensor:
# # #         """Оценивает, насколько паттерн похож на реальный."""
# # #         return self.model(pattern)
# # #
# # #     def evaluate_pattern(self, pattern: np.ndarray, device: torch.device = None) -> float:
# # #         """Оценивает один паттерн."""
# # #         if device is None:
# # #             device = next(self.parameters()).device
# # #
# # #         pattern_tensor = torch.FloatTensor(pattern).unsqueeze(0).to(device)
# # #         with torch.no_grad():
# # #             score = self.forward(pattern_tensor).cpu().item()
# # #         return score
# # #
# # #
# # # # models/gan.py - исправленная версия ExperienceEncoder
# # # class ExperienceEncoder:
# # #     """
# # #     Кодирует опыт (state, action, reward, next_state) в векторное представление,
# # #     которое может быть использовано как паттерн для GAN.
# # #     """
# # #
# # #     def __init__(self, state_dim: int = 8, action_dim: int = 4, max_pattern_len: int = 100, pattern_dim: int = None):
# # #         """
# # #         Args:
# # #             state_dim: Размерность состояния
# # #             action_dim: Размерность действия (количество возможных действий)
# # #             max_pattern_len: Максимальное количество паттернов
# # #             pattern_dim: Не используется, оставлен для совместимости
# # #         """
# # #         self.state_dim = state_dim
# # #         self.action_dim = action_dim
# # #         self.max_pattern_len = max_pattern_len
# # #         # pattern_dim вычисляется автоматически
# # #         self.pattern_dim = state_dim + action_dim + 1 + state_dim  # state + action_vec + reward + next_state
# # #
# # #     def encode_experience(self,
# # #                           experience: Tuple) -> np.ndarray:
# # #         """
# # #         Кодирует один переход в паттерн.
# # #         experience: (state, action, reward, next_state)
# # #         """
# # #         state, action, reward, next_state = experience
# # #
# # #         # Кодируем состояние
# # #         state_vec = np.array(state[:self.state_dim], dtype=np.float32)
# # #         next_state_vec = np.array(next_state[:self.state_dim], dtype=np.float32)
# # #
# # #         # Кодируем действие (one-hot)
# # #         action_vec = np.zeros(self.action_dim, dtype=np.float32)
# # #         if isinstance(action, int) and 0 <= action < self.action_dim:
# # #             action_vec[action] = 1.0
# # #
# # #         # Собираем паттерн
# # #         pattern = np.concatenate([
# # #             state_vec,
# # #             action_vec,
# # #             np.array([reward], dtype=np.float32),
# # #             next_state_vec
# # #         ])
# # #
# # #         # Нормализуем
# # #         norm = np.linalg.norm(pattern) + 1e-8
# # #         pattern = pattern / norm
# # #
# # #         return pattern
# # #
# # #     def encode_experiences(self,
# # #                            experiences: List[Tuple]) -> np.ndarray:
# # #         """Кодирует список переходов в матрицу паттернов."""
# # #         patterns = []
# # #         for exp in experiences:
# # #             pattern = self.encode_experience(exp)
# # #             patterns.append(pattern)
# # #
# # #         # Обрезаем до максимальной длины
# # #         if len(patterns) > self.max_pattern_len:
# # #             indices = random.sample(range(len(patterns)), self.max_pattern_len)
# # #             patterns = [patterns[i] for i in indices]
# # #
# # #         return np.array(patterns, dtype=np.float32)
# # #
# # #     def decode_pattern(self, pattern: np.ndarray) -> Dict[str, Any]:
# # #         """
# # #         Декодирует паттерн обратно в читаемый формат.
# # #         Используется для интерпретации сгенерированных паттернов.
# # #         """
# # #         state_dim = self.state_dim
# # #         action_dim = self.action_dim
# # #
# # #         # Извлекаем компоненты
# # #         state = pattern[:state_dim]
# # #         action_vec = pattern[state_dim:state_dim + action_dim]
# # #         reward = pattern[state_dim + action_dim]
# # #         next_state = pattern[state_dim + action_dim + 1:state_dim + action_dim + 1 + state_dim]
# # #
# # #         # Определяем действие
# # #         action = int(np.argmax(action_vec)) if np.max(action_vec) > 0.5 else -1
# # #
# # #         return {
# # #             'state': state.tolist(),
# # #             'action': action,
# # #             'reward': float(reward),
# # #             'next_state': next_state.tolist(),
# # #             'action_vec': action_vec.tolist()
# # #         }
# # #
# # #
# # # # class ExperienceEncoder:
# # # #     """
# # # #     Кодирует опыт (state, action, reward, next_state) в векторное представление,
# # # #     которое может быть использовано как паттерн для GAN.
# # # #     """
# # # #
# # # #     def __init__(self, state_dim: int = 8, action_dim: int = 4, max_pattern_len: int = 100):
# # # #         self.state_dim = state_dim
# # # #         self.action_dim = action_dim
# # # #         self.max_pattern_len = max_pattern_len
# # # #
# # # #     def encode_experience(self,
# # # #                           experience: Tuple) -> np.ndarray:
# # # #         """
# # # #         Кодирует один переход в паттерн.
# # # #         experience: (state, action, reward, next_state)
# # # #         """
# # # #         state, action, reward, next_state = experience
# # # #
# # # #         # Кодируем состояние
# # # #         state_vec = np.array(state[:self.state_dim], dtype=np.float32)
# # # #         next_state_vec = np.array(next_state[:self.state_dim], dtype=np.float32)
# # # #
# # # #         # Кодируем действие (one-hot)
# # # #         action_vec = np.zeros(self.action_dim, dtype=np.float32)
# # # #         if isinstance(action, int) and 0 <= action < self.action_dim:
# # # #             action_vec[action] = 1.0
# # # #
# # # #         # Собираем паттерн
# # # #         pattern = np.concatenate([
# # # #             state_vec,
# # # #             action_vec,
# # # #             np.array([reward], dtype=np.float32),
# # # #             next_state_vec
# # # #         ])
# # # #
# # # #         # Нормализуем
# # # #         pattern = pattern / (np.linalg.norm(pattern) + 1e-8)
# # # #
# # # #         return pattern
# # # #
# # # #     def encode_experiences(self,
# # # #                            experiences: List[Tuple]) -> np.ndarray:
# # # #         """Кодирует список переходов в матрицу паттернов."""
# # # #         patterns = []
# # # #         for exp in experiences:
# # # #             pattern = self.encode_experience(exp)
# # # #             patterns.append(pattern)
# # # #
# # # #         # Обрезаем до максимальной длины
# # # #         if len(patterns) > self.max_pattern_len:
# # # #             indices = random.sample(range(len(patterns)), self.max_pattern_len)
# # # #             patterns = [patterns[i] for i in indices]
# # # #
# # # #         return np.array(patterns, dtype=np.float32)
# # # #
# # # #     def decode_pattern(self, pattern: np.ndarray) -> Dict[str, Any]:
# # # #         """
# # # #         Декодирует паттерн обратно в читаемый формат.
# # # #         Используется для интерпретации сгенерированных паттернов.
# # # #         """
# # # #         state_dim = self.state_dim
# # # #         action_dim = self.action_dim
# # # #
# # # #         # Извлекаем компоненты
# # # #         state = pattern[:state_dim]
# # # #         action_vec = pattern[state_dim:state_dim + action_dim]
# # # #         reward = pattern[state_dim + action_dim]
# # # #         next_state = pattern[state_dim + action_dim + 1:state_dim + action_dim + 1 + state_dim]
# # # #
# # # #         # Определяем действие
# # # #         action = int(np.argmax(action_vec)) if np.max(action_vec) > 0.5 else -1
# # # #
# # # #         return {
# # # #             'state': state.tolist(),
# # # #             'action': action,
# # # #             'reward': float(reward),
# # # #             'next_state': next_state.tolist(),
# # # #             'action_vec': action_vec.tolist()
# # # #         }
# # #
# # #
# # # # models/gan.py - исправленная инициализация GAN
# # # class GAN:
# # #     """
# # #     Полноценный GAN для генерации новых моделей поведения на основе
# # #     накопленного опыта бота.
# # #     """
# # #
# # #     def __init__(self,
# # #                  latent_dim: int = 100,
# # #                  pattern_dim: int = 64,
# # #                  device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
# # #                  lr_generator: float = 0.0002,
# # #                  lr_discriminator: float = 0.0002,
# # #                  beta1: float = 0.5,
# # #                  beta2: float = 0.999,
# # #                  batch_size: int = 32,
# # #                  experience_buffer_size: int = 10000):
# # #         self.device = torch.device(device)
# # #         self.latent_dim = latent_dim
# # #         self.pattern_dim = pattern_dim
# # #         self.batch_size = batch_size
# # #
# # #         # Инициализируем модели
# # #         self.generator = Generator(latent_dim, pattern_dim).to(self.device)
# # #         self.discriminator = Discriminator(pattern_dim).to(self.device)
# # #
# # #         # Оптимизаторы
# # #         self.optimizer_g = optim.Adam(
# # #             self.generator.parameters(),
# # #             lr=lr_generator,
# # #             betas=(beta1, beta2)
# # #         )
# # #         self.optimizer_d = optim.Adam(
# # #             self.discriminator.parameters(),
# # #             lr=lr_discriminator,
# # #             betas=(beta1, beta2)
# # #         )
# # #
# # #         # Функция потерь
# # #         self.criterion = nn.BCELoss()
# # #
# # #         # Буфер для хранения опыта
# # #         self.experience_buffer = deque(maxlen=experience_buffer_size)
# # #         self.encoder = ExperienceEncoder()  # Убрали pattern_dim
# # #
# # #         # Счётчики для статистики
# # #         self.generator_losses = []
# # #         self.discriminator_losses = []
# # #         self.generation_count = 0
# # #         self.pattern_repository = PatternRepository()
# # #         self.pattern_evaluator = PatternEvaluator(self.discriminator, self.encoder)
# # #
# # #         print(f"GAN инициализирован на устройстве: {self.device}")
# # #
# # # # class GAN:
# # # #     """
# # # #     Полноценный GAN для генерации новых моделей поведения на основе
# # # #     накопленного опыта бота.
# # # #     """
# # # #
# # # #     def __init__(self,
# # # #                  latent_dim: int = 100,
# # # #                  pattern_dim: int = 64,
# # # #                  device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
# # # #                  lr_generator: float = 0.0002,
# # # #                  lr_discriminator: float = 0.0002,
# # # #                  beta1: float = 0.5,
# # # #                  beta2: float = 0.999,
# # # #                  batch_size: int = 32,
# # # #                  experience_buffer_size: int = 10000):
# # # #
# # # #         self.device = torch.device(device)
# # # #         self.latent_dim = latent_dim
# # # #         self.pattern_dim = pattern_dim
# # # #         self.batch_size = batch_size
# # # #
# # # #         # Инициализируем модели
# # # #         self.generator = Generator(latent_dim, pattern_dim).to(self.device)
# # # #         self.discriminator = Discriminator(pattern_dim).to(self.device)
# # # #
# # # #         # Оптимизаторы
# # # #         self.optimizer_g = optim.Adam(
# # # #             self.generator.parameters(),
# # # #             lr=lr_generator,
# # # #             betas=(beta1, beta2)
# # # #         )
# # # #         self.optimizer_d = optim.Adam(
# # # #             self.discriminator.parameters(),
# # # #             lr=lr_discriminator,
# # # #             betas=(beta1, beta2)
# # # #         )
# # # #
# # # #         # Функция потерь
# # # #         self.criterion = nn.BCELoss()
# # # #
# # # #         # Буфер для хранения опыта
# # # #         self.experience_buffer = deque(maxlen=experience_buffer_size)
# # # #         self.encoder = ExperienceEncoder(pattern_dim=pattern_dim)
# # # #
# # # #         # Счётчики для статистики
# # # #         self.generator_losses = []
# # # #         self.discriminator_losses = []
# # # #         self.generation_count = 0
# # # #
# # # #         print(f"GAN инициализирован на устройстве: {self.device}")
# # # #
# # # #     def add_experience(self, experience: Tuple):
# # # #         """Добавляет один переход в буфер."""
# # # #         self.experience_buffer.append(experience)
# # # #
# # # #     def add_experiences(self, experiences: List[Tuple]):
# # # #         """Добавляет список переходов в буфер."""
# # # #         self.experience_buffer.extend(experiences)
# # # #
# # # #     def prepare_training_data(self, experiences: Optional[List[Tuple]] = None) -> torch.Tensor:
# # # #         """Подготавливает данные для тренировки."""
# # # #         if experiences is None:
# # # #             experiences = list(self.experience_buffer)
# # # #
# # # #         if not experiences:
# # # #             return None
# # # #
# # # #         # Кодируем опыт в паттерны
# # # #         patterns = self.encoder.encode_experiences(experiences)
# # # #
# # # #         # Преобразуем в тензор
# # # #         patterns_tensor = torch.FloatTensor(patterns).to(self.device)
# # # #
# # # #         return patterns_tensor
# # # #
# # # #     def train_step(self, real_patterns: torch.Tensor) -> Tuple[float, float]:
# # # #         """Выполняет один шаг тренировки GAN."""
# # # #         batch_size = min(self.batch_size, real_patterns.size(0))
# # # #
# # # #         # Метки для реальных и фейковых данных
# # # #         real_labels = torch.ones(batch_size, 1, device=self.device)
# # # #         fake_labels = torch.zeros(batch_size, 1, device=self.device)
# # # #
# # # #         # --- Тренируем дискриминатор ---
# # # #         self.optimizer_d.zero_grad()
# # # #
# # # #         # Потери на реальных данных
# # # #         real_output = self.discriminator(real_patterns[:batch_size])
# # # #         d_loss_real = self.criterion(real_output, real_labels)
# # # #
# # # #         # Потери на фейковых данных
# # # #         z = torch.randn(batch_size, self.latent_dim, device=self.device)
# # # #         fake_patterns = self.generator(z)
# # # #         fake_output = self.discriminator(fake_patterns.detach())
# # # #         d_loss_fake = self.criterion(fake_output, fake_labels)
# # # #
# # # #         # Общая потеря дискриминатора
# # # #         d_loss = d_loss_real + d_loss_fake
# # # #         d_loss.backward()
# # # #         self.optimizer_d.step()
# # # #
# # # #         # --- Тренируем генератор ---
# # # #         self.optimizer_g.zero_grad()
# # # #
# # # #         # Генерируем фейковые паттерны
# # # #         z = torch.randn(batch_size, self.latent_dim, device=self.device)
# # # #         fake_patterns = self.generator(z)
# # # #         fake_output = self.discriminator(fake_patterns)
# # # #
# # # #         # Потеря генератора - хочет обмануть дискриминатор
# # # #         g_loss = self.criterion(fake_output, real_labels)
# # # #         g_loss.backward()
# # # #         self.optimizer_g.step()
# # # #
# # # #         return g_loss.item(), d_loss.item()
# # # #
# # # #     def train(self,
# # # #               experiences: Optional[List[Tuple]] = None,
# # # #               epochs: int = 10,
# # # #               verbose: bool = False) -> Dict[str, List[float]]:
# # # #         """
# # # #         Тренирует GAN на накопленном опыте.
# # # #         """
# # # #         if experiences is None:
# # # #             experiences = list(self.experience_buffer)
# # # #
# # # #         if not experiences:
# # # #             print("Нет данных для тренировки GAN.")
# # # #             return {'g_loss': [], 'd_loss': []}
# # # #
# # # #         patterns = self.prepare_training_data(experiences)
# # # #         if patterns is None:
# # # #             return {'g_loss': [], 'd_loss': []}
# # # #
# # # #         # Создаём DataLoader для батчей
# # # #         dataset = TensorDataset(patterns)
# # # #         dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
# # # #
# # # #         g_losses = []
# # # #         d_losses = []
# # # #
# # # #         for epoch in range(epochs):
# # # #             for batch in dataloader:
# # # #                 real_batch = batch[0]
# # # #                 g_loss, d_loss = self.train_step(real_batch)
# # # #                 g_losses.append(g_loss)
# # # #                 d_losses.append(d_loss)
# # # #
# # # #             if verbose and (epoch + 1) % 5 == 0:
# # # #                 avg_g_loss = np.mean(g_losses[-len(dataloader):])
# # # #                 avg_d_loss = np.mean(d_losses[-len(dataloader):])
# # # #                 print(f"GAN Epoch {epoch + 1}/{epochs}: G_loss={avg_g_loss:.4f}, D_loss={avg_d_loss:.4f}")
# # # #
# # # #         self.generator_losses.extend(g_losses)
# # # #         self.discriminator_losses.extend(d_losses)
# # # #
# # # #         return {'g_loss': g_losses, 'd_loss': d_losses}
# # # #
# # # #     def generate_pattern(self) -> np.ndarray:
# # # #         """Генерирует новый паттерн поведения."""
# # # #         self.generation_count += 1
# # # #         return self.generator.generate_pattern(self.device)
# # # #
# # # #     def generate_batch(self, n: int = 10) -> np.ndarray:
# # # #         """Генерирует батч новых паттернов."""
# # # #         self.generation_count += n
# # # #         return self.generator.generate_batch(n, self.device)
# # # #
# # # #     def generate_rule(self) -> Dict[str, Any]:
# # # #         """
# # # #         Генерирует новое правило рефлекса на основе сгенерированного паттерна.
# # # #         """
# # # #         pattern = self.generate_pattern()
# # # #
# # # #         # Декодируем паттерн
# # # #         decoded = self.encoder.decode_pattern(pattern)
# # # #
# # # #         # Преобразуем в правило рефлекса
# # # #         rule = {
# # # #             'sense_type': self._interpret_sense_type(decoded['state']),
# # # #             'signal_type': self._interpret_signal_type(decoded['state']),
# # # #             'action': self._interpret_action(decoded['action']),
# # # #             'threshold': float(abs(decoded['reward']))  # Используем reward как порог
# # # #         }
# # # #
# # # #         return rule
# # # #
# # # #     def generate_instinct_pattern(self) -> Dict[str, Any]:
# # # #         """
# # # #         Генерирует новый паттерн инстинкта на основе сгенерированного паттерна.
# # # #         """
# # # #         pattern = self.generate_pattern()
# # # #         decoded = self.encoder.decode_pattern(pattern)
# # # #
# # # #         # Преобразуем в паттерн инстинкта
# # # #         instinct_pattern = {
# # # #             'pattern': {
# # # #                 'signals': {
# # # #                     'sound': self._interpret_signal_type(decoded['state']),
# # # #                     'vision': self._interpret_vision_signal(decoded['state'])
# # # #                 }
# # # #             },
# # # #             'action': {
# # # #                 'action': self._interpret_action(decoded['action'])
# # # #             }
# # # #         }
# # # #
# # # #         return instinct_pattern
# # # #
# # # #     def save_models(self, path_prefix: str = 'models/gan'):
# # # #         """Сохраняет модели на диск."""
# # # #         torch.save({
# # # #             'generator_state_dict': self.generator.state_dict(),
# # # #             'discriminator_state_dict': self.discriminator.state_dict(),
# # # #             'optimizer_g_state_dict': self.optimizer_g.state_dict(),
# # # #             'optimizer_d_state_dict': self.optimizer_d.state_dict(),
# # # #             'generator_losses': self.generator_losses,
# # # #             'discriminator_losses': self.discriminator_losses,
# # # #             'generation_count': self.generation_count,
# # # #             'latent_dim': self.latent_dim,
# # # #             'pattern_dim': self.pattern_dim
# # # #         }, f'{path_prefix}_checkpoint.pth')
# # # #         print(f"Модели сохранены в {path_prefix}_checkpoint.pth")
# # # #
# # # #     def load_models(self, path_prefix: str = 'models/gan'):
# # # #         """Загружает модели с диска."""
# # # #         checkpoint = torch.load(f'{path_prefix}_checkpoint.pth', map_location=self.device)
# # # #
# # # #         self.generator.load_state_dict(checkpoint['generator_state_dict'])
# # # #         self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
# # # #         self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
# # # #         self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
# # # #         self.generator_losses = checkpoint['generator_losses']
# # # #         self.discriminator_losses = checkpoint['discriminator_losses']
# # # #         self.generation_count = checkpoint['generation_count']
# # # #
# # # #         print(f"Модели загружены из {path_prefix}_checkpoint.pth")
# # # #
# # # #     # --- Вспомогательные методы для интерпретации паттернов ---
# # # #
# # # #     def _interpret_sense_type(self, state: np.ndarray) -> str:
# # # #         """Интерпретирует тип сенсора из состояния."""
# # # #         # Определяем на основе первых элементов состояния
# # # #         sense_types = ['smell', 'sound', 'vision', 'touch', 'temperature']
# # # #         idx = int(abs(state[0]) * len(sense_types)) % len(sense_types)
# # # #         return sense_types[idx]
# # # #
# # # #     def _interpret_signal_type(self, state: np.ndarray) -> str:
# # # #         """Интерпретирует тип сигнала из состояния."""
# # # #         signal_types = ['food_smell', 'predator_smell', 'fire_smell',
# # # #                         'loud_crash', 'bright_flash', 'danger_signal']
# # # #         idx = int(abs(state[1]) * len(signal_types)) % len(signal_types)
# # # #         return signal_types[idx]
# # # #
# # # #     def _interpret_vision_signal(self, state: np.ndarray) -> str:
# # # #         """Интерпретирует визуальный сигнал из состояния."""
# # # #         vision_signals = ['bright_flash', 'darkness', 'motion', 'color_change']
# # # #         idx = int(abs(state[2]) * len(vision_signals)) % len(vision_signals)
# # # #         return vision_signals[idx]
# # # #
# # # #     def _interpret_action(self, action: int) -> str:
# # # #         """Интерпретирует действие из паттерна."""
# # # #         actions = ['grab', 'avoid', 'run_away', 'move_on', 'explore']
# # # #         if 0 <= action < len(actions):
# # # #             return actions[action]
# # # #         return actions[abs(action) % len(actions)]
# # # #
# # # #     def get_training_stats(self) -> Dict[str, Any]:
# # # #         """Возвращает статистику тренировки."""
# # # #         return {
# # # #             'generator_losses': self.generator_losses[-100:] if self.generator_losses else [],
# # # #             'discriminator_losses': self.discriminator_losses[-100:] if self.discriminator_losses else [],
# # # #             'total_generations': self.generation_count,
# # # #             'buffer_size': len(self.experience_buffer),
# # # #             'avg_g_loss': np.mean(self.generator_losses[-50:]) if self.generator_losses else 0,
# # # #             'avg_d_loss': np.mean(self.discriminator_losses[-50:]) if self.discriminator_losses else 0,
# # # #             'pattern_dim': self.pattern_dim,
# # # #             'latent_dim': self.latent_dim
# # # #         }
# # # #
# # # #     def visualize_patterns(self, n_patterns: int = 5) -> np.ndarray:
# # # #         """Визуализирует сгенерированные паттерны как тепловую карту."""
# # # #         patterns = self.generate_batch(n_patterns)
# # # #         # Нормализуем для отображения
# # # #         patterns_normalized = (patterns - patterns.min()) / (patterns.max() - patterns.min() + 1e-8)
# # # #         return patterns_normalized
# # # #
# # # #     def test_generator_quality(self, n_tests: int = 100) -> Dict[str, float]:
# # # #         """
# # # #         Тестирует качество генератора, генерируя паттерны и оценивая их
# # # #         дискриминатором.
# # # #         """
# # # #         if not self.discriminator:
# # # #             return {'avg_score': 0.0, 'std_score': 0.0, 'max_score': 0.0}
# # # #
# # # #         scores = []
# # # #         for _ in range(n_tests):
# # # #             pattern = self.generate_pattern()
# # # #             score = self.discriminator.evaluate_pattern(pattern)
# # # #             scores.append(score)
# # # #
# # # #         return {
# # # #             'avg_score': float(np.mean(scores)),
# # # #             'std_score': float(np.std(scores)),
# # # #             'max_score': float(np.max(scores)),
# # # #             'min_score': float(np.min(scores)),
# # # #             'above_threshold': float(np.mean([s > 0.7 for s in scores]))
# # # #         }
# # #
# # #
# # # # models/gan.py - исправленный PatternEvaluator
# # # class PatternEvaluator:
# # #     """
# # #     Оценивает качество сгенерированных паттернов и выбирает лучшие.
# # #     """
# # #
# # #     def __init__(self, discriminator: Discriminator, encoder: ExperienceEncoder):
# # #         self.discriminator = discriminator
# # #         self.encoder = encoder
# # #         self.device = next(discriminator.parameters()).device
# # #
# # #     def evaluate_pattern(self, pattern: np.ndarray) -> float:
# # #         """Оценивает качество одного паттерна."""
# # #         # Убеждаемся, что паттерн имеет правильную размерность
# # #         if pattern.ndim == 1:
# # #             pattern = pattern.reshape(1, -1)
# # #
# # #         pattern_tensor = torch.FloatTensor(pattern).to(self.device)
# # #         with torch.no_grad():
# # #             score = self.discriminator(pattern_tensor).cpu().item()
# # #         return score
# # #
# # #     def select_best_patterns(self,
# # #                              patterns: np.ndarray,
# # #                              n: int = 5) -> List[np.ndarray]:
# # #         """Выбирает n лучших паттернов по оценке дискриминатора."""
# # #         scores = [self.evaluate_pattern(p) for p in patterns]
# # #         best_indices = np.argsort(scores)[-n:]
# # #         return [patterns[i] for i in best_indices]
# # #
# # #     def validate_pattern(self,
# # #                          pattern: np.ndarray,
# # #                          min_score: float = 0.7) -> bool:
# # #         """
# # #         Проверяет, достаточно ли хорош паттерн.
# # #         Возвращает True, если паттерн качественный.
# # #         """
# # #         score = self.evaluate_pattern(pattern)
# # #         return score >= min_score
# # #
# # # # class PatternEvaluator:
# # # #     """
# # # #     Оценивает качество сгенерированных паттернов и выбирает лучшие.
# # # #     """
# # # #
# # # #     def __init__(self, discriminator: Discriminator, encoder: ExperienceEncoder):
# # # #         self.discriminator = discriminator
# # # #         self.encoder = encoder
# # # #
# # # #     def evaluate_pattern(self, pattern: np.ndarray) -> float:
# # # #         """Оценивает качество одного паттерна."""
# # # #         return self.discriminator.evaluate_pattern(pattern)
# # # #
# # # #     def select_best_patterns(self,
# # # #                              patterns: np.ndarray,
# # # #                              n: int = 5) -> List[np.ndarray]:
# # # #         """Выбирает n лучших паттернов по оценке дискриминатора."""
# # # #         scores = [self.evaluate_pattern(p) for p in patterns]
# # # #         best_indices = np.argsort(scores)[-n:]
# # # #         return [patterns[i] for i in best_indices]
# # # #
# # # #     def validate_pattern(self,
# # # #                          pattern: np.ndarray,
# # # #                          min_score: float = 0.7) -> bool:
# # # #         """
# # # #         Проверяет, достаточно ли хорош паттерн.
# # # #         Возвращает True, если паттерн качественный.
# # # #         """
# # # #         score = self.evaluate_pattern(pattern)
# # # #         return score >= min_score
# # #
# # #
# # # class PatternRepository:
# # #     """
# # #     Хранит и управляет сгенерированными паттернами, которые показали
# # #     хорошие результаты в симуляциях.
# # #     """
# # #
# # #     def __init__(self, max_stored: int = 1000):
# # #         self.patterns = []
# # #         self.scores = []
# # #         self.max_stored = max_stored
# # #
# # #     def add_pattern(self, pattern: np.ndarray, score: float):
# # #         """Добавляет паттерн с оценкой."""
# # #         self.patterns.append(pattern)
# # #         self.scores.append(score)
# # #
# # #         # Сортируем по убыванию оценки
# # #         combined = sorted(zip(self.patterns, self.scores), key=lambda x: x[1], reverse=True)
# # #         self.patterns = [p for p, _ in combined[:self.max_stored]]
# # #         self.scores = [s for _, s in combined[:self.max_stored]]
# # #
# # #     def get_best_patterns(self, n: int = 10) -> List[np.ndarray]:
# # #         """Возвращает n лучших паттернов."""
# # #         return self.patterns[:n]
# # #
# # #     def get_patterns_with_scores(self, min_score: float = 0.5) -> List[Tuple[np.ndarray, float]]:
# # #         """Возвращает паттерны с оценками выше порога."""
# # #         result = []
# # #         for pattern, score in zip(self.patterns, self.scores):
# # #             if score >= min_score:
# # #                 result.append((pattern, score))
# # #         return result
# # #
# # #     def get_average_score(self) -> float:
# # #         """Возвращает среднюю оценку всех паттернов."""
# # #         if not self.scores:
# # #             return 0.0
# # #         return np.mean(self.scores)
# # #
# # #     def to_dict(self) -> Dict[str, Any]:
# # #         """Сохраняет репозиторий в словарь."""
# # #         return {
# # #             'patterns': [p.tolist() for p in self.patterns],
# # #             'scores': self.scores
# # #         }
# # #
# # #     @classmethod
# # #     def from_dict(cls, data: Dict[str, Any]) -> 'PatternRepository':
# # #         """Загружает репозиторий из словаря."""
# # #         repo = cls()
# # #         repo.patterns = [np.array(p) for p in data['patterns']]
# # #         repo.scores = data['scores']
# # #         return repo
# # #
# # #
# # #
# # # # Первая версия - заглушка
# # # # class Generator:
# # # #     def __init__(self):
# # # #         self.predictor = any
# # # #
# # # # class Discriminator:
# # # #     def __init__(self):
# # # #         self.predictor = any