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
        # state (первые 21) → sense_type и signal_type
        # action (следующие 4) → action
        # reward (25-й) → threshold

        state_part = pattern[:21]
        action_part = pattern[21:25]
        reward_part = pattern[25] if len(pattern) > 25 else 0.5

        # Определяем сенсор на основе максимального значения в state
        max_state_idx = np.argmax(np.abs(state_part))

        # Определяем signal_type на основе индекса
        signal_types = [
            'food_smell',  # 0
            'predator_smell',  # 1
            'loud_crash',  # 2
            'bright_flash',  # 3
            'temperature_sense',  # 4
            'predator_roar',  # 5
            'danger_signal',  # 6
            'food_vision',  # 7
            'movement',  # 8
            'unknown'  # 9+
        ]

        # Индекс для signal_type
        signal_idx = max_state_idx % len(signal_types)
        signal_type = signal_types[signal_idx]

        # Определяем sense_type на основе signal_type
        sense_map = {
            'food_smell': 'smell',
            'predator_smell': 'smell',
            'loud_crash': 'sound',
            'bright_flash': 'vision',
            'temperature_sense': 'touch',
            'predator_roar': 'sound',
            'danger_signal': 'sound',
            'food_vision': 'vision',
            'movement': 'vision',
            'unknown': 'sense'
        }
        sense_type = sense_map.get(signal_type, 'sense')

        # Определяем действие
        action_idx = np.argmax(action_part)
        actions = ['grab', 'move_on', 'avoid', 'investigate']
        action = actions[action_idx % len(actions)]

        # Порог (нормализуем от 0.1 до 0.9)
        threshold = 0.3 + 0.4 * (reward_part + 1) / 2
        threshold = float(np.clip(threshold, 0.1, 0.9))

        # Приоритет (на основе силы сигнала)
        priority = float(0.5 + 0.5 * np.abs(state_part[max_state_idx]))
        priority = float(np.clip(priority, 0.3, 1.0))

        return {
            'sense_type': sense_type,
            'signal_type': signal_type,  # ← ТЕПЕРЬ ЗАПОЛНЯЕТСЯ!
            'threshold': threshold,
            'priority': priority,
            'action': action,
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


