# core/instinct_evolution.py (полностью исправленный)
"""
Генетический алгоритм для эволюции инстинктов.
Гибридный подход: GA + GAN для оптимального развития инстинктивного поведения.
"""

import numpy as np
import random
import json
import torch
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass


@dataclass
class InstinctIndividual:
    """Отдельный инстинкт в популяции."""
    pattern: np.ndarray  # 256-мерный паттерн
    fitness: float = 0.0  # Оценка качества
    generation: int = 0  # Поколение создания
    usage_count: int = 0  # Сколько раз использовался
    success_rate: float = 0.0  # Процент успешных применений
    id: int = None  # Уникальный идентификатор

    def __post_init__(self):
        if self.id is None:
            self.id = random.randint(0, 10 ** 9)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь для сохранения."""
        return {
            'pattern': self.pattern.tolist(),
            'fitness': self.fitness,
            'generation': self.generation,
            'usage_count': self.usage_count,
            'success_rate': self.success_rate,
            'id': self.id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InstinctIndividual':
        """Восстанавливает из словаря."""
        return cls(
            pattern=np.array(data['pattern']),
            fitness=data['fitness'],
            generation=data['generation'],
            usage_count=data['usage_count'],
            success_rate=data['success_rate'],
            id=data.get('id', random.randint(0, 10 ** 9))
        )


class InstinctEvaluator:
    """
    Оценщик качества инстинктов.
    Использует дискриминатор InstinctGAN и реальный опыт.
    """

    def __init__(self, gan_discriminator, experience_buffer):
        self.discriminator = gan_discriminator
        self.experience_buffer = experience_buffer

    def evaluate_pattern(self, pattern: np.ndarray) -> float:
        """
        Оценивает качество паттерна инстинкта.
        """
        # 1. Оценка дискриминатора
        discriminator_score = self._evaluate_by_discriminator(pattern)

        # 2. Оценка новизны
        novelty_score = self._evaluate_novelty(pattern)

        # 3. Оценка сложности
        complexity_score = self._evaluate_complexity(pattern)

        # Комбинированная оценка
        final_score = (
                0.6 * discriminator_score +
                0.3 * novelty_score +
                0.1 * complexity_score
        )

        return float(np.clip(final_score, 0.0, 1.0))

    def _evaluate_by_discriminator(self, pattern: np.ndarray) -> float:
        """Оценивает паттерн дискриминатором GAN."""
        try:
            if hasattr(self.discriminator, 'evaluate_pattern'):
                return self.discriminator.evaluate_pattern(pattern)
            else:
                # Убеждаемся, что паттерн имеет правильную размерность
                if len(pattern) != 256:
                    if len(pattern) > 256:
                        pattern = pattern[:256]
                    else:
                        padded = np.zeros(256)
                        padded[:len(pattern)] = pattern
                        pattern = padded

                pattern_tensor = torch.FloatTensor(pattern).unsqueeze(0)
                if hasattr(self.discriminator, 'to'):
                    device = next(self.discriminator.parameters()).device
                    pattern_tensor = pattern_tensor.to(device)
                    with torch.no_grad():
                        score = self.discriminator(pattern_tensor).cpu().item()
                    return score
                return 0.5
        except Exception:
            return 0.5

    def _evaluate_novelty(self, pattern: np.ndarray) -> float:
        """
        Оценивает новизну паттерна.
        Штрафует за слишком похожие на существующие.
        """
        if not self.experience_buffer:
            return 1.0

        buffer_size = min(len(self.experience_buffer), 50)
        sample = random.sample(self.experience_buffer, buffer_size)

        similarities = []
        for exp in sample:
            other = None

            if isinstance(exp, np.ndarray):
                if len(exp) == len(pattern):
                    other = exp
                else:
                    continue

            elif isinstance(exp, (list, tuple)) and len(exp) >= 4:
                try:
                    state = exp[0]
                    action = exp[1]
                    reward = exp[2]
                    next_state = exp[3]

                    # Приводим к массивам
                    if hasattr(state, '__len__'):
                        state_arr = np.array(state).flatten()
                    else:
                        state_arr = np.array([state])

                    if hasattr(next_state, '__len__'):
                        next_state_arr = np.array(next_state).flatten()
                    else:
                        next_state_arr = np.array([next_state])

                    action_arr = np.array([action])
                    reward_arr = np.array([reward])

                    combined = np.concatenate([state_arr, action_arr, reward_arr, next_state_arr])

                    if len(combined) >= 256:
                        other = combined[:256]
                    else:
                        other = np.zeros(256)
                        other[:len(combined)] = combined
                except Exception:
                    continue
            else:
                continue

            if other is not None and len(other) == 256 and len(pattern) == 256:
                try:
                    norm_pattern = np.linalg.norm(pattern) + 1e-8
                    norm_other = np.linalg.norm(other) + 1e-8
                    sim = np.dot(pattern, other) / (norm_pattern * norm_other)
                    similarities.append(sim)
                except Exception:
                    continue

        if similarities:
            max_similarity = max(similarities)
            return 1.0 - max_similarity
        else:
            return 1.0

    def _evaluate_complexity(self, pattern: np.ndarray) -> float:
        """
        Оценивает сложность паттерна.
        """
        if pattern is None or len(pattern) == 0:
            return 0.0

        min_val = np.min(pattern)
        max_val = np.max(pattern)
        if max_val - min_val < 1e-8:
            return 0.0

        normalized = (pattern - min_val) / (max_val - min_val + 1e-8)
        unique_ratio = len(np.unique(normalized)) / len(normalized)
        variance = np.var(pattern)
        complexity = 0.5 * unique_ratio + 0.5 * min(variance * 10, 1.0)

        return complexity

# class InstinctEvaluator:
#     """
#     Оценщик качества инстинктов.
#     Использует дискриминатор InstinctGAN и реальный опыт.
#     """
#
#     def __init__(self, gan_discriminator, experience_buffer):
#         self.discriminator = gan_discriminator
#         self.experience_buffer = experience_buffer
#
#     def evaluate_pattern(self, pattern: np.ndarray) -> float:
#         """
#         Оценивает качество паттерна инстинкта.
#         """
#         # 1. Оценка дискриминатора
#         discriminator_score = self._evaluate_by_discriminator(pattern)
#
#         # 2. Оценка новизны
#         novelty_score = self._evaluate_novelty(pattern)
#
#         # 3. Оценка сложности
#         complexity_score = self._evaluate_complexity(pattern)
#
#         # Комбинированная оценка
#         final_score = (
#                 0.6 * discriminator_score +
#                 0.3 * novelty_score +
#                 0.1 * complexity_score
#         )
#
#         return float(np.clip(final_score, 0.0, 1.0))
#
#     def _evaluate_by_discriminator(self, pattern: np.ndarray) -> float:
#         """Оценивает паттерн дискриминатором GAN."""
#         try:
#             if hasattr(self.discriminator, 'evaluate_pattern'):
#                 return self.discriminator.evaluate_pattern(pattern)
#             else:
#                 # Убеждаемся, что паттерн имеет правильную размерность
#                 if len(pattern) != 256:
#                     if len(pattern) > 256:
#                         pattern = pattern[:256]
#                     else:
#                         padded = np.zeros(256)
#                         padded[:len(pattern)] = pattern
#                         pattern = padded
#
#                 pattern_tensor = torch.FloatTensor(pattern).unsqueeze(0)
#                 if hasattr(self.discriminator, 'to'):
#                     device = next(self.discriminator.parameters()).device
#                     pattern_tensor = pattern_tensor.to(device)
#                     with torch.no_grad():
#                         score = self.discriminator(pattern_tensor).cpu().item()
#                     return score
#                 return 0.5
#         except Exception:
#             return 0.5
#
#     def _evaluate_novelty(self, pattern: np.ndarray) -> float:
#         """
#         Оценивает новизну паттерна.
#         Штрафует за слишком похожие на существующие.
#         """
#         if not self.experience_buffer:
#             return 1.0
#
#         buffer_size = min(len(self.experience_buffer), 50)
#         sample = random.sample(self.experience_buffer, buffer_size)
#
#         similarities = []
#         for exp in sample:
#             other = None
#
#             if isinstance(exp, np.ndarray):
#                 if len(exp) == len(pattern):
#                     other = exp
#                 else:
#                     continue
#
#             elif isinstance(exp, (list, tuple)) and len(exp) >= 4:
#                 try:
#                     state = exp[0]
#                     action = exp[1]
#                     reward = exp[2]
#                     next_state = exp[3]
#
#                     # Приводим к массивам
#                     if hasattr(state, '__len__'):
#                         state_arr = np.array(state).flatten()
#                     else:
#                         state_arr = np.array([state])
#
#                     if hasattr(next_state, '__len__'):
#                         next_state_arr = np.array(next_state).flatten()
#                     else:
#                         next_state_arr = np.array([next_state])
#
#                     action_arr = np.array([action])
#                     reward_arr = np.array([reward])
#
#                     combined = np.concatenate([state_arr, action_arr, reward_arr, next_state_arr])
#
#                     if len(combined) >= 256:
#                         other = combined[:256]
#                     else:
#                         other = np.zeros(256)
#                         other[:len(combined)] = combined
#                 except Exception:
#                     continue
#             else:
#                 continue
#
#             if other is not None and len(other) == 256 and len(pattern) == 256:
#                 try:
#                     norm_pattern = np.linalg.norm(pattern) + 1e-8
#                     norm_other = np.linalg.norm(other) + 1e-8
#                     sim = np.dot(pattern, other) / (norm_pattern * norm_other)
#                     similarities.append(sim)
#                 except Exception:
#                     continue
#
#         if similarities:
#             max_similarity = max(similarities)
#             return 1.0 - max_similarity
#         else:
#             return 1.0
#
#     def _evaluate_complexity(self, pattern: np.ndarray) -> float:
#         """
#         Оценивает сложность паттерна.
#         """
#         if pattern is None or len(pattern) == 0:
#             return 0.0
#
#         min_val = np.min(pattern)
#         max_val = np.max(pattern)
#         if max_val - min_val < 1e-8:
#             return 0.0
#
#         normalized = (pattern - min_val) / (max_val - min_val + 1e-8)
#         unique_ratio = len(np.unique(normalized)) / len(normalized)
#         variance = np.var(pattern)
#         complexity = 0.5 * unique_ratio + 0.5 * min(variance * 10, 1.0)
#
#         return complexity


class InstinctPopulation:
    """
    Популяция инстинктов с генетическим алгоритмом.
    Поддерживает эволюцию, скрещивание, мутации и отбор.
    """

    def __init__(self,
                 population_size: int = 20,
                 pattern_dim: int = 256,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elite_ratio: float = 0.2):
        """
        Args:
            population_size: Размер популяции
            pattern_dim: Размерность паттерна инстинкта
            mutation_rate: Вероятность мутации
            crossover_rate: Вероятность скрещивания
            elite_ratio: Доля элиты (лучших), сохраняемых без изменений
        """
        self.population_size = population_size
        self.pattern_dim = pattern_dim
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = max(1, int(population_size * elite_ratio))

        self.individuals: List[InstinctIndividual] = []
        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []

        # Инициализация случайной популяции
        self._initialize_population()

    def _initialize_population(self):
        """Создает начальную случайную популяцию."""
        for _ in range(self.population_size):
            pattern = self._generate_random_pattern()
            self.individuals.append(InstinctIndividual(pattern=pattern))

        print(f"✅ Инициализирована популяция инстинктов: {self.population_size} особей")

    def _generate_random_pattern(self) -> np.ndarray:
        """Генерирует случайный паттерн с нормализацией."""
        pattern = np.random.randn(self.pattern_dim)
        # Нормализуем к единичной длине
        norm = np.linalg.norm(pattern) + 1e-8
        return pattern / norm

    def evaluate_population(self, evaluator) -> None:
        """
        Оценивает всю популяцию.

        Args:
            evaluator: Объект с методом evaluate_pattern(pattern) -> float
        """
        for individual in self.individuals:
            individual.fitness = evaluator.evaluate_pattern(individual.pattern)

        # Обновляем историю
        fitnesses = [ind.fitness for ind in self.individuals]
        self.best_fitness_history.append(max(fitnesses))
        self.avg_fitness_history.append(np.mean(fitnesses))

    def select_parents(self, n_pairs: int) -> List[Tuple[InstinctIndividual, InstinctIndividual]]:
        """
        Выбирает пары родителей для скрещивания.
        Использует турнирный отбор.

        Args:
            n_pairs: Количество пар родителей

        Returns:
            Список пар (parent1, parent2)
        """
        pairs = []
        for _ in range(n_pairs):
            # Турнирный отбор: выбираем 3 случайных, берем лучшего
            def tournament_select():
                tournament = random.sample(self.individuals, min(3, len(self.individuals)))
                return max(tournament, key=lambda ind: ind.fitness)

            parent1 = tournament_select()
            parent2 = tournament_select()

            # Избегаем скрещивания с самим собой
            attempts = 0
            while (parent1.id == parent2.id or
                   np.array_equal(parent1.pattern, parent2.pattern)) and attempts < 10:
                parent2 = tournament_select()
                attempts += 1

            pairs.append((parent1, parent2))

        return pairs

    def crossover(self, parent1: InstinctIndividual, parent2: InstinctIndividual) -> Tuple[np.ndarray, np.ndarray]:
        """
        Одноточечный кроссовер для паттернов инстинктов.

        Args:
            parent1: Первый родитель
            parent2: Второй родитель

        Returns:
            Два паттерна-потомка
        """
        p1 = parent1.pattern
        p2 = parent2.pattern

        # Случайная точка разрыва
        point = np.random.randint(1, self.pattern_dim - 1)

        # Обмен частями
        child1 = np.concatenate([p1[:point], p2[point:]])
        child2 = np.concatenate([p2[:point], p1[point:]])

        return child1, child2

    def mutate(self, pattern: np.ndarray) -> np.ndarray:
        """
        Мутирует паттерн с адаптивной силой.

        Args:
            pattern: Исходный паттерн

        Returns:
            Мутировавший паттерн
        """
        mutated = pattern.copy()

        # Адаптивная мутация: сила зависит от поколения
        mutation_strength = 0.05 * (1.0 / (1.0 + self.generation * 0.01))

        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                # Добавляем шум
                noise = np.random.randn() * mutation_strength
                mutated[i] += noise

        # Нормализуем
        norm = np.linalg.norm(mutated) + 1e-8
        return mutated / norm

    def evolve_one_generation(self, gan_generated_patterns: Optional[List[np.ndarray]] = None):
        """
        Выполняет одну эволюционную итерацию.

        Args:
            gan_generated_patterns: Паттерны от GAN для добавления в популяцию
        """
        self.generation += 1

        # 1. Сохраняем элиту
        sorted_inds = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
        elite = sorted_inds[:self.elite_count]

        # 2. Добавляем паттерны от GAN (если есть)
        new_individuals = elite.copy()

        if gan_generated_patterns:
            for pattern in gan_generated_patterns:
                new_individuals.append(InstinctIndividual(pattern=pattern))

        # 3. Скрещивание для заполнения популяции
        n_needed = self.population_size - len(new_individuals)
        n_pairs = n_needed // 2 + (n_needed % 2)

        if n_pairs > 0 and len(self.individuals) > 1:
            parent_pairs = self.select_parents(n_pairs)

            for parent1, parent2 in parent_pairs:
                # Скрещивание
                if random.random() < self.crossover_rate:
                    child1_pattern, child2_pattern = self.crossover(parent1, parent2)
                else:
                    # Если не скрещиваемся, берем копии родителей
                    child1_pattern = parent1.pattern.copy()
                    child2_pattern = parent2.pattern.copy()

                # Мутация
                child1_pattern = self.mutate(child1_pattern)
                child2_pattern = self.mutate(child2_pattern)

                # Создаем индивидов
                child1 = InstinctIndividual(
                    pattern=child1_pattern,
                    generation=self.generation,
                    fitness=0.0  # Будет оценен позже
                )
                child2 = InstinctIndividual(
                    pattern=child2_pattern,
                    generation=self.generation,
                    fitness=0.0
                )

                new_individuals.append(child1)
                if len(new_individuals) < self.population_size:
                    new_individuals.append(child2)

        # 4. Обновляем популяцию
        self.individuals = new_individuals[:self.population_size]

        # 5. Обновляем статистику использования
        for ind in self.individuals:
            ind.generation = self.generation

    def get_best(self, n: int = 5) -> List[InstinctIndividual]:
        """Возвращает n лучших индивидов."""
        sorted_inds = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
        return sorted_inds[:n]

    def get_best_patterns(self, n: int = 5) -> List[np.ndarray]:
        """Возвращает паттерны n лучших индивидов."""
        return [ind.pattern for ind in self.get_best(n)]

    def get_population_stats(self) -> Dict[str, Any]:
        """Возвращает статистику популяции."""
        fitnesses = [ind.fitness for ind in self.individuals]

        return {
            'generation': self.generation,
            'population_size': len(self.individuals),
            'best_fitness': max(fitnesses) if fitnesses else 0,
            'avg_fitness': np.mean(fitnesses) if fitnesses else 0,
            'std_fitness': np.std(fitnesses) if fitnesses else 0,
            'min_fitness': min(fitnesses) if fitnesses else 0,
            'best_fitness_history': self.best_fitness_history[-20:],
            'avg_fitness_history': self.avg_fitness_history[-20:],
            'unique_patterns': len(set(tuple(ind.pattern) for ind in self.individuals))
        }

    def to_dict(self) -> Dict[str, Any]:
        """Сериализует популяцию для сохранения."""
        return {
            'generation': self.generation,
            'population_size': self.population_size,
            'pattern_dim': self.pattern_dim,
            'individuals': [ind.to_dict() for ind in self.individuals],
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InstinctPopulation':
        """Восстанавливает популяцию из словаря."""
        population = cls(
            population_size=data['population_size'],
            pattern_dim=data['pattern_dim']
        )
        population.generation = data['generation']
        population.individuals = [InstinctIndividual.from_dict(ind) for ind in data['individuals']]
        population.best_fitness_history = data['best_fitness_history']
        population.avg_fitness_history = data['avg_fitness_history']
        return population


# class InstinctEvaluator:
#     """
#     Оценщик качества инстинктов.
#     Использует дискриминатор InstinctGAN и реальный опыт.
#     """
#
#     def __init__(self, gan_discriminator, experience_buffer):
#         self.discriminator = gan_discriminator
#         self.experience_buffer = experience_buffer
#
#     def evaluate_pattern(self, pattern: np.ndarray) -> float:
#         """
#         Оценивает качество паттерна инстинкта.
#
#         Компоненты оценки:
#         1. Оценка дискриминатора GAN (0-1)
#         2. Новизна (штраф за повторение)
#         3. Сложность (предпочтение более сложным паттернам)
#         """
#         # 1. Оценка дискриминатора
#         discriminator_score = self._evaluate_by_discriminator(pattern)
#
#         # 2. Оценка новизны
#         novelty_score = self._evaluate_novelty(pattern)
#
#         # 3. Оценка сложности
#         complexity_score = self._evaluate_complexity(pattern)
#
#         # Комбинированная оценка
#         final_score = (
#                 0.6 * discriminator_score +
#                 0.3 * novelty_score +
#                 0.1 * complexity_score
#         )
#
#         return float(np.clip(final_score, 0.0, 1.0))
#
#     def _evaluate_by_discriminator(self, pattern: np.ndarray) -> float:
#         """Оценивает паттерн дискриминатором GAN."""
#         try:
#             # Проверяем, есть ли метод evaluate_pattern
#             if hasattr(self.discriminator, 'evaluate_pattern'):
#                 return self.discriminator.evaluate_pattern(pattern)
#             else:
#                 # Если нет, используем прямой forward
#                 # Убеждаемся, что паттерн имеет правильную размерность
#                 if len(pattern) != 256:
#                     if len(pattern) > 256:
#                         pattern = pattern[:256]
#                     else:
#                         padded = np.zeros(256)
#                         padded[:len(pattern)] = pattern
#                         pattern = padded
#
#                 pattern_tensor = torch.FloatTensor(pattern).unsqueeze(0)
#                 if hasattr(self.discriminator, 'to'):
#                     device = next(self.discriminator.parameters()).device
#                     pattern_tensor = pattern_tensor.to(device)
#                     with torch.no_grad():
#                         score = self.discriminator(pattern_tensor).cpu().item()
#                     return score
#                 return 0.5
#         except Exception as e:
#             return 0.5
#
#     def _evaluate_novelty(self, pattern: np.ndarray) -> float:
#         """
#         Оценивает новизну паттерна.
#         Штрафует за слишком похожие на существующие.
#         """
#         if not self.experience_buffer:
#             return 1.0
#
#         buffer_size = min(len(self.experience_buffer), 50)
#         sample = random.sample(self.experience_buffer, buffer_size)
#
#         similarities = []
#         for exp in sample:
#             other = None
#
#             # Если exp - это уже паттерн (np.ndarray)
#             if isinstance(exp, np.ndarray):
#                 if len(exp) == len(pattern):
#                     other = exp
#                 else:
#                     continue
#
#             # Если exp - это список или кортеж
#             elif isinstance(exp, (list, tuple)) and len(exp) > 0:
#                 if isinstance(exp[0], np.ndarray) and len(exp[0]) == len(pattern):
#                     other = exp[0]
#                 elif len(exp) == len(pattern):
#                     other = np.array(exp)
#                 else:
#                     continue
#
#             if other is not None and len(other) == len(pattern):
#                 try:
#                     sim = np.dot(pattern, other) / (
#                             np.linalg.norm(pattern) * np.linalg.norm(other) + 1e-8
#                     )
#                     similarities.append(sim)
#                 except Exception:
#                     continue
#
#         if similarities:
#             max_similarity = max(similarities)
#             return 1.0 - max_similarity
#         else:
#             return 1.0
#
#     def _evaluate_complexity(self, pattern: np.ndarray) -> float:
#         """
#         Оценивает сложность паттерна.
#         Предпочитает более сложные паттерны (больше разнообразия).
#         """
#         normalized = (pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern) + 1e-8)
#         unique_ratio = len(np.unique(normalized)) / len(normalized)
#         variance = np.var(pattern)
#         complexity = 0.5 * unique_ratio + 0.5 * min(variance * 10, 1.0)
#         return complexity


# class InstinctEvaluator:
#     """
#     Оценщик качества инстинктов.
#     Использует дискриминатор InstinctGAN и реальный опыт.
#     """
#
#     def __init__(self, gan_discriminator, experience_buffer):
#         self.discriminator = gan_discriminator
#         self.experience_buffer = experience_buffer
#
#     def evaluate_pattern(self, pattern: np.ndarray) -> float:
#         """
#         Оценивает качество паттерна инстинкта.
#         """
#         # 1. Оценка дискриминатора
#         discriminator_score = self._evaluate_by_discriminator(pattern)
#
#         # 2. Оценка новизны
#         novelty_score = self._evaluate_novelty(pattern)
#
#         # 3. Оценка сложности
#         complexity_score = self._evaluate_complexity(pattern)
#
#         # Комбинированная оценка
#         final_score = (
#                 0.6 * discriminator_score +
#                 0.3 * novelty_score +
#                 0.1 * complexity_score
#         )
#
#         return float(np.clip(final_score, 0.0, 1.0))
#
#     def _evaluate_by_discriminator(self, pattern: np.ndarray) -> float:
#         """Оценивает паттерн дискриминатором."""
#         try:
#             if hasattr(self.discriminator, 'evaluate_pattern'):
#                 return self.discriminator.evaluate_pattern(pattern)
#             else:
#                 pattern_tensor = torch.FloatTensor(pattern).unsqueeze(0)
#                 if hasattr(self.discriminator, 'to'):
#                     device = next(self.discriminator.parameters()).device
#                     pattern_tensor = pattern_tensor.to(device)
#                     with torch.no_grad():
#                         score = self.discriminator(pattern_tensor).cpu().item()
#                     return score
#                 return 0.5
#         except Exception:
#             return 0.5
#
#     def _evaluate_novelty(self, pattern: np.ndarray) -> float:
#         """
#         Оценивает новизну паттерна.
#         Штрафует за слишком похожие на существующие.
#         """
#         # Проверяем сходство с паттернами в буфере
#         if not self.experience_buffer:
#             return 1.0
#
#         # Берем случайные паттерны из буфера
#         buffer_size = min(len(self.experience_buffer), 50)
#         sample = random.sample(self.experience_buffer, buffer_size)
#
#         similarities = []
#         for exp in sample:
#             # Извлекаем паттерн из опыта
#             other = None
#
#             # Если exp - это уже паттерн (np.ndarray)
#             if isinstance(exp, np.ndarray):
#                 # Проверяем размерность
#                 if len(exp) == len(pattern):
#                     other = exp
#                 else:
#                     # Если размерность не совпадает, пропускаем
#                     continue
#
#             # Если exp - это список или кортеж
#             elif isinstance(exp, (list, tuple)) and len(exp) > 0:
#                 # Пробуем взять первый элемент, если это numpy array
#                 if isinstance(exp[0], np.ndarray) and len(exp[0]) == len(pattern):
#                     other = exp[0]
#                 # Или пробуем преобразовать весь exp в array
#                 elif len(exp) == len(pattern):
#                     other = np.array(exp)
#                 else:
#                     # Пропускаем, если размерность не совпадает
#                     continue
#
#             # Если нашли подходящий паттерн, вычисляем сходство
#             if other is not None and len(other) == len(pattern):
#                 try:
#                     sim = np.dot(pattern, other) / (
#                             np.linalg.norm(pattern) * np.linalg.norm(other) + 1e-8
#                     )
#                     similarities.append(sim)
#                 except Exception:
#                     continue
#
#         if similarities:
#             max_similarity = max(similarities)
#             # Чем меньше сходство, тем выше новизна
#             return 1.0 - max_similarity
#         else:
#             return 1.0
#
#
#     # def _evaluate_novelty(self, pattern: np.ndarray) -> float:
#     #     """Оценивает новизну паттерна."""
#     #     if not self.experience_buffer:
#     #         return 1.0
#     #
#     #     buffer_size = min(len(self.experience_buffer), 50)
#     #     sample = random.sample(self.experience_buffer, buffer_size)
#     #
#     #     similarities = []
#     #     for exp in sample:
#     #         if isinstance(exp, np.ndarray):
#     #             other = exp
#     #         elif isinstance(exp, (list, tuple)) and len(exp) > 0:
#     #             other = exp[0] if isinstance(exp[0], np.ndarray) else np.array(exp[0])
#     #         else:
#     #             continue
#     #
#     #         sim = np.dot(pattern, other) / (np.linalg.norm(pattern) * np.linalg.norm(other) + 1e-8)
#     #         similarities.append(sim)
#     #
#     #     if similarities:
#     #         max_similarity = max(similarities)
#     #         return 1.0 - max_similarity
#     #     else:
#     #         return 1.0
#
#     def _evaluate_complexity(self, pattern: np.ndarray) -> float:
#         """Оценивает сложность паттерна."""
#         normalized = (pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern) + 1e-8)
#         unique_ratio = len(np.unique(normalized)) / len(normalized)
#         variance = np.var(pattern)
#         complexity = 0.5 * unique_ratio + 0.5 * min(variance * 10, 1.0)
#         return complexity


class InstinctEvolutionEngine:
    """
    Движок эволюции инстинктов с интеграцией GAN.
    """

    def __init__(self,
                 population_size: int = 20,
                 pattern_dim: int = 256,
                 gan_generator=None,
                 gan_discriminator=None,
                 experience_buffer=None):

        self.population = InstinctPopulation(
            population_size=population_size,
            pattern_dim=pattern_dim
        )

        self.evaluator = InstinctEvaluator(
            gan_discriminator=gan_discriminator,
            experience_buffer=experience_buffer
        )

        self.gan_generator = gan_generator
        self.generation_patterns = []

    def evolve(self, n_generations: int = 10, gan_patterns_per_gen: int = 5) -> List[float]:
        """Запускает эволюцию инстинктов."""
        best_history = []

        for gen in range(n_generations):
            print(f"\n🧬 Поколение инстинктов {gen + 1}/{n_generations}")

            # 1. Оцениваем текущую популяцию
            self.population.evaluate_population(self.evaluator)
            stats = self.population.get_population_stats()

            print(f"  Лучший фитнес: {stats['best_fitness']:.4f}")
            print(f"  Средний фитнес: {stats['avg_fitness']:.4f}")
            print(f"  Разнообразие: {stats['unique_patterns']}/{self.population.population_size}")

            # 2. Генерируем паттерны от GAN
            gan_patterns = []
            if self.gan_generator:
                try:
                    # Пробуем generate_batch
                    if hasattr(self.gan_generator, 'generate_batch'):
                        gan_patterns = self.gan_generator.generate_batch(gan_patterns_per_gen)
                    else:
                        # Пробуем generate_pattern
                        for _ in range(gan_patterns_per_gen):
                            if hasattr(self.gan_generator, 'generate_pattern'):
                                pattern = self.gan_generator.generate_pattern()
                                gan_patterns.append(pattern)
                    print(f"  Сгенерировано {len(gan_patterns)} паттернов от GAN")
                except Exception as e:
                    print(f"  ⚠️ Ошибка генерации GAN: {e}")

            # 3. Эволюционируем
            self.population.evolve_one_generation(gan_patterns)

            # 4. Сохраняем историю
            best_history.append(stats['best_fitness'])

            # 5. Сохраняем лучшие паттерны
            if gen % 5 == 0:
                best_inds = self.population.get_best(3)
                self.generation_patterns.append({
                    'generation': gen,
                    'best_patterns': [ind.pattern for ind in best_inds],
                    'best_fitness': [ind.fitness for ind in best_inds]
                })

        print(f"\n✅ Эволюция инстинктов завершена за {n_generations} поколений")
        print(f"  Лучший фитнес: {max(best_history):.4f}")
        print(f"  Средний фитнес: {np.mean(best_history):.4f}")

        return best_history

    def get_best_instincts(self, n: int = 3) -> List[np.ndarray]:
        """Возвращает лучшие паттерны инстинктов."""
        return self.population.get_best_patterns(n)

    def get_population_stats(self) -> Dict[str, Any]:
        """Возвращает статистику популяции."""
        return self.population.get_population_stats()

    def save_state(self, filepath: str):
        """Сохраняет состояние популяции."""
        data = self.population.to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Состояние популяции сохранено в {filepath}")

    def load_state(self, filepath: str):
        """Загружает состояние популяции."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.population = InstinctPopulation.from_dict(data)
        print(f"✅ Состояние популяции загружено из {filepath}")

    # def select_parents(self, n_pairs: int) -> List[Tuple[InstinctIndividual, InstinctIndividual]]:
    #     """
    #     Выбирает пары родителей для скрещивания.
    #     Использует турнирный отбор.
    #
    #     Args:
    #         n_pairs: Количество пар родителей
    #
    #     Returns:
    #         Список пар (parent1, parent2)
    #     """
    #     pairs = []
    #     for _ in range(n_pairs):
    #         # Турнирный отбор: выбираем 3 случайных, берем лучшего
    #         def tournament_select():
    #             tournament = random.sample(self.individuals, min(3, len(self.individuals)))
    #             return max(tournament, key=lambda ind: ind.fitness)
    #
    #         parent1 = tournament_select()
    #         parent2 = tournament_select()
    #
    #         # Избегаем скрещивания с самим собой
    #         attempts = 0
    #         while parent1 == parent2 and attempts < 10:
    #             parent2 = tournament_select()
    #             attempts += 1
    #
    #         pairs.append((parent1, parent2))
    #
    #     return pairs

    def crossover(self, parent1: InstinctIndividual, parent2: InstinctIndividual) -> Tuple[np.ndarray, np.ndarray]:
        """
        Одноточечный кроссовер для паттернов инстинктов.

        Args:
            parent1: Первый родитель
            parent2: Второй родитель

        Returns:
            Два паттерна-потомка
        """
        p1 = parent1.pattern
        p2 = parent2.pattern

        # Случайная точка разрыва
        point = np.random.randint(1, self.pattern_dim - 1)

        # Обмен частями
        child1 = np.concatenate([p1[:point], p2[point:]])
        child2 = np.concatenate([p2[:point], p1[point:]])

        return child1, child2

    def mutate(self, pattern: np.ndarray) -> np.ndarray:
        """
        Мутирует паттерн с адаптивной силой.

        Args:
            pattern: Исходный паттерн

        Returns:
            Мутировавший паттерн
        """
        mutated = pattern.copy()

        # Адаптивная мутация: сила зависит от поколения
        mutation_strength = 0.05 * (1.0 / (1.0 + self.generation * 0.01))

        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                # Добавляем шум
                noise = np.random.randn() * mutation_strength
                mutated[i] += noise

        # Нормализуем
        norm = np.linalg.norm(mutated) + 1e-8
        return mutated / norm

    def evolve_one_generation(self, gan_generated_patterns: Optional[List[np.ndarray]] = None):
        """
        Выполняет одну эволюционную итерацию.

        Args:
            gan_generated_patterns: Паттерны от GAN для добавления в популяцию
        """
        self.generation += 1

        # 1. Сохраняем элиту
        sorted_inds = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
        elite = sorted_inds[:self.elite_count]

        # 2. Добавляем паттерны от GAN (если есть)
        new_individuals = elite.copy()

        if gan_generated_patterns:
            for pattern in gan_generated_patterns:
                new_individuals.append(InstinctIndividual(pattern=pattern))

        # 3. Скрещивание для заполнения популяции
        n_needed = self.population_size - len(new_individuals)
        n_pairs = n_needed // 2 + (n_needed % 2)

        if n_pairs > 0 and len(self.individuals) > 1:
            parent_pairs = self.select_parents(n_pairs)

            for parent1, parent2 in parent_pairs:
                # Скрещивание
                if random.random() < self.crossover_rate:
                    child1_pattern, child2_pattern = self.crossover(parent1, parent2)
                else:
                    # Если не скрещиваемся, берем копии родителей
                    child1_pattern = parent1.pattern.copy()
                    child2_pattern = parent2.pattern.copy()

                # Мутация
                child1_pattern = self.mutate(child1_pattern)
                child2_pattern = self.mutate(child2_pattern)

                # Создаем индивидов
                child1 = InstinctIndividual(
                    pattern=child1_pattern,
                    generation=self.generation,
                    fitness=0.0  # Будет оценен позже
                )
                child2 = InstinctIndividual(
                    pattern=child2_pattern,
                    generation=self.generation,
                    fitness=0.0
                )

                new_individuals.append(child1)
                if len(new_individuals) < self.population_size:
                    new_individuals.append(child2)

        # 4. Обновляем популяцию
        self.individuals = new_individuals[:self.population_size]

        # 5. Обновляем статистику использования
        for ind in self.individuals:
            ind.generation = self.generation

    def get_best(self, n: int = 5) -> List[InstinctIndividual]:
        """Возвращает n лучших индивидов."""
        sorted_inds = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
        return sorted_inds[:n]

    def get_best_patterns(self, n: int = 5) -> List[np.ndarray]:
        """Возвращает паттерны n лучших индивидов."""
        return [ind.pattern for ind in self.get_best(n)]

    def get_population_stats(self) -> Dict[str, Any]:
        """Возвращает статистику популяции."""
        fitnesses = [ind.fitness for ind in self.individuals]

        return {
            'generation': self.generation,
            'population_size': len(self.individuals),
            'best_fitness': max(fitnesses) if fitnesses else 0,
            'avg_fitness': np.mean(fitnesses) if fitnesses else 0,
            'std_fitness': np.std(fitnesses) if fitnesses else 0,
            'min_fitness': min(fitnesses) if fitnesses else 0,
            'best_fitness_history': self.best_fitness_history[-20:],
            'avg_fitness_history': self.avg_fitness_history[-20:],
            'unique_patterns': len(set(tuple(ind.pattern) for ind in self.individuals))
        }

    def to_dict(self) -> Dict[str, Any]:
        """Сериализует популяцию для сохранения."""
        return {
            'generation': self.generation,
            'population_size': self.population_size,
            'pattern_dim': self.pattern_dim,
            'individuals': [ind.to_dict() for ind in self.individuals],
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InstinctPopulation':
        """Восстанавливает популяцию из словаря."""
        population = cls(
            population_size=data['population_size'],
            pattern_dim=data['pattern_dim']
        )
        population.generation = data['generation']
        population.individuals = [InstinctIndividual.from_dict(ind) for ind in data['individuals']]
        population.best_fitness_history = data['best_fitness_history']
        population.avg_fitness_history = data['avg_fitness_history']
        return population


class InstinctEvaluator:
    """
    Оценщик качества инстинктов.
    Использует дискриминатор InstinctGAN и реальный опыт.
    """

    def __init__(self, gan_discriminator, experience_buffer):
        self.discriminator = gan_discriminator
        self.experience_buffer = experience_buffer

    def evaluate_pattern(self, pattern: np.ndarray) -> float:
        """
        Оценивает качество паттерна инстинкта.

        Компоненты оценки:
        1. Оценка дискриминатора GAN (0-1)
        2. Новизна (штраф за повторение)
        3. Сложность (предпочтение более сложным паттернам)
        """
        # 1. Оценка дискриминатора
        discriminator_score = self._evaluate_by_discriminator(pattern)

        # 2. Оценка новизны
        novelty_score = self._evaluate_novelty(pattern)

        # 3. Оценка сложности
        complexity_score = self._evaluate_complexity(pattern)

        # Комбинированная оценка
        final_score = (
                0.6 * discriminator_score +
                0.3 * novelty_score +
                0.1 * complexity_score
        )

        return float(np.clip(final_score, 0.0, 1.0))

    def _evaluate_by_discriminator(self, pattern: np.ndarray) -> float:
        """Оценивает паттерн дискриминатором GAN."""
        try:
            # Проверяем, есть ли метод evaluate_pattern
            if hasattr(self.discriminator, 'evaluate_pattern'):
                return self.discriminator.evaluate_pattern(pattern)
            else:
                # Если нет, используем прямой forward
                # Убеждаемся, что паттерн имеет правильную размерность
                if len(pattern) != 256:
                    # Если размерность не 256, обрезаем или дополняем
                    if len(pattern) > 256:
                        pattern = pattern[:256]
                    else:
                        # Дополняем нулями
                        padded = np.zeros(256)
                        padded[:len(pattern)] = pattern
                        pattern = padded

                pattern_tensor = torch.FloatTensor(pattern).unsqueeze(0)
                if hasattr(self.discriminator, 'to'):
                    device = next(self.discriminator.parameters()).device
                    pattern_tensor = pattern_tensor.to(device)
                    with torch.no_grad():
                        score = self.discriminator(pattern_tensor).cpu().item()
                    return score
                return 0.5
        except Exception as e:
            # print(f"  ⚠️ Ошибка оценки дискриминатором: {e}")
            return 0.5

    # def _evaluate_by_discriminator(self, pattern: np.ndarray) -> float:
    #     """Оценивает паттерн дискриминатором GAN."""
    #     try:
    #         # Проверяем, есть ли метод evaluate_pattern
    #         if hasattr(self.discriminator, 'evaluate_pattern'):
    #             return self.discriminator.evaluate_pattern(pattern)
    #         else:
    #             # Если нет, используем прямой forward
    #             pattern_tensor = torch.FloatTensor(pattern).unsqueeze(0)
    #             if hasattr(self.discriminator, 'to'):
    #                 device = next(self.discriminator.parameters()).device
    #                 pattern_tensor = pattern_tensor.to(device)
    #                 with torch.no_grad():
    #                     score = self.discriminator(pattern_tensor).cpu().item()
    #                 return score
    #             return 0.5
    #     except Exception as e:
    #         # print(f"  ⚠️ Ошибка оценки дискриминатором: {e}")
    #         return 0.5

    # def _evaluate_novelty(self, pattern: np.ndarray) -> float:
    #     """
    #     Оценивает новизну паттерна.
    #     Штрафует за слишком похожие на существующие.
    #     """
    #     # Проверяем сходство с паттернами в буфере
    #     if not self.experience_buffer:
    #         return 1.0
    #
    #     # Берем случайные паттерны из буфера
    #     buffer_size = min(len(self.experience_buffer), 50)
    #     sample = random.sample(self.experience_buffer, buffer_size)
    #
    #     similarities = []
    #     for exp in sample:
    #         # Если exp - это паттерн (np.ndarray)
    #         if isinstance(exp, np.ndarray):
    #             other = exp
    #         elif isinstance(exp, (list, tuple)) and len(exp) > 0:
    #             # Если это список, берем первый элемент
    #             other = exp[0] if isinstance(exp[0], np.ndarray) else np.array(exp[0])
    #         else:
    #             continue
    #
    #         # Косинусное сходство
    #         sim = np.dot(pattern, other) / (np.linalg.norm(pattern) * np.linalg.norm(other) + 1e-8)
    #         similarities.append(sim)
    #
    #     if similarities:
    #         max_similarity = max(similarities)
    #         # Чем меньше сходство, тем выше новизна
    #         return 1.0 - max_similarity
    #     else:
    #         return 1.0

    def _evaluate_complexity(self, pattern: np.ndarray) -> float:
        """
        Оценивает сложность паттерна.
        Предпочитает более сложные паттерны (больше разнообразия).
        """
        # Используем энтропию как меру сложности
        normalized = (pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern) + 1e-8)

        # Количество уникальных значений
        unique_ratio = len(np.unique(normalized)) / len(normalized)

        # Сложность по дисперсии
        variance = np.var(pattern)

        # Комбинируем
        complexity = 0.5 * unique_ratio + 0.5 * min(variance * 10, 1.0)

        return complexity


class InstinctEvolutionEngine:
    """
    Движок эволюции инстинктов с интеграцией GAN.
    """

    def __init__(self,
                 population_size: int = 20,
                 pattern_dim: int = 256,
                 gan_generator=None,
                 gan_discriminator=None,
                 experience_buffer=None):

        self.population = InstinctPopulation(
            population_size=population_size,
            pattern_dim=pattern_dim
        )

        self.evaluator = InstinctEvaluator(
            gan_discriminator=gan_discriminator,
            experience_buffer=experience_buffer
        )

        self.gan_generator = gan_generator
        self.generation_patterns = []  # Для анализа эволюции

    def evolve(self, n_generations: int = 10, gan_patterns_per_gen: int = 5) -> List[float]:
        """
        Запускает эволюцию инстинктов на несколько поколений.

        Args:
            n_generations: Количество поколений
            gan_patterns_per_gen: Сколько паттернов генерировать GAN за поколение

        Returns:
            История лучших фитнесов
        """
        best_history = []

        for gen in range(n_generations):
            print(f"\n🧬 Поколение инстинктов {gen + 1}/{n_generations}")

            # 1. Оцениваем текущую популяцию
            self.population.evaluate_population(self.evaluator)
            stats = self.population.get_population_stats()

            print(f"  Лучший фитнес: {stats['best_fitness']:.4f}")
            print(f"  Средний фитнес: {stats['avg_fitness']:.4f}")
            print(f"  Разнообразие: {stats['unique_patterns']}/{self.population.population_size}")

            # 2. Генерируем паттерны от GAN
            gan_patterns = []
            if self.gan_generator:
                try:
                    # Проверяем, есть ли метод generate_batch
                    if hasattr(self.gan_generator, 'generate_batch'):
                        gan_patterns = self.gan_generator.generate_batch(gan_patterns_per_gen)
                    else:
                        # Если нет, генерируем по одному
                        for _ in range(gan_patterns_per_gen):
                            pattern = self.gan_generator.generate_pattern()
                            gan_patterns.append(pattern)
                    print(f"  Сгенерировано {len(gan_patterns)} паттернов от GAN")
                except Exception as e:
                    print(f"  ⚠️ Ошибка генерации GAN: {e}")

            # 3. Эволюционируем
            self.population.evolve_one_generation(gan_patterns)

            # 4. Сохраняем историю
            best_history.append(stats['best_fitness'])

            # 5. Сохраняем лучшие паттерны
            if gen % 5 == 0:
                best_inds = self.population.get_best(3)
                self.generation_patterns.append({
                    'generation': gen,
                    'best_patterns': [ind.pattern for ind in best_inds],
                    'best_fitness': [ind.fitness for ind in best_inds]
                })

        print(f"\n✅ Эволюция инстинктов завершена за {n_generations} поколений")
        print(f"  Лучший фитнес: {max(best_history):.4f}")
        print(f"  Средний фитнес: {np.mean(best_history):.4f}")

        return best_history

    def get_best_instincts(self, n: int = 3) -> List[np.ndarray]:
        """Возвращает лучшие паттерны инстинктов."""
        return self.population.get_best_patterns(n)

    def get_population_stats(self) -> Dict[str, Any]:
        """Возвращает статистику популяции."""
        return self.population.get_population_stats()

    def save_state(self, filepath: str):
        """Сохраняет состояние популяции."""
        data = self.population.to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Состояние популяции сохранено в {filepath}")

    def load_state(self, filepath: str):
        """Загружает состояние популяции."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.population = InstinctPopulation.from_dict(data)
        print(f"✅ Состояние популяции загружено из {filepath}")


# # core/instinct_evolution.py
# """
# Генетический алгоритм для эволюции инстинктов.
# Гибридный подход: GA + GAN для оптимального развития инстинктивного поведения.
# """
#
# import numpy as np
# from typing import List, Dict, Any, Optional, Tuple
# from dataclasses import dataclass
# import random
# import json
#
#
# @dataclass
# class InstinctIndividual:
#     """Отдельный инстинкт в популяции."""
#     pattern: np.ndarray  # 256-мерный паттерн
#     fitness: float = 0.0  # Оценка качества
#     generation: int = 0  # Поколение создания
#     usage_count: int = 0  # Сколько раз использовался
#     success_rate: float = 0.0  # Процент успешных применений
#
#     def to_dict(self) -> Dict[str, Any]:
#         """Преобразует в словарь для сохранения."""
#         return {
#             'pattern': self.pattern.tolist(),
#             'fitness': self.fitness,
#             'generation': self.generation,
#             'usage_count': self.usage_count,
#             'success_rate': self.success_rate
#         }
#
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> 'InstinctIndividual':
#         """Восстанавливает из словаря."""
#         return cls(
#             pattern=np.array(data['pattern']),
#             fitness=data['fitness'],
#             generation=data['generation'],
#             usage_count=data['usage_count'],
#             success_rate=data['success_rate']
#         )
#
#
# class InstinctPopulation:
#     """
#     Популяция инстинктов с генетическим алгоритмом.
#     Поддерживает эволюцию, скрещивание, мутации и отбор.
#     """
#
#     def __init__(self,
#                  population_size: int = 20,
#                  pattern_dim: int = 256,
#                  mutation_rate: float = 0.1,
#                  crossover_rate: float = 0.8,
#                  elite_ratio: float = 0.2):
#         """
#         Args:
#             population_size: Размер популяции
#             pattern_dim: Размерность паттерна инстинкта
#             mutation_rate: Вероятность мутации
#             crossover_rate: Вероятность скрещивания
#             elite_ratio: Доля элиты (лучших), сохраняемых без изменений
#         """
#         self.population_size = population_size
#         self.pattern_dim = pattern_dim
#         self.mutation_rate = mutation_rate
#         self.crossover_rate = crossover_rate
#         self.elite_count = max(1, int(population_size * elite_ratio))
#
#         self.individuals: List[InstinctIndividual] = []
#         self.generation = 0
#         self.best_fitness_history: List[float] = []
#         self.avg_fitness_history: List[float] = []
#
#         # Инициализация случайной популяции
#         self._initialize_population()
#
#     def _initialize_population(self):
#         """Создает начальную случайную популяцию."""
#         for _ in range(self.population_size):
#             pattern = self._generate_random_pattern()
#             self.individuals.append(InstinctIndividual(pattern=pattern))
#
#         print(f"✅ Инициализирована популяция инстинктов: {self.population_size} особей")
#
#     def _generate_random_pattern(self) -> np.ndarray:
#         """Генерирует случайный паттерн с нормализацией."""
#         pattern = np.random.randn(self.pattern_dim)
#         # Нормализуем к единичной длине
#         norm = np.linalg.norm(pattern) + 1e-8
#         return pattern / norm
#
#     def evaluate_population(self, evaluator) -> None:
#         """
#         Оценивает всю популяцию.
#
#         Args:
#             evaluator: Объект с методом evaluate_pattern(pattern) -> float
#         """
#         for individual in self.individuals:
#             individual.fitness = evaluator.evaluate_pattern(individual.pattern)
#
#         # Обновляем историю
#         fitnesses = [ind.fitness for ind in self.individuals]
#         self.best_fitness_history.append(max(fitnesses))
#         self.avg_fitness_history.append(np.mean(fitnesses))
#
#     def select_parents(self, n_pairs: int) -> List[Tuple[InstinctIndividual, InstinctIndividual]]:
#         """
#         Выбирает пары родителей для скрещивания.
#         Использует турнирный отбор.
#
#         Args:
#             n_pairs: Количество пар родителей
#
#         Returns:
#             Список пар (parent1, parent2)
#         """
#         pairs = []
#         for _ in range(n_pairs):
#             # Турнирный отбор: выбираем 3 случайных, берем лучшего
#             def tournament_select():
#                 tournament = random.sample(self.individuals, min(3, len(self.individuals)))
#                 return max(tournament, key=lambda ind: ind.fitness)
#
#             parent1 = tournament_select()
#             parent2 = tournament_select()
#
#             # Избегаем скрещивания с самим собой
#             attempts = 0
#             while parent1 == parent2 and attempts < 10:
#                 parent2 = tournament_select()
#                 attempts += 1
#
#             pairs.append((parent1, parent2))
#
#         return pairs
#
#     def crossover(self, parent1: InstinctIndividual, parent2: InstinctIndividual) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Одноточечный кроссовер для паттернов инстинктов.
#
#         Args:
#             parent1: Первый родитель
#             parent2: Второй родитель
#
#         Returns:
#             Два паттерна-потомка
#         """
#         p1 = parent1.pattern
#         p2 = parent2.pattern
#
#         # Случайная точка разрыва
#         point = np.random.randint(1, self.pattern_dim - 1)
#
#         # Обмен частями
#         child1 = np.concatenate([p1[:point], p2[point:]])
#         child2 = np.concatenate([p2[:point], p1[point:]])
#
#         return child1, child2
#
#     def mutate(self, pattern: np.ndarray) -> np.ndarray:
#         """
#         Мутирует паттерн с адаптивной силой.
#
#         Args:
#             pattern: Исходный паттерн
#
#         Returns:
#             Мутировавший паттерн
#         """
#         mutated = pattern.copy()
#
#         # Адаптивная мутация: сила зависит от поколения
#         mutation_strength = 0.05 * (1.0 / (1.0 + self.generation * 0.01))
#
#         for i in range(len(mutated)):
#             if random.random() < self.mutation_rate:
#                 # Добавляем шум
#                 noise = np.random.randn() * mutation_strength
#                 mutated[i] += noise
#
#         # Нормализуем
#         norm = np.linalg.norm(mutated) + 1e-8
#         return mutated / norm
#
#     def evolve_one_generation(self, gan_generated_patterns: Optional[List[np.ndarray]] = None):
#         """
#         Выполняет одну эволюционную итерацию.
#
#         Args:
#             gan_generated_patterns: Паттерны от GAN для добавления в популяцию
#         """
#         self.generation += 1
#
#         # 1. Сохраняем элиту
#         sorted_inds = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
#         elite = sorted_inds[:self.elite_count]
#
#         # 2. Добавляем паттерны от GAN (если есть)
#         new_individuals = elite.copy()
#
#         if gan_generated_patterns:
#             for pattern in gan_generated_patterns:
#                 new_individuals.append(InstinctIndividual(pattern=pattern))
#
#         # 3. Скрещивание для заполнения популяции
#         n_needed = self.population_size - len(new_individuals)
#         n_pairs = n_needed // 2 + (n_needed % 2)
#
#         if n_pairs > 0 and len(self.individuals) > 1:
#             parent_pairs = self.select_parents(n_pairs)
#
#             for parent1, parent2 in parent_pairs:
#                 # Скрещивание
#                 if random.random() < self.crossover_rate:
#                     child1_pattern, child2_pattern = self.crossover(parent1, parent2)
#                 else:
#                     # Если не скрещиваемся, берем копии родителей
#                     child1_pattern = parent1.pattern.copy()
#                     child2_pattern = parent2.pattern.copy()
#
#                 # Мутация
#                 child1_pattern = self.mutate(child1_pattern)
#                 child2_pattern = self.mutate(child2_pattern)
#
#                 # Создаем индивидов
#                 child1 = InstinctIndividual(
#                     pattern=child1_pattern,
#                     generation=self.generation,
#                     fitness=0.0  # Будет оценен позже
#                 )
#                 child2 = InstinctIndividual(
#                     pattern=child2_pattern,
#                     generation=self.generation,
#                     fitness=0.0
#                 )
#
#                 new_individuals.append(child1)
#                 if len(new_individuals) < self.population_size:
#                     new_individuals.append(child2)
#
#         # 4. Обновляем популяцию
#         self.individuals = new_individuals[:self.population_size]
#
#         # 5. Обновляем статистику использования
#         for ind in self.individuals:
#             ind.generation = self.generation
#
#     def get_best(self, n: int = 5) -> List[InstinctIndividual]:
#         """Возвращает n лучших индивидов."""
#         sorted_inds = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
#         return sorted_inds[:n]
#
#     def get_best_patterns(self, n: int = 5) -> List[np.ndarray]:
#         """Возвращает паттерны n лучших индивидов."""
#         return [ind.pattern for ind in self.get_best(n)]
#
#     def get_population_stats(self) -> Dict[str, Any]:
#         """Возвращает статистику популяции."""
#         fitnesses = [ind.fitness for ind in self.individuals]
#
#         return {
#             'generation': self.generation,
#             'population_size': len(self.individuals),
#             'best_fitness': max(fitnesses) if fitnesses else 0,
#             'avg_fitness': np.mean(fitnesses) if fitnesses else 0,
#             'std_fitness': np.std(fitnesses) if fitnesses else 0,
#             'min_fitness': min(fitnesses) if fitnesses else 0,
#             'best_fitness_history': self.best_fitness_history[-20:],
#             'avg_fitness_history': self.avg_fitness_history[-20:],
#             'unique_patterns': len(set(tuple(ind.pattern) for ind in self.individuals))
#         }
#
#     def to_dict(self) -> Dict[str, Any]:
#         """Сериализует популяцию для сохранения."""
#         return {
#             'generation': self.generation,
#             'population_size': self.population_size,
#             'pattern_dim': self.pattern_dim,
#             'individuals': [ind.to_dict() for ind in self.individuals],
#             'best_fitness_history': self.best_fitness_history,
#             'avg_fitness_history': self.avg_fitness_history
#         }
#
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> 'InstinctPopulation':
#         """Восстанавливает популяцию из словаря."""
#         population = cls(
#             population_size=data['population_size'],
#             pattern_dim=data['pattern_dim']
#         )
#         population.generation = data['generation']
#         population.individuals = [InstinctIndividual.from_dict(ind) for ind in data['individuals']]
#         population.best_fitness_history = data['best_fitness_history']
#         population.avg_fitness_history = data['avg_fitness_history']
#         return population
#
#
# class InstinctEvaluator:
#     """
#     Оценщик качества инстинктов.
#     Использует дискриминатор GAN и реальный опыт.
#     """
#
#     def __init__(self, gan_discriminator, experience_buffer):
#         self.discriminator = gan_discriminator
#         self.experience_buffer = experience_buffer
#
#     def evaluate_pattern(self, pattern: np.ndarray) -> float:
#         """
#         Оценивает качество паттерна инстинкта.
#
#         Компоненты оценки:
#         1. Оценка дискриминатора GAN (0-1)
#         2. Новизна (штраф за повторение)
#         3. Сложность (предпочтение более сложным паттернам)
#         """
#         # 1. Оценка дискриминатора
#         discriminator_score = self._evaluate_by_discriminator(pattern)
#
#         # 2. Оценка новизны
#         novelty_score = self._evaluate_novelty(pattern)
#
#         # 3. Оценка сложности
#         complexity_score = self._evaluate_complexity(pattern)
#
#         # Комбинированная оценка
#         final_score = (
#                 0.6 * discriminator_score +
#                 0.3 * novelty_score +
#                 0.1 * complexity_score
#         )
#
#         return float(np.clip(final_score, 0.0, 1.0))
#
#     def _evaluate_by_discriminator(self, pattern: np.ndarray) -> float:
#         """Оценивает паттерн дискриминатором GAN."""
#         try:
#             return self.discriminator.evaluate_pattern(pattern)
#         except Exception:
#             return 0.5  # Нейтральная оценка при ошибке
#
#     def _evaluate_novelty(self, pattern: np.ndarray) -> float:
#         """
#         Оценивает новизну паттерна.
#         Штрафует за слишком похожие на существующие.
#         """
#         # Проверяем сходство с паттернами в буфере
#         if not self.experience_buffer:
#             return 1.0
#
#         # Берем случайные паттерны из буфера
#         buffer_size = min(len(self.experience_buffer), 50)
#         sample = random.sample(self.experience_buffer, buffer_size)
#
#         similarities = []
#         for exp in sample:
#             # Получаем паттерн из опыта
#             if isinstance(exp, tuple) and len(exp) >= 1:
#                 # Простое косинусное сходство
#                 sim = np.dot(pattern, exp[0]) / (np.linalg.norm(pattern) * np.linalg.norm(exp[0]) + 1e-8)
#                 similarities.append(sim)
#
#         if similarities:
#             max_similarity = max(similarities)
#             # Чем меньше сходство, тем выше новизна
#             return 1.0 - max_similarity
#         else:
#             return 1.0
#
#     def _evaluate_complexity(self, pattern: np.ndarray) -> float:
#         """
#         Оценивает сложность паттерна.
#         Предпочитает более сложные паттерны (больше разнообразия).
#         """
#         # Используем энтропию как меру сложности
#         normalized = (pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern) + 1e-8)
#
#         # Количество уникальных значений
#         unique_ratio = len(np.unique(normalized)) / len(normalized)
#
#         # Сложность по дисперсии
#         variance = np.var(pattern)
#
#         # Комбинируем
#         complexity = 0.5 * unique_ratio + 0.5 * min(variance * 10, 1.0)
#
#         return complexity
#
#
# class InstinctEvolutionEngine:
#     """
#     Движок эволюции инстинктов с интеграцией GAN.
#     """
#
#     def __init__(self,
#                  population_size: int = 20,
#                  pattern_dim: int = 256,
#                  gan_generator=None,
#                  gan_discriminator=None,
#                  experience_buffer=None):
#
#         self.population = InstinctPopulation(
#             population_size=population_size,
#             pattern_dim=pattern_dim
#         )
#
#         self.evaluator = InstinctEvaluator(
#             gan_discriminator=gan_discriminator,
#             experience_buffer=experience_buffer
#         )
#
#         self.gan_generator = gan_generator
#         self.generation_patterns = []  # Для анализа эволюции
#
#     def evolve(self, n_generations: int = 10, gan_patterns_per_gen: int = 5) -> List[float]:
#         """
#         Запускает эволюцию инстинктов на несколько поколений.
#
#         Args:
#             n_generations: Количество поколений
#             gan_patterns_per_gen: Сколько паттернов генерировать GAN за поколение
#
#         Returns:
#             История лучших фитнесов
#         """
#         best_history = []
#
#         for gen in range(n_generations):
#             print(f"\n🧬 Поколение инстинктов {gen + 1}/{n_generations}")
#
#             # 1. Оцениваем текущую популяцию
#             self.population.evaluate_population(self.evaluator)
#             stats = self.population.get_population_stats()
#
#             print(f"  Лучший фитнес: {stats['best_fitness']:.4f}")
#             print(f"  Средний фитнес: {stats['avg_fitness']:.4f}")
#             print(f"  Разнообразие: {stats['unique_patterns']}/{self.population.population_size}")
#
#             # 2. Генерируем паттерны от GAN
#             gan_patterns = []
#             if self.gan_generator:
#                 try:
#                     gan_patterns = self.gan_generator.generate_batch(gan_patterns_per_gen)
#                     print(f"  Сгенерировано {len(gan_patterns)} паттернов от GAN")
#                 except Exception as e:
#                     print(f"  ⚠️ Ошибка генерации GAN: {e}")
#
#             # 3. Эволюционируем
#             self.population.evolve_one_generation(gan_patterns)
#
#             # 4. Сохраняем историю
#             best_history.append(stats['best_fitness'])
#
#             # 5. Сохраняем лучшие паттерны
#             if gen % 5 == 0:
#                 best_inds = self.population.get_best(3)
#                 self.generation_patterns.append({
#                     'generation': gen,
#                     'best_patterns': [ind.pattern for ind in best_inds],
#                     'best_fitness': [ind.fitness for ind in best_inds]
#                 })
#
#         print(f"\n✅ Эволюция инстинктов завершена за {n_generations} поколений")
#         print(f"  Лучший фитнес: {max(best_history):.4f}")
#         print(f"  Средний фитнес: {np.mean(best_history):.4f}")
#
#         return best_history
#
#     def get_best_instincts(self, n: int = 3) -> List[np.ndarray]:
#         """Возвращает лучшие паттерны инстинктов."""
#         return self.population.get_best_patterns(n)
#
#     def get_population_stats(self) -> Dict[str, Any]:
#         """Возвращает статистику популяции."""
#         return self.population.get_population_stats()
#
#     def save_state(self, filepath: str):
#         """Сохраняет состояние популяции."""
#         data = self.population.to_dict()
#         with open(filepath, 'w') as f:
#             json.dump(data, f, indent=2)
#         print(f"✅ Состояние популяции сохранено в {filepath}")
#
#     def load_state(self, filepath: str):
#         """Загружает состояние популяции."""
#         with open(filepath, 'r') as f:
#             data = json.load(f)
#         self.population = InstinctPopulation.from_dict(data)
#         print(f"✅ Состояние популяции загружено из {filepath}")