# core/instinct_evolution.py
"""
[ru] Генетический алгоритм для эволюции инстинктов.  Гибридный подход: GA + GAN для оптимального развития инстинктивного поведения.
[en] Genetic algorithm for instinct evolution. Hybrid approach: GA + GAN for optimal development of instinctive behavior.
"""

import numpy as np
import random
import json
import torch
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass


@dataclass
class InstinctIndividual:
    """
    [ru] Отдельный инстинкт в популяции.
    [en] A single instinct in the population.
    """
    pattern: np.ndarray  # [ru] 256-мерный паттерн  [en] 256-dimensional pattern
    fitness: float = 0.0  # [ru] Оценка качества  [en] Quality score
    generation: int = 0  # [ru] Поколение создания  [en] Generation of creation
    usage_count: int = 0  # [ru] Сколько раз использовался  [en] How many times it was used
    success_rate: float = 0.0  # [ru] Процент успешных применений  [en] Percentage of successful applications
    id: int = None  # [ru] Уникальный идентификатор  [en] Unique identifier

    def __post_init__(self):
        if self.id is None:
            self.id = random.randint(0, 10 ** 9)

    def to_dict(self) -> Dict[str, Any]:
        """
        [ru] Преобразует в словарь для сохранения.
        [en] Converts to a dictionary for saving.
        """
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
        """
        [ru] Восстанавливает из словаря.
        [en] Restores from a dictionary.
        """
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
    [ru] Оценщик качества инстинктов. Использует дискриминатор InstinctGAN и реальный опыт.
    [en] Instinct quality evaluator. Uses InstinctGAN discriminator and real experience.
    """

    def __init__(self, gan_discriminator, experience_buffer):
        self.discriminator = gan_discriminator
        self.experience_buffer = experience_buffer

    def evaluate_pattern(self, pattern: np.ndarray) -> float:
        """
        [ru] Оценивает качество паттерна инстинкта.
        [en] Evaluates the quality of an instinct pattern.
        """
        # [ru] 1. Оценка дискриминатора
        # [en] 1. Discriminator score
        discriminator_score = self._evaluate_by_discriminator(pattern)

        # [ru] 2. Оценка новизны
        # [en] 2. Novelty score
        novelty_score = self._evaluate_novelty(pattern)

        # [ru] 3. Оценка сложности
        # [en] 3. Complexity score
        complexity_score = self._evaluate_complexity(pattern)

        # [ru] Комбинированная оценка
        # [en] Combined score
        final_score = (
                0.6 * discriminator_score +
                0.3 * novelty_score +
                0.1 * complexity_score
        )

        return float(np.clip(final_score, 0.0, 1.0))

    def _evaluate_by_discriminator(self, pattern: np.ndarray) -> float:
        """
        [ru] Оценивает паттерн дискриминатором GAN.
        [en] Evaluates the pattern using the GAN discriminator.
        """
        try:
            if hasattr(self.discriminator, 'evaluate_pattern'):
                return self.discriminator.evaluate_pattern(pattern)
            else:
                # [ru] Убеждаемся, что паттерн имеет правильную размерность
                # [en] Ensure the pattern has the correct dimensionality
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
        [ru] Оценивает новизну паттерна. Штрафует за слишком похожие на существующие.
        [en] Evaluates pattern novelty. Penalizes for being too similar to existing ones.
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

                    # [ru] Приводим к массивам
                    # [en] Convert to arrays
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
        [ru] Оценивает сложность паттерна.
        [en] Evaluates pattern complexity.
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

class InstinctPopulation:
    """
    [ru] Популяция инстинктов с генетическим алгоритмом. Поддерживает эволюцию, скрещивание, мутации и отбор.
    [en] Instinct population with a genetic algorithm. Supports evolution, crossover, mutation, and selection.
    """

    def __init__(self,
                 population_size: int = 20,
                 pattern_dim: int = 256,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elite_ratio: float = 0.2):
        """
        Args:
            [ru] population_size: Размер популяции
            [ru] pattern_dim: Размерность паттерна инстинкта
            [ru] mutation_rate: Вероятность мутации
            [ru] crossover_rate: Вероятность скрещивания
            [ru] elite_ratio: Доля элиты (лучших), сохраняемых без изменений
        Args:
            [en] population_size: Population size
            [en] pattern_dim: Instinct pattern dimensionality
            [en] mutation_rate: Mutation probability
            [en] crossover_rate: Crossover probability
            [en] elite_ratio: Proportion of elite (best) preserved unchanged
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

        # [ru] Инициализация случайной популяции
        # [en] Initialize random population
        self._initialize_population()

    def _initialize_population(self):
        """
        [ru] Создает начальную случайную популяцию.
        [en] Creates the initial random population.
        """
        for _ in range(self.population_size):
            pattern = self._generate_random_pattern()
            self.individuals.append(InstinctIndividual(pattern=pattern))

        print(f"[ru] Инициализирована популяция инстинктов: {self.population_size} особей")
        print(f"[en] Instinct population initialized: {self.population_size} individuals")

    def _generate_random_pattern(self) -> np.ndarray:
        """
        [ru] Генерирует случайный паттерн с нормализацией.
        [en] Generates a random pattern with normalization.
        """
        pattern = np.random.randn(self.pattern_dim)
        # [ru] Нормализуем к единичной длине
        # [en] Normalize to unit length
        norm = np.linalg.norm(pattern) + 1e-8
        return pattern / norm

    def evaluate_population(self, evaluator) -> None:
        """
        [ru] Оценивает всю популяцию.
        [en] Evaluates the entire population.

        Args:
            [ru] evaluator: Объект с методом evaluate_pattern(pattern) -> float
            [en] evaluator: Object with method evaluate_pattern(pattern) -> float
        """
        for individual in self.individuals:
            individual.fitness = evaluator.evaluate_pattern(individual.pattern)

        # [ru] Обновляем историю
        # [en] Update history
        fitnesses = [ind.fitness for ind in self.individuals]
        self.best_fitness_history.append(max(fitnesses))
        self.avg_fitness_history.append(np.mean(fitnesses))

    def select_parents(self, n_pairs: int) -> List[Tuple[InstinctIndividual, InstinctIndividual]]:
        """
        [ru] Выбирает пары родителей для скрещивания. Использует турнирный отбор.
        Args:
            [ru] n_pairs: Количество пар родителей
        Returns:
            [ru] Список пар (parent1, parent2)

        [en] Selects parent pairs for crossover. Uses tournament selection.
        Args:
            [en] n_pairs: Number of parent pairs
        Returns:
            [en] List of pairs (parent1, parent2)
        """
        pairs = []
        for _ in range(n_pairs):
            # [ru] Турнирный отбор: выбираем 3 случайных, берем лучшего
            # [en] Tournament selection: pick 3 random, take the best
            def tournament_select():
                tournament = random.sample(self.individuals, min(3, len(self.individuals)))
                return max(tournament, key=lambda ind: ind.fitness)

            parent1 = tournament_select()
            parent2 = tournament_select()

            # [ru] Избегаем скрещивания с самим собой
            # [en] Avoid self-crossover
            attempts = 0
            while (parent1.id == parent2.id or
                   np.array_equal(parent1.pattern, parent2.pattern)) and attempts < 10:
                parent2 = tournament_select()
                attempts += 1

            pairs.append((parent1, parent2))

        return pairs

    def crossover(self, parent1: InstinctIndividual, parent2: InstinctIndividual) -> Tuple[np.ndarray, np.ndarray]:
        """
        [ru] Одноточечный кроссовер для паттернов инстинктов.
        Args:
            [ru] parent1: Первый родитель
            [ru] parent2: Второй родитель
        Returns:
            [ru] Два паттерна-потомка

        [en] Single-point crossover for instinct patterns.
        Args:
            [en] parent1: First parent
            [en] parent2: Second parent
        Returns:
            [en] Two child patterns
        """
        p1 = parent1.pattern
        p2 = parent2.pattern

        # [ru] Случайная точка разрыва
        # [en] Random breakpoint
        point = np.random.randint(1, self.pattern_dim - 1)

        # [ru] Обмен частями
        # [en] Exchange parts
        child1 = np.concatenate([p1[:point], p2[point:]])
        child2 = np.concatenate([p2[:point], p1[point:]])

        return child1, child2

    def mutate(self, pattern: np.ndarray) -> np.ndarray:
        """
        [ru] Мутирует паттерн с адаптивной силой.
        Args:
            [ru] pattern: Исходный паттерн
        Returns:
            [ru] Мутировавший паттерн

        [en] Mutates pattern with adaptive strength.
        Args:
            [en] pattern: Original pattern
        Returns:
            [en] Mutated pattern
        """
        mutated = pattern.copy()

        # [ru] Адаптивная мутация: сила зависит от поколения
        # [en] Adaptive mutation: strength depends on generation
        mutation_strength = 0.05 * (1.0 / (1.0 + self.generation * 0.01))

        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                # [ru] Добавляем шум
                # [en] Add noise
                noise = np.random.randn() * mutation_strength
                mutated[i] += noise

        # [ru] Нормализуем
        # [en] Normalize
        norm = np.linalg.norm(mutated) + 1e-8
        return mutated / norm

    def evolve_one_generation(self, gan_generated_patterns: Optional[List[np.ndarray]] = None):
        """
        [ru] Выполняет одну эволюционную итерацию.
        Args:
            [ru] gan_generated_patterns: Паттерны от GAN для добавления в популяцию

        [en] Performs one evolutionary iteration.
        Args:
            [en] gan_generated_patterns: Patterns from GAN to add to the population
        """
        self.generation += 1

        # [ru] 1. Сохраняем элиту
        # [en] 1. Preserve elite
        sorted_inds = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
        elite = sorted_inds[:self.elite_count]

        # [ru] 2. Добавляем паттерны от GAN (если есть)
        # [en] 2. Add patterns from GAN (if any)
        new_individuals = elite.copy()

        if gan_generated_patterns:
            for pattern in gan_generated_patterns:
                new_individuals.append(InstinctIndividual(pattern=pattern))

        # [ru] 3. Скрещивание для заполнения популяции
        # [en] 3. Crossover to fill population
        n_needed = self.population_size - len(new_individuals)
        n_pairs = n_needed // 2 + (n_needed % 2)

        if n_pairs > 0 and len(self.individuals) > 1:
            parent_pairs = self.select_parents(n_pairs)

            for parent1, parent2 in parent_pairs:
                # [ru] Скрещивание
                # [en] Crossover
                if random.random() < self.crossover_rate:
                    child1_pattern, child2_pattern = self.crossover(parent1, parent2)
                else:
                    # [ru] Если не скрещиваемся, берем копии родителей
                    # [en] If not crossing over, take copies of parents
                    child1_pattern = parent1.pattern.copy()
                    child2_pattern = parent2.pattern.copy()

                # [ru] Мутация
                # [en] Mutation
                child1_pattern = self.mutate(child1_pattern)
                child2_pattern = self.mutate(child2_pattern)

                # [ru] Создаем индивидов
                # [en] Create individuals
                child1 = InstinctIndividual(
                    pattern=child1_pattern,
                    generation=self.generation,
                    fitness=0.0  # [ru] Будет оценен позже  [en] Will be evaluated later
                )
                child2 = InstinctIndividual(
                    pattern=child2_pattern,
                    generation=self.generation,
                    fitness=0.0
                )

                new_individuals.append(child1)
                if len(new_individuals) < self.population_size:
                    new_individuals.append(child2)

        # [ru] 4. Обновляем популяцию
        # [en] 4. Update population
        self.individuals = new_individuals[:self.population_size]

        # [ru] 5. Обновляем статистику использования
        # [en] 5. Update usage statistics
        for ind in self.individuals:
            ind.generation = self.generation

    def get_best(self, n: int = 5) -> List[InstinctIndividual]:
        """
        [ru] Возвращает n лучших индивидов.
        [en] Returns the n best individuals.
        """
        sorted_inds = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
        return sorted_inds[:n]

    def get_best_patterns(self, n: int = 5) -> List[np.ndarray]:
        """
        [ru] Возвращает паттерны n лучших индивидов.
        [en] Returns patterns of the n best individuals.
        """
        return [ind.pattern for ind in self.get_best(n)]

    def get_population_stats(self) -> Dict[str, Any]:
        """
        [ru] Возвращает статистику популяции.
        [en] Returns population statistics.
        """
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
        """
        [ru] Сериализует популяцию для сохранения.
        [en] Serializes population for saving.
        """
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
        """
        [ru] Восстанавливает популяцию из словаря.
        [en] Restores population from a dictionary.
        """
        population = cls(
            population_size=data['population_size'],
            pattern_dim=data['pattern_dim']
        )
        population.generation = data['generation']
        population.individuals = [InstinctIndividual.from_dict(ind) for ind in data['individuals']]
        population.best_fitness_history = data['best_fitness_history']
        population.avg_fitness_history = data['avg_fitness_history']
        return population

class InstinctEvolutionEngine:
    """
    [ru] Движок эволюции инстинктов с интеграцией GAN.
    [en] Instinct evolution engine with GAN integration.
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
        """
        [ru] Запускает эволюцию инстинктов.
        [en] Runs instinct evolution.
        """
        best_history = []

        for gen in range(n_generations):
            print(f"\n[ru] Поколение инстинктов {gen + 1}/{n_generations}")
            print(f"\n[en] Instinct generation {gen + 1}/{n_generations}")

            # [ru] 1. Оцениваем текущую популяцию
            # [en] 1. Evaluate current population
            self.population.evaluate_population(self.evaluator)
            stats = self.population.get_population_stats()

            print(f"  [ru] Лучший фитнес: {stats['best_fitness']:.4f}")
            print(f"  [ru] Средний фитнес: {stats['avg_fitness']:.4f}")
            print(f"  [ru] Разнообразие: {stats['unique_patterns']}/{self.population.population_size}")

            print(f"  [en] Best fitness: {stats['best_fitness']:.4f}")
            print(f"  [en] Average fitness: {stats['avg_fitness']:.4f}")
            print(f"  [en] Diversity: {stats['unique_patterns']}/{self.population.population_size}")

            # [ru] 2. Генерируем паттерны от GAN
            # [en] 2. Generate patterns from GAN
            gan_patterns = []
            if self.gan_generator:
                try:
                    # [ru] Пробуем generate_batch
                    # [en] Try generate_batch
                    if hasattr(self.gan_generator, 'generate_batch'):
                        gan_patterns = self.gan_generator.generate_batch(gan_patterns_per_gen)
                    else:
                        # [ru] Пробуем generate_pattern
                        # [en] Try generate_pattern
                        for _ in range(gan_patterns_per_gen):
                            if hasattr(self.gan_generator, 'generate_pattern'):
                                pattern = self.gan_generator.generate_pattern()
                                gan_patterns.append(pattern)
                    print(f"  [ru] Сгенерировано {len(gan_patterns)} паттернов от GAN")
                    print(f"  [en] Generated {len(gan_patterns)} patterns from GAN")
                except Exception as e:
                    print(f"  [ru] Ошибка генерации GAN: {e}")
                    print(f"  [en] GAN generation error: {e}")

            # [ru] 3. Эволюционируем
            # [en] 3. Evolve
            self.population.evolve_one_generation(gan_patterns)

            # [ru] 4. Сохраняем историю
            # [en] 4. Save history
            best_history.append(stats['best_fitness'])

            # [ru] 5. Сохраняем лучшие паттерны
            # [en] 5. Save best patterns
            if gen % 5 == 0:
                best_inds = self.population.get_best(3)
                self.generation_patterns.append({
                    'generation': gen,
                    'best_patterns': [ind.pattern for ind in best_inds],
                    'best_fitness': [ind.fitness for ind in best_inds]
                })

        print(f"\n[ru] Эволюция инстинктов завершена за {n_generations} поколений")
        print(f"  [ru] Лучший фитнес: {max(best_history):.4f}")
        print(f"  [ru] Средний фитнес: {np.mean(best_history):.4f}")

        print(f"\n[en] Instinct evolution completed in {n_generations} generations")
        print(f"  [en] Best fitness: {max(best_history):.4f}")
        print(f"  [en] Average fitness: {np.mean(best_history):.4f}")

        return best_history

    def get_best_instincts(self, n: int = 3) -> List[np.ndarray]:
        """
        [ru] Возвращает лучшие паттерны инстинктов.
        [en] Returns the best instinct patterns.
        """
        return self.population.get_best_patterns(n)

    def get_population_stats(self) -> Dict[str, Any]:
        """
        [ru] Возвращает статистику популяции.
        [en] Returns population statistics.
        """
        return self.population.get_population_stats()

    def save_state(self, filepath: str):
        """
        [ru] Сохраняет состояние популяции.
        [en] Saves population state.
        """
        data = self.population.to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[ru] Состояние популяции сохранено в {filepath}")
        print(f"[en] Population state saved to {filepath}")

    def load_state(self, filepath: str):
        """
        [ru] Загружает состояние популяции.
        [en] Loads population state.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.population = InstinctPopulation.from_dict(data)
        print(f"[ru] Состояние популяции загружено из {filepath}")
        print(f"[en] Population state loaded from {filepath}")


    def crossover(self, parent1: InstinctIndividual, parent2: InstinctIndividual) -> Tuple[np.ndarray, np.ndarray]:
        """
        [ru] Одноточечный кроссовер для паттернов инстинктов.
        Args:
            [ru] parent1: Первый родитель
            [ru] parent2: Второй родитель
        Returns:
            [ru] Два паттерна-потомка

        [en] Single-point crossover for instinct patterns.
        Args:
            [en] parent1: First parent
            [en] parent2: Second parent
        Returns:
            [en] Two child patterns
        """
        p1 = parent1.pattern
        p2 = parent2.pattern

        # [ru] Случайная точка разрыва
        # [en] Random breakpoint
        point = np.random.randint(1, self.pattern_dim - 1)

        # [ru] Обмен частями
        # [en] Exchange parts
        child1 = np.concatenate([p1[:point], p2[point:]])
        child2 = np.concatenate([p2[:point], p1[point:]])

        return child1, child2

    def mutate(self, pattern: np.ndarray) -> np.ndarray:
        """
        [ru] Мутирует паттерн с адаптивной силой.
        Args:
            [ru] pattern: Исходный паттерн
            [ru] Мутировавший паттерн

        [en] Mutates pattern with adaptive strength.
        Args:
            [en] pattern: Original pattern
        Returns:
            [en] Mutated pattern
        """
        mutated = pattern.copy()

        # [ru] Адаптивная мутация: сила зависит от поколения
        # [en] Adaptive mutation: strength depends on generation
        mutation_strength = 0.05 * (1.0 / (1.0 + self.generation * 0.01))

        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                # [ru] Добавляем шум
                # [en] Add noise
                noise = np.random.randn() * mutation_strength
                mutated[i] += noise

        # [ru] Нормализуем
        # [en] Normalize
        norm = np.linalg.norm(mutated) + 1e-8
        return mutated / norm

    def evolve_one_generation(self, gan_generated_patterns: Optional[List[np.ndarray]] = None):
        """
        [ru] Выполняет одну эволюционную итерацию.
        Args:
            [ru] gan_generated_patterns: Паттерны от GAN для добавления в популяцию

        [en] Performs one evolutionary iteration.
        Args:
            [en] gan_generated_patterns: Patterns from GAN to add to the population
        """
        self.generation += 1

        # [ru] 1. Сохраняем элиту
        # [en] 1. Preserve elite
        sorted_inds = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
        elite = sorted_inds[:self.elite_count]

        # [ru] 2. Добавляем паттерны от GAN (если есть)
        # [en] 2. Add patterns from GAN (if any)
        new_individuals = elite.copy()

        if gan_generated_patterns:
            for pattern in gan_generated_patterns:
                new_individuals.append(InstinctIndividual(pattern=pattern))

        # [ru] 3. Скрещивание для заполнения популяции
        # [en] 3. Crossover to fill population
        n_needed = self.population_size - len(new_individuals)
        n_pairs = n_needed // 2 + (n_needed % 2)

        if n_pairs > 0 and len(self.individuals) > 1:
            parent_pairs = self.select_parents(n_pairs)

            for parent1, parent2 in parent_pairs:
                # [ru] Скрещивание
                # [en] Crossover
                if random.random() < self.crossover_rate:
                    child1_pattern, child2_pattern = self.crossover(parent1, parent2)
                else:
                    # [ru] Если не скрещиваемся, берем копии родителей
                    # [en] If not crossing over, take copies of parents
                    child1_pattern = parent1.pattern.copy()
                    child2_pattern = parent2.pattern.copy()

                # [ru] Мутация
                # [en] Mutation
                child1_pattern = self.mutate(child1_pattern)
                child2_pattern = self.mutate(child2_pattern)

                # [ru] Создаем индивидов
                # [en] Create individuals
                child1 = InstinctIndividual(
                    pattern=child1_pattern,
                    generation=self.generation,
                    fitness=0.0  # [ru] Будет оценен позже  [en] Will be evaluated later
                )
                child2 = InstinctIndividual(
                    pattern=child2_pattern,
                    generation=self.generation,
                    fitness=0.0
                )

                new_individuals.append(child1)
                if len(new_individuals) < self.population_size:
                    new_individuals.append(child2)

        # [ru] 4. Обновляем популяцию
        # [en] 4. Update population
        self.individuals = new_individuals[:self.population_size]

        # [ru] 5. Обновляем статистику использования
        # [en] 5. Update usage statistics
        for ind in self.individuals:
            ind.generation = self.generation

    def get_best(self, n: int = 5) -> List[InstinctIndividual]:
        """
        [ru] Возвращает n лучших индивидов.
        [en] Returns the n best individuals.
        """
        sorted_inds = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
        return sorted_inds[:n]

    def get_best_patterns(self, n: int = 5) -> List[np.ndarray]:
        """
        [ru] Возвращает паттерны n лучших индивидов.
        [en] Returns patterns of the n best individuals.
        """
        return [ind.pattern for ind in self.get_best(n)]

    def get_population_stats(self) -> Dict[str, Any]:
        """
        [ru] Возвращает статистику популяции.
        [en] Returns population statistics.
        """
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
        """
        [ru] Сериализует популяцию для сохранения.
        [en] Serializes population for saving.
        """
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
        """
        [ru] Восстанавливает популяцию из словаря.
        [en] Restores population from a dictionary.
        """
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
    [ru] Оценщик качества инстинктов. Использует дискриминатор InstinctGAN и реальный опыт.
    [en] Instinct quality evaluator. Uses InstinctGAN discriminator and real experience.
    """

    def __init__(self, gan_discriminator, experience_buffer):
        self.discriminator = gan_discriminator
        self.experience_buffer = experience_buffer

    def evaluate_pattern(self, pattern: np.ndarray) -> float:
        """
        [ru] Оценивает качество паттерна инстинкта.
        [ru] Компоненты оценки:
        [ru] 1. Оценка дискриминатора GAN (0-1)
        [ru] 2. Новизна (штраф за повторение)
        [ru] 3. Сложность (предпочтение более сложным паттернам)

        [en] Evaluates the quality of an instinct pattern.
        [en] Evaluation components:
        [en] 1. GAN discriminator score (0-1)
        [en] 2. Novelty (penalty for repetition)
        [en] 3. Complexity (preference for more complex patterns)
        """
        # [ru] 1. Оценка дискриминатора
        # [en] 1. Discriminator score
        discriminator_score = self._evaluate_by_discriminator(pattern)

        # [ru] 2. Оценка новизны
        # [en] 2. Novelty score
        novelty_score = self._evaluate_novelty(pattern)

        # [ru] 3. Оценка сложности
        # [en] 3. Complexity score
        complexity_score = self._evaluate_complexity(pattern)

        # [ru] Комбинированная оценка
        # [en] Combined score
        final_score = (
                0.6 * discriminator_score +
                0.3 * novelty_score +
                0.1 * complexity_score
        )

        return float(np.clip(final_score, 0.0, 1.0))

    def _evaluate_by_discriminator(self, pattern: np.ndarray) -> float:
        """
        [ru] Оценивает паттерн дискриминатором GAN.
        [en] Evaluates the pattern using the GAN discriminator.
        """
        try:
            # [ru] Проверяем, есть ли метод evaluate_pattern
            # [en] Check if evaluate_pattern method exists
            if hasattr(self.discriminator, 'evaluate_pattern'):
                return self.discriminator.evaluate_pattern(pattern)
            else:
                # [ru] Если нет, используем прямой forward
                # [ru] Убеждаемся, что паттерн имеет правильную размерность
                # [en] If not, use direct forward
                # [en] Ensure the pattern has the correct dimensionality
                if len(pattern) != 256:
                    # [ru] Если размерность не 256, обрезаем или дополняем
                    # [en] If dimensionality is not 256, trim or pad
                    if len(pattern) > 256:
                        pattern = pattern[:256]
                    else:
                        # [ru] Дополняем нулями
                        # [en] Pad with zeros
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
            print(f" Ошибка оценки дискриминатором: {e}")
            return 0.5

    def _evaluate_complexity(self, pattern: np.ndarray) -> float:
        """
        [ru] Оценивает сложность паттерна. Предпочитает более сложные паттерны (больше разнообразия).
        [en] Evaluates pattern complexity. Prefers more complex patterns (more diversity).
        """
        # [ru] Используем энтропию как меру сложности
        # [en] Use entropy as a measure of complexity
        normalized = (pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern) + 1e-8)

        # [ru] Количество уникальных значений
        # [en] Number of unique values
        unique_ratio = len(np.unique(normalized)) / len(normalized)

        # [ru] Сложность по дисперсии
        # [en] Complexity by variance
        variance = np.var(pattern)

        # [ru] Комбинируем
        # [en] Combine
        complexity = 0.5 * unique_ratio + 0.5 * min(variance * 10, 1.0)

        return complexity


class InstinctEvolutionEngine:
    """
    [ru] Движок эволюции инстинктов с интеграцией GAN.
    [en] Instinct evolution engine with GAN integration.
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
        self.generation_patterns = []  # [ru] Для анализа эволюции  [en] For evolution analysis

    def evolve(self, n_generations: int = 10, gan_patterns_per_gen: int = 5) -> List[float]:
        """
        [ru] Запускает эволюцию инстинктов на несколько поколений.
        Args:
            [ru] n_generations: Количество поколений
            [ru] gan_patterns_per_gen: Сколько паттернов генерировать GAN за поколение
        Returns:
            [ru] История лучших фитнесов

        [en] Runs instinct evolution for several generations.
        Args:
            [en] n_generations: Number of generations
            [en] gan_patterns_per_gen: How many patterns GAN should generate per generation
        Returns:
            [en] History of best fitnesses
        """
        best_history = []

        for gen in range(n_generations):
            print(f"\n[ru] Поколение инстинктов {gen + 1}/{n_generations}")
            print(f"\n[en] Instinct generation {gen + 1}/{n_generations}")

            # [ru] 1. Оцениваем текущую популяцию
            # [en] 1. Evaluate current population
            self.population.evaluate_population(self.evaluator)
            stats = self.population.get_population_stats()

            print(f"  [ru] Лучший фитнес: {stats['best_fitness']:.4f}")
            print(f"  [ru] Средний фитнес: {stats['avg_fitness']:.4f}")
            print(f"  [ru] Разнообразие: {stats['unique_patterns']}/{self.population.population_size}")

            print(f"  [en] Best fitness: {stats['best_fitness']:.4f}")
            print(f"  [en] Average fitness: {stats['avg_fitness']:.4f}")
            print(f"  [en] Diversity: {stats['unique_patterns']}/{self.population.population_size}")

            # [ru] 2. Генерируем паттерны от GAN
            # [en] 2. Generate patterns from GAN
            gan_patterns = []
            if self.gan_generator:
                try:
                    # [ru] Проверяем, есть ли метод generate_batch
                    # [en] Check if generate_batch method exists
                    if hasattr(self.gan_generator, 'generate_batch'):
                        gan_patterns = self.gan_generator.generate_batch(gan_patterns_per_gen)
                    else:
                        # [ru] Если нет, генерируем по одному
                        # [en] If not, generate one by one
                        for _ in range(gan_patterns_per_gen):
                            pattern = self.gan_generator.generate_pattern()
                            gan_patterns.append(pattern)
                    print(f"  [ru] Сгенерировано {len(gan_patterns)} паттернов от GAN")
                    print(f"  [en] Generated {len(gan_patterns)} patterns from GAN")
                except Exception as e:
                    print(f"  [ru] Ошибка генерации GAN: {e}")
                    print(f"  [en] GAN generation error: {e}")

            # [ru] 3. Эволюционируем
            # [en] 3. Evolve
            self.population.evolve_one_generation(gan_patterns)

            # [ru] 4. Сохраняем историю
            # [en] 4. Save history
            best_history.append(stats['best_fitness'])

            # [ru] 5. Сохраняем лучшие паттерны
            # [en] 5. Save best patterns
            if gen % 5 == 0:
                best_inds = self.population.get_best(3)
                self.generation_patterns.append({
                    'generation': gen,
                    'best_patterns': [ind.pattern for ind in best_inds],
                    'best_fitness': [ind.fitness for ind in best_inds]
                })

        print(f"\n[ru] Эволюция инстинктов завершена за {n_generations} поколений")
        print(f"  [ru] Лучший фитнес: {max(best_history):.4f}")
        print(f"  [ru] Средний фитнес: {np.mean(best_history):.4f}")

        print(f"\n[en] Instinct evolution completed in {n_generations} generations")
        print(f"  [en] Best fitness: {max(best_history):.4f}")
        print(f"  [en] Average fitness: {np.mean(best_history):.4f}")

        return best_history

    def get_best_instincts(self, n: int = 3) -> List[np.ndarray]:
        """
        [ru] Возвращает лучшие паттерны инстинктов.
        [en] Returns the best instinct patterns.
        """
        return self.population.get_best_patterns(n)

    def get_population_stats(self) -> Dict[str, Any]:
        """
        [ru] Возвращает статистику популяции.
        [en] Returns population statistics.
        """
        return self.population.get_population_stats()

    def save_state(self, filepath: str):
        """
        [ru] Сохраняет состояние популяции.
        [en] Saves population state.
        """
        data = self.population.to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[ru] Состояние популяции сохранено в {filepath}")
        print(f"[en] Population state saved to {filepath}")

    def load_state(self, filepath: str):
        """
        [ru] Загружает состояние популяции.
        [en] Loads population state.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.population = InstinctPopulation.from_dict(data)
        print(f"[ru] Состояние популяции загружено из {filepath}")
        print(f"[en] Population state loaded from {filepath}")



# # core/instinct_evolution.py
# """
# Генетический алгоритм для эволюции инстинктов.
# Гибридный подход: GA + GAN для оптимального развития инстинктивного поведения.
# """
#
# import numpy as np
# import random
# import json
# import torch
# from typing import List, Dict, Optional, Tuple, Any, Union
# from dataclasses import dataclass
#
#
# @dataclass
# class InstinctIndividual:
#     """
#     Отдельный инстинкт в популяции.
#     """
#     pattern: np.ndarray  # 256-мерный паттерн
#     fitness: float = 0.0  # Оценка качества
#     generation: int = 0  # Поколение создания
#     usage_count: int = 0  # Сколько раз использовался
#     success_rate: float = 0.0  # Процент успешных применений
#     id: int = None  # Уникальный идентификатор
#
#     def __post_init__(self):
#         if self.id is None:
#             self.id = random.randint(0, 10 ** 9)
#
#     def to_dict(self) -> Dict[str, Any]:
#         """
#         Преобразует в словарь для сохранения.
#         """
#         return {
#             'pattern': self.pattern.tolist(),
#             'fitness': self.fitness,
#             'generation': self.generation,
#             'usage_count': self.usage_count,
#             'success_rate': self.success_rate,
#             'id': self.id
#         }
#
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> 'InstinctIndividual':
#         """
#         Восстанавливает из словаря.
#         """
#         return cls(
#             pattern=np.array(data['pattern']),
#             fitness=data['fitness'],
#             generation=data['generation'],
#             usage_count=data['usage_count'],
#             success_rate=data['success_rate'],
#             id=data.get('id', random.randint(0, 10 ** 9))
#         )
#
#
# class InstinctEvaluator:
#     """
#     Оценщик качества инстинктов. Использует дискриминатор InstinctGAN и реальный опыт.
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
#         """
#         Оценивает паттерн дискриминатором GAN.
#         """
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
#         Оценивает новизну паттерна. Штрафует за слишком похожие на существующие.
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
#         """
#         Создает начальную случайную популяцию.
#         """
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
#             while (parent1.id == parent2.id or
#                    np.array_equal(parent1.pattern, parent2.pattern)) and attempts < 10:
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
#         """
#         Возвращает n лучших индивидов.
#         """
#         sorted_inds = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
#         return sorted_inds[:n]
#
#     def get_best_patterns(self, n: int = 5) -> List[np.ndarray]:
#         """
#         Возвращает паттерны n лучших индивидов.
#         """
#         return [ind.pattern for ind in self.get_best(n)]
#
#     def get_population_stats(self) -> Dict[str, Any]:
#         """
#         Возвращает статистику популяции.
#         """
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
#         """
#         Сериализует популяцию для сохранения.
#         """
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
#         """
#         Восстанавливает популяцию из словаря.
#         """
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
#         self.generation_patterns = []
#
#     def evolve(self, n_generations: int = 10, gan_patterns_per_gen: int = 5) -> List[float]:
#         """
#         Запускает эволюцию инстинктов.
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
#                     # Пробуем generate_batch
#                     if hasattr(self.gan_generator, 'generate_batch'):
#                         gan_patterns = self.gan_generator.generate_batch(gan_patterns_per_gen)
#                     else:
#                         # Пробуем generate_pattern
#                         for _ in range(gan_patterns_per_gen):
#                             if hasattr(self.gan_generator, 'generate_pattern'):
#                                 pattern = self.gan_generator.generate_pattern()
#                                 gan_patterns.append(pattern)
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
#         """
#         Возвращает лучшие паттерны инстинктов.
#         """
#         return self.population.get_best_patterns(n)
#
#     def get_population_stats(self) -> Dict[str, Any]:
#         """
#         Возвращает статистику популяции.
#         """
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
#
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
#         """
#         Возвращает n лучших индивидов.
#         """
#         sorted_inds = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
#         return sorted_inds[:n]
#
#     def get_best_patterns(self, n: int = 5) -> List[np.ndarray]:
#         """
#         Возвращает паттерны n лучших индивидов.
#         """
#         return [ind.pattern for ind in self.get_best(n)]
#
#     def get_population_stats(self) -> Dict[str, Any]:
#         """
#         Возвращает статистику популяции.
#         """
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
#         """
#         Сериализует популяцию для сохранения.
#         """
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
#         """
#         Восстанавливает популяцию из словаря.
#         """
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
#         """
#         Оценивает паттерн дискриминатором GAN.
#         """
#         try:
#             # Проверяем, есть ли метод evaluate_pattern
#             if hasattr(self.discriminator, 'evaluate_pattern'):
#                 return self.discriminator.evaluate_pattern(pattern)
#             else:
#                 # Если нет, используем прямой forward
#                 # Убеждаемся, что паттерн имеет правильную размерность
#                 if len(pattern) != 256:
#                     # Если размерность не 256, обрезаем или дополняем
#                     if len(pattern) > 256:
#                         pattern = pattern[:256]
#                     else:
#                         # Дополняем нулями
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
#             # print(f"  ⚠️ Ошибка оценки дискриминатором: {e}")
#             return 0.5
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
#                     # Проверяем, есть ли метод generate_batch
#                     if hasattr(self.gan_generator, 'generate_batch'):
#                         gan_patterns = self.gan_generator.generate_batch(gan_patterns_per_gen)
#                     else:
#                         # Если нет, генерируем по одному
#                         for _ in range(gan_patterns_per_gen):
#                             pattern = self.gan_generator.generate_pattern()
#                             gan_patterns.append(pattern)
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
#         """
#         Возвращает лучшие паттерны инстинктов.
#         """
#         return self.population.get_best_patterns(n)
#
#     def get_population_stats(self) -> Dict[str, Any]:
#         """
#         Возвращает статистику популяции.
#         """
#         return self.population.get_population_stats()
#
#     def save_state(self, filepath: str):
#         """
#         Сохраняет состояние популяции.
#         """
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
#
