# core/evolution.py
"""
Движок эволюции с поддержкой GAN и визуализацией.
"""

import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Any

from core.population import Population
from db.connector import save_generation, save_genome
from models.gan import GAN, PatternEvaluator, PatternRepository


class EvolutionEngine:
    """
    Движок эволюции ботов с использованием GAN.
    """

    def __init__(self, world, population_size: int = 20, generations: int = 50,
                 steps_per_episode: int = 500, elite_count: int = 2,
                 mutation_rate: float = 0.1, use_gan: bool = True,
                 gan_training_epochs: int = 5, visualize: bool = True):
        """
        Инициализирует движок эволюции.

        Args:
            world: Объект мира
            population_size: Размер популяции
            generations: Количество поколений
            steps_per_episode: Шагов за эпизод
            elite_count: Количество элитных особей
            mutation_rate: Скорость мутации
            use_gan: Использовать GAN
            gan_training_epochs: Эпох обучения GAN
            visualize: Включить визуализацию
        """
        self.world = world
        self.population = Population(population_size, world)
        self.generations = generations
        self.steps_per_episode = steps_per_episode
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.best_fitness_history = []
        self.gan_stats = []
        self.visualize = visualize

        self.use_gan = use_gan
        self.gan_training_epochs = gan_training_epochs

        # ============================================================
        # ИНИЦИАЛИЗАЦИЯ GAN
        # ============================================================
        if use_gan:
            self.gan = GAN(
                latent_dim=128,
                pattern_dim=47,
                batch_size=16,
                state_dim=21,
                action_dim=4
            )

            self.pattern_repository = PatternRepository()
            self.pattern_evaluator = PatternEvaluator(
                self.gan.discriminator,
                self.gan.generator
            )

            # ============================================================
            # ИНИЦИАЛИЗАЦИЯ ИНСТИНКТОВ
            # ============================================================
            try:
                from models.instinct_gan import InstinctGAN
                from core.instinct_evolution import InstinctEvolutionEngine

                self.instinct_gan = InstinctGAN(
                    latent_dim=64,
                    pattern_dim=256,
                    batch_size=8
                )

                self.instinct_engine = InstinctEvolutionEngine(
                    population_size=20,
                    pattern_dim=256,
                    gan_generator=self.instinct_gan.generator,
                    gan_discriminator=self.instinct_gan.discriminator,
                    experience_buffer=self.gan.experience_buffer
                )
            except ImportError as e:
                print(f"⚠️ Инстинкты не загружены: {e}")
                self.instinct_gan = None
                self.instinct_engine = None
        else:
            self.gan = None
            self.pattern_repository = None
            self.pattern_evaluator = None
            self.instinct_gan = None
            self.instinct_engine = None

        # ============================================================
        # ИНИЦИАЛИЗАЦИЯ ВИЗУАЛИЗАЦИИ
        # ============================================================
        if self.visualize:
            try:
                from core.visualization import TrainingVisualizer
                self.visualizer = TrainingVisualizer()
                self.training_data = {
                    'generations': [],
                    'best_fitness': [],
                    'avg_fitness': [],
                    'g_loss': [],
                    'd_loss': [],
                    'diversity': [],
                    'reflex_success': {},
                    'patterns_generated': 0,
                    'patterns_stored': 0
                }
            except ImportError as e:
                print(f"⚠️ Визуализация не загружена: {e}")
                self.visualizer = None
                self.training_data = None
        else:
            self.visualizer = None
            self.training_data = None

    def _collect_training_data(self, generation: int, fitnesses: List[float],
                               experiences: List) -> None:
        """Собирает данные для визуализации."""
        if not self.visualize or self.training_data is None:
            return

        self.training_data['generations'].append(generation)
        self.training_data['best_fitness'].append(max(fitnesses))
        self.training_data['avg_fitness'].append(np.mean(fitnesses))

        # Разнообразие паттернов
        if self.pattern_repository:
            patterns = self.pattern_repository.get_all_patterns()
            if patterns:
                unique = len(set(tuple(p) for p in patterns))
                total = len(patterns)
                self.training_data['diversity'].append(unique / total if total > 0 else 0)
            else:
                self.training_data['diversity'].append(0)
        else:
            self.training_data['diversity'].append(0)

        # Статистика рефлексов из популяции
        for ind in self.population.individuals:
            if hasattr(ind, 'reflex_stats'):
                for action_id, stats in ind.reflex_stats.items():
                    if action_id not in self.training_data['reflex_success']:
                        self.training_data['reflex_success'][action_id] = []
                    rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
                    self.training_data['reflex_success'][action_id].append(rate)

        # Сохраняем потери GAN, если есть
        if self.gan_stats:
            last_stats = self.gan_stats[-1]
            self.training_data['g_loss'].append(last_stats.get('avg_g_loss', 0))
            self.training_data['d_loss'].append(last_stats.get('avg_d_loss', 0))
        else:
            self.training_data['g_loss'].append(0)
            self.training_data['d_loss'].append(0)

    def _generate_visualizations(self) -> None:
        """Генерирует все визуализации."""
        if not self.visualize or self.visualizer is None or self.training_data is None:
            return

        print("\n📊 Генерация визуализаций...")

        try:
            # Основные графики
            if self.training_data['generations']:
                self.visualizer.plot_fitness_progress(
                    self.training_data['generations'],
                    self.training_data['best_fitness'],
                    self.training_data['avg_fitness']
                )

            if self.training_data['g_loss'] and any(self.training_data['g_loss']):
                self.visualizer.plot_gan_progress(
                    self.training_data['generations'],
                    self.training_data['g_loss'],
                    self.training_data['d_loss']
                )

            if self.training_data['diversity'] and any(self.training_data['diversity']):
                self.visualizer.plot_pattern_diversity(
                    self.training_data['generations'],
                    self.training_data['diversity']
                )

            if self.training_data['reflex_success']:
                self.visualizer.plot_reflex_success(
                    self.training_data['reflex_success']
                )

            # Полная панель
            dashboard_data = {
                'generations': self.training_data['generations'],
                'best_fitness': self.training_data['best_fitness'],
                'avg_fitness': self.training_data['avg_fitness'],
                'g_loss': self.training_data['g_loss'] or [0] * len(self.training_data['generations']),
                'd_loss': self.training_data['d_loss'] or [0] * len(self.training_data['generations']),
                'diversity': self.training_data['diversity'] or [0] * len(self.training_data['generations']),
                'reflex_success': self.training_data['reflex_success'],
                'population_stats': self.population.get_statistics() if hasattr(self.population, 'get_statistics') else {},
                'patterns_generated': len(self.pattern_repository.get_all_patterns()) if self.pattern_repository else 0,
                'patterns_stored': 0
            }
            self.visualizer.create_dashboard(dashboard_data)

            # Анимация
            try:
                self.visualizer.create_animation(dashboard_data)
            except Exception as e:
                print(f"⚠️ Не удалось создать анимацию: {e}")

            print("✅ Все визуализации созданы!")

        except Exception as e:
            print(f"⚠️ Ошибка при генерации визуализаций: {e}")

    def _generate_and_evaluate_patterns(self, n_patterns: int = 20) -> None:
        """Генерирует и оценивает новые паттерны поведения."""
        if not self.gan or not self.pattern_repository:
            return

        try:
            patterns = self.gan.generate_batch(n_patterns)
            for pattern in patterns:
                if self.pattern_evaluator:
                    score = self.pattern_evaluator.evaluate_pattern(pattern)
                    if score >= 0.5:
                        self.pattern_repository.add_pattern(pattern, score)
            print(f"  Сгенерировано {n_patterns} паттернов, сохранено {len(self.pattern_repository.get_all_patterns())}")
        except Exception as e:
            print(f"⚠️ Ошибка при генерации паттернов: {e}")

    def _integrate_best_patterns(self) -> None:
        """Интегрирует лучшие сгенерированные паттерны в правила поведения."""
        if not self.pattern_repository or not self.gan:
            return

        try:
            best_patterns = self.pattern_repository.get_best_patterns(5)
            if not best_patterns:
                return

            new_rules = []
            for i, pattern in enumerate(best_patterns):
                rule = self.gan.generate_rule()
                score = self.pattern_repository.scores[i] if i < len(self.pattern_repository.scores) else 0.7
                rule['confidence'] = float(score)
                new_rules.append(rule)

            best_individuals = self.population.select_best(self.elite_count)
            for ind in best_individuals:
                for rule in new_rules[:3]:
                    if rule not in ind.reflex_module.rules:
                        ind.reflex_module.rules.append(rule)
                        print(f"  Добавлено новое правило рефлекса: {rule.get('action')}")
        except Exception as e:
            print(f"⚠️ Ошибка при интеграции паттернов: {e}")

    def run(self, save_to_db: bool = False) -> List[float]:
        """
        Запускает эволюцию.

        Args:
            save_to_db: Сохранять ли результаты в БД

        Returns:
            История лучших фитнесов
        """
        for gen in range(self.generations):
            start_time = time.time()
            print(f"=== Поколение {gen + 1}/{self.generations} ===")

            # Оцениваем всех особей
            fitnesses, experiences = self.population.evaluate_all(self.steps_per_episode)
            best_fit = max(fitnesses)
            avg_fit = sum(fitnesses) / len(fitnesses)
            self.best_fitness_history.append(best_fit)

            print(f"Лучший фитнес: {best_fit:.2f}, Средний: {avg_fit:.2f}")
            print(f"Собрано переходов: {len(experiences)}")

            # Сохраняем статистику в БД (опционально)
            if save_to_db:
                gen_id = save_generation(gen + 1, best_fit, avg_fit)
                best_ind = self.population.select_best(1)[0]
                if gen_id:
                    save_genome(gen_id, best_ind.genome.to_dict(), best_ind.fitness)

            # Собираем данные для визуализации
            if self.visualize:
                self._collect_training_data(gen + 1, fitnesses, experiences)

            # Обучение GAN на собранных данных
            if self.gan and experiences:
                print(f"Обучение GAN на {len(experiences)} переходах...")
                self.gan.add_experiences(experiences)

                if len(self.gan.experience_buffer) >= self.gan.batch_size:
                    gan_results = self.gan.train(
                        epochs=self.gan_training_epochs,
                        verbose=(gen % 5 == 0)
                    )

                    if gan_results and gan_results.get('g_loss') and gan_results.get('d_loss'):
                        avg_g_loss = np.mean(gan_results['g_loss'])
                        avg_d_loss = np.mean(gan_results['d_loss'])
                        min_g_loss = np.min(gan_results['g_loss'])
                        max_g_loss = np.max(gan_results['g_loss'])
                        min_d_loss = np.min(gan_results['d_loss'])
                        max_d_loss = np.max(gan_results['d_loss'])

                        stats = {
                            'generation': gen + 1,
                            'avg_g_loss': avg_g_loss,
                            'avg_d_loss': avg_d_loss,
                            'min_g_loss': min_g_loss,
                            'max_g_loss': max_g_loss,
                            'min_d_loss': min_d_loss,
                            'max_d_loss': max_d_loss,
                            'steps': len(gan_results['g_loss'])
                        }
                        self.gan_stats.append(stats)

                        print(f"\n{'=' * 50}")
                        print(f"📊 GAN статистика за поколение {gen + 1}:")
                        print(f"{'=' * 50}")
                        print(f"  Генератор (G):")
                        print(f"    Средняя потеря: {avg_g_loss:.4f}")
                        print(f"    Минимальная:    {min_g_loss:.4f}")
                        print(f"    Максимальная:   {max_g_loss:.4f}")
                        print(f"  Дискриминатор (D):")
                        print(f"    Средняя потеря: {avg_d_loss:.4f}")
                        print(f"    Минимальная:    {min_d_loss:.4f}")
                        print(f"    Максимальная:   {max_d_loss:.4f}")
                        print(f"  Количество шагов: {len(gan_results['g_loss'])}")
                        print(f"{'=' * 50}\n")

                        # Генерируем и оцениваем новые паттерны
                        self._generate_and_evaluate_patterns()
                        self._integrate_best_patterns()

            # Эволюция инстинктов каждые 5 поколений
            if gen % 5 == 0 and self.use_gan and self.instinct_engine is not None:
                if self.gan and len(self.gan.experience_buffer) > 10:
                    try:
                        print("\n🧬 Эволюция инстинктов...")
                        instinct_history = self.instinct_engine.evolve(
                            n_generations=2,
                            gan_patterns_per_gen=3
                        )
                    except Exception as e:
                        print(f"⚠️ Ошибка эволюции инстинктов: {e}")

            # Создаём следующее поколение (кроме последнего)
            if gen < self.generations - 1:
                self.population.next_generation(
                    elite_count=self.elite_count,
                    mutation_rate=self.mutation_rate
                )

            elapsed = time.time() - start_time
            print(f"Время поколения: {elapsed:.2f} сек.\n")

        # Финальная визуализация после всех поколений
        if self.visualize:
            self._generate_visualizations()

        return self.best_fitness_history

    def get_gan_stats(self) -> List[Dict]:
        """Возвращает накопленную статистику GAN."""
        return self.gan_stats



# # core/evolution.py - ИСПРАВЛЕННЫЙ (только изменённая часть)
#
# def __init__(self, world, population_size=20, generations=50,
#              steps_per_episode=500, elite_count=2, mutation_rate=0.1,
#              use_gan=True, gan_training_epochs=5,
#              visualize=True):
#     self.world = world
#     self.population = Population(population_size, world)
#     self.generations = generations
#     self.steps_per_episode = steps_per_episode
#     self.elite_count = elite_count
#     self.mutation_rate = mutation_rate
#     self.best_fitness_history = []
#     self.gan_stats = []
#     self.visualize = visualize
#
#     self.use_gan = use_gan
#     self.gan_training_epochs = gan_training_epochs
#
#     if use_gan:
#         self.gan = GAN(
#             latent_dim=128,
#             pattern_dim=47,
#             batch_size=16,
#             state_dim=21,
#             action_dim=4
#         )
#
#         self.pattern_repository = PatternRepository()
#         self.pattern_evaluator = PatternEvaluator(
#             self.gan.discriminator,
#             self.gan.generator
#         )
#
#         from models.instinct_gan import InstinctGAN
#         from core.instinct_evolution import InstinctEvolutionEngine
#
#         self.instinct_gan = InstinctGAN(
#             latent_dim=64,
#             pattern_dim=256,
#             batch_size=8
#         )
#
#         self.instinct_engine = InstinctEvolutionEngine(
#             population_size=20,
#             pattern_dim=256,
#             gan_generator=self.instinct_gan.generator,
#             gan_discriminator=self.instinct_gan.discriminator,
#             experience_buffer=self.gan.experience_buffer
#         )
#     else:
#         self.gan = None
#         self.pattern_repository = None
#         self.pattern_evaluator = None
#         self.instinct_gan = None
#         self.instinct_engine = None
#
#     # Визуализация
#     if self.visualize:
#         from core.visualization import TrainingVisualizer
#         self.visualizer = TrainingVisualizer()
#         self.training_data = {
#             'generations': [],
#             'best_fitness': [],
#             'avg_fitness': [],
#             'g_loss': [],
#             'd_loss': [],
#             'diversity': [],
#             'reflex_success': {},
#             'patterns_generated': 0,
#             'patterns_stored': 0
#         }
#     else:
#         self.visualizer = None
#         self.training_data = None
#
#
#
# # # core/evolution.py (исправленный)
# # from core.population import Population
# # from core.visualization import TrainingVisualizer
# # from db.connector import save_generation, save_genome
# # from models.gan import GAN, PatternEvaluator, PatternRepository
# # import time
# # import numpy as np
# # from typing import List, Dict, Optional, Tuple, Any, Union
# #
# #
# # class EvolutionEngine:
# #
# #     def __init__(self, world, population_size=20, generations=50,
# #                  steps_per_episode=500, elite_count=2, mutation_rate=0.1,
# #                  use_gan=True, gan_training_epochs=5,
# #                  visualize=True):
# #         self.world = world
# #         self.population = Population(population_size, world)
# #         self.generations = generations
# #         self.steps_per_episode = steps_per_episode
# #         self.elite_count = elite_count
# #         self.mutation_rate = mutation_rate
# #         self.best_fitness_history = []
# #         self.gan_stats = []
# #         self.visualize = visualize
# #
# #         self.use_gan = use_gan
# #         self.gan_training_epochs = gan_training_epochs
# #
# #         if use_gan:
# #             # Инициализируем GAN с pattern_dim=47
# #             self.gan = GAN(
# #                 latent_dim=128,
# #                 pattern_dim=47,
# #                 batch_size=16,
# #                 state_dim=21,
# #                 action_dim=4
# #             )
# #
# #             # Репозиторий паттернов
# #             self.pattern_repository = PatternRepository()
# #
# #             # PatternEvaluator использует дискриминатор и генератор
# #             # Если encoder нет, используем генератор как энкодер
# #             try:
# #                 self.pattern_evaluator = PatternEvaluator(
# #                     self.gan.discriminator,
# #                     self.gan.generator  # ← используем generator вместо encoder
# #                 )
# #             except AttributeError:
# #                 self.pattern_evaluator = None
# #                 print("⚠️ PatternEvaluator временно отключён")
# #
# #             from models.instinct_gan import InstinctGAN
# #             from core.instinct_evolution import InstinctEvolutionEngine
# #
# #             self.instinct_gan = InstinctGAN(
# #                 latent_dim=64,
# #                 pattern_dim=256,
# #                 batch_size=8
# #             )
# #
# #             self.instinct_engine = InstinctEvolutionEngine(
# #                 population_size=20,
# #                 pattern_dim=256,
# #                 gan_generator=self.instinct_gan.generator,
# #                 gan_discriminator=self.instinct_gan.discriminator,
# #                 experience_buffer=self.gan.experience_buffer
# #             )
# #         else:
# #             self.gan = None
# #             self.pattern_repository = None
# #             self.pattern_evaluator = None
# #             self.instinct_gan = None
# #             self.instinct_engine = None
# #
# #         # Визуализация
# #         if self.visualize:
# #             from core.visualization import TrainingVisualizer
# #             self.visualizer = TrainingVisualizer()
# #             self.training_data = {
# #                 'generations': [],
# #                 'best_fitness': [],
# #                 'avg_fitness': [],
# #                 'g_loss': [],
# #                 'd_loss': [],
# #                 'diversity': [],
# #                 'reflex_success': {},
# #                 'patterns_generated': 0,
# #                 'patterns_stored': 0
# #             }
# #         else:
# #             self.visualizer = None
# #             self.training_data = None
# #
# #
# #
# #     # def __init__(self, world, population_size=20, generations=50,
# #     #              steps_per_episode=500, elite_count=2, mutation_rate=0.1,
# #     #              use_gan=True, gan_training_epochs=5,
# #     #              visualize=True):
# #     #     self.world = world
# #     #     self.population = Population(population_size, world)
# #     #     self.generations = generations
# #     #     self.steps_per_episode = steps_per_episode
# #     #     self.elite_count = elite_count
# #     #     self.mutation_rate = mutation_rate
# #     #     self.best_fitness_history = []
# #     #     self.gan_stats = []
# #     #     self.visualize = visualize
# #     #
# #     #     self.use_gan = use_gan
# #     #     self.gan_training_epochs = gan_training_epochs
# #     #
# #     #     if use_gan:
# #     #         # GAN с фиксированной размерностью 21
# #     #         self.gan = GAN(
# #     #             latent_dim=128,
# #     #             pattern_dim=47,
# #     #             batch_size=16,
# #     #             state_dim=21,
# #     #             action_dim=4
# #     #         )
# #     #         self.pattern_repository = PatternRepository()
# #     #         self.pattern_evaluator = PatternEvaluator(
# #     #             self.gan.discriminator,
# #     #             self.gan.encoder
# #     #         )
# #     #
# #     #         from models.instinct_gan import InstinctGAN
# #     #         from core.instinct_evolution import InstinctEvolutionEngine
# #     #
# #     #         self.instinct_gan = InstinctGAN(
# #     #             latent_dim=64,
# #     #             pattern_dim=256,
# #     #             batch_size=8
# #     #         )
# #     #
# #     #         self.instinct_engine = InstinctEvolutionEngine(
# #     #             population_size=20,
# #     #             pattern_dim=256,
# #     #             gan_generator=self.instinct_gan.generator,
# #     #             gan_discriminator=self.instinct_gan.discriminator,
# #     #             experience_buffer=self.gan.experience_buffer
# #     #         )
# #     #     else:
# #     #         self.gan = None
# #     #         self.pattern_repository = None
# #     #         self.pattern_evaluator = None
# #     #         self.instinct_gan = None
# #     #         self.instinct_engine = None
# #     #
# #     #     # Визуализация
# #     #     if self.visualize:
# #     #         from core.visualization import TrainingVisualizer
# #     #         self.visualizer = TrainingVisualizer()
# #     #         self.training_data = {
# #     #             'generations': [],
# #     #             'best_fitness': [],
# #     #             'avg_fitness': [],
# #     #             'g_loss': [],
# #     #             'd_loss': [],
# #     #             'diversity': [],
# #     #             'reflex_success': {},
# #     #             'patterns_generated': 0,
# #     #             'patterns_stored': 0
# #     #         }
# #     #     else:
# #     #         self.visualizer = None
# #     #         self.training_data = None
# #
# #
# #     # def __init__(self, world, population_size=20, generations=50,
# #     #              steps_per_episode=500, elite_count=2, mutation_rate=0.1,
# #     #              use_gan=True, gan_training_epochs=5,
# #     #              visualize=True):  # ← ДОБАВЛЯЕМ visualize
# #     #     self.world = world
# #     #     self.population = Population(population_size, world)
# #     #     self.generations = generations
# #     #     self.steps_per_episode = steps_per_episode
# #     #     self.elite_count = elite_count
# #     #     self.mutation_rate = mutation_rate
# #     #     self.best_fitness_history = []
# #     #     self.gan_stats = []
# #     #     self.visualize = visualize  # ← СОХРАНЯЕМ
# #     #
# #     #     # ============================================================
# #     #     # 1. СНАЧАЛА ИНИЦИАЛИЗИРУЕМ GAN (основной)
# #     #     # ============================================================
# #     #     self.use_gan = use_gan
# #     #     self.gan_training_epochs = gan_training_epochs
# #     #
# #     #     if use_gan:
# #     #         self.gan = GAN(
# #     #             latent_dim=128,
# #     #             pattern_dim=21,
# #     #             batch_size=16,
# #     #             state_dim=21,
# #     #             action_dim=4
# #     #         )
# #     #
# #     #
# #     #     # if use_gan:
# #     #     #     # Определяем state_dim из мира
# #     #     #     test_state = world.get_state(None) if hasattr(world, 'get_state') else [0] * 21
# #     #     #     state_dim = len(test_state) if hasattr(test_state, '__len__') else 21
# #     #     #
# #     #     #     self.gan = GAN(
# #     #     #         latent_dim=128,
# #     #     #         pattern_dim=21,  # state + action (4 действия)
# #     #     #         # pattern_dim=state_dim + 4,  # state + action (4 действия)
# #     #     #         batch_size=16,
# #     #     #         state_dim=state_dim,
# #     #     #         action_dim=4
# #     #     #     )
# #     #
# #     #     # if use_gan:
# #     #     #     self.gan = GAN(
# #     #     #         latent_dim=128,
# #     #     #         pattern_dim=None,  # Автоматическое вычисление (21)
# #     #     #         batch_size=16,
# #     #     #         state_dim=8,
# #     #     #         action_dim=4
# #     #     #     )
# #     #     #     self.pattern_repository = PatternRepository()
# #     #     #     self.pattern_evaluator = PatternEvaluator(
# #     #     #         self.gan.discriminator,
# #     #     #         self.gan.encoder
# #     #     #     )
# #     #     #
# #     #     #     # ============================================================
# #     #     #     # 2. ПОСЛЕ СОЗДАНИЯ GAN - ИНИЦИАЛИЗИРУЕМ ИНСТИНКТЫ
# #     #     #     # ============================================================
# #     #     #     from models.instinct_gan import InstinctGAN
# #     #     #     from core.instinct_evolution import InstinctEvolutionEngine
# #     #     #
# #     #     #     self.instinct_gan = InstinctGAN(
# #     #     #         latent_dim=64,
# #     #     #         pattern_dim=256,
# #     #     #         batch_size=8
# #     #     #     )
# #     #     #
# #     #     #     self.instinct_engine = InstinctEvolutionEngine(
# #     #     #         population_size=20,
# #     #     #         pattern_dim=256,
# #     #     #         gan_generator=self.instinct_gan.generator,
# #     #     #         gan_discriminator=self.instinct_gan.discriminator,
# #     #     #         experience_buffer=self.gan.experience_buffer
# #     #     #     )
# #     #     # else:
# #     #     #     self.gan = None
# #     #     #     self.pattern_repository = None
# #     #     #     self.pattern_evaluator = None
# #     #     #     self.instinct_gan = None
# #     #     #     self.instinct_engine = None
# #     #
# #     #     # ============================================================
# #     #     # 3. ДОБАВЛЯЕМ ВИЗУАЛИЗАТОР
# #     #     # ============================================================
# #     #     if self.visualize:
# #     #         from core.visualization import TrainingVisualizer
# #     #         self.visualizer = TrainingVisualizer()
# #     #         self.training_data = {
# #     #             'generations': [],
# #     #             'best_fitness': [],
# #     #             'avg_fitness': [],
# #     #             'g_loss': [],
# #     #             'd_loss': [],
# #     #             'diversity': [],
# #     #             'reflex_success': {},
# #     #             'patterns_generated': 0,
# #     #             'patterns_stored': 0
# #     #         }
# #     #     else:
# #     #         self.visualizer = None
# #     #         self.training_data = None
# #
# #     def run(self, save_to_db=False):
# #         for gen in range(self.generations):
# #             start_time = time.time()
# #             print(f"=== Поколение {gen + 1}/{self.generations} ===")
# #
# #             # Оцениваем всех особей
# #             fitnesses, experiences = self.population.evaluate_all(self.steps_per_episode)
# #             best_fit = max(fitnesses)
# #             avg_fit = sum(fitnesses) / len(fitnesses)
# #             self.best_fitness_history.append(best_fit)
# #
# #             print(f"Лучший фитнес: {best_fit:.2f}, Средний: {avg_fit:.2f}")
# #             print(f"Собрано переходов: {len(experiences)}")
# #
# #             # Сохраняем статистику в БД (опционально)
# #             if save_to_db:
# #                 gen_id = save_generation(gen + 1, best_fit, avg_fit)
# #                 best_ind = self.population.select_best(1)[0]
# #                 if gen_id:
# #                     save_genome(gen_id, best_ind.genome.to_dict(), best_ind.fitness)
# #
# #             # Собираем данные для визуализации (исправлено: self.visualize)
# #             if self.visualize:
# #                 self._collect_training_data(gen + 1, fitnesses, experiences)
# #
# #             # Обучение GAN на собранных данных
# #             if self.gan and experiences:
# #                 print(f"Обучение GAN на {len(experiences)} переходах...")
# #                 self.gan.add_experiences(experiences)
# #
# #                 if len(self.gan.experience_buffer) >= self.gan.batch_size:
# #                     gan_results = self.gan.train(
# #                         epochs=self.gan_training_epochs,
# #                         verbose=(gen % 5 == 0)
# #                     )
# #
# #                     if gan_results and gan_results.get('g_loss') and gan_results.get('d_loss'):
# #                         avg_g_loss = np.mean(gan_results['g_loss'])
# #                         avg_d_loss = np.mean(gan_results['d_loss'])
# #                         min_g_loss = np.min(gan_results['g_loss'])
# #                         max_g_loss = np.max(gan_results['g_loss'])
# #                         min_d_loss = np.min(gan_results['d_loss'])
# #                         max_d_loss = np.max(gan_results['d_loss'])
# #
# #                         stats = {
# #                             'generation': gen + 1,
# #                             'avg_g_loss': avg_g_loss,
# #                             'avg_d_loss': avg_d_loss,
# #                             'min_g_loss': min_g_loss,
# #                             'max_g_loss': max_g_loss,
# #                             'min_d_loss': min_d_loss,
# #                             'max_d_loss': max_d_loss,
# #                             'steps': len(gan_results['g_loss'])
# #                         }
# #                         self.gan_stats.append(stats)
# #
# #                         print(f"\n{'=' * 50}")
# #                         print(f"📊 GAN статистика за поколение {gen + 1}:")
# #                         print(f"{'=' * 50}")
# #                         print(f"  Генератор (G):")
# #                         print(f"    Средняя потеря: {avg_g_loss:.4f}")
# #                         print(f"    Минимальная:    {min_g_loss:.4f}")
# #                         print(f"    Максимальная:   {max_g_loss:.4f}")
# #                         print(f"  Дискриминатор (D):")
# #                         print(f"    Средняя потеря: {avg_d_loss:.4f}")
# #                         print(f"    Минимальная:    {min_d_loss:.4f}")
# #                         print(f"    Максимальная:   {max_d_loss:.4f}")
# #                         print(f"  Количество шагов: {len(gan_results['g_loss'])}")
# #                         print(f"{'=' * 50}\n")
# #
# #                         # Генерируем и оцениваем новые паттерны
# #                         self._generate_and_evaluate_patterns()
# #                         self._integrate_best_patterns()
# #
# #             # Эволюция инстинктов каждые 5 поколений
# #             if gen % 5 == 0 and self.use_gan and self.instinct_engine is not None:
# #                 if self.gan and len(self.gan.experience_buffer) > 10:
# #                     try:
# #                         print("\n🧬 Эволюция инстинктов...")
# #                         instinct_history = self.instinct_engine.evolve(
# #                             n_generations=2,
# #                             gan_patterns_per_gen=3
# #                         )
# #                     except Exception as e:
# #                         print(f"⚠️ Ошибка эволюции инстинктов: {e}")
# #
# #             # Создаём следующее поколение (кроме последнего)
# #             if gen < self.generations - 1:
# #                 self.population.next_generation(
# #                     elite_count=self.elite_count,
# #                     mutation_rate=self.mutation_rate
# #                 )
# #
# #             elapsed = time.time() - start_time
# #             print(f"Время поколения: {elapsed:.2f} сек.\n")
# #
# #         # Финальная визуализация после всех поколений
# #         if self.visualize:
# #             self._generate_visualizations()
# #
# #         return self.best_fitness_history
# #
# #     # def __init__(self, world, population_size=20, generations=50,
# #     #              steps_per_episode=500, elite_count=2, mutation_rate=0.1,
# #     #              use_gan=True, gan_training_epochs=5):
# #     #     self.world = world
# #     #     self.population = Population(population_size, world)
# #     #     self.generations = generations
# #     #     self.steps_per_episode = steps_per_episode
# #     #     self.elite_count = elite_count
# #     #     self.mutation_rate = mutation_rate
# #     #     self.best_fitness_history = []
# #     #     self.gan_stats = []
# #     #
# #     #     # ============================================================
# #     #     # 1. СНАЧАЛА ИНИЦИАЛИЗИРУЕМ GAN (основной)
# #     #     # ============================================================
# #     #     self.use_gan = use_gan
# #     #     self.gan_training_epochs = gan_training_epochs
# #     #
# #     #     if use_gan:
# #     #         self.gan = GAN(
# #     #             latent_dim=128,
# #     #             pattern_dim=None,  # Автоматическое вычисление (21)
# #     #             batch_size=16,
# #     #             state_dim=8,
# #     #             action_dim=4
# #     #         )
# #     #         self.pattern_repository = PatternRepository()
# #     #         self.pattern_evaluator = PatternEvaluator(
# #     #             self.gan.discriminator,
# #     #             self.gan.encoder
# #     #         )
# #     #
# #     #         # ============================================================
# #     #         # 2. ПОСЛЕ СОЗДАНИЯ GAN - ИНИЦИАЛИЗИРУЕМ ИНСТИНКТЫ
# #     #         # ============================================================
# #     #         from models.instinct_gan import InstinctGAN
# #     #         from core.instinct_evolution import InstinctEvolutionEngine
# #     #
# #     #         self.instinct_gan = InstinctGAN(
# #     #             latent_dim=64,
# #     #             pattern_dim=256,
# #     #             batch_size=8
# #     #         )
# #     #
# #     #         self.instinct_engine = InstinctEvolutionEngine(
# #     #             population_size=20,
# #     #             pattern_dim=256,
# #     #             gan_generator=self.instinct_gan.generator,
# #     #             gan_discriminator=self.instinct_gan.discriminator,
# #     #             experience_buffer=self.gan.experience_buffer  # Теперь self.gan уже существует
# #     #         )
# #     #     else:
# #     #         self.gan = None
# #     #         self.pattern_repository = None
# #     #         self.pattern_evaluator = None
# #     #         self.instinct_gan = None
# #     #         self.instinct_engine = None
# #     #
# #     #     # НОВОЕ: Визуализация
# #     #     self.visualizer = TrainingVisualizer()
# #     #     self.training_data = {
# #     #         'generations': [],
# #     #         'best_fitness': [],
# #     #         'avg_fitness': [],
# #     #         'g_loss': [],
# #     #         'd_loss': [],
# #     #         'diversity': [],
# #     #         'reflex_success': {},
# #     #         'patterns_generated': 0,
# #     #         'patterns_stored': 0
# #     #     }
# #
# #     # def run(self, save_to_db=False):
# #     #     for gen in range(self.generations):
# #     #         start_time = time.time()
# #     #         print(f"=== Поколение {gen + 1}/{self.generations} ===")
# #     #
# #     #         # Оцениваем всех особей
# #     #         fitnesses, experiences = self.population.evaluate_all(self.steps_per_episode)
# #     #         best_fit = max(fitnesses)
# #     #         avg_fit = sum(fitnesses) / len(fitnesses)
# #     #         self.best_fitness_history.append(best_fit)
# #     #
# #     #         print(f"Лучший фитнес: {best_fit:.2f}, Средний: {avg_fit:.2f}")
# #     #         print(f"Собрано переходов: {len(experiences)}")
# #     #
# #     #         # Сохраняем статистику в БД (опционально)
# #     #         if save_to_db:
# #     #             gen_id = save_generation(gen + 1, best_fit, avg_fit)
# #     #             best_ind = self.population.select_best(1)[0]
# #     #             if gen_id:
# #     #                 save_genome(gen_id, best_ind.genome.to_dict(), best_ind.fitness)
# #     #
# #     #         # Собираем данные для визуализации
# #     #         if visualize:
# #     #             self._collect_training_data(gen + 1, fitnesses, experiences)
# #     #
# #     #         # Обучение GAN на собранных данных
# #     #         if self.gan and experiences:
# #     #             print(f"Обучение GAN на {len(experiences)} переходах...")
# #     #             self.gan.add_experiences(experiences)
# #     #
# #     #             if len(self.gan.experience_buffer) >= self.gan.batch_size:
# #     #                 gan_results = self.gan.train(
# #     #                     epochs=self.gan_training_epochs,
# #     #                     verbose=(gen % 5 == 0)
# #     #                 )
# #     #
# #     #                 if gan_results['g_loss'] and gan_results['d_loss']:
# #     #                     avg_g_loss = np.mean(gan_results['g_loss'])
# #     #                     avg_d_loss = np.mean(gan_results['d_loss'])
# #     #                     min_g_loss = np.min(gan_results['g_loss'])
# #     #                     max_g_loss = np.max(gan_results['g_loss'])
# #     #                     min_d_loss = np.min(gan_results['d_loss'])
# #     #                     max_d_loss = np.max(gan_results['d_loss'])
# #     #
# #     #                     stats = {
# #     #                         'generation': gen + 1,
# #     #                         'avg_g_loss': avg_g_loss,
# #     #                         'avg_d_loss': avg_d_loss,
# #     #                         'min_g_loss': min_g_loss,
# #     #                         'max_g_loss': max_g_loss,
# #     #                         'min_d_loss': min_d_loss,
# #     #                         'max_d_loss': max_d_loss,
# #     #                         'steps': len(gan_results['g_loss'])
# #     #                     }
# #     #                     self.gan_stats.append(stats)
# #     #
# #     #                     print(f"\n{'=' * 50}")
# #     #                     print(f"📊 GAN статистика за поколение {gen + 1}:")
# #     #                     print(f"{'=' * 50}")
# #     #                     print(f"  Генератор (G):")
# #     #                     print(f"    Средняя потеря: {avg_g_loss:.4f}")
# #     #                     print(f"    Минимальная:    {min_g_loss:.4f}")
# #     #                     print(f"    Максимальная:   {max_g_loss:.4f}")
# #     #                     print(f"  Дискриминатор (D):")
# #     #                     print(f"    Средняя потеря: {avg_d_loss:.4f}")
# #     #                     print(f"    Минимальная:    {min_d_loss:.4f}")
# #     #                     print(f"    Максимальная:   {max_d_loss:.4f}")
# #     #                     print(f"  Количество шагов: {len(gan_results['g_loss'])}")
# #     #                     print(f"{'=' * 50}\n")
# #     #
# #     #                     # Генерируем и оцениваем новые паттерны
# #     #                     self._generate_and_evaluate_patterns()
# #     #                     self._integrate_best_patterns()
# #     #
# #     #         # Эволюция инстинктов каждые 5 поколений
# #     #         if gen % 5 == 0 and self.use_gan and self.instinct_engine is not None:
# #     #             if self.gan and len(self.gan.experience_buffer) > 10:
# #     #                 try:
# #     #                     print("\n🧬 Эволюция инстинктов...")
# #     #                     instinct_history = self.instinct_engine.evolve(
# #     #                         n_generations=2,
# #     #                         gan_patterns_per_gen=3
# #     #                     )
# #     #                 except Exception as e:
# #     #                     print(f"⚠️ Ошибка эволюции инстинктов: {e}")
# #     #
# #     #         # Создаём следующее поколение (кроме последнего)
# #     #         if gen < self.generations - 1:
# #     #             self.population.next_generation(
# #     #                 elite_count=self.elite_count,
# #     #                 mutation_rate=self.mutation_rate
# #     #             )
# #     #
# #     #         elapsed = time.time() - start_time
# #     #         print(f"Время поколения: {elapsed:.2f} сек.\n")
# #     #
# #     #     return self.best_fitness_history
# #
# #     def _generate_visualizations(self):
# #         """Генерирует все визуализации."""
# #         print("\n📊 Генерация визуализаций...")
# #
# #         # Основные графики
# #         self.visualizer.plot_fitness_progress(
# #             self.training_data['generations'],
# #             self.training_data['best_fitness'],
# #             self.training_data['avg_fitness']
# #         )
# #
# #         if self.training_data['g_loss']:
# #             self.visualizer.plot_gan_progress(
# #                 self.training_data['generations'],
# #                 self.training_data['g_loss'],
# #                 self.training_data['d_loss']
# #             )
# #
# #         self.visualizer.plot_pattern_diversity(
# #             self.training_data['generations'],
# #             self.training_data['diversity']
# #         )
# #
# #         if self.training_data['reflex_success']:
# #             self.visualizer.plot_reflex_success(
# #                 self.training_data['reflex_success']
# #             )
# #
# #         # Полная панель
# #         dashboard_data = {
# #             'generations': self.training_data['generations'],
# #             'best_fitness': self.training_data['best_fitness'],
# #             'avg_fitness': self.training_data['avg_fitness'],
# #             'g_loss': self.training_data['g_loss'] or [0] * len(self.training_data['generations']),
# #             'd_loss': self.training_data['d_loss'] or [0] * len(self.training_data['generations']),
# #             'diversity': self.training_data['diversity'] or [0] * len(self.training_data['generations']),
# #             'reflex_success': self.training_data['reflex_success'],
# #             'population_stats': self.population.get_statistics(),
# #             'patterns_generated': len(self.pattern_repository.patterns) if self.pattern_repository else 0,
# #             'patterns_stored': len([p for p in self.pattern_repository.patterns if p.get('stored', False)]) if self.pattern_repository else 0
# #         }
# #         self.visualizer.create_dashboard(dashboard_data)
# #
# #         # Анимация
# #         try:
# #             self.visualizer.create_animation(dashboard_data)
# #         except Exception as e:
# #             print(f"⚠️ Не удалось создать анимацию: {e}")
# #
# #         print("✅ Все визуализации созданы!")
# #
# #     def _collect_training_data(self, generation: int, fitnesses: List[float],
# #                                experiences: List):
# #         """Собирает данные для визуализации."""
# #         self.training_data['generations'].append(generation)
# #         self.training_data['best_fitness'].append(max(fitnesses))
# #         self.training_data['avg_fitness'].append(np.mean(fitnesses))
# #
# #         # Разнообразие паттернов
# #         if self.pattern_repository:
# #             unique = len(set(tuple(p) for p in self.pattern_repository.patterns))
# #             total = len(self.pattern_repository.patterns)
# #             self.training_data['diversity'].append(unique / total if total > 0 else 0)
# #
# #         # Статистика рефлексов
# #         for ind in self.population.individuals:
# #             for action_id, stats in ind.reflex_stats.items():
# #                 if action_id not in self.training_data['reflex_success']:
# #                     self.training_data['reflex_success'][action_id] = []
# #                 rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
# #                 self.training_data['reflex_success'][action_id].append(rate)
# #
# #     def _generate_and_evaluate_patterns(self, n_patterns: int = 20):
# #         """Генерирует и оценивает новые паттерны поведения."""
# #         if not self.gan:
# #             return
# #
# #         try:
# #             patterns = self.gan.generate_batch(n_patterns)
# #             for pattern in patterns:
# #                 score = self.pattern_evaluator.evaluate_pattern(pattern)
# #                 if score >= 0.5:
# #                     self.pattern_repository.add_pattern(pattern, score)
# #             print(f"Сгенерировано {n_patterns} паттернов, сохранено {len(self.pattern_repository.patterns)}")
# #         except Exception as e:
# #             print(f"⚠️ Ошибка при генерации паттернов: {e}")
# #
# #     def _integrate_best_patterns(self):
# #         """Интегрирует лучшие сгенерированные паттерны в правила поведения."""
# #         if not self.pattern_repository or not self.gan:
# #             return
# #
# #         try:
# #             best_patterns = self.pattern_repository.get_best_patterns(5)
# #             if not best_patterns:
# #                 return
# #
# #             new_rules = []
# #             for i, pattern in enumerate(best_patterns):
# #                 rule = self.gan.generate_rule()
# #                 score = self.pattern_repository.scores[i] if i < len(self.pattern_repository.scores) else 0.7
# #                 rule['confidence'] = float(score)
# #                 new_rules.append(rule)
# #
# #             best_individuals = self.population.select_best(self.elite_count)
# #             for ind in best_individuals:
# #                 for rule in new_rules[:3]:
# #                     if rule not in ind.reflex_module.rules:
# #                         ind.reflex_module.rules.append(rule)
# #                         print(f"Добавлено новое правило рефлекса: {rule}")
# #         except Exception as e:
# #             print(f"⚠️ Ошибка при интеграции паттернов: {e}")
# #
# #     def get_gan_stats(self):
# #         """Возвращает накопленную статистику GAN."""
# #         return self.gan_stats