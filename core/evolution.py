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
            saved_count = 0
            for pattern in patterns:
                if self.pattern_evaluator:
                    score = self.pattern_evaluator.evaluate_pattern(pattern)
                    if score >= 0.01:  # ← БЫЛО 0.2, СТАЛО 0.01
                        self.pattern_repository.add_pattern(pattern, score)
                        saved_count += 1
            print(f"  Сгенерировано {n_patterns} паттернов, сохранено {saved_count}")
        except Exception as e:
            print(f"⚠️ Ошибка при генерации паттернов: {e}")


    # def _generate_and_evaluate_patterns(self, n_patterns: int = 20) -> None:
    #     """Генерирует и оценивает новые паттерны поведения."""
    #     if not self.gan or not self.pattern_repository:
    #         return
    #
    #     try:
    #         patterns = self.gan.generate_batch(n_patterns)
    #         for pattern in patterns:
    #             if self.pattern_evaluator:
    #                 score = self.pattern_evaluator.evaluate_pattern(pattern)
    #                 if score >= 0.5:
    #                     self.pattern_repository.add_pattern(pattern, score)
    #         print(f"  Сгенерировано {n_patterns} паттернов, сохранено {len(self.pattern_repository.get_all_patterns())}")
    #     except Exception as e:
    #         print(f"⚠️ Ошибка при генерации паттернов: {e}")


    def _integrate_best_patterns(self) -> None:
        """Интегрирует лучшие сгенерированные паттерны в правила поведения и сохраняет в БД."""
        if not self.pattern_repository or not self.gan:
            return

        try:
            best_patterns = self.pattern_repository.get_best_patterns(5)
            if not best_patterns:
                return

            from db.connector import save_generated_pattern  # ← ТОЛЬКО ЭТОТ ИМПОРТ

            # Текущее поколение (номер, а не ID из БД)
            generation_num = len(self.best_fitness_history)  # 1, 2, 3, ...

            new_rules = []
            for i, pattern in enumerate(best_patterns):
                # Генерируем правило из паттерна
                rule = self.gan.generate_rule()
                score = self.pattern_repository.scores[i] if i < len(self.pattern_repository.scores) else 0.7
                rule['confidence'] = float(score)
                new_rules.append(rule)

                # 💾 СОХРАНЯЕМ В БД
                try:
                    save_generated_pattern(generation_num, pattern, score, rule)
                    print(f"  💾 Паттерн сохранён в БД: {rule.get('action')} (score: {score:.3f})")
                except Exception as e:
                    print(f"  ⚠️ Ошибка сохранения паттерна в БД: {e}")

            # Добавляем правила в рефлексы лучших особей
            best_individuals = self.population.select_best(self.elite_count)
            for ind in best_individuals:
                for rule in new_rules[:3]:
                    if rule not in ind.reflex_module.rules:
                        ind.reflex_module.rules.append(rule)
                        print(f"  Добавлено новое правило рефлекса: {rule.get('action')}")

        except Exception as e:
            print(f"⚠️ Ошибка при интеграции паттернов: {e}")


    # def _integrate_best_patterns(self) -> None:
    #     """Интегрирует лучшие сгенерированные паттерны в правила поведения и сохраняет в БД."""
    #     if not self.pattern_repository or not self.gan:
    #         return
    #
    #     try:
    #         best_patterns = self.pattern_repository.get_best_patterns(5)
    #         if not best_patterns:
    #             return
    #
    #         # Получаем текущий generation_id (если сохраняем в БД)
    #         from db.connector import save_generated_pattern, get_current_generation_id
    #
    #         new_rules = []
    #         for i, pattern in enumerate(best_patterns):
    #             # Генерируем правило из паттерна
    #             rule = self.gan.generate_rule()
    #             score = self.pattern_repository.scores[i] if i < len(self.pattern_repository.scores) else 0.7
    #             rule['confidence'] = float(score)
    #             new_rules.append(rule)
    #
    #             # 💾 СОХРАНЯЕМ В БД
    #             try:
    #                 gen_id = len(self.best_fitness_history)  # текущее поколение
    #                 save_generated_pattern(gen_id, pattern, score, rule)
    #                 print(f"  💾 Паттерн сохранён в БД: {rule.get('action')} (score: {score:.3f})")
    #             except Exception as e:
    #                 print(f"  ⚠️ Ошибка сохранения паттерна в БД: {e}")
    #
    #         # Добавляем правила в рефлексы лучших особей
    #         best_individuals = self.population.select_best(self.elite_count)
    #         for ind in best_individuals:
    #             for rule in new_rules[:3]:
    #                 if rule not in ind.reflex_module.rules:
    #                     ind.reflex_module.rules.append(rule)
    #                     print(f"  Добавлено новое правило рефлекса: {rule.get('action')}")
    #
    #     except Exception as e:
    #         print(f"⚠️ Ошибка при интеграции паттернов: {e}")


    # def _integrate_best_patterns(self) -> None:
    #     """Интегрирует лучшие сгенерированные паттерны в правила поведения."""
    #     if not self.pattern_repository or not self.gan:
    #         return
    #
    #     try:
    #         best_patterns = self.pattern_repository.get_best_patterns(5)
    #         if not best_patterns:
    #             return
    #
    #         new_rules = []
    #         for i, pattern in enumerate(best_patterns):
    #             rule = self.gan.generate_rule()
    #             score = self.pattern_repository.scores[i] if i < len(self.pattern_repository.scores) else 0.7
    #             rule['confidence'] = float(score)
    #             new_rules.append(rule)
    #
    #         best_individuals = self.population.select_best(self.elite_count)
    #         for ind in best_individuals:
    #             for rule in new_rules[:3]:
    #                 if rule not in ind.reflex_module.rules:
    #                     ind.reflex_module.rules.append(rule)
    #                     print(f"  Добавлено новое правило рефлекса: {rule.get('action')}")
    #     except Exception as e:
    #         print(f"⚠️ Ошибка при интеграции паттернов: {e}")

    def run(self, save_to_db: bool = False) -> List[float]:
        """
        Запускает эволюцию с детальным логированием GA.
        """
        print("\n" + "=" * 70)
        print("🧬 ЗАПУСК ГЕНЕТИЧЕСКОГО АЛГОРИТМА")
        print("=" * 70)
        print(f"  Популяция: {len(self.population.individuals)} особей")
        print(f"  Поколений: {self.generations}")
        print(f"  Элита: {self.elite_count} лучших")
        print(f"  Скорость мутации: {self.mutation_rate}")
        print("=" * 70 + "\n")

        for gen in range(self.generations):
            start_time = time.time()
            print(f"\n{'=' * 70}")
            print(f"🧬 ПОКОЛЕНИЕ {gen + 1}/{self.generations}")
            print(f"{'=' * 70}")

            # ============================================================
            # 1. ОЦЕНКА ПОПУЛЯЦИИ
            # ============================================================
            print("📊 Оценка особей...")
            fitnesses, experiences = self.population.evaluate_all(self.steps_per_episode)
            best_fit = max(fitnesses)
            avg_fit = sum(fitnesses) / len(fitnesses)
            min_fit = min(fitnesses)
            self.best_fitness_history.append(best_fit)

            print(f"  Лучший фитнес: {best_fit:.2f}")
            print(f"  Средний фитнес: {avg_fit:.2f}")
            print(f"  Худший фитнес: {min_fit:.2f}")
            print(f"  Собрано переходов: {len(experiences)}")

            # ============================================================
            # 2. СОХРАНЕНИЕ В БД (опционально)
            # ============================================================
            if save_to_db:
                gen_id = save_generation(gen + 1, best_fit, avg_fit)
                best_ind = self.population.select_best(1)[0]
                if gen_id:
                    save_genome(gen_id, best_ind.genome.to_dict(), best_ind.fitness)
                    print(f"  💾 Поколение {gen + 1} сохранено в БД (ID: {gen_id})")

            # ============================================================
            # 3. СБОР ДАННЫХ ДЛЯ ВИЗУАЛИЗАЦИИ
            # ============================================================
            if self.visualize:
                self._collect_training_data(gen + 1, fitnesses, experiences)

            # ============================================================
            # 4. ОБУЧЕНИЕ GAN
            # ============================================================
            if self.gan and experiences:
                print(f"\n🎯 ОБУЧЕНИЕ GAN на {len(experiences)} переходах...")
                self.gan.add_experiences(experiences)

                if len(self.gan.experience_buffer) >= self.gan.batch_size:
                    gan_results = self.gan.train(
                        epochs=self.gan_training_epochs,
                        verbose=(gen % 5 == 0)
                    )

                    if gan_results and gan_results.get('g_loss') and gan_results.get('d_loss'):
                        avg_g_loss = np.mean(gan_results['g_loss'])
                        avg_d_loss = np.mean(gan_results['d_loss'])

                        # Логируем изменение GAN
                        if gen > 0 and self.gan_stats:
                            prev_g = self.gan_stats[-1].get('avg_g_loss', avg_g_loss)
                            prev_d = self.gan_stats[-1].get('avg_d_loss', avg_d_loss)
                            g_change = ((avg_g_loss - prev_g) / (prev_g + 0.001)) * 100
                            d_change = ((avg_d_loss - prev_d) / (prev_d + 0.001)) * 100
                            print(f"  📈 G Loss: {avg_g_loss:.4f} ({g_change:+.1f}% от прошлого)")
                            print(f"  📉 D Loss: {avg_d_loss:.4f} ({d_change:+.1f}% от прошлого)")
                        else:
                            print(f"  📈 G Loss: {avg_g_loss:.4f} (начальный)")
                            print(f"  📉 D Loss: {avg_d_loss:.4f} (начальный)")

                        stats = {
                            'generation': gen + 1,
                            'avg_g_loss': avg_g_loss,
                            'avg_d_loss': avg_d_loss,
                            'min_g_loss': np.min(gan_results['g_loss']),
                            'max_g_loss': np.max(gan_results['g_loss']),
                            'min_d_loss': np.min(gan_results['d_loss']),
                            'max_d_loss': np.max(gan_results['d_loss']),
                            'steps': len(gan_results['g_loss'])
                        }
                        self.gan_stats.append(stats)

                        # Генерируем и оцениваем новые паттерны
                        self._generate_and_evaluate_patterns()
                        self._integrate_best_patterns()

            # ============================================================
            # 5. ЭВОЛЮЦИЯ ИНСТИНКТОВ (каждые 5 поколений)
            # ============================================================
            if gen % 5 == 0 and self.use_gan and self.instinct_engine is not None:
                if self.gan and len(self.gan.experience_buffer) > 10:
                    print("\n🧬 ЭВОЛЮЦИЯ ИНСТИНКТОВ...")
                    try:
                        instinct_history = self.instinct_engine.evolve(
                            n_generations=2,
                            gan_patterns_per_gen=3
                        )
                        if instinct_history:
                            print(f"  Лучший фитнес инстинктов: {max(instinct_history):.4f}")
                            print(f"  Средний фитнес инстинктов: {np.mean(instinct_history):.4f}")
                    except Exception as e:
                        print(f"  ⚠️ Ошибка эволюции инстинктов: {e}")

            # ============================================================
            # 6. СОЗДАНИЕ СЛЕДУЮЩЕГО ПОКОЛЕНИЯ (GA)
            # ============================================================
            if gen < self.generations - 1:
                print("\n🧬 СОЗДАНИЕ СЛЕДУЮЩЕГО ПОКОЛЕНИЯ (GA)...")

                # Выбираем элиту
                best_individuals = self.population.select_best(self.elite_count)
                print(f"  Элита: {len(best_individuals)} особей, средний фитнес: {np.mean([ind.fitness for ind in best_individuals]):.2f}")

                # Создаём новое поколение
                self.population.next_generation(
                    elite_count=self.elite_count,
                    mutation_rate=self.mutation_rate
                )

                # Логируем изменения в геномах
                new_best = self.population.select_best(1)[0]
                old_best = best_individuals[0] if best_individuals else None

                if old_best and hasattr(old_best, 'genome') and hasattr(new_best, 'genome'):
                    old_params = old_best.genome.params
                    new_params = new_best.genome.params

                    # Проверяем, что изменилось
                    changes = []
                    for key in old_params:
                        if key in new_params and old_params[key] != new_params[key]:
                            if isinstance(old_params[key], (int, float)):
                                changes.append(f"{key}: {old_params[key]:.2f} → {new_params[key]:.2f}")

                    if changes:
                        print(f"  🧬 Изменения в геноме лучшей особи:")
                        for change in changes[:5]:
                            print(f"    • {change}")
                    else:
                        print(f"  🧬 Геном лучшей особи не изменился (мутации не произошло)")

                print(f"  Популяция: {len(self.population.individuals)} особей")

            elapsed = time.time() - start_time
            print(f"\n⏱️ Время поколения: {elapsed:.2f} сек.")

        # ============================================================
        # 7. ФИНАЛЬНАЯ СТАТИСТИКА
        # ============================================================
        print("\n" + "=" * 70)
        print("🏁 ЭВОЛЮЦИЯ ЗАВЕРШЕНА")
        print("=" * 70)
        print(f"  Лучший фитнес за всё время: {max(self.best_fitness_history):.2f}")
        print(f"  Финальный фитнес: {self.best_fitness_history[-1]:.2f}")
        print(f"  Всего поколений: {len(self.best_fitness_history)}")

        # Статистика GAN
        if self.gan_stats:
            final_g = self.gan_stats[-1].get('avg_g_loss', 0)
            final_d = self.gan_stats[-1].get('avg_d_loss', 0)
            print(f"\n  Финальные потери GAN:")
            print(f"    G Loss: {final_g:.4f}")
            print(f"    D Loss: {final_d:.4f}")

        print("=" * 70 + "\n")

        # Финальная визуализация
        if self.visualize:
            self._generate_visualizations()

        return self.best_fitness_history



    # def run(self, save_to_db: bool = False) -> List[float]:
    #     """
    #     Запускает эволюцию.
    #
    #     Args:
    #         save_to_db: Сохранять ли результаты в БД
    #
    #     Returns:
    #         История лучших фитнесов
    #     """
    #     for gen in range(self.generations):
    #         start_time = time.time()
    #         print(f"=== Поколение {gen + 1}/{self.generations} ===")
    #
    #         # Оцениваем всех особей
    #         fitnesses, experiences = self.population.evaluate_all(self.steps_per_episode)
    #         best_fit = max(fitnesses)
    #         avg_fit = sum(fitnesses) / len(fitnesses)
    #         self.best_fitness_history.append(best_fit)
    #
    #         print(f"Лучший фитнес: {best_fit:.2f}, Средний: {avg_fit:.2f}")
    #         print(f"Собрано переходов: {len(experiences)}")
    #
    #         # Сохраняем статистику в БД (опционально)
    #         if save_to_db:
    #             gen_id = save_generation(gen + 1, best_fit, avg_fit)
    #             best_ind = self.population.select_best(1)[0]
    #             if gen_id:
    #                 save_genome(gen_id, best_ind.genome.to_dict(), best_ind.fitness)
    #
    #         # Собираем данные для визуализации
    #         if self.visualize:
    #             self._collect_training_data(gen + 1, fitnesses, experiences)
    #
    #         # Обучение GAN на собранных данных
    #         if self.gan and experiences:
    #             print(f"Обучение GAN на {len(experiences)} переходах...")
    #             self.gan.add_experiences(experiences)
    #
    #             if len(self.gan.experience_buffer) >= self.gan.batch_size:
    #                 gan_results = self.gan.train(
    #                     epochs=self.gan_training_epochs,
    #                     verbose=(gen % 5 == 0)
    #                 )
    #
    #                 if gan_results and gan_results.get('g_loss') and gan_results.get('d_loss'):
    #                     avg_g_loss = np.mean(gan_results['g_loss'])
    #                     avg_d_loss = np.mean(gan_results['d_loss'])
    #                     min_g_loss = np.min(gan_results['g_loss'])
    #                     max_g_loss = np.max(gan_results['g_loss'])
    #                     min_d_loss = np.min(gan_results['d_loss'])
    #                     max_d_loss = np.max(gan_results['d_loss'])
    #
    #                     stats = {
    #                         'generation': gen + 1,
    #                         'avg_g_loss': avg_g_loss,
    #                         'avg_d_loss': avg_d_loss,
    #                         'min_g_loss': min_g_loss,
    #                         'max_g_loss': max_g_loss,
    #                         'min_d_loss': min_d_loss,
    #                         'max_d_loss': max_d_loss,
    #                         'steps': len(gan_results['g_loss'])
    #                     }
    #                     self.gan_stats.append(stats)
    #
    #                     print(f"\n{'=' * 50}")
    #                     print(f"📊 GAN статистика за поколение {gen + 1}:")
    #                     print(f"{'=' * 50}")
    #                     print(f"  Генератор (G):")
    #                     print(f"    Средняя потеря: {avg_g_loss:.4f}")
    #                     print(f"    Минимальная:    {min_g_loss:.4f}")
    #                     print(f"    Максимальная:   {max_g_loss:.4f}")
    #                     print(f"  Дискриминатор (D):")
    #                     print(f"    Средняя потеря: {avg_d_loss:.4f}")
    #                     print(f"    Минимальная:    {min_d_loss:.4f}")
    #                     print(f"    Максимальная:   {max_d_loss:.4f}")
    #                     print(f"  Количество шагов: {len(gan_results['g_loss'])}")
    #                     print(f"{'=' * 50}\n")
    #
    #                     # Генерируем и оцениваем новые паттерны
    #                     self._generate_and_evaluate_patterns()
    #                     self._integrate_best_patterns()
    #
    #         # Эволюция инстинктов каждые 5 поколений
    #         if gen % 5 == 0 and self.use_gan and self.instinct_engine is not None:
    #             if self.gan and len(self.gan.experience_buffer) > 10:
    #                 try:
    #                     print("\n🧬 Эволюция инстинктов...")
    #                     instinct_history = self.instinct_engine.evolve(
    #                         n_generations=2,
    #                         gan_patterns_per_gen=3
    #                     )
    #                 except Exception as e:
    #                     print(f"⚠️ Ошибка эволюции инстинктов: {e}")
    #
    #         # Создаём следующее поколение (кроме последнего)
    #         if gen < self.generations - 1:
    #             self.population.next_generation(
    #                 elite_count=self.elite_count,
    #                 mutation_rate=self.mutation_rate
    #             )
    #
    #         elapsed = time.time() - start_time
    #         print(f"Время поколения: {elapsed:.2f} сек.\n")
    #
    #     # Финальная визуализация после всех поколений
    #     if self.visualize:
    #         self._generate_visualizations()
    #
    #     return self.best_fitness_history
    #
    # def get_gan_stats(self) -> List[Dict]:
    #     """Возвращает накопленную статистику GAN."""
    #     return self.gan_stats



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