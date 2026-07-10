# core/evolution.py (исправленный)
from core.population import Population
from db.connector import save_generation, save_genome
from models.gan import GAN, PatternEvaluator, PatternRepository
import time
import numpy as np


class EvolutionEngine:
    def __init__(self, world, population_size=20, generations=50,
                 steps_per_episode=500, elite_count=2, mutation_rate=0.1,
                 use_gan=True, gan_training_epochs=5):
        self.world = world
        self.population = Population(population_size, world)
        self.generations = generations
        self.steps_per_episode = steps_per_episode
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.best_fitness_history = []
        self.gan_stats = []

        # ============================================================
        # 1. СНАЧАЛА ИНИЦИАЛИЗИРУЕМ GAN (основной)
        # ============================================================
        self.use_gan = use_gan
        self.gan_training_epochs = gan_training_epochs

        if use_gan:
            self.gan = GAN(
                latent_dim=128,
                pattern_dim=None,  # Автоматическое вычисление (21)
                batch_size=16,
                state_dim=8,
                action_dim=4
            )
            self.pattern_repository = PatternRepository()
            self.pattern_evaluator = PatternEvaluator(
                self.gan.discriminator,
                self.gan.encoder
            )

            # ============================================================
            # 2. ПОСЛЕ СОЗДАНИЯ GAN - ИНИЦИАЛИЗИРУЕМ ИНСТИНКТЫ
            # ============================================================
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
                experience_buffer=self.gan.experience_buffer  # Теперь self.gan уже существует
            )
        else:
            self.gan = None
            self.pattern_repository = None
            self.pattern_evaluator = None
            self.instinct_gan = None
            self.instinct_engine = None

    def run(self, save_to_db=False):
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

            # Обучение GAN на собранных данных
            if self.gan and experiences:
                print(f"Обучение GAN на {len(experiences)} переходах...")
                self.gan.add_experiences(experiences)

                if len(self.gan.experience_buffer) >= self.gan.batch_size:
                    gan_results = self.gan.train(
                        epochs=self.gan_training_epochs,
                        verbose=(gen % 5 == 0)
                    )

                    if gan_results['g_loss'] and gan_results['d_loss']:
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

        return self.best_fitness_history

    def _generate_and_evaluate_patterns(self, n_patterns: int = 20):
        """Генерирует и оценивает новые паттерны поведения."""
        if not self.gan:
            return

        try:
            patterns = self.gan.generate_batch(n_patterns)
            for pattern in patterns:
                score = self.pattern_evaluator.evaluate_pattern(pattern)
                if score >= 0.5:
                    self.pattern_repository.add_pattern(pattern, score)
            print(f"Сгенерировано {n_patterns} паттернов, сохранено {len(self.pattern_repository.patterns)}")
        except Exception as e:
            print(f"⚠️ Ошибка при генерации паттернов: {e}")

    def _integrate_best_patterns(self):
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
                        print(f"Добавлено новое правило рефлекса: {rule}")
        except Exception as e:
            print(f"⚠️ Ошибка при интеграции паттернов: {e}")

    def get_gan_stats(self):
        """Возвращает накопленную статистику GAN."""
        return self.gan_stats