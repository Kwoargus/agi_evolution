# scripts/compare_evolution_with_gan.py (исправленный)
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from core.world import World
from core.evolution import EvolutionEngine
from models.gan import GAN, PatternRepository
import pandas as pd


class EvolutionComparison:
    """
    Сравнивает эволюцию с GAN и без GAN.
    """

    def __init__(self, world, population_size=10, generations=15, steps_per_episode=200):
        self.world = world
        self.population_size = population_size
        self.generations = generations
        self.steps_per_episode = steps_per_episode

    def run_comparison(self):
        """Запускает два параллельных процесса эволюции."""
        print("=== Запуск сравнения эволюции с GAN и без GAN ===")

        # Эволюция без GAN
        print("\n1. Запуск эволюции БЕЗ GAN...")
        engine_no_gan = EvolutionEngine(
            world=self.world,
            population_size=self.population_size,
            generations=self.generations,
            steps_per_episode=self.steps_per_episode,
            elite_count=2,
            mutation_rate=0.15,
            use_gan=False
        )
        history_no_gan = engine_no_gan.run(save_to_db=False)

        # Эволюция с GAN
        print("\n2. Запуск эволюции С GAN...")
        engine_with_gan = EvolutionEngine(
            world=self.world,
            population_size=self.population_size,
            generations=self.generations,
            steps_per_episode=self.steps_per_episode,
            elite_count=2,
            mutation_rate=0.15,
            use_gan=True,
            gan_training_epochs=5
        )
        history_with_gan = engine_with_gan.run(save_to_db=False)

        # Сохраняем результаты
        self.results = {
            'no_gan': history_no_gan,
            'with_gan': history_with_gan,
            'engine_no_gan': engine_no_gan,
            'engine_with_gan': engine_with_gan
        }

        return self.results

    def visualize_comparison(self):
        """Визуализирует сравнение двух подходов."""
        if not hasattr(self, 'results'):
            print("Сначала запустите run_comparison()")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Сравнение эволюции с GAN и без GAN', fontsize=16)

        # 1. Сравнение фитнеса по поколениям
        ax1 = axes[0, 0]
        ax1.plot(self.results['no_gan'], label='Без GAN', color='red', linewidth=2)
        ax1.plot(self.results['with_gan'], label='С GAN', color='blue', linewidth=2)
        ax1.set_title('Лучший фитнес по поколениям')
        ax1.set_xlabel('Поколение')
        ax1.set_ylabel('Фитнес')
        ax1.legend()
        ax1.grid(True)

        # 2. Разница в фитнесе
        ax2 = axes[0, 1]
        diff = np.array(self.results['with_gan']) - np.array(self.results['no_gan'])
        ax2.bar(range(len(diff)), diff, color='green' if diff[-1] > 0 else 'red', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title('Разница в фитнесе (GAN - без GAN)')
        ax2.set_xlabel('Поколение')
        ax2.set_ylabel('Разница')
        ax2.grid(True)

        # 3. Обучение GAN (потери)
        ax3 = axes[1, 0]
        engine = self.results['engine_with_gan']
        if hasattr(engine, 'gan') and engine.gan:
            g_losses = engine.gan.generator_losses
            d_losses = engine.gan.discriminator_losses
            if g_losses and d_losses:
                ax3.plot(g_losses, label='Generator Loss', alpha=0.7, color='orange')
                ax3.plot(d_losses, label='Discriminator Loss', alpha=0.7, color='purple')
                ax3.set_title('Потери GAN')
                ax3.set_xlabel('Итерация')
                ax3.set_ylabel('Потеря')
                ax3.legend()
                ax3.grid(True)

        # 4. Статистика паттернов
        ax4 = axes[1, 1]
        if hasattr(engine, 'pattern_repository') and engine.pattern_repository:
            scores = engine.pattern_repository.scores
            if scores:
                ax4.hist(scores, bins=20, alpha=0.7, color='teal')
                ax4.axvline(x=0.7, color='red', linestyle='--', label='Порог качества')
                ax4.set_title('Распределение оценок паттернов')
                ax4.set_xlabel('Оценка')
                ax4.set_ylabel('Частота')
                ax4.legend()
                ax4.grid(True)

        plt.tight_layout()
        plt.show()

    def generate_report(self):
        """Генерирует текстовый отчёт о сравнении."""
        if not hasattr(self, 'results'):
            print("Сначала запустите run_comparison()")
            return

        report = []
        report.append("=" * 60)
        report.append("ОТЧЁТ О СРАВНЕНИИ ЭВОЛЮЦИИ")
        report.append("=" * 60)

        no_gan = self.results['no_gan']
        with_gan = self.results['with_gan']

        report.append(f"\n1. Общая информация:")
        report.append(f"   - Поколений: {len(no_gan)}")
        report.append(f"   - Размер популяции: {self.population_size}")

        report.append(f"\n2. Без GAN:")
        report.append(f"   - Лучший фитнес: {max(no_gan):.2f}")
        report.append(f"   - Средний фитнес: {np.mean(no_gan):.2f}")
        report.append(f"   - Стандартное отклонение: {np.std(no_gan):.2f}")
        report.append(f"   - Фитнес в последнем поколении: {no_gan[-1]:.2f}")

        report.append(f"\n3. С GAN:")
        report.append(f"   - Лучший фитнес: {max(with_gan):.2f}")
        report.append(f"   - Средний фитнес: {np.mean(with_gan):.2f}")
        report.append(f"   - Стандартное отклонение: {np.std(with_gan):.2f}")
        report.append(f"   - Фитнес в последнем поколении: {with_gan[-1]:.2f}")

        report.append(f"\n4. Разница (GAN - без GAN):")
        report.append(f"   - Разница в лучшем фитнесе: {max(with_gan) - max(no_gan):.2f}")
        report.append(f"   - Разница в среднем фитнесе: {np.mean(with_gan) - np.mean(no_gan):.2f}")
        report.append(f"   - Разница в последнем поколении: {with_gan[-1] - no_gan[-1]:.2f}")

        if no_gan[-1] > 0:
            improvement = (with_gan[-1] - no_gan[-1]) / no_gan[-1] * 100
            report.append(f"   - Улучшение (в %): {improvement:.1f}%")
        else:
            report.append(f"   - Улучшение (в %): N/A (базовый фитнес = 0)")

        # Дополнительная информация о GAN
        engine = self.results['engine_with_gan']
        if hasattr(engine, 'gan') and engine.gan:
            gan_stats = engine.gan.get_training_stats()
            report.append(f"\n5. Статистика GAN:")
            report.append(f"   - Размер буфера опыта: {gan_stats.get('buffer_size', 0)}")
            report.append(f"   - Всего сгенерировано паттернов: {gan_stats.get('total_generations', 0)}")
            report.append(f"   - Средняя потеря генератора: {gan_stats.get('avg_g_loss', 0):.4f}")
            report.append(f"   - Средняя потеря дискриминатора: {gan_stats.get('avg_d_loss', 0):.4f}")

            if hasattr(engine, 'pattern_repository') and engine.pattern_repository:
                repo = engine.pattern_repository
                report.append(f"\n6. Репозиторий паттернов:")
                report.append(f"   - Сохранено паттернов: {len(repo.patterns)}")
                report.append(f"   - Средняя оценка: {repo.get_average_score():.4f}")
                if repo.scores:
                    report.append(f"   - Максимальная оценка: {max(repo.scores):.4f}")
                    report.append(f"   - Паттернов выше порога: {sum(1 for s in repo.scores if s > 0.7)}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


def run_comparison_and_analyze():
    """Запускает сравнение и выводит результаты."""
    # Создаём мир
    world = World(width=800, height=600, world_size=20, cell_size=40)

    # Добавляем объекты в мир
    from core.objects import GameObject, Predator, Food

    # Огонь
    fire = GameObject(-5, -5, obj_type="fire", temperature=800, size=1.0)
    world.add_object(fire)

    # Хищник
    predator = Predator(8, 8, name="wolf", obj_type="predator", smell="predator_smell")
    world.add_object(predator)

    # Еда
    food1 = Food(-6, 6, name="apple", obj_type="food", smell="food_smell")
    food2 = Food(6, -6, name="apple", obj_type="food", smell="food_smell")
    world.add_object(food1)
    world.add_object(food2)

    # Запускаем сравнение
    comparator = EvolutionComparison(
        world=world,
        population_size=8,
        generations=10,
        steps_per_episode=150
    )

    results = comparator.run_comparison()

    # Визуализируем
    comparator.visualize_comparison()

    # Выводим отчёт
    report = comparator.generate_report()
    print(report)

    # Сохраняем отчёт в файл
    with open('evolution_comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("\nОтчёт сохранён в 'evolution_comparison_report.txt'")


if __name__ == "__main__":
    run_comparison_and_analyze()



# # scripts/compare_evolution_with_gan.py
# import sys
# import os
#
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# import matplotlib.pyplot as plt
# import numpy as np
# from core.world import World
# from core.evolution import EvolutionEngine
# from models.gan import GAN, PatternRepository
# import pandas as pd
#
#
# class EvolutionComparison:
#     """
#     Сравнивает эволюцию с GAN и без GAN.
#     """
#
#     def __init__(self, world, population_size=10, generations=15, steps_per_episode=200):
#         self.world = world
#         self.population_size = population_size
#         self.generations = generations
#         self.steps_per_episode = steps_per_episode
#
#     def run_comparison(self):
#         """Запускает два параллельных процесса эволюции."""
#         print("=== Запуск сравнения эволюции с GAN и без GAN ===")
#
#         # Эволюция без GAN
#         print("\n1. Запуск эволюции БЕЗ GAN...")
#         engine_no_gan = EvolutionEngine(
#             world=self.world,
#             population_size=self.population_size,
#             generations=self.generations,
#             steps_per_episode=self.steps_per_episode,
#             elite_count=2,
#             mutation_rate=0.15,
#             use_gan=False
#         )
#         history_no_gan = engine_no_gan.run(save_to_db=False)
#
#         # Эволюция с GAN
#         print("\n2. Запуск эволюции С GAN...")
#         engine_with_gan = EvolutionEngine(
#             world=self.world,
#             population_size=self.population_size,
#             generations=self.generations,
#             steps_per_episode=self.steps_per_episode,
#             elite_count=2,
#             mutation_rate=0.15,
#             use_gan=True,
#             gan_training_epochs=5
#         )
#         history_with_gan = engine_with_gan.run(save_to_db=False)
#
#         # Сохраняем результаты
#         self.results = {
#             'no_gan': history_no_gan,
#             'with_gan': history_with_gan,
#             'engine_no_gan': engine_no_gan,
#             'engine_with_gan': engine_with_gan
#         }
#
#         return self.results
#
#     def visualize_comparison(self):
#         """Визуализирует сравнение двух подходов."""
#         if not hasattr(self, 'results'):
#             print("Сначала запустите run_comparison()")
#             return
#
#         fig, axes = plt.subplots(2, 2, figsize=(14, 10))
#         fig.suptitle('Сравнение эволюции с GAN и без GAN', fontsize=16)
#
#         # 1. Сравнение фитнеса по поколениям
#         ax1 = axes[0, 0]
#         ax1.plot(self.results['no_gan'], label='Без GAN', color='red', linewidth=2)
#         ax1.plot(self.results['with_gan'], label='С GAN', color='blue', linewidth=2)
#         ax1.set_title('Лучший фитнес по поколениям')
#         ax1.set_xlabel('Поколение')
#         ax1.set_ylabel('Фитнес')
#         ax1.legend()
#         ax1.grid(True)
#
#         # 2. Разница в фитнесе
#         ax2 = axes[0, 1]
#         diff = np.array(self.results['with_gan']) - np.array(self.results['no_gan'])
#         ax2.bar(range(len(diff)), diff, color='green' if diff[-1] > 0 else 'red', alpha=0.7)
#         ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
#         ax2.set_title('Разница в фитнесе (GAN - без GAN)')
#         ax2.set_xlabel('Поколение')
#         ax2.set_ylabel('Разница')
#         ax2.grid(True)
#
#         # 3. Обучение GAN (потери)
#         ax3 = axes[1, 0]
#         engine = self.results['engine_with_gan']
#         if hasattr(engine, 'gan') and engine.gan:
#             g_losses = engine.gan.generator_losses
#             d_losses = engine.gan.discriminator_losses
#             if g_losses and d_losses:
#                 ax3.plot(g_losses, label='Generator Loss', alpha=0.7, color='orange')
#                 ax3.plot(d_losses, label='Discriminator Loss', alpha=0.7, color='purple')
#                 ax3.set_title('Потери GAN')
#                 ax3.set_xlabel('Итерация')
#                 ax3.set_ylabel('Потеря')
#                 ax3.legend()
#                 ax3.grid(True)
#
#         # 4. Статистика паттернов
#         ax4 = axes[1, 1]
#         if hasattr(engine, 'pattern_repository') and engine.pattern_repository:
#             scores = engine.pattern_repository.scores
#             if scores:
#                 ax4.hist(scores, bins=20, alpha=0.7, color='teal')
#                 ax4.axvline(x=0.7, color='red', linestyle='--', label='Порог качества')
#                 ax4.set_title('Распределение оценок паттернов')
#                 ax4.set_xlabel('Оценка')
#                 ax4.set_ylabel('Частота')
#                 ax4.legend()
#                 ax4.grid(True)
#
#         plt.tight_layout()
#         plt.show()
#
#     def generate_report(self):
#         """Генерирует текстовый отчёт о сравнении."""
#         if not hasattr(self, 'results'):
#             print("Сначала запустите run_comparison()")
#             return
#
#         report = []
#         report.append("=" * 60)
#         report.append("ОТЧЁТ О СРАВНЕНИИ ЭВОЛЮЦИИ")
#         report.append("=" * 60)
#
#         no_gan = self.results['no_gan']
#         with_gan = self.results['with_gan']
#
#         report.append(f"\n1. Общая информация:")
#         report.append(f"   - Поколений: {len(no_gan)}")
#         report.append(f"   - Размер популяции: {self.population_size}")
#
#         report.append(f"\n2. Без GAN:")
#         report.append(f"   - Лучший фитнес: {max(no_gan):.2f}")
#         report.append(f"   - Средний фитнес: {np.mean(no_gan):.2f}")
#         report.append(f"   - Стандартное отклонение: {np.std(no_gan):.2f}")
#         report.append(f"   - Фитнес в последнем поколении: {no_gan[-1]:.2f}")
#
#         report.append(f"\n3. С GAN:")
#         report.append(f"   - Лучший фитнес: {max(with_gan):.2f}")
#         report.append(f"   - Средний фитнес: {np.mean(with_gan):.2f}")
#         report.append(f"   - Стандартное отклонение: {np.std(with_gan):.2f}")
#         report.append(f"   - Фитнес в последнем поколении: {with_gan[-1]:.2f}")
#
#         report.append(f"\n4. Разница (GAN - без GAN):")
#         report.append(f"   - Разница в лучшем фитнесе: {max(with_gan) - max(no_gan):.2f}")
#         report.append(f"   - Разница в среднем фитнесе: {np.mean(with_gan) - np.mean(no_gan):.2f}")
#         report.append(f"   - Разница в последнем поколении: {with_gan[-1] - no_gan[-1]:.2f}")
#
#         improvement = (with_gan[-1] - no_gan[-1]) / (no_gan[-1] + 1e-8) * 100
#         report.append(f"   - Улучшение (в %): {improvement:.1f}%")
#
#         # Дополнительная информация о GAN
#         engine = self.results['engine_with_gan']
#         if hasattr(engine, 'gan') and engine.gan:
#             gan_stats = engine.gan.get_training_stats()
#             report.append(f"\n5. Статистика GAN:")
#             report.append(f"   - Размер буфера опыта: {gan_stats.get('buffer_size', 0)}")
#             report.append(f"   - Всего сгенерировано паттернов: {gan_stats.get('total_generations', 0)}")
#             report.append(f"   - Средняя потеря генератора: {gan_stats.get('avg_g_loss', 0):.4f}")
#             report.append(f"   - Средняя потеря дискриминатора: {gan_stats.get('avg_d_loss', 0):.4f}")
#
#             if hasattr(engine, 'pattern_repository') and engine.pattern_repository:
#                 repo = engine.pattern_repository
#                 report.append(f"\n6. Репозиторий паттернов:")
#                 report.append(f"   - Сохранено паттернов: {len(repo.patterns)}")
#                 report.append(f"   - Средняя оценка: {repo.get_average_score():.4f}")
#                 if repo.scores:
#                     report.append(f"   - Максимальная оценка: {max(repo.scores):.4f}")
#                     report.append(f"   - Паттернов выше порога: {sum(1 for s in repo.scores if s > 0.7)}")
#
#         report.append("\n" + "=" * 60)
#
#         return "\n".join(report)
#
#
# def run_comparison_and_analyze():
#     """Запускает сравнение и выводит результаты."""
#     # Создаём мир
#     world = World(width=800, height=600, world_size=20, cell_size=40)
#
#     # Добавляем объекты в мир
#     from core.objects import GameObject, Predator, Food
#
#     # Огонь
#     fire = GameObject(-5, -5, obj_type="fire", temperature=800, size=1.0)
#     world.add_object(fire)
#
#     # Хищник
#     predator = Predator(8, 8, name="wolf", obj_type="predator", smell="predator_smell")
#     world.add_object(predator)
#
#     # Еда
#     food1 = Food(-6, 6, name="apple", obj_type="food", smell="food_smell")
#     food2 = Food(6, -6, name="apple", obj_type="food", smell="food_smell")
#     world.add_object(food1)
#     world.add_object(food2)
#
#     # Запускаем сравнение
#     comparator = EvolutionComparison(
#         world=world,
#         population_size=8,
#         generations=10,
#         steps_per_episode=150
#     )
#
#     results = comparator.run_comparison()
#
#     # Визуализируем
#     comparator.visualize_comparison()
#
#     # Выводим отчёт
#     report = comparator.generate_report()
#     print(report)
#
#     # Сохраняем отчёт в файл
#     with open('evolution_comparison_report.txt', 'w', encoding='utf-8') as f:
#         f.write(report)
#     print("\nОтчёт сохранён в 'evolution_comparison_report.txt'")
#
#
# if __name__ == "__main__":
#     run_comparison_and_analyze()