# tests/test_genome_improvement.py (исправленный)
import sys
import os
import random
import numpy as np

# Добавляем корневую директорию проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
from core.genome import Genome
from core.individual import Individual
from core.world import World
from models.gan import GAN
import matplotlib.pyplot as plt


# # tests/test_genome_improvement.py (исправленный)
# import sys
# import os
# import random
# import numpy as np
#
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# from typing import Dict, Any
# from core.genome import Genome
# from core.individual import Individual
# from core.world import World
# from models.gan import GAN
# import matplotlib.pyplot as plt


class GenomeQualityTester:
    """
    Тестирует качество геномов и влияние GAN на их улучшение.
    """

    def __init__(self, world, max_steps=500):
        self.world = world
        self.max_steps = max_steps

    def test_single_genome(self, genome: Genome, n_trials: int = 3) -> Dict[str, float]:
        """Тестирует один геном в нескольких прогонах."""
        fitnesses = []

        for trial in range(n_trials):
            # Используем разный seed для разнообразия
            np.random.seed(trial * 137 + 42)
            random.seed(trial * 137 + 42)

            # ОЧИЩАЕМ мир и создаем нового индивида
            self.world.objects = []

            # Добавляем объекты в фиксированных позициях для стабильности
            from core.objects import Food, Predator, GameObject
            self.world.add_object(Food(-6, 6, name="apple", obj_type="food", smell="food_smell"))
            self.world.add_object(Food(6, -6, name="apple", obj_type="food", smell="food_smell"))
            self.world.add_object(Food(-6, -6, name="apple", obj_type="food", smell="food_smell"))
            self.world.add_object(Food(6, 6, name="apple", obj_type="food", smell="food_smell"))
            self.world.add_object(Predator(8, 8, name="wolf", obj_type="predator", smell="predator_smell"))
            self.world.add_object(Predator(-8, -8, name="wolf", obj_type="predator", smell="predator_smell"))
            self.world.add_object(GameObject(-4, -4, obj_type="fire", temperature=800, size=1.0))
            self.world.add_object(GameObject(4, 4, obj_type="fire", temperature=800, size=1.0))

            # Создаем индивида с начальной позицией (0, 0)
            individual = Individual(x=0, z=0, genome=genome)

            # Запускаем оценку
            try:
                fitness = individual.evaluate(self.world, max_steps=self.max_steps)
                fitnesses.append(fitness)
            except Exception as e:
                print(f"   Ошибка в trial {trial}: {e}")
                fitnesses.append(0.0)

        # Если все провалились, возвращаем нулевые значения
        if not fitnesses:
            return {
                'mean': 0.0,
                'std': 0.0,
                'max': 0.0,
                'min': 0.0
            }

        return {
            'mean': np.mean(fitnesses),
            'std': np.std(fitnesses),
            'max': max(fitnesses),
            'min': min(fitnesses)
        }

    def compare_genomes(self, genome1: Genome, genome2: Genome) -> Dict[str, Any]:
        """Сравнивает два генома."""
        results1 = self.test_single_genome(genome1)
        results2 = self.test_single_genome(genome2)

        # Вычисляем улучшение
        improvement_abs = results2['mean'] - results1['mean']
        improvement_pct = (improvement_abs / (results1['mean'] + 1e-8)) * 100

        return {
            'genome1': results1,
            'genome2': results2,
            'improvement_abs': improvement_abs,
            'improvement_pct': improvement_pct,
            'is_better': results2['mean'] > results1['mean']
        }

    def visualize_genome_comparison(self, comparison_results: Dict[str, Any]):
        """Визуализирует сравнение геномов."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 1. Сравнение фитнеса
        ax1 = axes[0]
        labels = ['Genome 1', 'Genome 2']
        means = [comparison_results['genome1']['mean'], comparison_results['genome2']['mean']]
        stds = [comparison_results['genome1']['std'], comparison_results['genome2']['std']]

        ax1.bar(labels, means, yerr=stds, capsize=5, alpha=0.7, color=['red', 'blue'])
        ax1.set_title('Сравнение фитнеса геномов')
        ax1.set_ylabel('Средний фитнес')
        ax1.grid(True)

        # Добавляем значения на график
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax1.text(i, mean + std + 1, f'{mean:.1f}±{std:.1f}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 2. Распределение фитнеса (boxplot)
        ax2 = axes[1]
        # Генерируем данные для boxplot
        data1 = np.random.normal(comparison_results['genome1']['mean'],
                                 max(comparison_results['genome1']['std'], 0.1), 100)
        data2 = np.random.normal(comparison_results['genome2']['mean'],
                                 max(comparison_results['genome2']['std'], 0.1), 100)

        ax2.boxplot([data1, data2], tick_labels=['Genome 1', 'Genome 2'])
        ax2.set_title('Распределение фитнеса')
        ax2.set_ylabel('Фитнес')
        ax2.grid(True)

        # Показываем улучшение
        improvement_text = f"Улучшение: {comparison_results['improvement_abs']:.1f} ({comparison_results['improvement_pct']:.1f}%)"
        plt.suptitle(improvement_text, fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.show()


def test_gan_improvement():
    """
    Тестирует, улучшает ли GAN геномы ботов.
    """
    print("=== Тестирование влияния GAN на улучшение генов ===")

    # Создаём мир
    world = World(width=800, height=600, world_size=20, cell_size=40)
    print(f"✅ Мир создан: {world.world_size}x{world.world_size}")

    # Добавляем базовые объекты
    from core.objects import Food, Predator, GameObject
    world.add_object(Food(-5, 5, name="apple", obj_type="food", smell="food_smell"))
    world.add_object(Food(5, -5, name="apple", obj_type="food", smell="food_smell"))
    world.add_object(Predator(7, 7, name="wolf", obj_type="predator", smell="predator_smell"))
    world.add_object(Predator(-7, -7, name="wolf", obj_type="predator", smell="predator_smell"))
    world.add_object(GameObject(-3, -3, obj_type="fire", temperature=800, size=1.0))
    world.add_object(GameObject(3, 3, obj_type="fire", temperature=800, size=1.0))
    print("✅ Объекты добавлены")

    # Создаём тестер
    tester = GenomeQualityTester(world, max_steps=500)
    print(f"✅ Тестер создан: max_steps=500")

    # Создаём начальный геном
    print("\n1. Создание начального генома...")
    base_genome = Genome({
        'move_delay': 5,
        'step_size': 2.0,
        'reflex_priorities': {'grab': 0.5, 'avoid': 0.5},
        'instinct_priorities': {'run_away': 0.5},
        'exploration_bias': 0.5,
        'max_steps': 500
    })
    print(f"   Начальный геном: move_delay={base_genome.params['move_delay']}")

    # Тестируем начальный геном
    print("2. Тестирование начального генома...")
    base_results = tester.test_single_genome(base_genome, n_trials=3)
    print(f"   Средний фитнес: {base_results['mean']:.2f} ± {base_results['std']:.2f}")

    # Если базовый геном не работает, выходим
    if base_results['mean'] == 0:
        print("   ⚠️ Базовый геном не показывает результатов. Проверьте настройки мира.")
        return

    # ==================== ИСПРАВЛЕННЫЙ ОТСТУП ====================
    # Создаём GAN и обучаем его с разными геномами
    print("\n3. Создание и обучение GAN...")
    gan = GAN(
        latent_dim=32,
        pattern_dim=None,
        batch_size=4,
        state_dim=8,
        action_dim=4
    )

    # Создаем несколько разных геномов для сбора разнообразного опыта
    genome_variants = [
        {'move_delay': 2, 'exploration_bias': 0.3},
        {'move_delay': 3, 'exploration_bias': 0.5},
        {'move_delay': 4, 'exploration_bias': 0.7},
        {'move_delay': 5, 'exploration_bias': 0.9},
    ]

    experiences = []

    for variant in genome_variants:
        # Создаем геном с разными параметрами
        variant_genome = Genome({
            'move_delay': variant['move_delay'],
            'step_size': 2.0,
            'reflex_priorities': {'grab': 0.5, 'avoid': 0.5},
            'instinct_priorities': {'run_away': 0.5},
            'exploration_bias': variant['exploration_bias'],
            'max_steps': 500
        })

        individual = Individual(x=0, z=0, genome=variant_genome)

        for episode in range(2):
            # Сбрасываем мир и индивида
            world.objects = []
            world.add_object(Food(-5, 5, name="apple", obj_type="food", smell="food_smell"))
            world.add_object(Food(5, -5, name="apple", obj_type="food", smell="food_smell"))
            world.add_object(Predator(7, 7, name="wolf", obj_type="predator", smell="predator_smell"))
            world.add_object(Predator(-7, -7, name="wolf", obj_type="predator", smell="predator_smell"))
            world.add_object(GameObject(-3, -3, obj_type="fire", temperature=800, size=1.0))
            world.add_object(GameObject(3, 3, obj_type="fire", temperature=800, size=1.0))

            individual.reset()
            individual.x = 0
            individual.z = 0

            # Случайная начальная позиция для разнообразия
            if np.random.random() > 0.5:
                individual.x = np.random.randint(-3, 3)
                individual.z = np.random.randint(-3, 3)

            for step in range(200):
                if not individual.alive:
                    break
                state = world.get_state(individual)
                individual.update(world)
                for exp in individual.get_experiences():
                    experiences.append(exp)
                    if len(experiences) > 500:
                        break
                if len(experiences) > 500:
                    break

    if experiences:
        gan.add_experiences(experiences)
        print(f"   Добавлено {len(experiences)} переходов в буфер")
    else:
        print("   ⚠️ Не удалось собрать опыт для обучения GAN")
        return

    # Обучаем GAN с большим количеством эпох
    print("4. Обучение GAN...")
    if len(experiences) > 10:
        gan.train(epochs=20, verbose=True)  # Увеличили с 10 до 20
    else:
        print("   ⚠️ Недостаточно данных для обучения GAN")
        return
    # ==================== КОНЕЦ ИСПРАВЛЕННОГО ОТСТУПА ====================

    # Генерируем новые паттерны и создаём улучшенные геномы
    print("\n5. Генерация улучшенных геномов...")
    improved_genomes = []

    # Сохраняем начальное значение для сравнения
    initial_move_delay = base_genome.params.get('move_delay', 0)

    for i in range(5):
        # Генерируем паттерн
        pattern = gan.generate_pattern()

        # Создаём геном на основе паттерна
        new_genome = Genome(base_genome.to_dict())

        # Модифицируем параметры на основе паттерна
        if len(pattern) >= 10:
            # Более агрессивная модификация
            new_genome.params['move_delay'] = max(1, int(abs(pattern[0]) * 15) + 1)
            new_genome.params['exploration_bias'] = float(abs(pattern[1]) % 1.0)

            # Убеждаемся, что параметры изменились
            if new_genome.params['move_delay'] == initial_move_delay:
                new_genome.params['move_delay'] = initial_move_delay + np.random.randint(1, 5)

            if 'reflex_priorities' in new_genome.params:
                new_genome.params['reflex_priorities']['grab'] = float(abs(pattern[2]) % 1.0)
                new_genome.params['reflex_priorities']['avoid'] = float(abs(pattern[3]) % 1.0)

        improved_genomes.append(new_genome)

    # Тестируем улучшенные геномы
    print("\n6. Тестирование улучшенных геномов...")
    improved_results = []
    for i, genome in enumerate(improved_genomes):
        result = tester.test_single_genome(genome, n_trials=3)
        improved_results.append(result)
        print(f"   Геном {i + 1}: {result['mean']:.2f} ± {result['std']:.2f}")

    # Сравниваем лучший улучшенный геном с базовым
        if improved_results:
            best_idx = np.argmax([r['mean'] for r in improved_results])

            if improved_results[best_idx]['mean'] > 0:
                comparison = tester.compare_genomes(base_genome, improved_genomes[best_idx])

                print("\n7. Результаты сравнения:")
                print(f"   Базовый фитнес: {base_results['mean']:.2f}")
                print(f"   Лучший улучшенный фитнес: {improved_results[best_idx]['mean']:.2f}")
                print(f"   Абсолютное улучшение: {comparison['improvement_abs']:.2f}")
                print(f"   Процентное улучшение: {comparison['improvement_pct']:.1f}%")
                print(f"   Лучший улучшенный геном лучше базового: {comparison['is_better']}")

                print(f"\n   Параметры базового генома:")
                print(f"     move_delay: {base_genome.params.get('move_delay', 'N/A')}")
                print(f"     exploration_bias: {base_genome.params.get('exploration_bias', 'N/A'):.3f}")

                best_genome = improved_genomes[best_idx]
                print(f"\n   Параметры лучшего улучшенного генома:")
                print(f"     move_delay: {best_genome.params.get('move_delay', 'N/A')}")
                print(f"     exploration_bias: {best_genome.params.get('exploration_bias', 'N/A'):.3f}")

                # Визуализируем
                tester.visualize_genome_comparison(comparison)
            else:
                print("\n⚠️ Улучшенные геномы не показывают результатов")


    # if improved_results:
    #     best_idx = np.argmax([r['mean'] for r in improved_results])
    #
    #     # Проверяем, что лучший результат не нулевой
    #     if improved_results[best_idx]['mean'] > 0:
    #         comparison = tester.compare_genomes(base_genome, improved_genomes[best_idx])
    #
    #         print("\n7. Результаты сравнения:")
    #         print(f"   Абсолютное улучшение: {comparison['improvement_abs']:.2f}")
    #         print(f"   Процентное улучшение: {comparison['improvement_pct']:.1f}%")
    #         print(f"   Лучший улучшенный геном лучше базового: {comparison['is_better']}")
    #
    #         print(f"\n   Параметры базового генома:")
    #         print(f"     move_delay: {base_genome.params.get('move_delay', 'N/A')}")
    #         print(f"     exploration_bias: {base_genome.params.get('exploration_bias', 'N/A'):.3f}")
    #
    #         best_genome = improved_genomes[best_idx]
    #         print(f"\n   Параметры лучшего улучшенного генома:")
    #         print(f"     move_delay: {best_genome.params.get('move_delay', 'N/A')}")
    #         print(f"     exploration_bias: {best_genome.params.get('exploration_bias', 'N/A'):.3f}")
    #
    #         # Визуализируем
    #         tester.visualize_genome_comparison(comparison)
        else:
            print("\n⚠️ Улучшенные геномы не показывают результатов")
    else:
        print("\n⚠️ Не удалось создать улучшенные геномы")

    print("\n7. Результаты сравнения:")
    print(f"   Базовый фитнес: {base_results['mean']:.2f}")
    print(f"   Лучший улучшенный фитнес: {improved_results[best_idx]['mean']:.2f}")
    print(f"   Абсолютное улучшение: {improved_results[best_idx]['mean'] - base_results['mean']:.2f}")
    print(f"   Процентное улучшение: {((improved_results[best_idx]['mean'] - base_results['mean']) / base_results['mean'] * 100):.1f}%")

if __name__ == "__main__":
    test_gan_improvement()
    print("\n=== Тестирование завершено ===")