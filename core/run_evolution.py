# run_evolution.py
import sys
import os

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.world import World
from core.evolution import EvolutionEngine


def main():
    # Создаём мир
    objects_config = [
        # Еда (разбросана по всему миру)
        {'type': 'food', 'x': 2, 'z': 2},
        {'type': 'food', 'x': 4, 'z': 4},
        {'type': 'food', 'x': 6, 'z': 6},
        {'type': 'food', 'x': 8, 'z': 8},
        {'type': 'food', 'x': 10, 'z': 10},
        {'type': 'food', 'x': 12, 'z': 12},
        {'type': 'food', 'x': 14, 'z': 14},
        {'type': 'food', 'x': 16, 'z': 16},
        {'type': 'food', 'x': 18, 'z': 18},
        {'type': 'food', 'x': 20, 'z': 20},
        {'type': 'food', 'x': 22, 'z': 22},
        {'type': 'food', 'x': 24, 'z': 24},
        {'type': 'food', 'x': 26, 'z': 26},
        {'type': 'food', 'x': 28, 'z': 28},
        {'type': 'food', 'x': 30, 'z': 30},

        # Хищники (опасность)
        {'type': 'predator', 'x': 32, 'z': 32},
        {'type': 'predator', 'x': 34, 'z': 34},
        {'type': 'predator', 'x': 36, 'z': 36},

        # Огонь (опасность)
        {'type': 'fire', 'x': 38, 'z': 38},
        {'type': 'fire', 'x': 40, 'z': 40},
        {'type': 'fire', 'x': 42, 'z': 42},
    ]

    world = World(
        width=1200,
        height=800,
        world_size=50,  # ← УВЕЛИЧИВАЕМ мир
        cell_size=40,
        objects_config=objects_config
    )







    # # Создаём мир с объектами
    # objects_config = [
    #     {'type': 'food', 'x': 2, 'z': 2},
    #     {'type': 'food', 'x': 4, 'z': 4},
    #     {'type': 'food', 'x': 6, 'z': 6},
    #     {'type': 'food', 'x': 8, 'z': 8},
    #     {'type': 'food', 'x': 10, 'z': 10},
    #     {'type': 'food', 'x': 12, 'z': 12},
    #     {'type': 'food', 'x': 14, 'z': 14},
    #     {'type': 'food', 'x': 16, 'z': 16},
    #     {'type': 'predator', 'x': 18, 'z': 18},
    #     {'type': 'predator', 'x': 20, 'z': 20},
    #     {'type': 'fire', 'x': 22, 'z': 22},
    # ]
    #
    # world = World(width=1200, height=800, world_size=30, cell_size=40,
    #               objects_config=objects_config)

    # world = World(width=1200, height=800, world_size=20, cell_size=40)

    # Создаём движок эволюции с визуализацией
    engine = EvolutionEngine(
        world=world,
        population_size=20,
        generations=50,
        steps_per_episode=1000,  # ← БЫЛО 500, СТАЛО 1000
        elite_count=2,
        mutation_rate=0.15,  # ← УВЕЛИЧИВАЕМ МУТАЦИЮ
        use_gan=True,
        gan_training_epochs=10,
        visualize=True
    )



    # engine = EvolutionEngine(
    #     world=world,
    #     population_size=20,
    #     generations=50,
    #     steps_per_episode=500,
    #     elite_count=2,
    #     mutation_rate=0.1,
    #     use_gan=True,
    #     gan_training_epochs=5,
    #     visualize=True  # ← ВКЛЮЧАЕМ ВИЗУАЛИЗАЦИЮ
    # )

    # Запускаем эволюцию
    print("🚀 ЗАПУСК ЭВОЛЮЦИИ...")
    best_fitness_history = engine.run(save_to_db=False)

    print("\n✅ ЭВОЛЮЦИЯ ЗАВЕРШЕНА!")
    print(f"Лучший фитнес: {max(best_fitness_history):.2f}")
    print(f"Финальный фитнес: {best_fitness_history[-1]:.2f}")
    print(f"Графики сохранены в папке: training_plots/")


if __name__ == "__main__":
    main()