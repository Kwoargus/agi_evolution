# run_evolution.py
import sys
import os

# [ru] Добавляем путь к проекту
# [en] Add the project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.world import World
from core.evolution import EvolutionEngine


def main():
    # [ru] Создаём мир
    # [en] Create the world
    objects_config = [
        # [ru] Еда (разбросана по всему миру)
        # [en] Food (scattered around the world)
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

        # [ru] Хищники (опасность)
        # [en] Predators (danger)
        {'type': 'predator', 'x': 32, 'z': 32},
        {'type': 'predator', 'x': 34, 'z': 34},
        {'type': 'predator', 'x': 36, 'z': 36},

        # [ru] Огонь (опасность)
        # [en] Fire (danger)
        {'type': 'fire', 'x': 38, 'z': 38},
        {'type': 'fire', 'x': 40, 'z': 40},
        {'type': 'fire', 'x': 42, 'z': 42},
    ]

    world = World(
        width=1200,
        height=800,
        # [ru] ← УВЕЛИЧИВАЕМ мир
        # [en] ← INCREASING the world
        world_size=50,
        cell_size=40,
        objects_config=objects_config
    )


    # [ru] Создаём движок эволюции с визуализацией
    # [en] Create the evolution engine with visualization
    engine = EvolutionEngine(
        world=world,
        population_size=20,
        generations=50,
        # [ru] ← БЫЛО 500, СТАЛО 1000
        # [en] ← WAS 500, NOW 1000
        steps_per_episode=1000,
        elite_count=2,
        # [ru] ← УВЕЛИЧИВАЕМ МУТАЦИЮ
        # [en] ← INCREASING MUTATION
        mutation_rate=0.15,
        use_gan=True,
        gan_training_epochs=10,
        visualize=True
    )


    # [ru] Запускаем эволюцию
    # [en] Run the evolution
    print("[ru] ЗАПУСК ЭВОЛЮЦИИ...")
    print("[en] STARTING EVOLUTION...")
    best_fitness_history = engine.run(save_to_db=False)

    print("[ru] \n ЭВОЛЮЦИЯ ЗАВЕРШЕНА!")
    print("[en] \n EVOLUTION COMPLETED!")

    print(f"[ru] Лучший фитнес: {max(best_fitness_history):.2f}")
    print(f"[ru] Финальный фитнес: {best_fitness_history[-1]:.2f}")
    print(f"[ru] Графики сохранены в папке: training_plots/")

    print(f"[en] Best fitness: {max(best_fitness_history):.2f}")
    print(f"[en] Final fitness: {best_fitness_history[-1]:.2f}")
    print(f"[en] Graphs saved in folder: training_plots/")


if __name__ == "__main__":
    main()



# # run_evolution.py
# import sys
# import os
#
# # Добавляем путь к проекту
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#
# from core.world import World
# from core.evolution import EvolutionEngine
#
#
# def main():
#     # Создаём мир
#     objects_config = [
#         # Еда (разбросана по всему миру)
#         {'type': 'food', 'x': 2, 'z': 2},
#         {'type': 'food', 'x': 4, 'z': 4},
#         {'type': 'food', 'x': 6, 'z': 6},
#         {'type': 'food', 'x': 8, 'z': 8},
#         {'type': 'food', 'x': 10, 'z': 10},
#         {'type': 'food', 'x': 12, 'z': 12},
#         {'type': 'food', 'x': 14, 'z': 14},
#         {'type': 'food', 'x': 16, 'z': 16},
#         {'type': 'food', 'x': 18, 'z': 18},
#         {'type': 'food', 'x': 20, 'z': 20},
#         {'type': 'food', 'x': 22, 'z': 22},
#         {'type': 'food', 'x': 24, 'z': 24},
#         {'type': 'food', 'x': 26, 'z': 26},
#         {'type': 'food', 'x': 28, 'z': 28},
#         {'type': 'food', 'x': 30, 'z': 30},
#
#         # Хищники (опасность)
#         {'type': 'predator', 'x': 32, 'z': 32},
#         {'type': 'predator', 'x': 34, 'z': 34},
#         {'type': 'predator', 'x': 36, 'z': 36},
#
#         # Огонь (опасность)
#         {'type': 'fire', 'x': 38, 'z': 38},
#         {'type': 'fire', 'x': 40, 'z': 40},
#         {'type': 'fire', 'x': 42, 'z': 42},
#     ]
#
#     world = World(
#         width=1200,
#         height=800,
#         world_size=50,  # ← УВЕЛИЧИВАЕМ мир
#         cell_size=40,
#         objects_config=objects_config
#     )
#
#
#     # Создаём движок эволюции с визуализацией
#     engine = EvolutionEngine(
#         world=world,
#         population_size=20,
#         generations=50,
#         steps_per_episode=1000,  # ← БЫЛО 500, СТАЛО 1000
#         elite_count=2,
#         mutation_rate=0.15,  # ← УВЕЛИЧИВАЕМ МУТАЦИЮ
#         use_gan=True,
#         gan_training_epochs=10,
#         visualize=True
#     )
#
#
#     # Запускаем эволюцию
#     print("🚀 ЗАПУСК ЭВОЛЮЦИИ...")
#     best_fitness_history = engine.run(save_to_db=False)
#
#     print("\n✅ ЭВОЛЮЦИЯ ЗАВЕРШЕНА!")
#     print(f"Лучший фитнес: {max(best_fitness_history):.2f}")
#     print(f"Финальный фитнес: {best_fitness_history[-1]:.2f}")
#     print(f"Графики сохранены в папке: training_plots/")
#
#
# if __name__ == "__main__":
#     main()