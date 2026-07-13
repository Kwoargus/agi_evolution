# run_evolution.py
import sys
import os

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.world import World
from core.evolution import EvolutionEngine


def main():
    # Создаём мир
    world = World(width=1200, height=800, world_size=20, cell_size=40)

    # Создаём движок эволюции с визуализацией
    engine = EvolutionEngine(
        world=world,
        population_size=20,
        generations=50,
        steps_per_episode=500,
        elite_count=2,
        mutation_rate=0.1,
        use_gan=True,
        gan_training_epochs=5,
        visualize=True  # ← ВКЛЮЧАЕМ ВИЗУАЛИЗАЦИЮ
    )

    # Запускаем эволюцию
    print("🚀 ЗАПУСК ЭВОЛЮЦИИ...")
    best_fitness_history = engine.run(save_to_db=False)

    print("\n✅ ЭВОЛЮЦИЯ ЗАВЕРШЕНА!")
    print(f"Лучший фитнес: {max(best_fitness_history):.2f}")
    print(f"Финальный фитнес: {best_fitness_history[-1]:.2f}")
    print(f"Графики сохранены в папке: training_plots/")


if __name__ == "__main__":
    main()