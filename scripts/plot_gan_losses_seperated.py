import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np


def plot_gan_losses(g_losses, d_losses, title="GAN Training Losses", save_path=None):
    """
    Визуализирует потери GAN.

    Args:
        g_losses: Список потерь генератора
        d_losses: Список потерь дискриминатора
        title: Заголовок графика
        save_path: Путь для сохранения графика (опционально)
    """
    if not g_losses or not d_losses:
        print("⚠️ Нет данных для визуализации")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # График потерь
    ax1.plot(g_losses, label='Generator Loss', color='blue', alpha=0.7)
    ax1.plot(d_losses, label='Discriminator Loss', color='red', alpha=0.7)
    ax1.set_title('Потери GAN')
    ax1.set_xlabel('Шаг обучения')
    ax1.set_ylabel('Потеря')
    ax1.legend()
    ax1.grid(True)

    # Статистика на графике
    ax1.text(0.02, 0.98,
             f'G: min={np.min(g_losses):.3f}, max={np.max(g_losses):.3f}, mean={np.mean(g_losses):.3f}\n'
             f'D: min={np.min(d_losses):.3f}, max={np.max(d_losses):.3f}, mean={np.mean(d_losses):.3f}',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Распределение потерь
    ax2.hist(g_losses, bins=30, alpha=0.5, label='Generator', color='blue')
    ax2.hist(d_losses, bins=30, alpha=0.5, label='Discriminator', color='red')
    ax2.set_title('Распределение потерь')
    ax2.set_xlabel('Потеря')
    ax2.set_ylabel('Частота')
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ График сохранён в {save_path}")

    plt.show()


def plot_fitness_history(history, title="История фитнеса", save_path=None):
    """Визуализирует историю фитнеса."""
    plt.figure(figsize=(10, 6))
    plt.plot(history, marker='o', linewidth=2, color='green')
    plt.title(title)
    plt.xlabel('Поколение')
    plt.ylabel('Лучший фитнес')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ График фитнеса сохранён в {save_path}")

    plt.show()


# Пример использования после завершения эволюции:
if __name__ == "__main__":
    # Этот код можно запустить отдельно, загрузив сохраненные данные
    # или использовать вместе с evolution.py

    from core.world import World
    from core.evolution import EvolutionEngine

    # Создаём мир
    world = World(width=800, height=600, world_size=20, cell_size=40)

    # Добавляем объекты
    from core.objects import GameObject, Predator, Food

    world.add_object(GameObject(-5, -5, obj_type="fire", temperature=800, size=1.0))
    world.add_object(Predator(5, 5, name="wolf", obj_type="predator", smell="predator_smell"))
    world.add_object(Food(-6, 6, name="apple", obj_type="food", smell="food_smell"))
    world.add_object(Food(6, -6, name="apple", obj_type="food", smell="food_smell"))

    # Запускаем эволюцию
    engine = EvolutionEngine(
        world=world,
        population_size=10,
        generations=15,
        steps_per_episode=200,
        elite_count=2,
        mutation_rate=0.15,
        use_gan=True,
        gan_training_epochs=3
    )

    history = engine.run(save_to_db=False)

    # Визуализируем результаты
    if engine.gan:
        plot_gan_losses(
            engine.gan.generator_losses,
            engine.gan.discriminator_losses,
            f"GAN Losses - {engine.generations} поколений",
            save_path='gan_losses_plot.png'
        )

    plot_fitness_history(
        history,
        title=f"Эволюция - Лучший фитнес ({engine.generations} поколений)",
        save_path='fitness_history.png'
    )