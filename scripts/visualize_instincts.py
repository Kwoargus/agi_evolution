# scripts/visualize_instincts.py
"""
Визуализация инстинктов бота.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from core.instinct_evolution import InstinctEvolutionEngine
from models.instinct_gan import InstinctGAN


def visualize_instincts():
    """
    Визуализирует эволюцию инстинктов.
    """
    print("🧬 Визуализация эволюции инстинктов...")

    # Создаем InstinctGAN
    instinct_gan = InstinctGAN(latent_dim=64, pattern_dim=256, batch_size=8)

    # Генерируем синтетические паттерны
    synthetic_patterns = []
    for _ in range(200):
        pattern = np.random.randn(256)
        pattern = pattern / (np.linalg.norm(pattern) + 1e-8)
        synthetic_patterns.append(pattern)

    instinct_gan.add_patterns(synthetic_patterns)
    instinct_gan.train(epochs=5, verbose=True)

    # Создаем движок эволюции инстинктов
    engine = InstinctEvolutionEngine(
        population_size=20,
        pattern_dim=256,
        gan_generator=instinct_gan.generator,
        gan_discriminator=instinct_gan.discriminator,
        experience_buffer=instinct_gan.pattern_buffer
    )

    # Запускаем эволюцию
    history = engine.evolve(n_generations=10, gan_patterns_per_gen=5)

    # Визуализируем
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Эволюция инстинктов', fontsize=16)

    # 1. График эволюции
    ax1 = axes[0, 0]
    ax1.plot(history, marker='o', linewidth=2, color='blue')
    ax1.set_title('Лучший фитнес по поколениям')
    ax1.set_xlabel('Поколение')
    ax1.set_ylabel('Фитнес')
    ax1.grid(True)

    # 2. Распределение фитнеса
    ax2 = axes[0, 1]
    stats = engine.get_population_stats()
    fitnesses = [ind.fitness for ind in engine.population.individuals]
    ax2.hist(fitnesses, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(stats['best_fitness'], color='red', linestyle='--', label=f'Best: {stats["best_fitness"]:.3f}')
    ax2.axvline(stats['avg_fitness'], color='blue', linestyle='--', label=f'Avg: {stats["avg_fitness"]:.3f}')
    ax2.set_title('Распределение фитнеса')
    ax2.set_xlabel('Фитнес')
    ax2.set_ylabel('Количество')
    ax2.legend()
    ax2.grid(True)

    # 3. Тепловая карта лучшего паттерна
    ax3 = axes[0, 2]
    best_patterns = engine.get_best_instincts(1)
    if best_patterns:
        pattern = best_patterns[0]
        pattern_reshaped = np.resize(pattern, (16, 16))
        im = ax3.imshow(pattern_reshaped, cmap='viridis', aspect='auto')
        ax3.set_title('Лучший паттерн инстинкта')
        ax3.set_xlabel('Измерение')
        ax3.set_ylabel('Измерение')
        plt.colorbar(im, ax=ax3)

    # 4. Статистика популяции
    ax4 = axes[1, 0]
    ax4.axis('off')
    stats_text = f"""
    ╔══════════════════════════════════════╗
    ║    СТАТИСТИКА ПОПУЛЯЦИИ ИНСТИНКТОВ   ║
    ╠══════════════════════════════════════╣
    ║  Поколение:        {stats['generation']:<6}           ║
    ║  Размер:           {stats['population_size']:<6}           ║
    ║  Лучший фитнес:    {stats['best_fitness']:.4f}  ║
    ║  Средний фитнес:   {stats['avg_fitness']:.4f}  ║
    ║  Разнообразие:     {stats['unique_patterns']}/{stats['population_size']:<6}      ║
    ╚══════════════════════════════════════╝
    """
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')

    # 5. Потери GAN
    ax5 = axes[1, 1]
    g_losses = instinct_gan.generator_losses
    d_losses = instinct_gan.discriminator_losses
    if g_losses and d_losses:
        ax5.plot(g_losses, label='Generator', color='orange', alpha=0.7)
        ax5.plot(d_losses, label='Discriminator', color='purple', alpha=0.7)
        ax5.set_title('Потери InstinctGAN')
        ax5.set_xlabel('Шаг')
        ax5.set_ylabel('Потеря')
        ax5.legend()
        ax5.grid(True)

    # 6. Примеры инстинктов
    ax6 = axes[1, 2]
    patterns = engine.get_best_instincts(5)
    if patterns:
        # Показываем 5 лучших паттернов в виде строк
        for i, pattern in enumerate(patterns):
            ax6.plot(pattern[:100] + i * 2, alpha=0.7, label=f'#{i + 1}')
        ax6.set_title('Лучшие паттерны инстинктов (первые 100 значений)')
        ax6.set_xlabel('Индекс')
        ax6.set_ylabel('Значение + смещение')
        ax6.legend()
        ax6.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_instincts()