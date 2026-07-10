# scripts/test_instinct_evolution.py (исправленный)
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from core.instinct_evolution import InstinctEvolutionEngine, InstinctPopulation
from models.instinct_gan import InstinctGAN  # <-- Импортируем InstinctGAN
import torch
from typing import List, Dict, Optional, Tuple, Any, Union


def test_instinct_evolution():
    """
    Тестирует эволюцию инстинктов с GA и GAN.
    """
    print("=== Тестирование эволюции инстинктов ===")

    # ============================================================
    # 1. СОЗДАЕМ InstinctGAN (специально для паттернов 256-dim)
    # ============================================================
    print("\nСоздание InstinctGAN...")
    instinct_gan = InstinctGAN(
        latent_dim=64,
        pattern_dim=256,
        batch_size=8
    )

    # ============================================================
    # 2. ГЕНЕРИРУЕМ СИНТЕТИЧЕСКИЕ ПАТТЕРНЫ
    # ============================================================
    print("Генерация синтетических паттернов для обучения...")
    synthetic_patterns = []
    for _ in range(300):
        # Генерируем случайный паттерн размерности 256
        pattern = np.random.randn(256)
        # Нормализуем
        pattern = pattern / (np.linalg.norm(pattern) + 1e-8)
        synthetic_patterns.append(pattern)

    # Добавляем паттерны в буфер
    instinct_gan.add_patterns(synthetic_patterns)
    print(f"✅ Добавлено {len(synthetic_patterns)} паттернов в буфер")

    # ============================================================
    # 3. ОБУЧАЕМ InstinctGAN
    # ============================================================
    print("\nОбучение InstinctGAN...")
    results = instinct_gan.train(epochs=10, verbose=True)
    print(f"✅ Обучение завершено")
    if results['g_loss'] and results['d_loss']:
        print(f"  Средняя потеря G: {np.mean(results['g_loss']):.4f}")
        print(f"  Средняя потеря D: {np.mean(results['d_loss']):.4f}")

    # ============================================================
    # 4. СОЗДАЕМ ДВИЖОК ЭВОЛЮЦИИ ИНСТИНКТОВ
    # ============================================================
    engine = InstinctEvolutionEngine(
        population_size=20,
        pattern_dim=256,
        gan_generator=instinct_gan.generator,
        gan_discriminator=instinct_gan.discriminator,
        experience_buffer=instinct_gan.pattern_buffer  # Используем pattern_buffer
    )

    print("✅ Движок эволюции инстинктов создан")

    # ============================================================
    # 5. ЗАПУСКАЕМ ЭВОЛЮЦИЮ
    # ============================================================
    print("\n🚀 Запуск эволюции инстинктов...")
    history = engine.evolve(n_generations=10, gan_patterns_per_gen=5)

    # ============================================================
    # 6. ПОЛУЧАЕМ РЕЗУЛЬТАТЫ
    # ============================================================
    best_instincts = engine.get_best_instincts(3)
    print(f"\n🏆 Лучшие паттерны инстинктов:")
    for i, pattern in enumerate(best_instincts):
        print(f"  {i + 1}. Длина: {len(pattern)}, среднее: {np.mean(pattern):.4f}, std: {np.std(pattern):.4f}")
        print(f"     Первые 10 значений: {pattern[:10]}...")

    # ============================================================
    # 7. ВИЗУАЛИЗАЦИЯ
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. График эволюции
    ax1 = axes[0, 0]
    ax1.plot(history, marker='o', linewidth=2, color='blue')
    ax1.set_title('Эволюция инстинктов')
    ax1.set_xlabel('Поколение')
    ax1.set_ylabel('Лучший фитнес')
    ax1.grid(True)
    ax1.axhline(y=max(history), color='red', linestyle='--', label=f'Max: {max(history):.3f}')
    ax1.legend()

    # 2. Распределение фитнеса
    ax2 = axes[0, 1]
    stats = engine.get_population_stats()
    fitnesses = [ind.fitness for ind in engine.population.individuals]
    ax2.hist(fitnesses, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(stats['best_fitness'], color='red', linestyle='--', linewidth=2, label=f'Best: {stats["best_fitness"]:.3f}')
    ax2.axvline(stats['avg_fitness'], color='blue', linestyle='--', linewidth=2, label=f'Avg: {stats["avg_fitness"]:.3f}')
    ax2.set_title('Распределение фитнеса в популяции')
    ax2.set_xlabel('Фитнес')
    ax2.set_ylabel('Количество')
    ax2.legend()
    ax2.grid(True)

    # 3. Тепловая карта лучшего паттерна
    ax3 = axes[1, 0]
    if best_instincts:
        best_pattern = best_instincts[0]
        # Ресайзим до 16x16 для визуализации
        size = 16
        pattern_reshaped = np.resize(best_pattern, (size, size))
        im = ax3.imshow(pattern_reshaped, cmap='viridis', aspect='auto')
        ax3.set_title('Лучший паттерн инстинкта (256-dim)')
        ax3.set_xlabel('Измерение')
        ax3.set_ylabel('Измерение')
        plt.colorbar(im, ax=ax3)

    # 4. Статистика популяции
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""
    ╔══════════════════════════════════════╗
    ║    СТАТИСТИКА ПОПУЛЯЦИИ ИНСТИНКТОВ   ║
    ╠══════════════════════════════════════╣
    ║  Поколение:        {stats['generation']:<6}           ║
    ║  Размер популяции: {stats['population_size']:<6}           ║
    ║                                    ║
    ║  Лучший фитнес:    {stats['best_fitness']:.4f}  ║
    ║  Средний фитнес:   {stats['avg_fitness']:.4f}  ║
    ║  Std фитнес:       {stats['std_fitness']:.4f}  ║
    ║  Min фитнес:       {stats['min_fitness']:.4f}  ║
    ║                                    ║
    ║  Уникальных паттернов: {stats['unique_patterns']:<6}      ║
    ║  Разнообразие:     {stats['unique_patterns'] / stats['population_size'] * 100:.1f}%           ║
    ╚══════════════════════════════════════╝
    """
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.suptitle('Эволюция инстинктов с GA + InstinctGAN', fontsize=16)
    plt.tight_layout()
    plt.show()

    # ============================================================
    # 8. СОХРАНЯЕМ СОСТОЯНИЕ
    # ============================================================
    engine.save_state('instinct_population_state.json')
    instinct_gan.save_models('models/instinct_gan')

    print("\n✅ Тестирование эволюции инстинктов завершено!")
    print(f"   Лучший фитнес: {max(history):.4f}")
    print(f"   Средний фитнес: {np.mean(history):.4f}")
    print(f"   Уникальных паттернов: {stats['unique_patterns']}/{stats['population_size']}")


if __name__ == "__main__":
    test_instinct_evolution()