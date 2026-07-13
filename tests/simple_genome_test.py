# scripts/simple_genome_test.py (исправленный)
import sys
import os

# Добавляем корневую директорию проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.genome import Genome
from core.individual import Individual
from core.world import World
from models.gan import GAN
from typing import List, Dict, Optional, Tuple, Any, Union

def simple_test():
    """Упрощенный тест влияния GAN на геномы."""
    print("=" * 60)
    print("ПРОСТОЙ ТЕСТ GAN И ГЕНОМОВ")
    print("=" * 60)

    # Создаем мир
    world = World(width=800, height=600, world_size=20, cell_size=40)
    print("✅ Мир создан")

    # Создаем GAN с правильными параметрами
    gan = GAN(
        latent_dim=32,
        pattern_dim=None,  # Автоматическое определение (21)
        batch_size=4,
        state_dim=8,
        action_dim=4
    )
    print(f"✅ GAN создан: pattern_dim={gan.pattern_dim}")

    # Генерируем синтетические данные для обучения
    print("\nГенерация синтетических данных...")
    experiences = []
    for i in range(200):
        state = np.random.randn(8)
        action = np.random.randint(0, 4)
        reward = np.random.randn() * 2 + 5
        next_state = np.random.randn(8)
        experiences.append((state, action, reward, next_state))

    gan.add_experiences(experiences)
    print(f"✅ Добавлено {len(experiences)} переходов")

    # Обучаем GAN
    print("\nОбучение GAN...")
    results = gan.train(epochs=5, verbose=True)

    if results['g_loss'] and results['d_loss']:
        print(f"✅ Обучение завершено")
        print(f"   Средняя потеря G: {np.mean(results['g_loss']):.4f}")
        print(f"   Средняя потеря D: {np.mean(results['d_loss']):.4f}")

    # Генерируем правила рефлексов
    print("\nГенерация правил рефлексов:")
    for i in range(5):
        rule = gan.generate_rule()
        print(f"  {i + 1}. {rule}")

    # Генерируем паттерны инстинктов
    print("\nГенерация паттернов инстинктов:")
    for i in range(3):
        instinct = gan.generate_instinct_pattern()
        print(f"  {i + 1}. {instinct}")

    # Получаем статистику
    stats = gan.get_training_stats()
    print(f"\nСтатистика GAN:")
    print(f"  Буфер опыта: {stats['buffer_size']}")
    print(f"  Всего генераций: {stats['total_generations']}")

    print("\n✅ Тест завершен успешно!")


if __name__ == "__main__":
    simple_test()



# # scripts/simple_genome_test.py
# import sys
# import os
#
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# import numpy as np
# from core.genome import Genome
# from core.individual import Individual
# from core.world import World
# from models.gan import GAN
#
#
# def simple_test():
#     """Упрощенный тест влияния GAN на геномы."""
#     print("Простой тест GAN и геномов...")
#
#     # Создаем мир
#     world = World(width=800, height=600, world_size=20, cell_size=40)
#
#     # Создаем GAN
#     gan = GAN(latent_dim=32, pattern_dim=None, batch_size=4)
#
#     # Генерируем синтетические данные для обучения
#     experiences = []
#     for _ in range(100):
#         state = np.random.randn(8)
#         action = np.random.randint(0, 4)
#         reward = np.random.randn() * 2 + 5
#         next_state = np.random.randn(8)
#         experiences.append((state, action, reward, next_state))
#
#     gan.add_experiences(experiences)
#     gan.train(epochs=5, verbose=True)
#
#     # Генерируем правила
#     print("\nСгенерированные правила рефлексов:")
#     for i in range(5):
#         rule = gan.generate_rule()
#         print(f"  {i + 1}. {rule}")
#
#     print("\n✅ Тест завершен!")
#
#
# if __name__ == "__main__":
#     simple_test()