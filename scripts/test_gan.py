# scripts/test_gan.py (исправленный)
import sys
import os
# Добавляем корневую директорию проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Добавляем корневую директорию проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from models.gan import GAN, ExperienceEncoder, PatternRepository


def test_gan():
    """Тестирует работу GAN с синтетическими данными."""
    print("Тестирование GAN...")

    # Создаём GAN
    gan = GAN(
        latent_dim=64,
        pattern_dim=None,  # Автоматическое определение
        batch_size=8,
        state_dim=8,
        action_dim=4
    )

    # Генерируем синтетические данные
    print("Генерация синтетических данных...")
    synthetic_experiences = []
    for _ in range(200):
        state = np.random.randn(8)
        action = np.random.randint(0, 4)
        reward = np.random.randn() * 2 + 5  # Положительные награды
        next_state = np.random.randn(8)
        synthetic_experiences.append((state, action, reward, next_state))

    # Обучаем GAN
    print("Обучение GAN...")
    gan.add_experiences(synthetic_experiences)
    results = gan.train(epochs=20, verbose=True)

    # Генерируем новые паттерны
    print("Генерация новых паттернов...")
    patterns = gan.generate_batch(10)
    print(f"Сгенерировано {len(patterns)} паттернов")

    # Создаём репозиторий и оцениваем паттерны
    repo = PatternRepository()
    evaluator = gan.pattern_evaluator
    for pattern in patterns:
        score = evaluator.evaluate_pattern(pattern)
        repo.add_pattern(pattern, score)

    print(f"Лучшая оценка паттерна: {max(repo.scores):.4f}")
    print(f"Средняя оценка: {repo.get_average_score():.4f}")

    # Генерируем правила
    print("\nГенерация правил из паттернов:")
    for i in range(3):
        rule = gan.generate_rule()
        print(f"  Правило {i + 1}: {rule}")

    print("\nТестирование завершено!")


if __name__ == "__main__":
    test_gan()



# # scripts/test_gan.py
# import sys
# import os
#
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# import numpy as np
# from models.gan import GAN, ExperienceEncoder, PatternRepository
#
#
# def test_gan():
#     """Тестирует работу GAN с синтетическими данными."""
#     print("Тестирование GAN...")
#
#     # Создаём GAN - pattern_dim теперь вычисляется автоматически
#     gan = GAN(
#         latent_dim=64,
#         pattern_dim=None,  # Автоматическое определение
#         batch_size=8,
#         state_dim=8,
#         action_dim=4
#     )
#
#     # Генерируем синтетические данные
#     print("Генерация синтетических данных...")
#     synthetic_experiences = []
#     for _ in range(200):
#         state = np.random.randn(8)
#         action = np.random.randint(0, 4)
#         reward = np.random.randn() * 2 + 5  # Положительные награды
#         next_state = np.random.randn(8)
#         synthetic_experiences.append((state, action, reward, next_state))
#
#     # Обучаем GAN
#     print("Обучение GAN...")
#     gan.add_experiences(synthetic_experiences)
#     results = gan.train(epochs=20, verbose=True)
#
#     # Генерируем новые паттерны
#     print("Генерация новых паттернов...")
#     patterns = gan.generate_batch(10)
#     print(f"Сгенерировано {len(patterns)} паттернов")
#
#     # Создаём репозиторий и оцениваем паттерны
#     repo = PatternRepository()
#     evaluator = gan.pattern_evaluator
#     for pattern in patterns:
#         score = evaluator.evaluate_pattern(pattern)
#         repo.add_pattern(pattern, score)
#
#     print(f"Лучшая оценка паттерна: {max(repo.scores):.4f}")
#     print(f"Средняя оценка: {repo.get_average_score():.4f}")
#
#     # Генерируем правила
#     print("\nГенерация правил из паттернов:")
#     for i in range(3):
#         rule = gan.generate_rule()
#         print(f"  Правило {i + 1}: {rule}")
#
#     print("\nТестирование завершено!")
#
#
# if __name__ == "__main__":
#     test_gan()
#
#
#
# # # scripts/test_gan.py
# # import sys
# # import os
# #
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# #
# # import torch
# # import numpy as np
# # from models.gan import GAN, ExperienceEncoder, PatternRepository
# #
# #
# # def test_gan():
# #     """Тестирует работу GAN с синтетическими данными."""
# #     print("Тестирование GAN...")
# #
# #     # Создаём GAN
# #     gan = GAN(
# #         latent_dim=64,
# #         pattern_dim=32,
# #         batch_size=8
# #     )
# #
# #     # Генерируем синтетические данные
# #     print("Генерация синтетических данных...")
# #     synthetic_experiences = []
# #     for _ in range(100):
# #         state = np.random.randn(8)
# #         action = np.random.randint(0, 4)
# #         reward = np.random.randn()
# #         next_state = np.random.randn(8)
# #         synthetic_experiences.append((state, action, reward, next_state))
# #
# #     # Обучаем GAN
# #     print("Обучение GAN...")
# #     gan.add_experiences(synthetic_experiences)
# #     results = gan.train(epochs=20, verbose=True)
# #
# #     # Генерируем новые паттерны
# #     print("Генерация новых паттернов...")
# #     patterns = gan.generate_batch(10)
# #     print(f"Сгенерировано {len(patterns)} паттернов")
# #
# #     # Создаём репозиторий и оцениваем паттерны
# #     repo = PatternRepository()
# #     evaluator = gan.pattern_evaluator
# #     for pattern in patterns:
# #         score = evaluator.evaluate_pattern(pattern)
# #         repo.add_pattern(pattern, score)
# #
# #     print(f"Лучшая оценка паттерна: {max(repo.scores):.4f}")
# #     print(f"Средняя оценка: {repo.get_average_score():.4f}")
# #
# #     # Генерируем правила
# #     print("\nГенерация правил из паттернов:")
# #     for i in range(3):
# #         rule = gan.generate_rule()
# #         print(f"  Правило {i + 1}: {rule}")
# #
# #     print("\nТестирование завершено!")
# #
# #
# # if __name__ == "__main__":
# #     test_gan()