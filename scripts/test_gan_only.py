# scripts/test_gan_only.py (исправленный)
"""
Тестирует GAN без использования pygame.
"""
import sys
import os
# Добавляем корневую директорию проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

print("Тестирование GAN без pygame...")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except Exception as e:
    print(f"❌ NumPy: {e}")
    sys.exit(1)

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ PyTorch: {e}")
    sys.exit(1)

try:
    from models.gan import GAN, ExperienceEncoder, PatternRepository
    print("✅ GAN модуль загружен")
except Exception as e:
    print(f"❌ GAN: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Создаём GAN с правильными параметрами (state_dim=8, action_dim=4)
try:
    gan = GAN(
        latent_dim=32,
        pattern_dim=None,  # Автоматически вычислится как 8+4+1+8=21
        batch_size=4,
        state_dim=8,  # <-- ВАЖНО: явно указываем размерность состояния
        action_dim=4  # <-- ВАЖНО: явно указываем размерность действий
    )
    print(f"✅ GAN создан на устройстве: {gan.device}")
    print(f"   Pattern dim: {gan.pattern_dim}")
    print(f"   Latent dim: {gan.latent_dim}")
except Exception as e:
    print(f"❌ Ошибка создания GAN: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Генерируем паттерн
try:
    pattern = gan.generate_pattern()
    print(f"✅ Сгенерирован паттерн размером: {pattern.shape}")
    print(f"   Первые 5 значений: {pattern[:5]}")
except Exception as e:
    print(f"❌ Ошибка генерации паттерна: {e}")
    sys.exit(1)

# Генерируем батч
try:
    patterns = gan.generate_batch(n=5)
    print(f"✅ Сгенерирован батч: {patterns.shape}")
except Exception as e:
    print(f"❌ Ошибка генерации батча: {e}")
    sys.exit(1)

# Создаём синтетические данные и обучаем
try:
    print("\nСоздание синтетических данных для обучения...")
    experiences = []
    for _ in range(100):
        # Используем state_dim=8 и action_dim=4
        state = np.random.randn(8)  # <-- 8 компонент состояния
        action = np.random.randint(0, 4)  # <-- 4 действия
        reward = np.random.randn() * 2 + 5
        next_state = np.random.randn(8)  # <-- 8 компонент следующего состояния
        experiences.append((state, action, reward, next_state))

    gan.add_experiences(experiences)
    print(f"✅ Добавлено {len(experiences)} переходов")

    print("Обучение GAN...")
    results = gan.train(epochs=5, verbose=True)
    print(f"✅ Обучение завершено")
    if results['g_loss'] and results['d_loss']:
        print(f"   Потери генератора (последние): {results['g_loss'][-5:]}")
        print(f"   Потери дискриминатора (последние): {results['d_loss'][-5:]}")
except Exception as e:
    print(f"❌ Ошибка обучения: {e}")
    import traceback
    traceback.print_exc()

# Генерируем правило
try:
    rule = gan.generate_rule()
    print(f"\n✅ Сгенерировано правило рефлекса:")
    print(f"   sense_type: {rule['sense_type']}")
    print(f"   signal_type: {rule['signal_type']}")
    print(f"   action: {rule['action']}")
    print(f"   threshold: {rule['threshold']:.2f}")
except Exception as e:
    print(f"❌ Ошибка генерации правила: {e}")

# Получаем статистику
try:
    stats = gan.get_training_stats()
    print(f"\n✅ Статистика GAN:")
    print(f"   Размер буфера: {stats['buffer_size']}")
    print(f"   Всего генераций: {stats['total_generations']}")
    if stats.get('avg_g_loss'):
        print(f"   Средняя потеря генератора: {stats['avg_g_loss']:.4f}")
    if stats.get('avg_d_loss'):
        print(f"   Средняя потеря дискриминатора: {stats['avg_d_loss']:.4f}")
except Exception as e:
    print(f"❌ Ошибка получения статистики: {e}")

print("\n✅ Все тесты пройдены успешно!")



# # test_gan_only.py
# """
# Тестирует GAN без использования pygame.
# """
# import sys
# import os
# import numpy as np
#
# print("Тестирование GAN без pygame...")
#
# # Проверяем numpy
# try:
#     import numpy as np
#
#     print(f"✅ NumPy: {np.__version__}")
# except Exception as e:
#     print(f"❌ NumPy: {e}")
#     sys.exit(1)
#
# # Проверяем torch
# try:
#     import torch
#
#     print(f"✅ PyTorch: {torch.__version__}")
#     print(f"   CUDA: {torch.cuda.is_available()}")
#     if torch.cuda.is_available():
#         print(f"   GPU: {torch.cuda.get_device_name(0)}")
# except Exception as e:
#     print(f"❌ PyTorch: {e}")
#     sys.exit(1)
#
# # Импортируем GAN
# try:
#     from models.gan import GAN, ExperienceEncoder, PatternRepository
#
#     print("✅ GAN модуль загружен")
# except Exception as e:
#     print(f"❌ GAN: {e}")
#     import traceback
#
#     traceback.print_exc()
#     sys.exit(1)
#
# # Создаём GAN с меньшими размерами для теста
# try:
#     gan = GAN(latent_dim=16, pattern_dim=12, batch_size=4)
#     print(f"✅ GAN создан на устройстве: {gan.device}")
# except Exception as e:
#     print(f"❌ Ошибка создания GAN: {e}")
#     import traceback
#
#     traceback.print_exc()
#     sys.exit(1)
#
# # Генерируем паттерн
# try:
#     pattern = gan.generate_pattern()
#     print(f"✅ Сгенерирован паттерн размером: {pattern.shape}")
#     print(f"   Первые 5 значений: {pattern[:5]}")
# except Exception as e:
#     print(f"❌ Ошибка генерации паттерна: {e}")
#     import traceback
#
#     traceback.print_exc()
#     sys.exit(1)
#
# # Генерируем батч
# try:
#     patterns = gan.generate_batch(n=5)
#     print(f"✅ Сгенерирован батч: {patterns.shape}")
# except Exception as e:
#     print(f"❌ Ошибка генерации батча: {e}")
#     import traceback
#
#     traceback.print_exc()
#     sys.exit(1)
#
# # Создаём синтетические данные и обучаем
# try:
#     print("\nСоздание синтетических данных для обучения...")
#     experiences = []
#     for _ in range(100):
#         state = np.random.randn(8)
#         action = np.random.randint(0, 4)
#         reward = np.random.randn()
#         next_state = np.random.randn(8)
#         experiences.append((state, action, reward, next_state))
#
#     gan.add_experiences(experiences)
#     print(f"✅ Добавлено {len(experiences)} переходов")
#
#     print("Обучение GAN...")
#     results = gan.train(epochs=5, verbose=True)
#     print(f"✅ Обучение завершено")
#     if results['g_loss'] and results['d_loss']:
#         print(f"   Потери генератора (последние 5): {results['g_loss'][-5:]}")
#         print(f"   Потери дискриминатора (последние 5): {results['d_loss'][-5:]}")
# except Exception as e:
#     print(f"❌ Ошибка обучения: {e}")
#     import traceback
#
#     traceback.print_exc()
#     # Не завершаем скрипт, продолжаем с генерацией
#
# # Генерируем правило
# try:
#     rule = gan.generate_rule()
#     print(f"\n✅ Сгенерировано правило рефлекса:")
#     print(f"   sense_type: {rule['sense_type']}")
#     print(f"   signal_type: {rule['signal_type']}")
#     print(f"   action: {rule['action']}")
#     print(f"   threshold: {rule['threshold']:.2f}")
# except Exception as e:
#     print(f"❌ Ошибка генерации правила: {e}")
#
# # Получаем статистику
# try:
#     stats = gan.get_training_stats()
#     print(f"\n✅ Статистика GAN:")
#     print(f"   Размер буфера: {stats['buffer_size']}")
#     print(f"   Всего генераций: {stats['total_generations']}")
#     if stats.get('avg_g_loss'):
#         print(f"   Средняя потеря генератора: {stats['avg_g_loss']:.4f}")
#     if stats.get('avg_d_loss'):
#         print(f"   Средняя потеря дискриминатора: {stats['avg_d_loss']:.4f}")
# except Exception as e:
#     print(f"❌ Ошибка получения статистики: {e}")
#
# # Сохраняем модели
# try:
#     os.makedirs('../models', exist_ok=True)
#     gan.save_models('models/test_gan')
#     print("✅ Модели сохранены")
# except Exception as e:
#     print(f"❌ Ошибка сохранения: {e}")
#
# print("\n✅ Все тесты пройдены успешно!")
#
#
# # # test_gan_only.py
# # """
# # Тестирует GAN без использования pygame.
# # """
# # import sys
# #
# # print("Тестирование GAN без pygame...")
# #
# # try:
# #     # Пытаемся импортировать numpy
# #     import numpy as np
# #
# #     print(f"✅ NumPy: {np.__version__}")
# # except Exception as e:
# #     print(f"❌ NumPy: {e}")
# #     sys.exit(1)
# #
# # try:
# #     # Пытаемся импортировать torch
# #     import torch
# #
# #     print(f"✅ PyTorch: {torch.__version__}")
# #     print(f"   CUDA: {torch.cuda.is_available()}")
# # except Exception as e:
# #     print(f"❌ PyTorch: {e}")
# #     sys.exit(1)
# #
# # try:
# #     # Импортируем GAN
# #     from models.gan import GAN, ExperienceEncoder
# #
# #     print("✅ GAN модуль загружен")
# # except Exception as e:
# #     print(f"❌ GAN: {e}")
# #     sys.exit(1)
# #
# # # Создаём GAN
# # try:
# #     gan = GAN(latent_dim=32, pattern_dim=16)
# #     print(f"✅ GAN создан на устройстве: {gan.device}")
# #
# #     # Генерируем паттерн
# #     pattern = gan.generate_pattern()
# #     print(f"✅ Сгенерирован паттерн размером: {pattern.shape}")
# #
# #     print("\n✅ Все тесты пройдены успешно!")
# # except Exception as e:
# #     print(f"❌ Ошибка при тестировании GAN: {e}")
# #     import traceback
# #
# #     traceback.print_exc()
# #     sys.exit(1)