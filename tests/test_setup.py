# test_setup.py (обновлённая версия)
"""
Простой скрипт для проверки, что все модули установлены и работают.
"""
import sys
import os
# Добавляем корневую директорию проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union

def check_imports():
    """Проверяет, что все необходимые модули импортируются."""
    print("Проверка импортов...")

    modules = [
        'torch',
        'numpy',
        'matplotlib',
        'pygame',
        'psycopg2'
    ]

    missing = []
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module} - {str(e)}")
            missing.append(module)

    if missing:
        print(f"\n⚠️ Установите недостающие модули:")
        for m in missing:
            print(f"  pip install {m}")
        return False

    print("\n✅ Все модули установлены!")
    return True


def check_project_structure():
    """Проверяет структуру проекта."""
    print("\nПроверка структуры проекта...")

    required_dirs = [
        'core',
        'models',
        'db',
        'scripts',
        'tests'
    ]

    required_files = [
        'core/world.py',
        'core/individual.py',
        'core/genome.py',
        'core/evolution.py',
        'models/gan.py',
        'models/reflex_module.py',
        'models/instinct_module.py'
    ]

    missing_dirs = []
    for d in required_dirs:
        if os.path.exists(d):
            print(f"  ✅ {d}/")
        else:
            print(f"  ❌ {d}/ - НЕ НАЙДЕН")
            missing_dirs.append(d)

    missing_files = []
    for f in required_files:
        if os.path.exists(f):
            print(f"  ✅ {f}")
        else:
            print(f"  ❌ {f} - НЕ НАЙДЕН")
            missing_files.append(f)

    if missing_dirs or missing_files:
        print("\n⚠️ Некоторые файлы или директории не найдены!")
        return False

    print("\n✅ Структура проекта корректна!")
    return True


def test_gan_import():
    """Тестирует импорт GAN модуля."""
    print("\nТестирование импорта GAN...")
    try:
        from models.gan import GAN, ExperienceEncoder, PatternRepository
        print("  ✅ GAN модуль импортирован успешно")
        return True
    except ImportError as e:
        print(f"  ❌ Ошибка импорта GAN: {e}")
        return False


def test_torch():
    """Тестирует PyTorch."""
    print("\nТестирование PyTorch...")
    try:
        import torch
        print(f"  ✅ PyTorch версия: {torch.__version__}")
        print(f"  ✅ CUDA доступна: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  ✅ CUDA устройство: {torch.cuda.get_device_name(0)}")
        return True
    except Exception as e:
        print(f"  ❌ Ошибка PyTorch: {e}")
        return False


def test_gan_creation():
    """Тестирует создание GAN."""
    print("\nТестирование создания GAN...")
    try:
        from models.gan import GAN
        # Создаём GAN с минимальными параметрами
        gan = GAN(latent_dim=32, pattern_dim=16, batch_size=4)
        print("  ✅ GAN создан успешно")
        print(f"  ✅ Устройство: {gan.device}")
        print(f"  ✅ Encoder создан: {gan.encoder is not None}")
        print(f"  ✅ Репозиторий создан: {gan.pattern_repository is not None}")
        return True
    except Exception as e:
        print(f"  ❌ Ошибка создания GAN: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pattern_repository():
    """Тестирует репозиторий паттернов."""
    print("\nТестирование PatternRepository...")
    try:
        from models.gan import PatternRepository
        repo = PatternRepository()

        # Добавляем паттерн
        pattern = np.random.randn(16)
        repo.add_pattern(pattern, 0.8)

        # Проверяем, что паттерн добавлен
        if len(repo.patterns) == 1:
            print("  ✅ PatternRepository работает корректно")
            return True
        else:
            print("  ❌ PatternRepository не добавил паттерн")
            return False
    except Exception as e:
        print(f"  ❌ Ошибка в PatternRepository: {e}")
        return False


def main():
    """Главная функция проверки."""
    print("=" * 60)
    print("ПРОВЕРКА НАСТРОЙКИ ПРОЕКТА")
    print("=" * 60)

    # Импортируем numpy для тестов
    try:
        import numpy as np
    except ImportError:
        print("❌ NumPy не установлен! Установите: pip install numpy<2.0")
        return

    checks = [
        ("Импорты модулей", check_imports()),
        ("Структура проекта", check_project_structure()),
        ("Импорт GAN", test_gan_import()),
        ("PyTorch", test_torch()),
        ("Создание GAN", test_gan_creation()),
        ("PatternRepository", test_pattern_repository())
    ]

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ПРОВЕРКИ")
    print("=" * 60)

    all_ok = True
    for name, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
        if not result:
            all_ok = False

    if all_ok:
        print("\n✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!")
        print("   Вы можете запускать эксперименты.")
    else:
        print("\n❌ НЕКОТОРЫЕ ПРОВЕРКИ НЕ ПРОЙДЕНЫ!")
        print("   Устраните проблемы и попробуйте снова.")
        print("\nЕсли вы используете виртуальное окружение:")
        print("  source venv/bin/activate  # Linux/macOS")
        print("  venv\\Scripts\\activate     # Windows")


if __name__ == "__main__":
    main()



# # test_setup.py
# """
# Простой скрипт для проверки, что все модули установлены и работают.
# Запустите его перед основными экспериментами.
# """
# import sys
# import os
#
#
# def check_imports():
#     """Проверяет, что все необходимые модули импортируются."""
#     print("Проверка импортов...")
#
#     modules = [
#         'torch',
#         'numpy',
#         'matplotlib',
#         'pygame',
#         'psycopg2'
#     ]
#
#     missing = []
#     for module in modules:
#         try:
#             __import__(module)
#             print(f"  ✅ {module}")
#         except ImportError:
#             print(f"  ❌ {module} - НЕ УСТАНОВЛЕН")
#             missing.append(module)
#
#     if missing:
#         print(f"\n⚠️ Установите недостающие модули:")
#         for m in missing:
#             print(f"  pip install {m}")
#         return False
#
#     print("\n✅ Все модули установлены!")
#     return True
#
#
# def check_project_structure():
#     """Проверяет структуру проекта."""
#     print("\nПроверка структуры проекта...")
#
#     required_dirs = [
#         'core',
#         'models',
#         'db',
#         'scripts',
#         'tests'
#     ]
#
#     required_files = [
#         'core/world.py',
#         'core/individual.py',
#         'core/genome.py',
#         'core/evolution.py',
#         'models/gan.py',
#         'models/reflex_module.py',
#         'models/instinct_module.py'
#     ]
#
#     missing_dirs = []
#     for d in required_dirs:
#         if os.path.exists(d):
#             print(f"  ✅ {d}/")
#         else:
#             print(f"  ❌ {d}/ - НЕ НАЙДЕН")
#             missing_dirs.append(d)
#
#     missing_files = []
#     for f in required_files:
#         if os.path.exists(f):
#             print(f"  ✅ {f}")
#         else:
#             print(f"  ❌ {f} - НЕ НАЙДЕН")
#             missing_files.append(f)
#
#     if missing_dirs or missing_files:
#         print("\n⚠️ Некоторые файлы или директории не найдены!")
#         print("   Убедитесь, что вы находитесь в корневой директории проекта.")
#         return False
#
#     print("\n✅ Структура проекта корректна!")
#     return True
#
#
# def test_gan_import():
#     """Тестирует импорт GAN модуля."""
#     print("\nТестирование импорта GAN...")
#     try:
#         from models.gan import GAN, ExperienceEncoder, PatternRepository
#         print("  ✅ GAN модуль импортирован успешно")
#         return True
#     except ImportError as e:
#         print(f"  ❌ Ошибка импорта GAN: {e}")
#         return False
#
#
# def test_torch():
#     """Тестирует PyTorch."""
#     print("\nТестирование PyTorch...")
#     try:
#         import torch
#         print(f"  ✅ PyTorch версия: {torch.__version__}")
#         print(f"  ✅ CUDA доступна: {torch.cuda.is_available()}")
#         if torch.cuda.is_available():
#             print(f"  ✅ CUDA устройство: {torch.cuda.get_device_name(0)}")
#         return True
#     except Exception as e:
#         print(f"  ❌ Ошибка PyTorch: {e}")
#         return False
#
#
# def test_gan_creation():
#     """Тестирует создание GAN."""
#     print("\nТестирование создания GAN...")
#     try:
#         from models.gan import GAN
#         gan = GAN(latent_dim=32, pattern_dim=16)
#         print("  ✅ GAN создан успешно")
#         print(f"  ✅ Устройство: {gan.device}")
#         return True
#     except Exception as e:
#         print(f"  ❌ Ошибка создания GAN: {e}")
#         return False
#
#
# def main():
#     """Главная функция проверки."""
#     print("=" * 60)
#     print("ПРОВЕРКА НАСТРОЙКИ ПРОЕКТА")
#     print("=" * 60)
#
#     checks = [
#         ("Импорты модулей", check_imports()),
#         ("Структура проекта", check_project_structure()),
#         ("Импорт GAN", test_gan_import()),
#         ("PyTorch", test_torch()),
#         ("Создание GAN", test_gan_creation())
#     ]
#
#     print("\n" + "=" * 60)
#     print("РЕЗУЛЬТАТЫ ПРОВЕРКИ")
#     print("=" * 60)
#
#     all_ok = True
#     for name, result in checks:
#         status = "✅" if result else "❌"
#         print(f"{status} {name}")
#         if not result:
#             all_ok = False
#
#     if all_ok:
#         print("\n✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ! Вы можете запускать эксперименты.")
#     else:
#         print("\n❌ НЕКОТОРЫЕ ПРОВЕРКИ НЕ ПРОЙДЕНЫ!")
#         print("   Устраните проблемы и попробуйте снова.")
#         print("\nЕсли вы используете виртуальное окружение, убедитесь, что оно активировано:")
#         print("  source venv/bin/activate  # Linux/macOS")
#         print("  venv\\Scripts\\activate     # Windows")
#
#
# if __name__ == "__main__":
#     main()