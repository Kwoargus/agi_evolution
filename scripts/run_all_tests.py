# run_all_tests.py (исправленный для работы из любой директории)
# !/usr/bin/env python3
"""
Скрипт для запуска всех тестов и экспериментов GAN.
"""
import sys
import os
# Добавляем корневую директорию проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import subprocess

# Определяем корневую директорию проекта (поднимаемся на уровень выше, если скрипт в scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Если скрипт находится в папке scripts, поднимаемся на уровень выше
if os.path.basename(SCRIPT_DIR) == 'scripts':
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
else:
    PROJECT_ROOT = SCRIPT_DIR

# Правильные пути к директориям
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')
TESTS_DIR = os.path.join(PROJECT_ROOT, 'tests')
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, 'experiments')

# Определяем команду Python
PYTHON_CMD = sys.executable or "python3"


def find_script(script_name, search_dirs=None):
    """
    Ищет скрипт в нескольких директориях.

    Args:
        script_name: Имя файла скрипта (например, 'test_gan.py')
        search_dirs: Список директорий для поиска

    Returns:
        Полный путь к скрипту или None, если не найден
    """
    if search_dirs is None:
        search_dirs = [SCRIPTS_DIR, TESTS_DIR, EXPERIMENTS_DIR, PROJECT_ROOT]

    for directory in search_dirs:
        script_path = os.path.join(directory, script_name)
        if os.path.exists(script_path):
            return script_path

    return None


def run_script(script_path, description):
    """Запускает Python скрипт и возвращает результат."""
    print(f"\n{'=' * 70}")
    print(f"ЗАПУСК: {description}")
    print(f"Скрипт: {script_path}")
    print(f"{'=' * 70}")

    if not os.path.exists(script_path):
        print(f"❌ Файл не найден: {script_path}")
        return False

    try:
        # Запускаем скрипт с тем же интерпретатором Python
        result = subprocess.run(
            [PYTHON_CMD, script_path],
            cwd=PROJECT_ROOT,
            capture_output=False,  # Показываем вывод в реальном времени
            text=True
        )

        if result.returncode == 0:
            print(f"\n✅ {description} успешно выполнен")
            return True
        else:
            print(f"\n❌ {description} завершился с ошибкой (код: {result.returncode})")
            return False

    except Exception as e:
        print(f"❌ Ошибка при запуске {description}: {e}")
        return False


def run_all_tests():
    """Запускает все тесты."""
    print("=" * 70)
    print("ЗАПУСК ВСЕХ ЭКСПЕРИМЕНТОВ ДЛЯ ПРОВЕРКИ GAN")
    print("=" * 70)
    print(f"Python: {PYTHON_CMD}")
    print(f"Корневая директория: {PROJECT_ROOT}")
    print(f"Директория скриптов: {SCRIPTS_DIR}")
    print(f"Директория тестов: {TESTS_DIR}")
    print(f"Директория экспериментов: {EXPERIMENTS_DIR}")

    # Проверяем существование директорий
    for dir_name, dir_path in [('scripts', SCRIPTS_DIR), ('tests', TESTS_DIR), ('experiments', EXPERIMENTS_DIR)]:
        if os.path.exists(dir_path):
            print(f"✅ Директория {dir_name} существует: {dir_path}")
        else:
            print(f"❌ Директория {dir_name} не найдена: {dir_path}")

    # Список тестов для запуска
    tests = [
        {
            'script_name': 'test_gan.py',
            'description': 'Тестирование GAN',
            'search_dirs': [SCRIPTS_DIR, TESTS_DIR]
        },
        {
            'script_name': 'test_gan_only.py',
            'description': 'Тестирование GAN (упрощенное)',
            'search_dirs': [SCRIPTS_DIR, TESTS_DIR]
        },
        {
            'script_name': 'test_gan_integration.py',
            'description': 'Интеграционное тестирование GAN',
            'search_dirs': [TESTS_DIR, SCRIPTS_DIR]
        },
        {
            'script_name': 'test_genome_improvement.py',
            'description': 'Тестирование улучшения геномов GAN',
            'search_dirs': [SCRIPTS_DIR, TESTS_DIR]
        },
        {
            'script_name': 'simple_genome_test.py',
            'description': 'Простой тест геномов',
            'search_dirs': [SCRIPTS_DIR, TESTS_DIR]
        },
        {
            'script_name': 'run_evolution.py',
            'description': 'Запуск эволюции с GAN',
            'search_dirs': [EXPERIMENTS_DIR, PROJECT_ROOT]
        },
        {
            'script_name': 'compare_evolution_with_gan.py',
            'description': 'Сравнение эволюции с GAN и без',
            'search_dirs': [SCRIPTS_DIR, TESTS_DIR]
        },
        {
            'script_name': 'visualize_gan_training.py',
            'description': 'Визуализация обучения GAN',
            'search_dirs': [SCRIPTS_DIR, TESTS_DIR]
        }
    ]

    results = []

    for test in tests:
        # Ищем скрипт
        script_path = find_script(test['script_name'], test.get('search_dirs'))

        if script_path is None:
            print(f"\n❌ Скрипт не найден: {test['script_name']}")
            print(f"   Искали в: {test.get('search_dirs', ['scripts', 'tests', 'experiments'])}")

            # Показываем содержимое директорий для отладки
            for dir_path in test.get('search_dirs', []):
                if os.path.exists(dir_path):
                    print(f"   Содержимое {dir_path}:")
                    for file in os.listdir(dir_path):
                        if file.endswith('.py'):
                            print(f"     - {file}")

            results.append((test['description'], False))
            continue

        success = run_script(script_path, test['description'])
        results.append((test['description'], success))

        # Пауза между тестами
        print("\nНажмите Enter для продолжения или 'q' для выхода...")
        user_input = input().strip().lower()
        if user_input == 'q':
            print("Прерывание выполнения...")
            break

    # Выводим итоговый отчет
    print("\n" + "=" * 70)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 70)

    successful = sum(1 for _, success in results if success)
    total = len(results)

    for description, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {description}")

    print("\n" + "=" * 70)
    print(f"ВСЕГО: {successful}/{total} тестов пройдено успешно")

    if successful == total:
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    else:
        print("⚠️ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ")

    print("=" * 70)


def run_specific_test(test_name):
    """Запускает конкретный тест по имени."""
    test_map = {
        'gan': ('test_gan.py', [SCRIPTS_DIR, TESTS_DIR]),
        'gan_only': ('test_gan_only.py', [SCRIPTS_DIR, TESTS_DIR]),
        'integration': ('test_gan_integration.py', [TESTS_DIR, SCRIPTS_DIR]),
        'genome': ('test_genome_improvement.py', [SCRIPTS_DIR, TESTS_DIR]),
        'simple_genome': ('simple_genome_test.py', [SCRIPTS_DIR, TESTS_DIR]),
        'evolution': ('run_evolution.py', [EXPERIMENTS_DIR, PROJECT_ROOT]),
        'compare': ('compare_evolution_with_gan.py', [SCRIPTS_DIR, TESTS_DIR]),
        'visualize': ('visualize_gan_training.py', [SCRIPTS_DIR, TESTS_DIR])
    }

    if test_name in test_map:
        script_name, search_dirs = test_map[test_name]
        script_path = find_script(script_name, search_dirs)

        if script_path:
            run_script(script_path, f"Тест: {test_name}")
        else:
            print(f"❌ Скрипт не найден: {script_name}")
            print(f"   Искали в: {search_dirs}")
    else:
        print(f"❌ Тест '{test_name}' не найден")
        print(f"Доступные тесты: {', '.join(test_map.keys())}")


def list_available_tests():
    """Выводит список доступных тестов."""
    print("\nДоступные тесты:")
    print("  gan          - Тестирование GAN")
    print("  gan_only     - Тестирование GAN (упрощенное)")
    print("  integration  - Интеграционное тестирование GAN")
    print("  genome       - Тестирование улучшения геномов GAN")
    print("  simple_genome - Простой тест геномов")
    print("  evolution    - Запуск эволюции с GAN")
    print("  compare      - Сравнение эволюции с GAN и без")
    print("  visualize    - Визуализация обучения GAN")
    print("\nИли запустите без аргументов для выполнения всех тестов.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '--list' or sys.argv[1] == '-l':
            list_available_tests()
        else:
            run_specific_test(sys.argv[1])
    else:
        run_all_tests()




# # run_all_tests.py
# # !/usr/bin/env python3
# """
# Скрипт для запуска всех тестов и экспериментов GAN.
# """
# import os
# import sys
# import subprocess
#
# # Определяем корневую директорию проекта
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
#
# # Правильные пути к директориям
# SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')
# TESTS_DIR = os.path.join(PROJECT_ROOT, 'tests')
# EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, 'experiments')
#
# # Определяем команду Python
# PYTHON_CMD = sys.executable or "python3"
#
#
# def find_script(script_name, search_dirs=None):
#     """
#     Ищет скрипт в нескольких директориях.
#
#     Args:
#         script_name: Имя файла скрипта (например, 'test_gan.py')
#         search_dirs: Список директорий для поиска
#
#     Returns:
#         Полный путь к скрипту или None, если не найден
#     """
#     if search_dirs is None:
#         search_dirs = [SCRIPTS_DIR, TESTS_DIR, EXPERIMENTS_DIR, PROJECT_ROOT]
#
#     for directory in search_dirs:
#         script_path = os.path.join(directory, script_name)
#         if os.path.exists(script_path):
#             return script_path
#
#     return None
#
#
# def run_script(script_path, description):
#     """Запускает Python скрипт и возвращает результат."""
#     print(f"\n{'=' * 70}")
#     print(f"ЗАПУСК: {description}")
#     print(f"Скрипт: {script_path}")
#     print(f"{'=' * 70}")
#
#     if not os.path.exists(script_path):
#         print(f"❌ Файл не найден: {script_path}")
#         return False
#
#     try:
#         # Запускаем скрипт с тем же интерпретатором Python
#         result = subprocess.run(
#             [PYTHON_CMD, script_path],
#             cwd=PROJECT_ROOT,
#             capture_output=False,  # Показываем вывод в реальном времени
#             text=True
#         )
#
#         if result.returncode == 0:
#             print(f"\n✅ {description} успешно выполнен")
#             return True
#         else:
#             print(f"\n❌ {description} завершился с ошибкой (код: {result.returncode})")
#             return False
#
#     except Exception as e:
#         print(f"❌ Ошибка при запуске {description}: {e}")
#         return False
#
#
# def run_all_tests():
#     """Запускает все тесты."""
#     print("=" * 70)
#     print("ЗАПУСК ВСЕХ ЭКСПЕРИМЕНТОВ ДЛЯ ПРОВЕРКИ GAN")
#     print("=" * 70)
#     print(f"Python: {PYTHON_CMD}")
#     print(f"Корневая директория: {PROJECT_ROOT}")
#     print(f"Директория скриптов: {SCRIPTS_DIR}")
#     print(f"Директория тестов: {TESTS_DIR}")
#     print(f"Директория экспериментов: {EXPERIMENTS_DIR}")
#
#     # Список тестов для запуска
#     tests = [
#         {
#             'script_name': 'test_gan.py',
#             'description': 'Тестирование GAN',
#             'search_dirs': [SCRIPTS_DIR, TESTS_DIR]
#         },
#         {
#             'script_name': 'test_gan_only.py',
#             'description': 'Тестирование GAN (упрощенное)',
#             'search_dirs': [SCRIPTS_DIR, TESTS_DIR]
#         },
#         {
#             'script_name': 'test_gan_integration.py',
#             'description': 'Интеграционное тестирование GAN',
#             'search_dirs': [TESTS_DIR, SCRIPTS_DIR]
#         },
#         {
#             'script_name': 'test_genome_improvement.py',
#             'description': 'Тестирование улучшения геномов GAN',
#             'search_dirs': [SCRIPTS_DIR, TESTS_DIR]
#         },
#         {
#             'script_name': 'simple_genome_test.py',
#             'description': 'Простой тест геномов',
#             'search_dirs': [SCRIPTS_DIR, TESTS_DIR]
#         },
#         {
#             'script_name': 'run_evolution.py',
#             'description': 'Запуск эволюции с GAN',
#             'search_dirs': [EXPERIMENTS_DIR, PROJECT_ROOT]
#         },
#         {
#             'script_name': 'compare_evolution_with_gan.py',
#             'description': 'Сравнение эволюции с GAN и без',
#             'search_dirs': [SCRIPTS_DIR, TESTS_DIR]
#         },
#         {
#             'script_name': 'visualize_gan_training.py',
#             'description': 'Визуализация обучения GAN',
#             'search_dirs': [SCRIPTS_DIR, TESTS_DIR]
#         }
#     ]
#
#     results = []
#
#     for test in tests:
#         # Ищем скрипт
#         script_path = find_script(test['script_name'], test.get('search_dirs'))
#
#         if script_path is None:
#             print(f"\n❌ Скрипт не найден: {test['script_name']}")
#             print(f"   Искали в: {test.get('search_dirs', ['scripts', 'tests', 'experiments'])}")
#             results.append((test['description'], False))
#             continue
#
#         success = run_script(script_path, test['description'])
#         results.append((test['description'], success))
#
#         # Пауза между тестами
#         print("\nНажмите Enter для продолжения или 'q' для выхода...")
#         user_input = input().strip().lower()
#         if user_input == 'q':
#             print("Прерывание выполнения...")
#             break
#
#     # Выводим итоговый отчет
#     print("\n" + "=" * 70)
#     print("ИТОГОВЫЙ ОТЧЕТ")
#     print("=" * 70)
#
#     successful = sum(1 for _, success in results if success)
#     total = len(results)
#
#     for description, success in results:
#         status = "✅" if success else "❌"
#         print(f"{status} {description}")
#
#     print("\n" + "=" * 70)
#     print(f"ВСЕГО: {successful}/{total} тестов пройдено успешно")
#
#     if successful == total:
#         print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
#     else:
#         print("⚠️ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ")
#
#     print("=" * 70)
#
#
# def run_specific_test(test_name):
#     """Запускает конкретный тест по имени."""
#     test_map = {
#         'gan': ('test_gan.py', [SCRIPTS_DIR, TESTS_DIR]),
#         'gan_only': ('test_gan_only.py', [SCRIPTS_DIR, TESTS_DIR]),
#         'integration': ('test_gan_integration.py', [TESTS_DIR, SCRIPTS_DIR]),
#         'genome': ('test_genome_improvement.py', [SCRIPTS_DIR, TESTS_DIR]),
#         'simple_genome': ('simple_genome_test.py', [SCRIPTS_DIR, TESTS_DIR]),
#         'evolution': ('run_evolution.py', [EXPERIMENTS_DIR, PROJECT_ROOT]),
#         'compare': ('compare_evolution_with_gan.py', [SCRIPTS_DIR, TESTS_DIR]),
#         'visualize': ('visualize_gan_training.py', [SCRIPTS_DIR, TESTS_DIR])
#     }
#
#     if test_name in test_map:
#         script_name, search_dirs = test_map[test_name]
#         script_path = find_script(script_name, search_dirs)
#
#         if script_path:
#             run_script(script_path, f"Тест: {test_name}")
#         else:
#             print(f"❌ Скрипт не найден: {script_name}")
#             print(f"   Искали в: {search_dirs}")
#     else:
#         print(f"❌ Тест '{test_name}' не найден")
#         print(f"Доступные тесты: {', '.join(test_map.keys())}")
#
#
# def list_available_tests():
#     """Выводит список доступных тестов."""
#     print("\nДоступные тесты:")
#     print("  gan          - Тестирование GAN")
#     print("  gan_only     - Тестирование GAN (упрощенное)")
#     print("  integration  - Интеграционное тестирование GAN")
#     print("  genome       - Тестирование улучшения геномов GAN")
#     print("  simple_genome - Простой тест геномов")
#     print("  evolution    - Запуск эволюции с GAN")
#     print("  compare      - Сравнение эволюции с GAN и без")
#     print("  visualize    - Визуализация обучения GAN")
#     print("\nИли запустите без аргументов для выполнения всех тестов.")
#
#
# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         if sys.argv[1] == '--list' or sys.argv[1] == '-l':
#             list_available_tests()
#         else:
#             run_specific_test(sys.argv[1])
#     else:
#         run_all_tests()


# # run_all_tests.py
# # !/usr/bin/env python3
# """
# Скрипт для запуска всех тестов и экспериментов GAN.
# """
# import os
# import sys
# import subprocess
#
# # Определяем корневую директорию проекта
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# TESTS_DIR = os.path.join(PROJECT_ROOT, 'tests')
# SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')
# EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, 'experiments')
#
# # Определяем команду Python
# PYTHON_CMD = sys.executable or "python3"
#
#
# def run_script(script_path, description):
#     """Запускает Python скрипт и возвращает результат."""
#     print(f"\n{'=' * 70}")
#     print(f"ЗАПУСК: {description}")
#     print(f"Скрипт: {script_path}")
#     print(f"{'=' * 70}")
#
#     if not os.path.exists(script_path):
#         print(f"❌ Файл не найден: {script_path}")
#         return False
#
#     try:
#         # Запускаем скрипт с тем же интерпретатором Python
#         result = subprocess.run(
#             [PYTHON_CMD, script_path],
#             cwd=PROJECT_ROOT,
#             capture_output=False,  # Показываем вывод в реальном времени
#             text=True
#         )
#
#         if result.returncode == 0:
#             print(f"\n✅ {description} успешно выполнен")
#             return True
#         else:
#             print(f"\n❌ {description} завершился с ошибкой (код: {result.returncode})")
#             return False
#
#     except Exception as e:
#         print(f"❌ Ошибка при запуске {description}: {e}")
#         return False
#
#
# def run_all_tests():
#     """Запускает все тесты."""
#     print("=" * 70)
#     print("ЗАПУСК ВСЕХ ЭКСПЕРИМЕНТОВ ДЛЯ ПРОВЕРКИ GAN")
#     print("=" * 70)
#     print(f"Python: {PYTHON_CMD}")
#     print(f"Корневая директория: {PROJECT_ROOT}")
#
#     # Список тестов для запуска
#     tests = [
#         {
#             'script': os.path.join(SCRIPTS_DIR, 'test_gan.py'),
#             'description': 'Тестирование GAN'
#         },
#         {
#             'script': os.path.join(SCRIPTS_DIR, 'test_gan_only.py'),
#             'description': 'Тестирование GAN (упрощенное)'
#         },
#         {
#             'script': os.path.join(TESTS_DIR, 'test_gan_integration.py'),
#             'description': 'Интеграционное тестирование GAN'
#         },
#         {
#             'script': os.path.join(SCRIPTS_DIR, 'test_genome_improvement.py'),
#             'description': 'Тестирование улучшения геномов GAN'
#         },
#         {
#             'script': os.path.join(SCRIPTS_DIR, 'simple_genome_test.py'),
#             'description': 'Простой тест геномов'
#         },
#         {
#             'script': os.path.join(EXPERIMENTS_DIR, 'run_evolution.py'),
#             'description': 'Запуск эволюции с GAN'
#         },
#         {
#             'script': os.path.join(SCRIPTS_DIR, 'compare_evolution_with_gan.py'),
#             'description': 'Сравнение эволюции с GAN и без'
#         },
#         {
#             'script': os.path.join(SCRIPTS_DIR, 'visualize_gan_training.py'),
#             'description': 'Визуализация обучения GAN'
#         }
#     ]
#
#     results = []
#
#     for test in tests:
#         success = run_script(test['script'], test['description'])
#         results.append((test['description'], success))
#
#         # Пауза между тестами
#         print("\nНажмите Enter для продолжения или 'q' для выхода...")
#         user_input = input().strip().lower()
#         if user_input == 'q':
#             print("Прерывание выполнения...")
#             break
#
#     # Выводим итоговый отчет
#     print("\n" + "=" * 70)
#     print("ИТОГОВЫЙ ОТЧЕТ")
#     print("=" * 70)
#
#     successful = sum(1 for _, success in results if success)
#     total = len(results)
#
#     for description, success in results:
#         status = "✅" if success else "❌"
#         print(f"{status} {description}")
#
#     print("\n" + "=" * 70)
#     print(f"ВСЕГО: {successful}/{total} тестов пройдено успешно")
#
#     if successful == total:
#         print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
#     else:
#         print("⚠️ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ")
#
#     print("=" * 70)
#
#
# def run_specific_test(test_name):
#     """Запускает конкретный тест по имени."""
#     test_map = {
#         'gan': os.path.join(SCRIPTS_DIR, 'test_gan.py'),
#         'gan_only': os.path.join(SCRIPTS_DIR, 'test_gan_only.py'),
#         'integration': os.path.join(TESTS_DIR, 'test_gan_integration.py'),
#         'genome': os.path.join(SCRIPTS_DIR, 'test_genome_improvement.py'),
#         'simple_genome': os.path.join(SCRIPTS_DIR, 'simple_genome_test.py'),
#         'evolution': os.path.join(EXPERIMENTS_DIR, 'run_evolution.py'),
#         'compare': os.path.join(SCRIPTS_DIR, 'compare_evolution_with_gan.py'),
#         'visualize': os.path.join(SCRIPTS_DIR, 'visualize_gan_training.py')
#     }
#
#     if test_name in test_map:
#         run_script(test_map[test_name], f"Тест: {test_name}")
#     else:
#         print(f"❌ Тест '{test_name}' не найден")
#         print(f"Доступные тесты: {', '.join(test_map.keys())}")
#
#
# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         # Если передан аргумент, запускаем конкретный тест
#         run_specific_test(sys.argv[1])
#     else:
#         # Иначе запускаем все тесты
#         run_all_tests()
#
#
#
# # import sys
# # import os
# # import subprocess
# # import platform
# #
# #
# # # Определяем правильную команду Python
# # def get_python_command():
# #     """Определяет, какая команда Python доступна в системе."""
# #     # Проверяем python3
# #     try:
# #         subprocess.run(['python3', '--version'], capture_output=True, check=True)
# #         return 'python3'
# #     except (subprocess.CalledProcessError, FileNotFoundError):
# #         pass
# #
# #     # Проверяем python
# #     try:
# #         subprocess.run(['python', '--version'], capture_output=True, check=True)
# #         return 'python'
# #     except (subprocess.CalledProcessError, FileNotFoundError):
# #         pass
# #
# #     # Проверяем python на Windows
# #     if platform.system() == 'Windows':
# #         try:
# #             subprocess.run(['py', '--version'], capture_output=True, check=True)
# #             return 'py'
# #         except (subprocess.CalledProcessError, FileNotFoundError):
# #             pass
# #
# #     raise RuntimeError("Не удалось найти Python. Убедитесь, что Python установлен и доступен в PATH.")
# #
# #
# # # Получаем путь к текущему скрипту
# # SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# # os.chdir(SCRIPT_DIR)  # Переходим в корневую директорию проекта
# #
# # PYTHON_CMD = get_python_command()
# # print(f"Используется команда Python: {PYTHON_CMD}")
# #
# #
# # def run_script(script_path):
# #     """Запускает Python скрипт с правильной командой."""
# #     full_path = os.path.join(SCRIPT_DIR, script_path)
# #
# #     # Проверяем, существует ли файл
# #     if not os.path.exists(full_path):
# #         print(f"❌ Файл не найден: {full_path}")
# #         return False
# #
# #     try:
# #         result = subprocess.run(
# #             [PYTHON_CMD, full_path],
# #             capture_output=False,
# #             text=True,
# #             cwd=SCRIPT_DIR
# #         )
# #         return result.returncode == 0
# #     except Exception as e:
# #         print(f"❌ Ошибка при запуске {script_path}: {e}")
# #         return False
# #
# #
# # def run_all_experiments():
# #     """Запускает все эксперименты для проверки GAN."""
# #
# #     print("=" * 70)
# #     print("ЗАПУСК ВСЕХ ЭКСПЕРИМЕНТОВ ДЛЯ ПРОВЕРКИ GAN")
# #     print("=" * 70)
# #
# #     experiments = [
# #         ("Тестирование GAN", "scripts/test_gan.py"),
# #         ("Сравнение эволюции с GAN и без GAN", "scripts/compare_evolution_with_gan.py"),
# #         ("Тестирование улучшения генов", "scripts/test_genome_improvement.py"),
# #         ("Интеграционные тесты", "tests/test_gan_integration.py")
# #     ]
# #
# #     results = []
# #
# #     for name, script_path in experiments:
# #         print(f"\n\n{'=' * 70}")
# #         print(f"ЗАПУСК: {name}")
# #         print(f"Скрипт: {script_path}")
# #         print("=" * 70)
# #
# #         success = run_script(script_path)
# #
# #         if success:
# #             print(f"✅ {name} завершён успешно")
# #             results.append((name, True))
# #         else:
# #             print(f"❌ {name} завершился с ошибкой")
# #             results.append((name, False))
# #
# #         print("\nНажмите Enter для продолжения...")
# #         input()
# #
# #     # Выводим итоговый отчёт
# #     print("\n\n" + "=" * 70)
# #     print("ИТОГОВЫЙ ОТЧЁТ")
# #     print("=" * 70)
# #
# #     for name, success in results:
# #         status = "✅ УСПЕШНО" if success else "❌ ОШИБКА"
# #         print(f"{name}: {status}")
# #
# #     total = len(results)
# #     passed = sum(1 for _, success in results if success)
# #     print(f"\nПройдено: {passed}/{total} экспериментов")
# #
# #
# # if __name__ == "__main__":
# #     try:
# #         run_all_experiments()
# #     except KeyboardInterrupt:
# #         print("\n\n⚠️ Эксперименты прерваны пользователем")
# #     except Exception as e:
# #         print(f"\n\n❌ Критическая ошибка: {e}")
#
# # # run_all_tests.py
# # import sys
# # import os
# #
# # sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# #
# # import subprocess
# # import time
# #
# #
# # def run_all_experiments():
# #     """Запускает все эксперименты для проверки GAN."""
# #
# #     print("=" * 70)
# #     print("ЗАПУСК ВСЕХ ЭКСПЕРИМЕНТОВ ДЛЯ ПРОВЕРКИ GAN")
# #     print("=" * 70)
# #
# #     experiments = [
# #         ("Тестирование GAN", "python scripts/test_gan.py"),
# #         ("Сравнение эволюции с GAN и без GAN", "python scripts/compare_evolution_with_gan.py"),
# #         ("Тестирование улучшения генов", "python scripts/test_genome_improvement.py"),
# #         ("Интеграционные тесты", "python tests/test_gan_integration.py")
# #     ]
# #
# #     for name, command in experiments:
# #         print(f"\n\n{'=' * 70}")
# #         print(f"ЗАПУСК: {name}")
# #         print(f"Команда: {command}")
# #         print("=" * 70)
# #
# #         try:
# #             result = subprocess.run(command, shell=True, capture_output=False, text=True)
# #             if result.returncode != 0:
# #                 print(f"⚠️ Эксперимент '{name}' завершился с ошибкой (код {result.returncode})")
# #         except Exception as e:
# #             print(f"❌ Ошибка при запуске эксперимента '{name}': {e}")
# #
# #         print("\nНажмите Enter для продолжения к следующему эксперименту...")
# #         input()
# #
# #
# # if __name__ == "__main__":
# #     run_all_experiments()