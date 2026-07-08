# check_setup.py

import sys
import os
# Добавляем корневую директорию проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Проверяет установку всех зависимостей.
"""
import sys
import importlib


def check_module(module_name):
    """Проверяет наличие модуля."""
    try:
        importlib.import_module(module_name)
        return True, "✅"
    except ImportError as e:
        return False, f"❌ {e}"


def main():
    print("=" * 60)
    print("ПРОВЕРКА УСТАНОВКИ ЗАВИСИМОСТЕЙ")
    print("=" * 60)

    print(f"Python версия: {sys.version}")

    modules = [
        'numpy',
        'matplotlib',
        'psycopg2',
        'torch',
        'torchvision',
        'pygame',
        'pandas'
    ]

    missing = []
    print("\nПроверка модулей:")
    for mod in modules:
        ok, status = check_module(mod)
        print(f"  {status} {mod}")
        if not ok:
            missing.append(mod)

    if missing:
        print(f"\n⚠️ Отсутствуют модули: {', '.join(missing)}")
        print("\nУстановите их:")
        for mod in missing:
            if mod == 'pygame':
                print(f"  sudo apt-get install python3-pygame")
            else:
                print(f"  pip install {mod}")
        return False
    else:
        print("\n✅ Все модули установлены!")
        return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)