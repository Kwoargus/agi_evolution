# install_deps.py
"""
Скрипт для установки зависимостей с учётом особенностей системы.
"""
import subprocess
import sys
import os
# Добавляем корневую директорию проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platform


def get_python_version():
    """Возвращает версию Python."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def install_with_venv():
    """Устанавливает зависимости в виртуальное окружение."""
    print("=" * 60)
    print("УСТАНОВКА ЗАВИСИМОСТЕЙ")
    print("=" * 60)

    python_version = get_python_version()
    print(f"Версия Python: {python_version}")

    if python_version.startswith('3.14'):
        print("\n⚠️ Обнаружена альфа/бета версия Python 3.14.")
        print("   Рекомендуется использовать Python 3.10 для лучшей совместимости.")

        choice = input("   Продолжить с Python 3.14? (y/n): ")
        if choice.lower() != 'y':
            print("\nУстановите Python 3.10 и создайте новое виртуальное окружение:")
            print("  sudo apt-get install python3.10 python3.10-venv python3.10-dev")
            print("  python3.10 -m venv venv")
            print("  source venv/bin/activate")
            return False

    # Проверяем, есть ли системные зависимости
    print("\nПроверка системных зависимостей...")
    deps_installed = True

    try:
        subprocess.run(['sdl2-config', '--version'], capture_output=True, check=True)
        print("  ✅ SDL2 установлен")
    except:
        print("  ❌ SDL2 не найден")
        print("     Установите: sudo apt-get install libsdl2-dev")
        deps_installed = False

    if not deps_installed:
        print("\n⚠️ Установите системные зависимости:")
        print("  sudo apt-get update")
        print("  sudo apt-get install -y python3-dev libsdl2-dev libsdl2-image-dev")
        print("  sudo apt-get install -y libsdl2-mixer-dev libsdl2-ttf-dev libjpeg-dev libpng-dev")
        print("  sudo apt-get install -y libfreetype6-dev libportmidi-dev")

        choice = input("\nПродолжить установку? (y/n): ")
        if choice.lower() != 'y':
            return False

    # Устанавливаем зависимости
    packages = [
        'numpy<2.0',
        'matplotlib',
        'psycopg2-binary',
        'torch',
        'torchvision',
        'pandas'
    ]

    # pygame устанавливаем отдельно (может быть проблемным)
    print("\nУстановка основных пакетов...")
    for pkg in packages:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', pkg], check=True)
            print(f"  ✅ {pkg}")
        except Exception as e:
            print(f"  ❌ {pkg}: {e}")

    # Пробуем установить pygame
    print("\nУстановка pygame...")
    try:
        # Пробуем установить через pip
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pygame'], check=True)
        print("  ✅ pygame установлен")
    except:
        print("  ❌ pygame не установлен через pip")
        print("\nПопробуйте установить pygame альтернативными способами:")
        print("1. Через системный менеджер:")
        print("   sudo apt-get install python3-pygame")
        print("2. Через pip с бинарниками:")
        print("   pip install pygame --only-binary pygame")
        print("3. Через apt-get:")
        print("   pip install --user pygame")

    print("\n" + "=" * 60)
    print("УСТАНОВКА ЗАВЕРШЕНА")
    print("=" * 60)

    return True


if __name__ == "__main__":
    install_with_venv()