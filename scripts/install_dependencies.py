"""
Скрипт для установки всех необходимых зависимостей.
"""
import subprocess
import sys
import os
# Добавляем корневую директорию проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platform


def get_python_command():
    """Определяет команду Python."""
    try:
        subprocess.run(['python3', '--version'], capture_output=True, check=True)
        return 'python3'
    except:
        return 'python'


def install_dependencies():
    """Устанавливает все необходимые пакеты."""
    print("=" * 60)
    print("УСТАНОВКА ЗАВИСИМОСТЕЙ")
    print("=" * 60)

    python_cmd = get_python_command()
    print(f"Используется Python: {python_cmd}")

    # Проверяем pip
    try:
        subprocess.run([python_cmd, '-m', 'pip', '--version'], check=True, capture_output=True)
        print("✅ pip найден")
    except:
        print("❌ pip не найден. Установите pip.")
        return False

    dependencies = [
        'torch',
        'numpy',
        'matplotlib',
        'pygame',
        'psycopg2-binary',
        'pandas',
        'scipy'
    ]

    print("\nУстановка пакетов:")
    for dep in dependencies:
        print(f"  - {dep}")

    # Устанавливаем зависимости
    for dep in dependencies:
        print(f"\nУстановка {dep}...")
        try:
            subprocess.run(
                [python_cmd, '-m', 'pip', 'install', dep],
                check=True,
                capture_output=False
            )
            print(f"✅ {dep} установлен")
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка установки {dep}: {e}")
            return False

    # Проверяем установку
    print("\nПроверка установленных пакетов:")
    try:
        result = subprocess.run(
            [python_cmd, '-m', 'pip', 'list'],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except:
        pass

    print("\n✅ Все зависимости установлены!")
    return True


if __name__ == "__main__":
    if not install_dependencies():
        print("\n❌ Установка зависимостей не удалась.")
        sys.exit(1)