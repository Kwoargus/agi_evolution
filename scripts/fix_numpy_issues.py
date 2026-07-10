# fix_numpy_issues.py
"""
Скрипт для решения проблем с несовместимостью NumPy версий.
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


def fix_numpy():
    """Исправляет проблемы с NumPy."""
    print("=" * 60)
    print("ИСПРАВЛЕНИЕ ПРОБЛЕМ С NumPy")
    print("=" * 60)

    python_cmd = get_python_command()
    print(f"Используется Python: {python_cmd}")

    # Проверяем текущую версию NumPy
    try:
        result = subprocess.run(
            [python_cmd, '-c', 'import numpy; print(numpy.__version__)'],
            capture_output=True,
            text=True,
            check=True
        )
        current_version = result.stdout.strip()
        print(f"Текущая версия NumPy: {current_version}")
    except:
        current_version = None

    # Если версия 2.x, downgrade до 1.x
    if current_version and current_version.startswith('2.'):
        print("\n⚠️ Обнаружена NumPy 2.x. Некоторые модули не совместимы.")
        print("   Выполняется downgrade до NumPy 1.x...")

        try:
            # Удаляем текущую версию
            subprocess.run(
                [python_cmd, '-m', 'pip', 'uninstall', 'numpy', '-y'],
                check=True,
                capture_output=False
            )
            print("✅ NumPy удалён")

            # Устанавливаем совместимую версию
            subprocess.run(
                [python_cmd, '-m', 'pip', 'install', 'numpy<2.0'],
                check=True,
                capture_output=False
            )
            print("✅ NumPy 1.x установлен")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка: {e}")
            return False
    else:
        print("✅ Версия NumPy совместима")
        return True


def reinstall_matplotlib():
    """Переустанавливает matplotlib."""
    print("\nПереустановка matplotlib...")
    python_cmd = get_python_command()

    try:
        subprocess.run(
            [python_cmd, '-m', 'pip', 'uninstall', 'matplotlib', '-y'],
            check=True,
            capture_output=False
        )
        subprocess.run(
            [python_cmd, '-m', 'pip', 'install', 'matplotlib'],
            check=True,
            capture_output=False
        )
        print("✅ matplotlib переустановлен")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка: {e}")
        return False


def install_psycopg2():
    """Устанавливает psycopg2."""
    print("\nУстановка psycopg2...")
    python_cmd = get_python_command()

    try:
        # Пробуем установить psycopg2-binary
        subprocess.run(
            [python_cmd, '-m', 'pip', 'install', 'psycopg2-binary'],
            check=True,
            capture_output=False
        )
        print("✅ psycopg2-binary установлен")
        return True
    except subprocess.CalledProcessError:
        try:
            # Пробуем установить psycopg2
            subprocess.run(
                [python_cmd, '-m', 'pip', 'install', 'psycopg2'],
                check=True,
                capture_output=False
            )
            print("✅ psycopg2 установлен")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка установки psycopg2: {e}")
            print("   Возможно, нужны системные зависимости:")
            print("   sudo apt-get install libpq-dev python3-dev")
            return False


if __name__ == "__main__":
    print("Исправление проблем с зависимостями...\n")

    # Фиксим NumPy
    numpy_fixed = fix_numpy()

    # Переустанавливаем matplotlib
    matplotlib_fixed = reinstall_matplotlib()

    # Устанавливаем psycopg2
    psycopg2_fixed = install_psycopg2()

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ИСПРАВЛЕНИЯ")
    print("=" * 60)
    print(f"NumPy: {'✅ Исправлен' if numpy_fixed else '❌ Ошибка'}")
    print(f"matplotlib: {'✅ Переустановлен' if matplotlib_fixed else '❌ Ошибка'}")
    print(f"psycopg2: {'✅ Установлен' if psycopg2_fixed else '❌ Ошибка'}")

    if numpy_fixed and matplotlib_fixed and psycopg2_fixed:
        print("\n✅ Все проблемы исправлены! Запустите test_setup.py снова.")
    else:
        print("\n⚠️ Некоторые проблемы не удалось исправить автоматически.")
        print("   Попробуйте выполнить вручную:")
        print("   pip3 install numpy<2.0 matplotlib psycopg2-binary")