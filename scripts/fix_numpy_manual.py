# fix_numpy_manual.py
"""
Ручное исправление проблем с NumPy.
"""
import subprocess
import sys
import os
# Добавляем корневую директорию проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_python_command():
    """Определяет команду Python."""
    try:
        subprocess.run(['python3', '--version'], capture_output=True, check=True)
        return 'python3'
    except:
        return 'python'


def check_current_numpy():
    """Проверяет текущую версию NumPy."""
    python_cmd = get_python_command()
    try:
        result = subprocess.run(
            [python_cmd, '-c', 'import numpy; print(numpy.__version__)'],
            capture_output=True,
            text=True,
            check=True
        )
        version = result.stdout.strip()
        print(f"Текущая версия NumPy: {version}")
        return version
    except:
        print("NumPy не установлен")
        return None


def fix_with_virtualenv():
    """Создаёт виртуальное окружение и устанавливает совместимые версии."""
    print("\n" + "=" * 60)
    print("СОЗДАНИЕ ВИРТУАЛЬНОГО ОКРУЖЕНИЯ")
    print("=" * 60)

    # Проверяем, есть ли venv
    if not os.path.exists('../venv'):
        print("Создание виртуального окружения...")
        try:
            subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
            print("✅ Виртуальное окружение создано")
        except Exception as e:
            print(f"❌ Ошибка создания виртуального окружения: {e}")
            return False

    # Определяем путь к python в venv
    if sys.platform == 'win32':
        venv_python = os.path.join('../venv', 'Scripts', 'python.exe')
        venv_pip = os.path.join('../venv', 'Scripts', 'pip.exe')
    else:
        venv_python = os.path.join('../venv', 'bin', 'python')
        venv_pip = os.path.join('../venv', 'bin', 'pip')

    if not os.path.exists(venv_python):
        print(f"❌ Не найден python в виртуальном окружении: {venv_python}")
        return False

    print(f"Используется Python из venv: {venv_python}")

    # Устанавливаем зависимости в venv
    print("\nУстановка зависимостей в виртуальное окружение...")

    packages = [
        'numpy<2.0',
        'matplotlib',
        'psycopg2-binary',
        'torch',
        'pygame',
        'pandas'
    ]

    for pkg in packages:
        print(f"  Установка {pkg}...")
        try:
            subprocess.run([venv_pip, 'install', pkg], check=True, capture_output=False)
            print(f"    ✅ {pkg} установлен")
        except Exception as e:
            print(f"    ❌ Ошибка установки {pkg}: {e}")

    print("\n" + "=" * 60)
    print("ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ")
    print("=" * 60)
    print("\nДля использования виртуального окружения:")
    print("  source venv/bin/activate  # Linux/macOS")
    print("  venv\\Scripts\\activate     # Windows")
    print("\nДля проверки версии NumPy:")
    print("  python -c 'import numpy; print(numpy.__version__)'")
    print("\nДля запуска экспериментов:")
    print("  python run_all_tests.py")

    return True


def fix_with_user_install():
    """Устанавливает совместимую версию в пользовательскую директорию."""
    print("\n" + "=" * 60)
    print("УСТАНОВКА В ПОЛЬЗОВАТЕЛЬСКУЮ ДИРЕКТОРИЮ")
    print("=" * 60)

    python_cmd = get_python_command()

    # Устанавливаем numpy с приоритетом пользовательской установки
    print("Установка numpy<2.0 в пользовательскую директорию...")
    try:
        subprocess.run(
            [python_cmd, '-m', 'pip', 'install', '--user', '--upgrade', 'numpy<2.0'],
            check=True,
            capture_output=False
        )
        print("✅ numpy установлен в пользовательскую директорию")

        # Проверяем
        result = subprocess.run(
            [python_cmd, '-c', 'import numpy; print(numpy.__version__)'],
            capture_output=True,
            text=True
        )
        print(f"  Версия после установки: {result.stdout.strip()}")
        return True
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


def fix_with_sudo():
    """Использует sudo для удаления numpy (на Linux/macOS)."""
    print("\n" + "=" * 60)
    print("ИСПОЛЬЗОВАНИЕ SUDO")
    print("=" * 60)

    if sys.platform == 'win32':
        print("Sudo не доступен на Windows")
        return False

    python_cmd = get_python_command()

    print("Удаление numpy с sudo...")
    try:
        subprocess.run(
            ['sudo', python_cmd, '-m', 'pip', 'uninstall', 'numpy', '-y'],
            check=True,
            capture_output=False
        )
        print("✅ numpy удалён")

        print("Установка numpy<2.0...")
        subprocess.run(
            ['sudo', python_cmd, '-m', 'pip', 'install', 'numpy<2.0'],
            check=True,
            capture_output=False
        )
        print("✅ numpy установлен")
        return True
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("   Возможно, нужно ввести пароль sudo")
        return False


def fix_with_alternative_import():
    """Использует обходной путь для импорта numpy."""
    print("\n" + "=" * 60)
    print("ОБХОДНОЙ ПУТЬ ДЛЯ NUMPY")
    print("=" * 60)

    # Проверяем, можно ли использовать numpy без проблем
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} уже работает")
        return True
    except Exception as e:
        print(f"❌ NumPy не работает: {e}")

        # Пробуем импортировать через другой путь
        try:
            import sys
            import site
            site_packages = site.getsitepackages()
            print(f"Пути site-packages: {site_packages}")

            # Пробуем добавить пользовательские пути
            user_site = site.getusersitepackages()
            if user_site not in sys.path:
                sys.path.insert(0, user_site)
                print(f"Добавлен пользовательский путь: {user_site}")

            import numpy as np
            print(f"✅ NumPy {np.__version__} теперь работает")
            return True
        except Exception as e2:
            print(f"❌ Всё равно не работает: {e2}")
            return False


def main():
    """Главная функция."""
    print("=" * 60)
    print("ИСПРАВЛЕНИЕ ПРОБЛЕМ С NUMPY")
    print("=" * 60)

    # Проверяем текущую версию
    version = check_current_numpy()

    if version and version.startswith('2.'):
        print("\n⚠️ Обнаружена NumPy 2.x. Рекомендую использовать виртуальное окружение.")

        # Предлагаем варианты
        print("\nВыберите вариант исправления:")
        print("1. Создать виртуальное окружение (рекомендуется)")
        print("2. Установить в пользовательскую директорию")
        print("3. Использовать sudo (Linux/macOS)")
        print("4. Попробовать обходной путь")
        print("5. Пропустить (продолжить с проблемой)")

        choice = input("\nВаш выбор (1-5): ").strip()

        if choice == '1':
            fix_with_virtualenv()
        elif choice == '2':
            fix_with_user_install()
        elif choice == '3':
            fix_with_sudo()
        elif choice == '4':
            fix_with_alternative_import()
        else:
            print("\n⚠️ Продолжаем с существующей версией NumPy.")
            print("   Если будут проблемы, создайте виртуальное окружение.")
    else:
        print("✅ NumPy версия совместима")

    # Проверяем результат
    print("\n" + "=" * 60)
    print("ПРОВЕРКА ПОСЛЕ ИСПРАВЛЕНИЯ")
    print("=" * 60)

    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} работает!")
        return True
    except Exception as e:
        print(f"❌ NumPy всё ещё не работает: {e}")
        print("\nРекомендую использовать виртуальное окружение:")
        print("  python3 -m venv venv")
        print("  source venv/bin/activate")
        print("  pip install numpy<2.0 matplotlib psycopg2-binary torch pygame")
        return False


if __name__ == "__main__":
    main()