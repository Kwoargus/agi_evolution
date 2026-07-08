# run.py - универсальный запускатор
import sys
import os
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON_CMD = sys.executable or "python3"


def run_script(script_path):
    """Запускает Python скрипт из корневой директории."""
    full_path = os.path.join(PROJECT_ROOT, script_path)
    if not os.path.exists(full_path):
        print(f"❌ Файл не найден: {full_path}")
        return False

    # Добавляем PROJECT_ROOT в PYTHONPATH
    env = os.environ.copy()
    env['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + env.get('PYTHONPATH', '')

    result = subprocess.run(
        [PYTHON_CMD, full_path],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=False
    )
    return result.returncode == 0


if __name__ == "__main__":
    scripts = [
        ("scripts/test_gan.py", "Тестирование GAN"),
        ("experiments/run_evolution.py", "Запуск эволюции"),
        ("tests/test_genome_improvement.py", "Тестирование геномов"),
    ]

    for script, description in scripts:
        print(f"\n{'=' * 60}")
        print(f"Запуск: {description}")
        print(f"{'=' * 60}")
        run_script(script)