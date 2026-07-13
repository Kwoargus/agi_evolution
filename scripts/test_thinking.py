# scripts/test_thinking.py
"""
Тестирование системы мышления.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.bot_with_thinking import ThinkingBot


def test_understanding():
    """Тестирует этап понимания."""
    print("\n" + "=" * 60)
    print("🧠 ТЕСТ ПОНИМАНИЯ")
    print("=" * 60)

    bot = ThinkingBot(0, 0)

    # Задачи для тестирования
    tasks = [
        "Нужно создать летательный аппарат для перевозки грузов",
        "Требуется разработать систему управления движением робота",
        "Необходимо создать механизм для подъема тяжелых предметов",
        "Нужен двигатель для преобразования тепловой энергии в механическую"
    ]

    for task in tasks:
        print(f"\n📝 Задача: {task}")
        result = bot.think_understand(task)
        print(f"   Концепты: {result['concepts']}")
        print(f"   Найдено узлов: {result['nodes_found']}")
        print(f"   Модель: {result['model_created']}")


def test_research():
    """Тестирует этап исследования."""
    print("\n" + "=" * 60)
    print("🔬 ТЕСТ ИССЛЕДОВАНИЯ")
    print("=" * 60)

    bot = ThinkingBot(0, 0)

    # Проблемы для тестирования
    problems = [
        "Как создать летательный аппарат тяжелее воздуха?",
        "Как сделать систему, которая может поднимать грузы?",
        "Как создать эффективный механизм передачи энергии?"
    ]

    for problem in problems:
        print(f"\n📝 Проблема: {problem}")
        result = bot.think_research(problem)
        print(f"   Требования: {result['requirements']}")
        print(f"   Аналогий: {result['analogies']}")
        print(f"   Гипотез сгенерировано: {result['hypotheses_generated']}")
        print(f"   Гипотез подтверждено: {result['hypotheses_validated']}")


def main():
    print("=" * 60)
    print("🧠 ТЕСТИРОВАНИЕ СИСТЕМЫ МЫШЛЕНИЯ")
    print("=" * 60)

    test_understanding()
    test_research()

    print("\n" + "=" * 60)
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 60)


if __name__ == "__main__":
    main()