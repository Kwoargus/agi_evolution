# tests/test_real_understanding.py
"""
Тест понимания с реальными данными из БД.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thinking.understanding import understand_task


def test_real_understanding():
    """Тест понимания с реальными данными из БД."""
    print("\n" + "=" * 70)
    print("🧠 ТЕСТ ПОНИМАНИЯ С РЕАЛЬНЫМИ ДАННЫМИ ИЗ БД")
    print("=" * 70)

    # Задачи для тестирования
    tasks = [
        "Нужен летательный аппарат для перевозки грузов",
        "Требуется разработать систему управления движением робота",
        "Необходимо создать механизм для подъема тяжелых предметов",
        "Нужен двигатель для преобразования тепловой энергии в механическую"
    ]

    for i, task in enumerate(tasks, 1):
        print(f"\n{'=' * 70}")
        print(f"📝 ЗАДАЧА {i}: {task}")
        print("=" * 70)

        result = understand_task(task, load_from_db=True)

        print(f"\n📊 РЕЗУЛЬТАТ:")
        print(f"   Концептов извлечено: {len(result.extracted_concepts)}")
        print(f"   Найдено узлов: {len(result.found_nodes)}")

        if result.found_nodes:
            print(f"\n   Найденные узлы:")
            for node in result.found_nodes[:5]:
                print(f"      - {node.name} ({node.node_type})")
                if node.properties:
                    print(f"        Свойства: {', '.join(node.properties[:3])}")

        if result.mental_model:
            print(f"\n   Ментальная модель:")
            print(f"      ID: {result.mental_model.id}")
            print(f"      Имя: {result.mental_model.name}")
            print(f"      Свойств: {len(result.mental_model.properties)}")
            if result.mental_model.properties:
                print(f"      Свойства: {', '.join(result.mental_model.properties[:5])}")
        else:
            print(f"\n   ❌ Модель не создана!")

        if result.errors:
            print(f"\n   ⚠️ Ошибки: {result.errors}")

        print(f"\n   ⏱️ Время: {result.execution_time:.2f}с")


if __name__ == "__main__":
    test_real_understanding()