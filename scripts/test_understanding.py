# scripts/test_understanding.py
"""
Тестирование модуля "Понимание".
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thinking.understanding import UnderstandingEngine, understand_task
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from db.knowledge_db import KnowledgeDB


def test_understanding():
    """Тестирует модуль понимания."""
    print("=" * 60)
    print("🧠 ТЕСТИРОВАНИЕ МОДУЛЯ ПОНИМАНИЯ")
    print("=" * 60)

    # Загружаем ГЗ
    print("\n📥 Загрузка графа знаний...")
    graph = GlobalKnowledgeGraph()
    db = KnowledgeDB()
    nodes = db.load_all_nodes()
    for node in nodes:
        graph.add_node(node)
    edges = db.load_all_edges()
    for edge in edges:
        graph.add_edge(edge)

    print(f"   Узлов: {len(graph.nodes)}")
    print(f"   Связей: {len(graph.edges)}")

    # Создаем движок понимания
    engine = UnderstandingEngine(graph)

    # Тестовые задачи
    test_tasks = [
        "Нужно создать летательный аппарат для перевозки грузов",
        "Требуется разработать систему управления движением робота",
        "Необходимо создать механизм для подъема тяжелых предметов",
        "Нужен двигатель для преобразования тепловой энергии в механическую",
        "Требуется разработать систему автоматического управления полетом",
        "Необходимо создать устройство для измерения давления в трубопроводе",
        "Нужна система для передачи данных на большие расстояния",
        "Требуется разработать механизм для изменения скорости вращения",
        "Необходимо создать систему охлаждения для мощного двигателя"
    ]

    print("\n" + "=" * 60)
    print("📝 ТЕСТОВЫЕ ЗАДАЧИ")
    print("=" * 60)

    results = []
    for i, task in enumerate(test_tasks):
        print(f"\n{i + 1}. {task}")
        result = engine.understand(task)
        results.append(result)

        print(f"   Концептов: {len(result.extracted_concepts)}")
        print(f"   Узлов: {len(result.found_nodes)}")
        print(f"   Модель: {result.new_model.id if result.new_model else 'Нет'}")
        print(f"   Время: {result.execution_time:.2f}с")

        if result.errors:
            print(f"   Ошибки: {result.errors}")

    # Статистика
    stats = engine.get_statistics()
    print("\n" + "=" * 60)
    print("📊 СТАТИСТИКА")
    print("=" * 60)
    print(f"   Всего операций: {stats['total_understandings']}")
    print(f"   Успешно: {stats['successful']}")
    print(f"   Ошибок: {stats['failed']}")
    print(f"   Среднее время: {stats['avg_time']:.2f}с")
    print(f"   Среднее концептов: {stats['avg_concepts']:.1f}")
    print(f"   Среднее узлов: {stats['avg_nodes']:.1f}")

    # Вывод лучшего результата
    best = max(results, key=lambda x: len(x.found_nodes))
    if best and best.found_nodes:
        print("\n🏆 ЛУЧШИЙ РЕЗУЛЬТАТ")
        print(f"   Задача: {best.task_description}")
        print(f"   Найдено узлов: {len(best.found_nodes)}")
        print(f"   Концепты: {best.extracted_concepts}")
        print(f"   Узлы: {[n.name for n in best.found_nodes[:5]]}")


def test_single_task():
    """Тест одной задачи с детальным выводом."""
    print("=" * 60)
    print("🔍 ДЕТАЛЬНЫЙ ТЕСТ ОДНОЙ ЗАДАЧИ")
    print("=" * 60)

    task = "Нужен двигатель для самолета, который будет поднимать грузы"

    print(f"\n📝 Задача: {task}")
    result = understand_task(task)

    print("\n📊 РЕЗУЛЬТАТ:")
    print(f"   Статус: {result.status.value}")
    print(f"   Концепты: {result.extracted_concepts}")
    print(f"   Найдено узлов: {len(result.found_nodes)}")

    if result.found_nodes:
        print("\n   Найденные узлы:")
        for node in result.found_nodes:
            print(f"      - {node.name} ({node.node_type})")
            print(f"        Свойства: {node.properties[:3]}")
            if node.functions:
                print(f"        Функции: {[f.name for f in node.functions[:2]]}")

    if result.mental_model:
        print(f"\n   Ментальная модель:")
        print(f"      ID: {result.mental_model.id}")
        print(f"      Имя: {result.mental_model.name}")
        print(f"      Свойств: {len(result.mental_model.properties)}")

    if result.errors:
        print(f"\n   Ошибки: {result.errors}")

    print(f"\n   Время выполнения: {result.execution_time:.2f}с")


if __name__ == "__main__":
    test_understanding()
    test_single_task()