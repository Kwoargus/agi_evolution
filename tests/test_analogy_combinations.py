# tests/test_analogy_combinations.py
"""
Тест поиска комбинаций узлов в AnalogyEngine.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from core.knowledge.individual_knowledge_graph import IndividualKnowledgeGraph
from core.knowledge.analogy_engine import AnalogyEngine
from db.knowledge_db import KnowledgeDB


def test_combination_search():
    """Тест поиска комбинаций узлов."""
    print("\n" + "=" * 70)
    print("🔍 ТЕСТ ПОИСКА КОМБИНАЦИЙ УЗЛОВ")
    print("=" * 70)

    # Загружаем ГЗ из БД
    print("\n📥 Загрузка ГЗ из БД...")
    gg = GlobalKnowledgeGraph()
    gg.load_from_db()
    print(f"   ✅ Загружено: {len(gg.nodes)} узлов, {len(gg.edges)} связей")

    # Создаём ИГЗ
    ig = IndividualKnowledgeGraph(bot_id="test_bot")

    # Создаём движок аналогий
    engine = AnalogyEngine(gg, ig)

    # Тестовые задачи
    test_cases = [
        {
            'task': "Нужен летательный аппарат для перевозки грузов",
            'properties': [
                'создавать подъёмную силу',  # → крыло
                'создавать тягу',  # → двигатель
                'перевозить грузы',  # → фюзеляж/самолёт
                'управляемый полёт'  # → рули/автопилот
            ]
        },
        {
            'task': "Требуется система управления движением робота",
            'properties': [
                'управлять движением',  # → контроллер
                'обрабатывать сенсорные данные',  # → датчики
                'принимать решения',  # → ИИ
                'исполнять команды'  # → приводы
            ]
        },
        {
            'task': "Нужен механизм для подъема тяжелых предметов",
            'properties': [
                'создавать подъёмную силу',  # → лебёдка/гидравлика
                'удерживать груз',  # → захват/крепление
                'управлять подъёмом',  # → контроллер
                'преобразовывать энергию'  # → двигатель
            ]
        }
    ]

    # test_cases = [
    #     {
    #         'task': "Нужен летательный аппарат для перевозки грузов",
    #         'properties': ['летать', 'перевозить грузы', 'крыло', 'двигатель']
    #     },
    #     {
    #         'task': "Требуется система управления движением робота",
    #         'properties': ['управление', 'движение', 'робот', 'система']
    #     },
    #     {
    #         'task': "Нужен механизм для подъема тяжелых предметов",
    #         'properties': ['подъем', 'механизм', 'тяжелый']
    #     }
    # ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"📝 ЗАДАЧА {i}: {test_case['task']}")
        print("=" * 70)
        print(f"   Требуемые свойства: {test_case['properties']}")

        # Ищем аналогии
        analogies = engine.find_analogies(
            task_description=test_case['task'],
            required_properties=test_case['properties'],
            max_results=10,
            min_nodes=2,
            max_nodes=4
        )

        print(f"\n📊 Найдено комбинаций: {len(analogies)}")

        for j, analogy in enumerate(analogies[:5], 1):
            print(f"\n   Комбинация {j}:")
            print(f"      ID: {analogy.id}")
            print(f"      Источник: {analogy.metadata.get('source', 'unknown')}")
            print(f"      Узлов: {len(analogy.nodes)}")
            print(f"      Свойств: {len(analogy.properties)}")
            print(f"      Оценка: {analogy.metadata.get('score', 0):.3f}")
            print(f"      Покрытие: {analogy.metadata.get('coverage_ratio', 0):.3f}")

            # Показываем узлы
            node_names = [n.name for n in analogy.nodes[:5]]
            print(f"      Узлы: {', '.join(node_names)}")

            # Показываем покрытые свойства
            covered = analogy.metadata.get('covered_properties', [])
            if covered:
                print(f"      Покрытые свойства: {', '.join(covered[:5])}")

            # Показываем недостающие свойства
            missing = analogy.metadata.get('missing_properties', [])
            if missing:
                print(f"      Недостающие свойства: {', '.join(missing[:5])}")

        if not analogies:
            print("   ⚠️ Комбинации не найдены")

    print("\n" + "=" * 70)
    print("✅ ТЕСТ ЗАВЕРШЕН")


if __name__ == "__main__":
    test_combination_search()