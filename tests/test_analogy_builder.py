# tests/test_analogy_builder.py
"""
Тест построения аналогов.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from core.knowledge.individual_knowledge_graph import IndividualKnowledgeGraph
from core.knowledge.analogy_builder import AnalogyBuilder
from core.thinking.functional_extractor import FunctionalExtractor


def test_analogy_builder():
    print("\n" + "=" * 70)
    print("🏗️ ТЕСТ ПОСТРОЕНИЯ АНАЛОГОВ")
    print("=" * 70)

    # Загружаем ГЗ
    gg = GlobalKnowledgeGraph()
    gg.load_from_db()
    print(f"✅ Загружено {len(gg.nodes)} узлов")

    ig = IndividualKnowledgeGraph(bot_id="test_bot")
    builder = AnalogyBuilder(gg, ig)

    # Извлекаем функции из задачи
    task = "Нужен летательный аппарат с двигателем для перевозки пилота"
    extractor = FunctionalExtractor(use_llm=False)
    extractor._build_indexes()
    print("DEBUG: 'двигатель' in keyword_to_function:", 'двигатель' in extractor.keyword_to_function)
    print("DEBUG: keyword_to_function keys:", list(extractor.keyword_to_function.keys()))
    functions = extractor.extract_as_properties(task)

    print(f"\n📝 Задача: {task}")
    print(f"📝 Функции: {functions}")

    analogies = builder.build_analogy(
        functional_properties=functions,
        min_nodes=2,
        max_nodes=4
    )

    print(f"\n📊 Построено аналогов: {len(analogies)}")

    for i, analogy in enumerate(analogies[:5], 1):
        print(f"\n{'=' * 50}")
        print(f"📌 АНАЛОГ {i}")
        print("=" * 50)
        print(f"   ID: {analogy.id}")
        print(f"   Узлов: {len(analogy.nodes)}")
        print(f"   Качество: {analogy.metadata.get('quality_score', 0):.3f}")
        print(f"   Покрытие: {analogy.metadata.get('match_ratio', 0):.3f}")

        print(f"\n   Узлы:")
        for node in analogy.nodes:
            print(f"      - {node.name} ({node.node_type})")

        if 'main_node' in analogy.metadata:
            print(f"\n   Структура:")
            print(f"      Главный: {analogy.metadata.get('main_node')}")
            print(f"      Части: {analogy.metadata.get('parts', [])}")

        if 'connections' in analogy.metadata:
            print(f"\n   Связей: {len(analogy.metadata['connections'])}")
            for conn in analogy.metadata['connections'][:3]:
                print(f"      - {conn.get('description', '')}")

        print(f"\n   Недостающие функции: {analogy.metadata.get('missing_functions', [])}")

    print("\n" + "=" * 70)
    print("✅ ТЕСТ ЗАВЕРШЕН")


if __name__ == "__main__":
    test_analogy_builder()


# # tests/test_analogy_builder.py
# """
# Тест построения аналогов.
# """
#
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
# from core.knowledge.individual_knowledge_graph import IndividualKnowledgeGraph
# from core.knowledge.analogy_builder import AnalogyBuilder
#
#
# def test_analogy_builder():
#     print("\n" + "=" * 70)
#     print("🏗️ ТЕСТ ПОСТРОЕНИЯ АНАЛОГОВ")
#     print("=" * 70)
#
#     # Загружаем ГЗ
#     gg = GlobalKnowledgeGraph()
#     gg.load_from_db()
#     print(f"✅ Загружено {len(gg.nodes)} узлов")
#
#     ig = IndividualKnowledgeGraph(bot_id="test_bot")
#
#     builder = AnalogyBuilder(gg, ig)
#
#     # Функциональные свойства
#     functions = [
#         'создавать подъёмную силу',
#         'перевозить',
#         'поднимать груз',
#         'создавать тягу'
#     ]
#
#     print(f"\n📝 Функции: {functions}")
#
#     analogies = builder.build_analogy(
#         functional_properties=functions,
#         min_nodes=2,
#         max_nodes=4
#     )
#
#     print(f"\n📊 Построено аналогов: {len(analogies)}")
#
#     for i, analogy in enumerate(analogies[:5], 1):
#         print(f"\n{'=' * 50}")
#         print(f"📌 АНАЛОГ {i}")
#         print("=" * 50)
#         print(f"   ID: {analogy.id}")
#         print(f"   Узлов: {len(analogy.nodes)}")
#         print(f"   Качество: {analogy.metadata.get('quality_score', 0):.3f}")
#         print(f"   Покрытие: {analogy.metadata.get('match_ratio', 0):.3f}")
#
#         print(f"\n   Узлы:")
#         for node in analogy.nodes:
#             print(f"      - {node.name} ({node.node_type})")
#
#         if 'main_node' in analogy.metadata:
#             print(f"\n   Структура:")
#             print(f"      Главный: {analogy.metadata.get('main_node')}")
#             print(f"      Части: {analogy.metadata.get('parts', [])}")
#
#         if 'connections' in analogy.metadata:
#             print(f"\n   Связей: {len(analogy.metadata['connections'])}")
#             for conn in analogy.metadata['connections'][:3]:
#                 print(f"      - {conn.get('description', '')}")
#
#         print(f"\n   Недостающие функции: {analogy.metadata.get('missing_functions', [])}")
#
#     print("\n" + "=" * 70)
#     print("✅ ТЕСТ ЗАВЕРШЕН")
#
#
# if __name__ == "__main__":
#     test_analogy_builder()