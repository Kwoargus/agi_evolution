# tests/test_hypothesis_validator.py
"""
Тест проверки гипотез.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from core.knowledge.individual_knowledge_graph import IndividualKnowledgeGraph
from core.knowledge.analogy_builder import AnalogyBuilder
from core.knowledge.hypothesis_generator import HypothesisGenerator
from core.knowledge.hypothesis_validator import HypothesisValidator
from core.thinking.functional_extractor import FunctionalExtractor


def test_hypothesis_validation():
    print("\n" + "=" * 70)
    print("🧪 ТЕСТ ПРОВЕРКИ ГИПОТЕЗ")
    print("=" * 70)

    # Загружаем ГЗ
    gg = GlobalKnowledgeGraph()
    gg.load_from_db()
    print(f"✅ Загружено {len(gg.nodes)} узлов")

    ig = IndividualKnowledgeGraph(bot_id="test_bot")

    # 1. Извлекаем функции
    task = "Нужен летательный аппарат с двигателем для перевозки пилота"
    extractor = FunctionalExtractor(use_llm=False)
    functions = extractor.extract_as_properties(task)
    print(f"\n📝 Задача: {task}")
    print(f"📝 Функции: {functions}")

    # 2. Строим аналоги
    builder = AnalogyBuilder(gg, ig)
    analogies = builder.build_analogy(
        functional_properties=functions,
        min_nodes=2,
        max_nodes=4
    )

    # 3. Генерируем гипотезы
    generator = HypothesisGenerator(gg, ig)
    hypotheses = generator.generate_hypotheses(
        analogies=analogies,
        required_functions=functions,
        max_hypotheses=5
    )

    print(f"\n📊 Сгенерировано гипотез: {len(hypotheses)}")

    # 4. Проверяем гипотезы
    validator = HypothesisValidator(gg)

    for i, hyp in enumerate(hypotheses, 1):
        print(f"\n{'=' * 50}")
        print(f"📌 ПРОВЕРКА ГИПОТЕЗЫ {i}")
        print("=" * 50)

        is_valid, score, missing = validator.validate(hyp, functions)
        print(f"   Статус: {'✅ ВАЛИДНА' if is_valid else '❌ ОТКЛОНЕНА'}")
        print(f"   Оценка: {score:.2f}")
        print(f"   Недостающие функции: {missing}")

        suggestions = validator.suggest_improvements(hyp, functions)
        if suggestions:
            print(f"\n   💡 Рекомендации:")
            for s in suggestions:
                print(f"      - {s}")

    print("\n" + "=" * 70)
    print("✅ ТЕСТ ЗАВЕРШЕН")


if __name__ == "__main__":
    test_hypothesis_validation()



# # tests/test_hypothesis_generator.py
# """
# Тест генерации гипотез.
# """
#
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
# from core.knowledge.individual_knowledge_graph import IndividualKnowledgeGraph
# from core.knowledge.analogy_builder import AnalogyBuilder
# from core.knowledge.hypothesis_generator import HypothesisGenerator
# from core.thinking.functional_extractor import FunctionalExtractor
#
#
# def test_hypothesis_generation():
#     print("\n" + "=" * 70)
#     print("🧠 ТЕСТ ГЕНЕРАЦИИ ГИПОТЕЗ")
#     print("=" * 70)
#
#     # Загружаем ГЗ
#     gg = GlobalKnowledgeGraph()
#     gg.load_from_db()
#     print(f"✅ Загружено {len(gg.nodes)} узлов")
#
#     ig = IndividualKnowledgeGraph(bot_id="test_bot")
#
#     # 1. Извлекаем функции
#     task = "Нужен летательный аппарат с двигателем для перевозки грузов"
#     extractor = FunctionalExtractor(use_llm=False)
#     functions = extractor.extract_as_properties(task)
#     print(f"\n📝 Задача: {task}")
#     print(f"📝 Функции: {functions}")
#
#     # 2. Строим аналоги
#     builder = AnalogyBuilder(gg, ig)
#     analogies = builder.build_analogy(
#         functional_properties=functions,
#         min_nodes=2,
#         max_nodes=4
#     )
#     print(f"\n📊 Найдено аналогов: {len(analogies)}")
#     for idx, a in enumerate(analogies):
#         print(f"   {idx+1}. Узлов: {len(a.nodes)}, оценка: {a.metadata.get('quality_score', 0):.3f}")
#
#     # 3. Генерируем гипотезы
#     generator = HypothesisGenerator(gg, ig)
#     hypotheses = generator.generate_hypotheses(
#         analogies=analogies,
#         required_functions=functions,
#         max_hypotheses=5
#     )
#
#     print(f"\n📊 Сгенерировано гипотез: {len(hypotheses)}")
#
#     for i, hyp in enumerate(hypotheses, 1):
#         print(f"\n{'=' * 50}")
#         print(f"📌 ГИПОТЕЗА {i}")
#         print("=" * 50)
#         print(f"   ID: {hyp.id}")
#         print(f"   Оценка: {hyp.predicted_score:.3f}")
#         print(f"   Статус: {hyp.status.value}")
#         print(f"   Описание: {hyp.description}")
#
#         print(f"\n   Узлы:")
#         for node in hyp.source_combination.nodes:
#             print(f"      - {node.name} ({node.node_type})")
#
#         if hyp.metadata.get('added_nodes'):
#             print(f"\n   Добавленные узлы:")
#             for name in hyp.metadata['added_nodes']:
#                 print(f"      + {name}")
#
#         print(f"\n   Покрытые функции: {hyp.metadata.get('covered_functions', [])}")
#         print(f"   Недостающие функции: {hyp.metadata.get('missing_functions', [])}")
#
#     print("\n" + "=" * 70)
#     print("✅ ТЕСТ ЗАВЕРШЕН")
#
#
# if __name__ == "__main__":
#     test_hypothesis_generation()