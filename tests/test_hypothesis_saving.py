# tests/test_hypothesis_saving.py
"""
Тест сохранения гипотез.
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
from db.knowledge_db import KnowledgeDB


def test_hypothesis_saving():
    print("\n" + "=" * 70)
    print("💾 ТЕСТ СОХРАНЕНИЯ ГИПОТЕЗ")
    print("=" * 70)

    # Загружаем ГЗ
    gg = GlobalKnowledgeGraph()
    gg.load_from_db()
    print(f"✅ Загружено {len(gg.nodes)} узлов")

    ig = IndividualKnowledgeGraph(bot_id="test_bot_saving")

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
        task_description=task,  # ← ПЕРЕДАЁМ ЗАДАЧУ
        max_hypotheses=5
    )

    print(f"\n📊 Сгенерировано гипотез: {len(hypotheses)}")

    # 4. Проверяем и валидируем
    validator = HypothesisValidator(gg)

    validated = []
    for hyp in hypotheses:
        is_valid, score, missing = validator.validate(hyp, functions)
        if is_valid:
            validated.append(hyp)
            print(f"✅ Гипотеза {hyp.id} валидна (score: {score:.2f})")
        else:
            print(f"❌ Гипотеза {hyp.id} отклонена (score: {score:.2f})")

    # 5. Сохраняем валидные гипотезы
    print(f"\n💾 Сохранение {len(validated)} валидных гипотез...")

    saved = generator.save_validated_hypotheses(validated)

    print(f"\n✅ Сохранено гипотез: {saved}")

    # 6. Проверяем, что гипотезы сохранились в БД
    db = KnowledgeDB()
    all_hypotheses = db.load_all_hypotheses()

    print(f"\n📊 Всего гипотез в БД: {len(all_hypotheses)}")

    # 7. Проверяем, что сохранились в ИГЗ
    stats = ig.get_statistics()
    print(f"\n📊 ИГЗ статистика:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 70)
    print("✅ ТЕСТ ЗАВЕРШЕН")


if __name__ == "__main__":
    test_hypothesis_saving()