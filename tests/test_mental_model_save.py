# tests/test_mental_model_save.py
from core.thinking.understanding import UnderstandingEngine
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from core.knowledge.individual_knowledge_graph import IndividualKnowledgeGraph
from db.knowledge_db import KnowledgeDB


def test_save_mental_model():
    print("🧠 ТЕСТ СОХРАНЕНИЯ МЕНТАЛЬНОЙ МОДЕЛИ")
    print("=" * 60)

    # Создаём графы
    gg = GlobalKnowledgeGraph()
    ig = IndividualKnowledgeGraph(bot_id="test_bot")

    # Создаём движок понимания
    engine = UnderstandingEngine(gg, ig)

    # Задача
    task = "Нужен летательный аппарат для перевозки грузов"

    # Понимание
    result = engine.understand(task)

    print(f"\n📌 Результат:")
    print(f"   Модель создана: {result.mental_model is not None}")
    print(f"   ID модели: {result.mental_model.id if result.mental_model else 'Нет'}")

    # Проверяем БД
    db = KnowledgeDB()
    models = db.load_all_mental_models()

    print(f"\n📌 Ментальные модели в БД: {len(models)}")

    # Проверяем, есть ли наша модель
    found = False
    for m in models:
        if m.id == result.mental_model.id:
            found = True
            print(f"   ✅ Модель найдена в БД: {m.name}")
            break

    if found:
        print("\n✅ ТЕСТ ПРОЙДЕН! Модель сохранена в БД.")
    else:
        print("\n❌ ТЕСТ НЕ ПРОЙДЕН! Модель не найдена в БД.")


if __name__ == "__main__":
    test_save_mental_model()