# tests/test_individual_graph.py
from core.knowledge.individual_knowledge_graph import IndividualKnowledgeGraph
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph

# Создаём ИГЗ
ig = IndividualKnowledgeGraph(bot_id="test_bot")

# Добавляем ментальную модель
ig.add_mental_model("model_1", {
    'name': 'Тестовая модель',
    'properties': ['летать', 'быстрый'],
    'task': 'Создать летательный аппарат'
})

# Ищем по свойствам
results = ig.find_mental_models_by_properties(['летать'])
print(f"Найдено моделей: {len(results)}")
assert len(results) > 0
assert results[0]['matches'] == ['летать']