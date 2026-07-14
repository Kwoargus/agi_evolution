# tests/test_analogy_engine.py
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from core.knowledge.individual_knowledge_graph import IndividualKnowledgeGraph
from core.knowledge.analogy_engine import AnalogyEngine

# Создаём графы
gg = GlobalKnowledgeGraph()
ig = IndividualKnowledgeGraph(bot_id="test_bot")

# Добавляем ментальную модель в ИГЗ
ig.add_mental_model("model_1", {
    'name': 'Модель самолёта',
    'properties': ['летать', 'быстрый', 'грузоподъёмный'],
    'task': 'Создать летательный аппарат'
})

# Создаём движок аналогий
engine = AnalogyEngine(gg, ig)

# Ищем аналогии
analogies = engine.find_analogies(
    task_description="Нужен быстрый летательный аппарат",
    required_properties=['летать', 'быстрый']
)

print(f"Найдено аналогий: {len(analogies)}")
for a in analogies:
    print(f"  {a.id}: источник={a.metadata.get('source')}, скор={a.metadata.get('score', 0):.2f}")