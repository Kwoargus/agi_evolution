# scripts/create_core_knowledge.py
from core.knowledge import KnowledgeNode


def create_core_knowledge():
    """
    Создает базовый набор знаний для AGI.
    """

    core_knowledge = {
        # Физические объекты и их свойства
        "air": {
            "properties": ["gas", "fluid", "exerts_pressure"],
            "functions": ["transmits_force", "creates_lift"]
        },
        "wing": {
            "properties": ["curved", "aerodynamic"],
            "functions": ["creates_lift", "redirects_airflow"]
        },
        "engine": {
            "properties": ["mechanical", "produces_power"],
            "functions": ["converts_energy", "drives_propeller"]
        },
        # ... и так далее
    }

    for name, data in core_knowledge.items():
        node = KnowledgeNode(
            id=name,
            name=name,
            node_type='object',
            properties=data['properties'],
            description=f"Basic knowledge about {name}"
        )
        save_node(node)