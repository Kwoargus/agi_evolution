# scripts/load_and_expand_knowledge.py
"""
Загрузка знаний из ConceptNet и расширение через LLM.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.load_conceptnet import ConceptNetLoader
from core.knowledge.llm_expander import expand_knowledge_with_llm
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from db.knowledge_db import KnowledgeDB


def load_and_expand():
    """
    Полный процесс: загрузка ConceptNet + расширение LLM.
    """
    print("=" * 60)
    print("ЗАГРУЗКА И РАСШИРЕНИЕ ЗНАНИЙ")
    print("=" * 60)

    # 1. Загружаем базовые знания из ConceptNet
    print("\n1. Загрузка из ConceptNet...")
    loader = ConceptNetLoader()
    loader.load_engineering_knowledge()

    # 2. Расширяем через LLM
    print("\n2. Расширение через LLM...")

    # Базовые знания для расширения
    base_knowledge = {
        "airplane": {
            "properties": ["flying", "fixed_wing", "heavier_than_air"],
            "functions": ["transport", "carry_people", "fly"],
            "related": ["air", "wing", "engine", "propeller", "fuel"]
        },
        "helicopter": {
            "properties": ["flying", "rotary_wing", "vertical_takeoff"],
            "functions": ["transport", "hover", "land_anywhere"],
            "related": ["air", "rotor", "engine", "tail_rotor"]
        },
        "glider": {
            "properties": ["flying", "fixed_wing", "no_engine"],
            "functions": ["soar", "glide", "stay_airborne"],
            "related": ["air", "wing", "thermal", "wind"]
        }
    }

    new_nodes = expand_knowledge_with_llm(base_knowledge)

    print(f"\n✅ Добавлено {len(new_nodes)} новых узлов из LLM")

    # 3. Сохраняем в БД
    print("\n3. Сохранение в БД...")
    db = KnowledgeDB()
    for node in new_nodes:
        db.save_node(node)

    print("✅ Готово!")


if __name__ == "__main__":
    load_and_expand()