# pipelines/knowledge_pipeline.py
"""
Конвейер построения графа знаний.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from core.knowledge.llm_expander import KnowledgeExpander
from db.knowledge_db import KnowledgeDB
from scripts.load_conceptnet import ConceptNetLoader


class KnowledgePipeline:
    """
    Полный конвейер построения графа знаний.
    """

    def __init__(self):
        self.graph = GlobalKnowledgeGraph()
        self.db = KnowledgeDB()
        self.expander = KnowledgeExpander(self.graph)

    def run(self, load_from_conceptnet: bool = True, expand_with_llm: bool = True):
        """
        Запускает конвейер.

        Args:
            load_from_conceptnet: Загружать из ConceptNet
            expand_with_llm: Расширять через LLM
        """
        print("🚀 ЗАПУСК КОНВЕЙЕРА ПОСТРОЕНИЯ ГЗ")
        print("=" * 60)

        # 1. Загрузка из ConceptNet
        if load_from_conceptnet:
            print("\n📥 Шаг 1: Загрузка из ConceptNet...")
            loader = ConceptNetLoader()
            loader.load_engineering_knowledge()
            print("   ✅ ConceptNet загружен")

        # 2. Загрузка из БД
        print("\n📤 Шаг 2: Загрузка из БД в граф...")
        self.graph.load_from_db()
        print(f"   ✅ Загружено {len(self.graph.nodes)} узлов")

        # 3. Расширение через LLM
        if expand_with_llm:
            print("\n🧠 Шаг 3: Расширение через LLM...")
            self._expand_with_llm()
            print("   ✅ LLM-расширение выполнено")

        # 4. Сохранение в БД
        print("\n💾 Шаг 4: Сохранение в БД...")
        self.graph.save_to_db()
        print("   ✅ Сохранено")

        print("\n" + "=" * 60)
        print(f"📊 ИТОГИ: {len(self.graph.nodes)} узлов, {len(self.graph.edges)} связей")
        print("=" * 60)

    def _expand_with_llm(self):
        """Расширяет граф через LLM."""
        # Здесь должен быть реальный вызов LLM
        # Для демонстрации используем тестовые данные

        test_data = {
            "airplane": {
                "properties": ["flying", "fixed_wing"],
                "functions": ["transport", "fly"],
                "related": ["air", "wing", "engine"]
            },
            "helicopter": {
                "properties": ["flying", "rotary_wing"],
                "functions": ["transport", "hover"],
                "related": ["air", "rotor", "engine"]
            }
        }

        self.expander.expand_with_llm(test_data)


def main():
    pipeline = KnowledgePipeline()
    pipeline.run(load_from_conceptnet=True, expand_with_llm=True)


if __name__ == "__main__":
    main()