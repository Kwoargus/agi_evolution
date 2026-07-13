# core/bot_with_thinking.py
"""
Бот с мышлением (понимание + исследование).
"""

from core.individual import Individual
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from core.knowledge.individual_knowledge_graph import IndividualKnowledgeGraph
from core.thinking.understanding import UnderstandingEngine
from core.thinking.research import ResearchEngine


class ThinkingBot(Individual):
    """
    Бот с мышлением.
    """

    def __init__(self, x: float, z: float):
        super().__init__(x, z)

        # Графы знаний
        self.global_graph = GlobalKnowledgeGraph()
        self.individual_graph = IndividualKnowledgeGraph()

        # Движки мышления
        self.understanding_engine = UnderstandingEngine(
            self.global_graph,
            self.individual_graph
        )
        self.research_engine = ResearchEngine(
            self.global_graph
        )

        # Загружаем данные в ГЗ
        self._load_knowledge()

    def _load_knowledge(self):
        """Загружает знания из БД."""
        from db.knowledge_db import KnowledgeDB
        db = KnowledgeDB()

        nodes = db.load_all_nodes()
        for node in nodes:
            self.global_graph.add_node(node)

        edges = db.load_all_edges()
        for edge in edges:
            self.global_graph.add_edge(edge)

        print(f"🧠 Загружено: {len(self.global_graph.nodes)} узлов, {len(self.global_graph.edges)} связей")

    def think_understand(self, task_description: str) -> dict:
        """
        Понимание задачи.
        """
        result = self.understanding_engine.understand(task_description)

        return {
            "task": task_description,
            "concepts": result.extracted_concepts,
            "nodes_found": len(result.found_nodes),
            "model_created": result.new_model.id if result.new_model else None,
            "experience": result.experience
        }

    def think_research(self, problem_description: str) -> dict:
        """
        Исследование проблемы.
        """
        result = self.research_engine.research(problem_description)

        return {
            "problem": problem_description,
            "requirements": result.required_properties,
            "analogies": len(result.found_analogies),
            "hypotheses_generated": len(result.generated_hypotheses),
            "hypotheses_validated": len(result.validated_hypotheses),
            "best_hypothesis": result.validated_hypotheses[0].id if result.validated_hypotheses else None
        }