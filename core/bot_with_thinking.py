# core/bot_with_thinking.py
"""
[ru] Бот с мышлением (понимание + исследование).
[en] Bot with thinking (understanding + research).
"""

from core.individual import Individual
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from core.knowledge.individual_knowledge_graph import IndividualKnowledgeGraph
from core.thinking.understanding import UnderstandingEngine
from core.thinking.research import ResearchEngine
import uuid


class ThinkingBot(Individual):
    """
    [ru] Бот с мышлением.
    [en] Bot with thinking.
    """

    def __init__(self, x: float, z: float, bot_id: str = None):
        super().__init__(x, z)

        self.bot_id = bot_id or str(uuid.uuid4())[:8]

        # [ru] Графы знаний
        # [en] Knowledge graphs
        self.global_graph = GlobalKnowledgeGraph()
        self.individual_graph = IndividualKnowledgeGraph(bot_id=self.bot_id)

        # [ru] ЗАГРУЖАЕМ ГЗ ИЗ БД
        # [en] LOAD KG FROM DB
        print("[ru] Загрузка ГЗ из БД...")
        print("[en] Loading the KG from the database...")
        self.global_graph.load_from_db()
        print(f" [ru] Загружено: {len(self.global_graph.nodes)} узлов, {len(self.global_graph.edges)} связей")
        print(f" [en] Downloaded: {len(self.global_graph.nodes)} nodes, {len(self.global_graph.edges)} edges")

        # [ru] Движки мышления с ЗАГРУЖЕННЫМ ГЗ
        # [en] Thinking engines with LOADED KG
        self.understanding_engine = UnderstandingEngine(
            # [ru] ← ПЕРЕДАЁМ ЗАГРУЖЕННЫЙ ГЗ
            # [en] ← PASS THE LOADED KG
            global_graph=self.global_graph,
            individual_graph=self.individual_graph,
            # [ru] ← НЕ ЗАГРУЖАЕМ ПОВТОРНО
            # [en] ← DO NOT LOAD AGAIN
            load_from_db=False
        )
        self.research_engine = ResearchEngine(
            self.global_graph,
            self.individual_graph
        )

        # [ru] Загружаем ИГЗ из БД (если есть)
        # [en] Load IKG from DB (if exists)
        self._load_individual_graph()


    def _load_individual_graph(self):
        """
        [ru] Загружает индивидуальный граф из БД.
        [en] Loads the individual graph from the database.
        """
        try:
            from db.knowledge_db import KnowledgeDB
            db = KnowledgeDB()

            graph_data = db.load_individual_graph(self.bot_id)
            if graph_data:
                self.individual_graph = IndividualKnowledgeGraph.from_dict(graph_data)
                print(f" Загружен ИГЗ: {len(self.individual_graph.knowledge)} записей, "
                      f"{len(self.individual_graph.mental_models)} моделей")
        except Exception as e:
            print(f"[ru] Ошибка загрузки ИГЗ: {e}")
            print(f"[en] Loading error GKG: {e}")

    def _load_knowledge(self):
        """
        [ru] Загружает знания из БД.
        [en] Loads knowledge from the database.
        """
        from db.knowledge_db import KnowledgeDB
        db = KnowledgeDB()

        nodes = db.load_all_nodes()
        for node in nodes:
            self.global_graph.add_node(node)

        edges = db.load_all_edges()
        for edge in edges:
            self.global_graph.add_edge(edge)

        print(f"[ru] Загружено: {len(self.global_graph.nodes)} узлов, {len(self.global_graph.edges)} связей")
        print(f"[ru] Downloaded: {len(self.global_graph.nodes)} nodes, {len(self.global_graph.edges)} edges")

    def think_understand(self, task_description: str) -> dict:
        """
        [ru] Понимание задачи.
        [en] Task understanding.
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
        [ru] Исследование проблемы.
        [en] Problem research.
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


# # core/bot_with_thinking.py
# """
# Бот с мышлением (понимание + исследование).
# """
#
# from core.individual import Individual
# from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
# from core.knowledge.individual_knowledge_graph import IndividualKnowledgeGraph
# from core.thinking.understanding import UnderstandingEngine
# from core.thinking.research import ResearchEngine
# import uuid
#
#
# class ThinkingBot(Individual):
#     """
#     Бот с мышлением.
#     """
#
#     def __init__(self, x: float, z: float, bot_id: str = None):
#         super().__init__(x, z)
#
#         self.bot_id = bot_id or str(uuid.uuid4())[:8]
#
#         # Графы знаний
#         self.global_graph = GlobalKnowledgeGraph()
#         self.individual_graph = IndividualKnowledgeGraph(bot_id=self.bot_id)
#
#         # ЗАГРУЖАЕМ ГЗ ИЗ БД
#         print(" Загрузка ГЗ из БД...")
#         self.global_graph.load_from_db()
#         print(f"   ✅ Загружено: {len(self.global_graph.nodes)} узлов, {len(self.global_graph.edges)} связей")
#
#         # Движки мышления с ЗАГРУЖЕННЫМ ГЗ
#         self.understanding_engine = UnderstandingEngine(
#             global_graph=self.global_graph,  # ← ПЕРЕДАЁМ ЗАГРУЖЕННЫЙ ГЗ
#             individual_graph=self.individual_graph,
#             load_from_db=False  # ← НЕ ЗАГРУЖАЕМ ПОВТОРНО
#         )
#         self.research_engine = ResearchEngine(
#             self.global_graph,
#             self.individual_graph
#         )
#
#         # Загружаем ИГЗ из БД (если есть)
#         self._load_individual_graph()
#
#
#     def _load_individual_graph(self):
#         """Загружает индивидуальный граф из БД."""
#         try:
#             from db.knowledge_db import KnowledgeDB
#             db = KnowledgeDB()
#
#             graph_data = db.load_individual_graph(self.bot_id)
#             if graph_data:
#                 self.individual_graph = IndividualKnowledgeGraph.from_dict(graph_data)
#                 print(f"🧠 Загружен ИГЗ: {len(self.individual_graph.knowledge)} записей, "
#                       f"{len(self.individual_graph.mental_models)} моделей")
#         except Exception as e:
#             print(f"⚠️ Ошибка загрузки ИГЗ: {e}")
#
#     def _load_knowledge(self):
#         """Загружает знания из БД."""
#         from db.knowledge_db import KnowledgeDB
#         db = KnowledgeDB()
#
#         nodes = db.load_all_nodes()
#         for node in nodes:
#             self.global_graph.add_node(node)
#
#         edges = db.load_all_edges()
#         for edge in edges:
#             self.global_graph.add_edge(edge)
#
#         print(f"🧠 Загружено: {len(self.global_graph.nodes)} узлов, {len(self.global_graph.edges)} связей")
#
#     def think_understand(self, task_description: str) -> dict:
#         """
#         Понимание задачи.
#         """
#         result = self.understanding_engine.understand(task_description)
#
#         return {
#             "task": task_description,
#             "concepts": result.extracted_concepts,
#             "nodes_found": len(result.found_nodes),
#             "model_created": result.new_model.id if result.new_model else None,
#             "experience": result.experience
#         }
#
#     def think_research(self, problem_description: str) -> dict:
#         """
#         Исследование проблемы.
#         """
#         result = self.research_engine.research(problem_description)
#
#         return {
#             "problem": problem_description,
#             "requirements": result.required_properties,
#             "analogies": len(result.found_analogies),
#             "hypotheses_generated": len(result.generated_hypotheses),
#             "hypotheses_validated": len(result.validated_hypotheses),
#             "best_hypothesis": result.validated_hypotheses[0].id if result.validated_hypotheses else None
#         }