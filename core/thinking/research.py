# core/thinking/research.py
"""
[ru] Модуль "Исследование" - генерация и проверка гипотез.
[en] "Research" module - hypothesis generation and testing.
"""

import random
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from core.knowledge import IndividualKnowledgeGraph
from core.knowledge.knowledge_node import KnowledgeNode
from core.knowledge.knowledge_edge import KnowledgeEdge, EdgeType
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from core.knowledge.combination import Combination
from core.knowledge.hypothesis import Hypothesis, HypothesisStatus

from core.knowledge.analogy_engine import AnalogyEngine


class HypothesisStatus(Enum):
    PROPOSED = "proposed"
    TESTING = "testing"
    VALIDATED = "validated"
    REJECTED = "rejected"


@dataclass
class ResearchResult:
    """
    [ru] Результат этапа исследования.
    [en] Result of the research phase.
    """
    problem_description: str
    required_properties: List[str]
    found_analogies: List[Combination]
    generated_hypotheses: List[Hypothesis]
    validated_hypotheses: List[Hypothesis]
    created_at: float = field(default_factory=time.time)


class ResearchEngine:
    """
    [ru] Движок "Исследования" - генерация и проверка гипотез.
    [en] "Research" engine - hypothesis generation and testing.
    """

    def __init__(self, global_graph: GlobalKnowledgeGraph,
                 individual_graph: IndividualKnowledgeGraph = None,
                 test_environment: Any = None):
        self.global_graph = global_graph
        self.test_environment = test_environment
        self.analogy_cache = {}

        # [ru] ИНИЦИАЛИЗИРУЕМ AnalogyEngine
        # [en] INITIALIZE AnalogyEngine
        self.analogy_engine = AnalogyEngine(
            global_graph=self.global_graph,
            individual_graph=self.individual_graph)

    def extract_requirements(self, problem_description: str) -> List[str]:
        """
        [ru] Извлекает требования из описания проблемы.
        [en] Extracts requirements from the problem description.
        """

        # [ru] Простейший подход: поиск ключевых слов
        # [en] Simplest approach: keyword search
        requirements = []

        requirement_keywords = {
            "летать": ["flying", "airborne", "flight"],
            "двигаться": ["mobile", "moving", "transport"],
            "поднимать": ["lifting", "hoisting", "elevating"],
            "вращать": ["rotating", "spinning", "turning"],
            "управлять": ["controlled", "steered", "guided"],
            "передавать": ["transmitting", "transferring", "conveying"],
            "преобразовывать": ["converting", "transforming", "changing"],
            "накапливать": ["storing", "accumulating", "buffering"],
            "защищать": ["protecting", "shielding", "guarding"],
            "соединять": ["connecting", "joining", "linking"],
        }

        text_lower = problem_description.lower()
        for keyword, aliases in requirement_keywords.items():
            if keyword in text_lower or any(a in text_lower for a in aliases):
                requirements.append(keyword)

        return requirements

    def find_analogies(self, required_properties: List[str]) -> List[Combination]:
        """
        [ru] Находит аналогии в ГЗ по требуемым свойствам.
        [en] Finds analogies in the Knowledge Graph by required properties.
        """
        analogies = []

        # [ru] Ищем узлы с требуемыми свойствами
        # [en] Search for nodes with required properties
        found_nodes = []
        for prop in required_properties:
            # [ru] Ищем узлы, у которых есть это свойство
            # [en] Search for nodes that have this property
            for node in self.global_graph.nodes.values():
                if prop in [p.lower() for p in node.properties]:
                    found_nodes.append(node)

        # [ru] Группируем найденные узлы по типам
        # [en] Group found nodes by type
        nodes_by_type = {}
        for node in found_nodes:
            if node.node_type not in nodes_by_type:
                nodes_by_type[node.node_type] = []
            nodes_by_type[node.node_type].append(node)

        # [ru] Создаем комбинации из узлов разных типов
        # [en] Create combinations of nodes of different types
        for node_type, nodes in nodes_by_type.items():
            if len(nodes) >= 2:
                # [ru] Берем 2-3 узла одного типа
                # [en] Take 2-3 nodes of the same type
                selected = random.sample(nodes, min(3, len(nodes)))
                combo = Combination(
                    id=f"analogy_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}",
                    nodes=selected,
                    properties=list(set([p for n in selected for p in n.properties]))
                )
                analogies.append(combo)

        # [ru] Добавляем кросс-типовые комбинации
        # [en] Add cross-type combinations
        types = list(nodes_by_type.keys())
        for _ in range(len(types) * 2):
            if len(types) < 2:
                break
            t1, t2 = random.sample(types, 2)
            n1 = random.choice(nodes_by_type[t1])
            n2 = random.choice(nodes_by_type[t2])

            combo = Combination(
                id=f"analogy_cross_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}",
                nodes=[n1, n2],
                properties=list(set(n1.properties + n2.properties))
            )
            analogies.append(combo)

        # [ru] Ограничиваем количество
        # [en] Limit the quantity
        return analogies[:10]

    def generate_hypothesis(self, analogy: Combination,
                            required_properties: List[str]) -> Hypothesis:
        """
        [ru] Генерирует гипотезу на основе аналогии.
        [en] Generates a hypothesis based on an analogy.
        """
        # [ru] Создаем модификации аналогии
        # [en] Create analogy modifications
        modifications = []

        # [ru] 1. Добавляем недостающие свойства
        # [en] 1. Add missing properties
        analogy_props = set(analogy.properties)
        missing = set(required_properties) - analogy_props
        if missing:
            modifications.append(f"add_{'_'.join(missing)}")

        # [ru] 2. Заменяем часть узлов
        # [en] 2. Replace part of the nodes
        if len(analogy.nodes) > 1:
            modifications.append("replace_part")

        # [ru] 3. Комбинируем с другим узлом
        # [en] 3. Combine with another node
        random_node = random.choice(list(self.global_graph.nodes.values()))
        if random_node.id not in [n.id for n in analogy.nodes]:
            modifications.append(f"combine_with_{random_node.name}")

        # [ru] Создаем гипотезу
        # [en] Create the hypothesis
        hypothesis_id = f"hyp_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

        return Hypothesis(
            id=hypothesis_id,
            task_description="Generated from analogy",
            source_combination=analogy,
            modifications=modifications,
            predicted_score=random.uniform(0.3, 0.8),
            status=HypothesisStatus.PROPOSED,
            metadata={
                "generated_at": time.time(),
                "source": "research_engine",
                "required_properties": required_properties
            }
        )

    def test_hypothesis(self, hypothesis: Hypothesis) -> Tuple[bool, float]:
        """
        [ru] Проверяет гипотезу в тестовой среде.
        [en] Tests the hypothesis in a test environment.
        """
        # [ru] В реальной системе здесь была бы симуляция. Для демонстрации используем простую оценку
        # [en] In a real system, there would be a simulation here. For demonstration, we use a simple evaluation

        # [ru] Оцениваем на основе покрытия требований
        # [en] Evaluate based on requirement coverage
        required = hypothesis.metadata.get("required_properties", [])
        if not required:
            return True, 0.7

        # [ru] Проверяем, какие требования покрыты
        # [en] Check which requirements are covered
        analogy_props = set(hypothesis.source_combination.properties)
        covered = sum(1 for r in required if r in analogy_props)
        score = covered / len(required) if required else 0.5

        # [ru] Добавляем случайность
        # [en] Add randomness
        score += random.uniform(-0.1, 0.1)
        score = max(0, min(1, score))

        # [ru] Проверяем, достаточно ли хороша гипотеза
        # [en] Check if the hypothesis is good enough
        validated = score >= 0.7

        # [ru] Обновляем гипотезу
        # [en] Update the hypothesis
        hypothesis.actual_score = score
        if validated:
            hypothesis.status = HypothesisStatus.VALIDATED
        else:
            hypothesis.status = HypothesisStatus.REJECTED

        return validated, score

    def research(self, problem_description: str) -> ResearchResult:
        """
        [ru] Основной метод "Исследования" с использованием AnalogyEngine.
        [en] Main "Research" method using AnalogyEngine.
        """
        print("[ru] ЗАПУСК ИССЛЕДОВАНИЯ...")
        print("[en] LAUNCHING THE RESEARCH...")
        print(f"[ru] Проблема: {problem_description[:100]}...")
        print(f"[en] Problem: {problem_description[:100]}...")

        # [ru] 1. Извлекаем требования
        # [en] 1. Extract requirements
        requirements = self.extract_requirements(problem_description)
        print(f" [ru] Требований: {len(requirements)}")
        print(f" [en] Requirements: {len(requirements)}")

        # [ru] 2. Находим аналогии через AnalogyEngine
        # [en] 2. Find analogies via AnalogyEngine
        analogies = self.analogy_engine.find_analogies(
            task_description=problem_description,
            required_properties=requirements,
            max_results=10
        )
        print(f"[ru] Найдено аналогий: {len(analogies)}")
        print(f"[ru] Analogies found: {len(analogies)}")

        # [ru] Логируем источники
        # [en] Log sources
        sources = {}
        for a in analogies:
            source = a.metadata.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        print(f" [ru] Источники: {sources}")
        print(f" [en] Sources: {sources}")

        # [ru] 3. Генерируем гипотезы на основе аналогий
        # [en] 3. Generate hypotheses based on analogies
        hypotheses = []
        for analogy in analogies[:5]:
            # [ru] Модифицируем аналогию
            # [en] Modify the analogy
            modifications = []

            # [ru] Если аналогия из ИГЗ, пробуем дополнить из ГГЗ
            # [en] If analogy is from IKG, try to supplement from GKG
            if analogy.metadata.get('source') == 'individual_graph':
                modifications.append("add_feature_надежный")
                modifications.append("replace_part")

            # [ru] Модифицируем
            # [en] Modify
            modified = self.analogy_engine.modify_analogy(analogy, modifications)

            # [ru] Создаём гипотезу
            # [en] Create hypothesis
            hyp = self.generate_hypothesis(modified, requirements)
            hypotheses.append(hyp)

        print(f" [ru] Сгенерировано гипотез: {len(hypotheses)}")
        print(f" [en] Hypotheses generated: {len(hypotheses)}")

        # [ru] 4. Проверяем гипотезы
        # [en] 4. Test hypotheses
        validated = []
        for hyp in hypotheses:
            is_valid, score = self.test_hypothesis(hyp)
            if is_valid:
                validated.append(hyp)
                print(f" [ru] Гипотеза {hyp.id}: {score:.2f} - валидна")
                print(f" [en] Hypothesis {hyp.id}: {score:.2f} - is valid")
            else:
                print(f" [ru] Гипотеза {hyp.id}: {score:.2f} - отклонена")
                print(f" [en] Hypothesis {hyp.id}: {score:.2f} - rejected")

        print(f" [ru] Валидных гипотез: {len(validated)}")
        print(f" [en] Valid hypotheses: {len(validated)}")

        return ResearchResult(
            problem_description=problem_description,
            required_properties=requirements,
            found_analogies=analogies,
            generated_hypotheses=hypotheses,
            validated_hypotheses=validated
        )






# # core/thinking/research.py
# """
# Модуль "Исследование" - генерация и проверка гипотез.
# """
#
# import random
# import time
# import hashlib
# from typing import List, Dict, Any, Optional, Tuple
# from dataclasses import dataclass, field
# from enum import Enum
#
# from core.knowledge import IndividualKnowledgeGraph
# from core.knowledge.knowledge_node import KnowledgeNode
# from core.knowledge.knowledge_edge import KnowledgeEdge, EdgeType
# from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
# from core.knowledge.combination import Combination
# from core.knowledge.hypothesis import Hypothesis, HypothesisStatus
#
# from core.knowledge.analogy_engine import AnalogyEngine
#
#
# class HypothesisStatus(Enum):
#     PROPOSED = "proposed"
#     TESTING = "testing"
#     VALIDATED = "validated"
#     REJECTED = "rejected"
#
#
# @dataclass
# class ResearchResult:
#     """
#     Результат этапа исследования.
#     """
#     problem_description: str
#     required_properties: List[str]
#     found_analogies: List[Combination]
#     generated_hypotheses: List[Hypothesis]
#     validated_hypotheses: List[Hypothesis]
#     created_at: float = field(default_factory=time.time)
#
#
# class ResearchEngine:
#     """
#     Движок "Исследования" - генерация и проверка гипотез.
#     """
#
#     def __init__(self, global_graph: GlobalKnowledgeGraph,
#                  individual_graph: IndividualKnowledgeGraph = None,
#                  test_environment: Any = None):
#         self.global_graph = global_graph
#         self.test_environment = test_environment
#         self.analogy_cache = {}
#
#         # ИНИЦИАЛИЗИРУЕМ AnalogyEngine
#         self.analogy_engine = AnalogyEngine(
#             global_graph=self.global_graph,
#             individual_graph=self.individual_graph)
#
#     def extract_requirements(self, problem_description: str) -> List[str]:
#         """
#         Извлекает требования из описания проблемы.
#         """
#
#         # Простейший подход: поиск ключевых слов
#         requirements = []
#
#         requirement_keywords = {
#             "летать": ["flying", "airborne", "flight"],
#             "двигаться": ["mobile", "moving", "transport"],
#             "поднимать": ["lifting", "hoisting", "elevating"],
#             "вращать": ["rotating", "spinning", "turning"],
#             "управлять": ["controlled", "steered", "guided"],
#             "передавать": ["transmitting", "transferring", "conveying"],
#             "преобразовывать": ["converting", "transforming", "changing"],
#             "накапливать": ["storing", "accumulating", "buffering"],
#             "защищать": ["protecting", "shielding", "guarding"],
#             "соединять": ["connecting", "joining", "linking"],
#         }
#
#         text_lower = problem_description.lower()
#         for keyword, aliases in requirement_keywords.items():
#             if keyword in text_lower or any(a in text_lower for a in aliases):
#                 requirements.append(keyword)
#
#         return requirements
#
#     def find_analogies(self, required_properties: List[str]) -> List[Combination]:
#         """
#         Находит аналогии в ГЗ по требуемым свойствам.
#         """
#         analogies = []
#
#         # Ищем узлы с требуемыми свойствами
#         found_nodes = []
#         for prop in required_properties:
#             # Ищем узлы, у которых есть это свойство
#             for node in self.global_graph.nodes.values():
#                 if prop in [p.lower() for p in node.properties]:
#                     found_nodes.append(node)
#
#         # Группируем найденные узлы по типам
#         nodes_by_type = {}
#         for node in found_nodes:
#             if node.node_type not in nodes_by_type:
#                 nodes_by_type[node.node_type] = []
#             nodes_by_type[node.node_type].append(node)
#
#         # Создаем комбинации из узлов разных типов
#         for node_type, nodes in nodes_by_type.items():
#             if len(nodes) >= 2:
#                 # Берем 2-3 узла одного типа
#                 selected = random.sample(nodes, min(3, len(nodes)))
#                 combo = Combination(
#                     id=f"analogy_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}",
#                     nodes=selected,
#                     properties=list(set([p for n in selected for p in n.properties]))
#                 )
#                 analogies.append(combo)
#
#         # Добавляем кросс-типовые комбинации
#         types = list(nodes_by_type.keys())
#         for _ in range(len(types) * 2):
#             if len(types) < 2:
#                 break
#             t1, t2 = random.sample(types, 2)
#             n1 = random.choice(nodes_by_type[t1])
#             n2 = random.choice(nodes_by_type[t2])
#
#             combo = Combination(
#                 id=f"analogy_cross_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}",
#                 nodes=[n1, n2],
#                 properties=list(set(n1.properties + n2.properties))
#             )
#             analogies.append(combo)
#
#         return analogies[:10]  # Ограничиваем количество
#
#     def generate_hypothesis(self, analogy: Combination,
#                             required_properties: List[str]) -> Hypothesis:
#         """
#         Генерирует гипотезу на основе аналогии.
#         """
#         # Создаем модификации аналогии
#         modifications = []
#
#         # 1. Добавляем недостающие свойства
#         analogy_props = set(analogy.properties)
#         missing = set(required_properties) - analogy_props
#         if missing:
#             modifications.append(f"add_{'_'.join(missing)}")
#
#         # 2. Заменяем часть узлов
#         if len(analogy.nodes) > 1:
#             modifications.append("replace_part")
#
#         # 3. Комбинируем с другим узлом
#         random_node = random.choice(list(self.global_graph.nodes.values()))
#         if random_node.id not in [n.id for n in analogy.nodes]:
#             modifications.append(f"combine_with_{random_node.name}")
#
#         # Создаем гипотезу
#         hypothesis_id = f"hyp_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
#
#         return Hypothesis(
#             id=hypothesis_id,
#             task_description="Generated from analogy",
#             source_combination=analogy,
#             modifications=modifications,
#             predicted_score=random.uniform(0.3, 0.8),
#             status=HypothesisStatus.PROPOSED,
#             metadata={
#                 "generated_at": time.time(),
#                 "source": "research_engine",
#                 "required_properties": required_properties
#             }
#         )
#
#     def test_hypothesis(self, hypothesis: Hypothesis) -> Tuple[bool, float]:
#         """
#         Проверяет гипотезу в тестовой среде.
#         """
#         # В реальной системе здесь была бы симуляция
#         # Для демонстрации используем простую оценку
#
#         # Оцениваем на основе покрытия требований
#         required = hypothesis.metadata.get("required_properties", [])
#         if not required:
#             return True, 0.7
#
#         # Проверяем, какие требования покрыты
#         analogy_props = set(hypothesis.source_combination.properties)
#         covered = sum(1 for r in required if r in analogy_props)
#         score = covered / len(required) if required else 0.5
#
#         # Добавляем случайность
#         score += random.uniform(-0.1, 0.1)
#         score = max(0, min(1, score))
#
#         # Проверяем, достаточно ли хороша гипотеза
#         validated = score >= 0.7
#
#         # Обновляем гипотезу
#         hypothesis.actual_score = score
#         if validated:
#             hypothesis.status = HypothesisStatus.VALIDATED
#         else:
#             hypothesis.status = HypothesisStatus.REJECTED
#
#         return validated, score
#
#     def research(self, problem_description: str) -> ResearchResult:
#         """
#         Основной метод "Исследования" с использованием AnalogyEngine.
#         """
#         print("🔬 ЗАПУСК ИССЛЕДОВАНИЯ...")
#         print(f"   Проблема: {problem_description[:100]}...")
#
#         # 1. Извлекаем требования
#         requirements = self.extract_requirements(problem_description)
#         print(f"   📍 Требований: {len(requirements)}")
#
#         # 2. Находим аналогии через AnalogyEngine
#         analogies = self.analogy_engine.find_analogies(
#             task_description=problem_description,
#             required_properties=requirements,
#             max_results=10
#         )
#         print(f"   📍 Найдено аналогий: {len(analogies)}")
#
#         # Логируем источники
#         sources = {}
#         for a in analogies:
#             source = a.metadata.get('source', 'unknown')
#             sources[source] = sources.get(source, 0) + 1
#         print(f"   📍 Источники: {sources}")
#
#         # 3. Генерируем гипотезы на основе аналогий
#         hypotheses = []
#         for analogy in analogies[:5]:
#             # Модифицируем аналогию
#             modifications = []
#
#             # Если аналогия из ИГЗ, пробуем дополнить из ГГЗ
#             if analogy.metadata.get('source') == 'individual_graph':
#                 modifications.append("add_feature_надежный")
#                 modifications.append("replace_part")
#
#             # Модифицируем
#             modified = self.analogy_engine.modify_analogy(analogy, modifications)
#
#             # Создаём гипотезу
#             hyp = self.generate_hypothesis(modified, requirements)
#             hypotheses.append(hyp)
#
#         print(f"   📍 Сгенерировано гипотез: {len(hypotheses)}")
#
#         # 4. Проверяем гипотезы
#         validated = []
#         for hyp in hypotheses:
#             is_valid, score = self.test_hypothesis(hyp)
#             if is_valid:
#                 validated.append(hyp)
#                 print(f"   ✅ Гипотеза {hyp.id}: {score:.2f} - валидна")
#             else:
#                 print(f"   ❌ Гипотеза {hyp.id}: {score:.2f} - отклонена")
#
#         print(f"   📍 Валидных гипотез: {len(validated)}")
#
#         return ResearchResult(
#             problem_description=problem_description,
#             required_properties=requirements,
#             found_analogies=analogies,
#             generated_hypotheses=hypotheses,
#             validated_hypotheses=validated
#         )
