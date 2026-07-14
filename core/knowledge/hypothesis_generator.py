# core/knowledge/hypothesis_generator.py
"""
Генератор гипотез на основе аналогов.
Создаёт новые комбинации узлов для покрытия функциональных требований.
"""

from typing import List, Dict, Set, Optional, Tuple
import hashlib
import time
from itertools import product

from core.knowledge import KnowledgeEdge, EdgeType
from core.knowledge.combination import Combination
from core.knowledge.hypothesis import Hypothesis, HypothesisStatus
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from core.knowledge.individual_knowledge_graph import IndividualKnowledgeGraph
from core.knowledge.knowledge_node import KnowledgeNode


class HypothesisGenerator:
    """
    Генерирует гипотезы (новые модели) на основе аналогов и функциональных требований.
    """

    def __init__(self, global_graph: GlobalKnowledgeGraph,
                 individual_graph: Optional[IndividualKnowledgeGraph] = None):
        self.global_graph = global_graph
        self.individual_graph = individual_graph or IndividualKnowledgeGraph()
        self.cache = {}

    def save_validated_hypotheses(self, hypotheses: List[Hypothesis]) -> int:
        from db.knowledge_db import KnowledgeDB
        db = KnowledgeDB()

        saved_count = 0

        for hyp in hypotheses:
            if hyp.status != HypothesisStatus.VALIDATED:
                continue

            # 1. Сохраняем комбинацию (если её нет)
            combo = hyp.source_combination
            if not db.save_combination(combo):
                print(f"⚠️ Не удалось сохранить комбинацию {combo.id}")
                continue

            # 2. Сохраняем гипотезу
            if self.individual_graph:
                self.individual_graph.add_hypothesis(hyp)

            if db.save_hypothesis(hyp):
                saved_count += 1
                print(f"💾 Гипотеза {hyp.id} сохранена")

        return saved_count

    # def save_validated_hypotheses(self, hypotheses: List[Hypothesis]) -> int:
    #     """
    #     Сохраняет подтверждённые гипотезы в ИГЗ и БД.
    #
    #     Args:
    #         hypotheses: Список гипотез
    #
    #     Returns:
    #         Количество сохранённых гипотез
    #     """
    #     from db.knowledge_db import KnowledgeDB
    #
    #     saved_count = 0
    #     db = KnowledgeDB()
    #
    #     for hyp in hypotheses:
    #         # Сохраняем только подтверждённые
    #         if hyp.status != HypothesisStatus.VALIDATED:
    #             continue
    #
    #         # Сохраняем в ИГЗ
    #         if self.individual_graph:
    #             self.individual_graph.add_hypothesis(hyp)
    #
    #         # Сохраняем в БД
    #         if db.save_hypothesis(hyp):
    #             saved_count += 1
    #             print(f"💾 Гипотеза {hyp.id} сохранена")
    #
    #     return saved_count

    def save_hypothesis_to_kg(self, hypothesis: Hypothesis) -> bool:
        """
        Сохраняет гипотезу как новый узел в ГЗ.

        Args:
            hypothesis: Гипотеза для сохранения

        Returns:
            True если сохранение успешно
        """
        try:
            # Создаём новый узел в ГЗ
            new_node = KnowledgeNode(
                id=f"hyp_node_{hypothesis.id[:8]}",
                name=f"Модель: {hypothesis.task_description[:30]}",
                node_type="hypothesis",
                properties=hypothesis.source_combination.properties,
                description=hypothesis.description
            )

            self.global_graph.add_node(new_node)

            # Добавляем связи с исходными узлами
            for node in hypothesis.source_combination.nodes:
                edge = KnowledgeEdge(
                    id=f"hyp_edge_{new_node.id}_{node.id}",
                    source_id=new_node.id,
                    target_id=node.id,
                    edge_type=EdgeType.HAS_PART,
                    weight=0.8,
                    description=f"Содержит {node.name}"
                )
                self.global_graph.add_edge(edge)

            # Сохраняем в БД
            from db.knowledge_db import KnowledgeDB
            db = KnowledgeDB()
            db.save_node(new_node)

            print(f"✅ Гипотеза сохранена как узел в ГЗ: {new_node.name}")
            return True

        except Exception as e:
            print(f"❌ Ошибка сохранения гипотезы в ГЗ: {e}")
            return False

    def generate_hypotheses(self,
                            analogies: List[Combination],
                            required_functions: List[str],
                            task_description: str = "",  # ← ДОБАВЛЯЕМ
                            max_hypotheses: int = 5) -> List[Hypothesis]:
        """
        Генерирует гипотезы на основе списка аналогов.

        Args:
            analogies: Список найденных аналогов (Combination)
            required_functions: Список функциональных свойств
            task_description: Оригинальный текст задачи (для сохранения)
            max_hypotheses: Максимальное количество гипотез для возврата

        Returns:
            Список гипотез (объектов Hypothesis)
        """
        hypotheses = []

        for analogy in analogies:
            # 1. Анализ покрытия
            covered, missing = self._analyze_coverage(analogy, required_functions)

            # 2. Если все функции покрыты — гипотеза уже есть
            if not missing:
                hyp = self._create_hypothesis_from_analogy(
                    analogy, required_functions, covered, task_description  # ← ПЕРЕДАЁМ
                )
                hypotheses.append(hyp)
                continue

            # 3. Поиск недостающих узлов
            missing_nodes = self._find_nodes_for_functions(missing)

            # 4. Генерация комбинаций с добавлением недостающих узлов
            for combo_nodes in self._generate_node_combinations(missing_nodes, max_add=3):
                new_nodes = list(analogy.nodes) + combo_nodes
                new_props = self._collect_properties(new_nodes)

                # Оценка покрытия после добавления
                new_covered, new_missing = self._analyze_coverage_by_props(new_props, required_functions)

                if len(new_covered) > len(covered):
                    hyp = self._create_hypothesis(
                        source_analogy=analogy,
                        nodes=new_nodes,
                        properties=new_props,
                        covered_functions=new_covered,
                        missing_functions=new_missing,
                        required_functions=required_functions,
                        added_nodes=combo_nodes,
                        task_description=task_description  # ← ПЕРЕДАЁМ
                    )
                    hypotheses.append(hyp)

        # Сортируем гипотезы по качеству
        hypotheses.sort(key=lambda h: h.predicted_score, reverse=True)
        return hypotheses[:max_hypotheses]

    def _analyze_coverage(self, analogy: Combination,
                          required_functions: List[str]) -> Tuple[Set[str], Set[str]]:
        """Анализирует покрытие функций узлами аналогии."""
        analogy_props = set(p.lower() for p in analogy.properties)
        req_set = set(f.lower() for f in required_functions)

        covered = req_set & analogy_props
        missing = req_set - covered

        return covered, missing

    def _analyze_coverage_by_props(self, properties: List[str],
                                   required_functions: List[str]) -> Tuple[Set[str], Set[str]]:
        """Анализирует покрытие функций набором свойств."""
        props_set = set(p.lower() for p in properties)
        req_set = set(f.lower() for f in required_functions)

        covered = req_set & props_set
        missing = req_set - covered

        return covered, missing

    def _find_nodes_for_functions(self, missing_functions: Set[str]) -> Dict[str, List[KnowledgeNode]]:
        """
        Для каждой недостающей функции находит подходящие узлы в ГЗ.
        """
        result = {}

        for func in missing_functions:
            candidates = []

            # 1. Поиск по свойству (точное совпадение)
            nodes_by_prop = self.global_graph.find_by_properties([func])
            candidates.extend(nodes_by_prop)

            # 2. Поиск по имени (если функция входит в имя узла)
            for node in self.global_graph.nodes.values():
                if func.lower() in node.name.lower() and node not in candidates:
                    candidates.append(node)

            # 3. Поиск по синонимам (можно расширить)
            # Например: 'поднимать груз' → 'лебедка', 'кран'
            # Пока просто используем найденные

            # Ограничиваем количество кандидатов (берём топ-5)
            result[func] = candidates[:5]

        return result

    def _generate_node_combinations(self, missing_nodes_dict: Dict[str, List[KnowledgeNode]],
                                    max_add: int = 3) -> List[List[KnowledgeNode]]:
        """
        Генерирует комбинации недостающих узлов (по одному из каждой функции).
        """
        # Если нет недостающих функций, возвращаем пустой список
        if not missing_nodes_dict:
            return [[]]

        # Собираем все возможные комбинации (по одному узлу для каждой функции)
        funcs = list(missing_nodes_dict.keys())
        # Берём функции, у которых есть кандидаты
        valid_funcs = [f for f in funcs if missing_nodes_dict.get(f)]
        if not valid_funcs:
            return []

        # Список списков кандидатов для каждой функции
        candidate_lists = [missing_nodes_dict[f] for f in valid_funcs]

        # Генерируем все комбинации (берём по одному из каждого списка)
        combinations_list = []
        for combo in product(*candidate_lists):
            nodes_list = list(combo)
            if len(nodes_list) <= max_add:
                combinations_list.append(nodes_list)

        # Если комбинаций слишком много, ограничиваем
        if len(combinations_list) > 20:
            combinations_list = combinations_list[:20]

        return combinations_list

    def _collect_properties(self, nodes: List[KnowledgeNode]) -> List[str]:
        """Собирает все свойства из списка узлов."""
        props = set()
        for node in nodes:
            props.update(node.properties)
        return list(props)

    def _create_hypothesis_from_analogy(self, analogy: Combination,
                                        required_functions: List[str],
                                        covered: Set[str],
                                        task_description: str = "") -> Hypothesis:
        """Создаёт гипотезу из аналогии (если все функции покрыты)."""
        total = len(required_functions)
        score = len(covered) / total if total > 0 else 1.0

        # Используем переданную задачу или заглушку
        task_desc = task_description if task_description else f"Гипотеза на основе аналогии {analogy.id}"

        hyp = Hypothesis(
            id=f"hyp_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}",
            task_description=task_desc,  # ← ИСПОЛЬЗУЕМ
            source_combination=analogy,
            modifications=[],
            description=f"Готовая модель на основе {len(analogy.nodes)} узлов",
            predicted_score=score,
            status=HypothesisStatus.PROPOSED,
            metadata={
                'source': 'hypothesis_generator',
                'analogy_id': analogy.id,
                'covered_functions': list(covered),
                'node_count': len(analogy.nodes),
                'node_names': [n.name for n in analogy.nodes]
            }
        )
        return hyp

    def _create_hypothesis(self,
                           source_analogy: Combination,
                           nodes: List[KnowledgeNode],
                           properties: List[str],
                           covered_functions: Set[str],
                           missing_functions: Set[str],
                           required_functions: List[str],
                           added_nodes: List[KnowledgeNode],
                           task_description: str = "") -> Hypothesis:
        """Создаёт новую гипотезу с добавленными узлами."""
        total = len(required_functions)
        coverage_ratio = len(covered_functions) / total if total > 0 else 1.0

        quality_score = coverage_ratio
        edge_count = self._count_edges_between_nodes(nodes)
        if edge_count > 0:
            quality_score += 0.1 * min(edge_count / len(nodes), 0.5)
        if len(nodes) > 10:
            quality_score -= 0.1
        quality_score = max(0.0, min(1.0, quality_score))

        # Используем переданную задачу или заглушку
        task_desc = task_description if task_description else f"Гипотеза на основе аналогии {source_analogy.id}"

        hyp = Hypothesis(
            id=f"hyp_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}",
            task_description=task_desc,  # ← ИСПОЛЬЗУЕМ
            source_combination=source_analogy,
            modifications=[f"add_{n.name}" for n in added_nodes],
            description=f"Модель из {len(nodes)} узлов, покрывает {len(covered_functions)}/{len(required_functions)} функций",
            predicted_score=quality_score,
            status=HypothesisStatus.PROPOSED,
            metadata={
                'source': 'hypothesis_generator',
                'analogy_id': source_analogy.id,
                'covered_functions': list(covered_functions),
                'missing_functions': list(missing_functions),
                'added_nodes': [n.name for n in added_nodes],
                'node_count': len(nodes),
                'node_names': [n.name for n in nodes],
                'edge_count': edge_count,
                'coverage_ratio': coverage_ratio
            }
        )
        return hyp

    def _count_edges_between_nodes(self, nodes: List[KnowledgeNode]) -> int:
        """Считает количество связей между узлами в списке."""
        if len(nodes) < 2:
            return 0

        node_ids = set(n.id for n in nodes)
        edge_count = 0

        for edge in self.global_graph.edges.values():
            if edge.source_id in node_ids and edge.target_id in node_ids:
                edge_count += 1

        return edge_count

    def save_hypothesis(self, hypothesis: Hypothesis) -> bool:
        """Сохраняет гипотезу в ИГЗ."""
        if not self.individual_graph:
            return False

        record = {
            'id': hypothesis.id,
            'type': 'hypothesis',
            'source_combination_id': hypothesis.source_combination.id,
            'description': hypothesis.description,
            'predicted_score': hypothesis.predicted_score,
            'metadata': hypothesis.metadata,
            'created_at': time.time()
        }
        self.individual_graph.add_knowledge(record)
        return True



# # core/knowledge/hypothesis_generator.py
# from core.knowledge import IndividualKnowledgeGraph
# from core.knowledge.hypothesis import Hypothesis
# from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
# from typing import List, Dict, Any, Tuple, Optional
# import numpy as np
#
# class HypothesisGenerator:
#     """
#     Генератор гипотез на основе GAN.
#     """
#
#     def __init__(self, global_graph: GlobalKnowledgeGraph,
#                  individual_graph: Optional[IndividualKnowledgeGraph] = None):
#         self.global_graph = global_graph
#         self.generator = self._build_generator()
#         self.discriminator = self._build_discriminator()
#
#     def generate_hypotheses(self,
#                             task_description: str,
#                             required_properties: List[str],
#                             n_hypotheses: int = 5) -> List[Hypothesis]:
#         """
#         Генерирует гипотезы для задачи.
#         """
#         # 1. Находим аналогии
#         analogies = self.analogy_engine.find_analogies(task_description, required_properties)
#
#         # 2. Для каждой аналогии генерируем модификации
#         hypotheses = []
#         for analogy in analogies[:n_hypotheses * 2]:
#             # Генератор предлагает модификации
#             modifications = self.generator.generate(analogy, task_description)
#             for mod in modifications:
#                 hypothesis = self._create_hypothesis(analogy, mod)
#                 hypotheses.append(hypothesis)
#
#         # 3. Оцениваем дискриминатором
#         scored = [(h, self.discriminator.evaluate(h, task_description)) for h in hypotheses]
#         scored.sort(key=lambda x: x[1], reverse=True)
#
#         return scored[:n_hypotheses]