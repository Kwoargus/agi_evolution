# core/knowledge/hypothesis_generator.py
"""
[ru] Генератор гипотез на основе аналогов. Создаёт новые комбинации узлов для покрытия функциональных требований.
[en] Hypothesis generator based on analogies. Creates new node combinations to cover functional requirements.
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
    [ru] Генерирует гипотезы (новые модели) на основе аналогов и функциональных требований.
    [en] Generates hypotheses (new models) based on analogies and functional requirements.
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

            # [ru] 1. Сохраняем комбинацию (если её нет)
            # [en] 1. Save the combination (if it does not exist)
            combo = hyp.source_combination
            if not db.save_combination(combo):
                print(f"[ru] Не удалось сохранить комбинацию {combo.id}")
                print(f"[en] Failed to save combination {combo.id}")
                continue

            # [ru] 2. Сохраняем гипотезу
            # [en] 2. Save the hypothesis
            if self.individual_graph:
                self.individual_graph.add_hypothesis(hyp)

            if db.save_hypothesis(hyp):
                saved_count += 1
                print(f"[ru] Гипотеза {hyp.id} сохранена")
                print(f"[en] Hypothesis {hyp.id} saved")

        return saved_count

    def save_hypothesis_to_kg(self, hypothesis: Hypothesis) -> bool:
        """
        [ru] Сохраняет гипотезу как новый узел в ГЗ.
        Args:
            [ru] hypothesis: Гипотеза для сохранения
        Returns:
            [ru] True если сохранение успешно

        [en] Saves the hypothesis as a new node in the Knowledge Graph.
        Args:
            [en] hypothesis: Hypothesis to save
        Returns:
            [en] True if saving was successful
        """
        try:
            # [ru] Создаём новый узел в ГЗ
            # [en] Create a new node in the Knowledge Graph
            new_node = KnowledgeNode(
                id=f"hyp_node_{hypothesis.id[:8]}",
                name=f"Модель: {hypothesis.task_description[:30]}",
                node_type="hypothesis",
                properties=hypothesis.source_combination.properties,
                description=hypothesis.description
            )

            self.global_graph.add_node(new_node)

            # [ru] Добавляем связи с исходными узлами
            # [en] Add connections with the source nodes
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

            # [ru] Сохраняем в БД
            # [en] Save to the database
            from db.knowledge_db import KnowledgeDB
            db = KnowledgeDB()
            db.save_node(new_node)

            print(f"✅ Гипотеза сохранена как узел в ГЗ: {new_node.name}")
            return True

        except Exception as e:
            print(f"[ru] Ошибка сохранения гипотезы в ГЗ: {e}")
            print(f"[en] Hypothesis persistence error in the KG: {e}")
            return False

    def generate_hypotheses(self,
                            analogies: List[Combination],
                            required_functions: List[str],
                            task_description: str = "",  # ← [ru] ДОБАВЛЯЕМ  [en] ADDING
                            max_hypotheses: int = 5) -> List[Hypothesis]:
        """
        [ru] Генерирует гипотезы на основе списка аналогов.
        Args:
            [ru] analogies: Список найденных аналогов (Combination)
            [ru] required_functions: Список функциональных свойств
            [ru] task_description: Оригинальный текст задачи (для сохранения)
            [ru] max_hypotheses: Максимальное количество гипотез для возврата
        Returns:
            [ru] Список гипотез (объектов Hypothesis)

        [en] Generates hypotheses based on a list of analogies.
        Args:
            [en] analogies: List of found analogies (Combination)
            [en] required_functions: List of functional properties
            [en] task_description: Original task text (for preservation)
            [en] max_hypotheses: Maximum number of hypotheses to return

        Returns:
            [en] List of hypotheses (Hypothesis objects)
        """
        hypotheses = []

        for analogy in analogies:
            # [ru] 1. Анализ покрытия
            # [en] 1. Coverage analysis
            covered, missing = self._analyze_coverage(analogy, required_functions)

            # [ru] 2. Если все функции покрыты — гипотеза уже есть
            # [en] 2. If all functions are covered — the hypothesis already exists
            if not missing:
                hyp = self._create_hypothesis_from_analogy(
                    analogy, required_functions, covered, task_description  # ← [ru] ПЕРЕДАЁМ  [en] PASSING
                )
                hypotheses.append(hyp)
                continue

            # [ru] 3. Поиск недостающих узлов
            # [en] 3. Search for missing nodes
            missing_nodes = self._find_nodes_for_functions(missing)

            # [ru] 4. Генерация комбинаций с добавлением недостающих узлов
            # [en] 4. Generate combinations by adding missing nodes
            for combo_nodes in self._generate_node_combinations(missing_nodes, max_add=3):
                new_nodes = list(analogy.nodes) + combo_nodes
                new_props = self._collect_properties(new_nodes)

                # [ru] Оценка покрытия после добавления
                # [en] Coverage evaluation after addition
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
                        task_description=task_description  # ← [ru] ПЕРЕДАЁМ  [en] PASSING
                    )
                    hypotheses.append(hyp)

        # [ru] Сортируем гипотезы по качеству
        # [en] Sort hypotheses by quality
        hypotheses.sort(key=lambda h: h.predicted_score, reverse=True)
        return hypotheses[:max_hypotheses]

    def _analyze_coverage(self, analogy: Combination,
                          required_functions: List[str]) -> Tuple[Set[str], Set[str]]:
        """
        [ru] Анализирует покрытие функций узлами аналогии.
        [en] Analyzes function coverage by the analogy nodes.
        """
        analogy_props = set(p.lower() for p in analogy.properties)
        req_set = set(f.lower() for f in required_functions)

        covered = req_set & analogy_props
        missing = req_set - covered

        return covered, missing

    def _analyze_coverage_by_props(self, properties: List[str],
                                   required_functions: List[str]) -> Tuple[Set[str], Set[str]]:
        """
        [ru] Анализирует покрытие функций набором свойств.
        [en] Analyzes function coverage by a set of properties.
        """
        props_set = set(p.lower() for p in properties)
        req_set = set(f.lower() for f in required_functions)

        covered = req_set & props_set
        missing = req_set - covered

        return covered, missing

    def _find_nodes_for_functions(self, missing_functions: Set[str]) -> Dict[str, List[KnowledgeNode]]:
        """
        [ru] Для каждой недостающей функции находит подходящие узлы в ГЗ.
        [en] For each missing function, finds suitable nodes in the Knowledge Graph.
        """
        result = {}

        for func in missing_functions:
            candidates = []

            # [ru] 1. Поиск по свойству (точное совпадение)
            # [en] 1. Search by property (exact match)
            nodes_by_prop = self.global_graph.find_by_properties([func])
            candidates.extend(nodes_by_prop)

            # [ru] 2. Поиск по имени (если функция входит в имя узла)
            # [en] 2. Search by name (if the function is part of the node name)
            for node in self.global_graph.nodes.values():
                if func.lower() in node.name.lower() and node not in candidates:
                    candidates.append(node)

            # [ru] 3. Поиск по синонимам (можно расширить)
            # [ru] Например: 'поднимать груз' → 'лебедка', 'кран'
            # [ru] Пока просто используем найденные
            # [en] 3. Search by synonyms (can be extended)
            # [en] For example: 'lift load' → 'winch', 'crane'
            # [en] For now, just use the found ones

            # [ru] Ограничиваем количество кандидатов (берём топ-5)
            # [en] Limit the number of candidates (take top-5)
            result[func] = candidates[:5]

        return result

    def _generate_node_combinations(self, missing_nodes_dict: Dict[str, List[KnowledgeNode]],
                                    max_add: int = 3) -> List[List[KnowledgeNode]]:
        """
        [ru] Генерирует комбинации недостающих узлов (по одному из каждой функции).
        [en] Generates combinations of missing nodes (one from each function).
        """
        # [ru] Если нет недостающих функций, возвращаем пустой список
        # [en] If there are no missing functions, return an empty list
        if not missing_nodes_dict:
            return [[]]

        # [ru] Собираем все возможные комбинации (по одному узлу для каждой функции)
        # [en] Collect all possible combinations (one node per function)
        funcs = list(missing_nodes_dict.keys())
        # [ru] Берём функции, у которых есть кандидаты
        # [en] Take functions that have candidates
        valid_funcs = [f for f in funcs if missing_nodes_dict.get(f)]
        if not valid_funcs:
            return []

        # [ru] Список списков кандидатов для каждой функции
        # [en] List of candidate lists for each function
        candidate_lists = [missing_nodes_dict[f] for f in valid_funcs]

        # [ru] Генерируем все комбинации (берём по одному из каждого списка)
        # [en] Generate all combinations (take one from each list)
        combinations_list = []
        for combo in product(*candidate_lists):
            nodes_list = list(combo)
            if len(nodes_list) <= max_add:
                combinations_list.append(nodes_list)

        # [ru] Если комбинаций слишком много, ограничиваем
        # [en] If there are too many combinations, limit them
        if len(combinations_list) > 20:
            combinations_list = combinations_list[:20]

        return combinations_list

    def _collect_properties(self, nodes: List[KnowledgeNode]) -> List[str]:
        """
        [ru] Собирает все свойства из списка узлов.
        [en] Collects all properties from a list of nodes.
        """
        props = set()
        for node in nodes:
            props.update(node.properties)
        return list(props)

    def _create_hypothesis_from_analogy(self, analogy: Combination,
                                        required_functions: List[str],
                                        covered: Set[str],
                                        task_description: str = "") -> Hypothesis:
        """
        [ru] Создаёт гипотезу из аналогии (если все функции покрыты).
        [en] Creates a hypothesis from an analogy (if all functions are covered).
        """
        total = len(required_functions)
        score = len(covered) / total if total > 0 else 1.0

        # [ru] Используем переданную задачу или заглушку
        # [en] Use the passed task or a stub
        task_desc = task_description if task_description else f" [ru] Гипотеза на основе аналогии {analogy.id} [en] Hypothesis based on analogy{analogy.id}"

        hyp = Hypothesis(
            id=f"hyp_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}",
            task_description=task_desc,  # ← [ru] ИСПОЛЬЗУЕМ  [en] USING
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
        """
        [ru] Создаёт новую гипотезу с добавленными узлами.
        [en] Creates a new hypothesis with added nodes.
        """
        total = len(required_functions)
        coverage_ratio = len(covered_functions) / total if total > 0 else 1.0

        quality_score = coverage_ratio
        edge_count = self._count_edges_between_nodes(nodes)
        if edge_count > 0:
            quality_score += 0.1 * min(edge_count / len(nodes), 0.5)
        if len(nodes) > 10:
            quality_score -= 0.1
        quality_score = max(0.0, min(1.0, quality_score))

        # [ru] Используем переданную задачу или заглушку
        # [en] Use the passed task or a stub
        task_desc = task_description if task_description else f"Гипотеза на основе аналогии {source_analogy.id}"

        hyp = Hypothesis(
            id=f"hyp_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}",
            task_description=task_desc,  # ← [ru] ИСПОЛЬЗУЕМ  [en] USING
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
        """
        [ru] Считает количество связей между узлами в списке.
        [en] Counts the number of edges between nodes in the list.
        """
        if len(nodes) < 2:
            return 0

        node_ids = set(n.id for n in nodes)
        edge_count = 0

        for edge in self.global_graph.edges.values():
            if edge.source_id in node_ids and edge.target_id in node_ids:
                edge_count += 1

        return edge_count

    def save_hypothesis(self, hypothesis: Hypothesis) -> bool:
        """
        [ru] Сохраняет гипотезу в ИГЗ.
        [en] Saves the hypothesis to the Individual Knowledge Graph.
        """
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
# """
# Генератор гипотез на основе аналогов.
# Создаёт новые комбинации узлов для покрытия функциональных требований.
# """
#
# from typing import List, Dict, Set, Optional, Tuple
# import hashlib
# import time
# from itertools import product
#
# from core.knowledge import KnowledgeEdge, EdgeType
# from core.knowledge.combination import Combination
# from core.knowledge.hypothesis import Hypothesis, HypothesisStatus
# from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
# from core.knowledge.individual_knowledge_graph import IndividualKnowledgeGraph
# from core.knowledge.knowledge_node import KnowledgeNode
#
#
# class HypothesisGenerator:
#     """
#     Генерирует гипотезы (новые модели) на основе аналогов и функциональных требований.
#     """
#
#     def __init__(self, global_graph: GlobalKnowledgeGraph,
#                  individual_graph: Optional[IndividualKnowledgeGraph] = None):
#         self.global_graph = global_graph
#         self.individual_graph = individual_graph or IndividualKnowledgeGraph()
#         self.cache = {}
#
#     def save_validated_hypotheses(self, hypotheses: List[Hypothesis]) -> int:
#         from db.knowledge_db import KnowledgeDB
#         db = KnowledgeDB()
#
#         saved_count = 0
#
#         for hyp in hypotheses:
#             if hyp.status != HypothesisStatus.VALIDATED:
#                 continue
#
#             # 1. Сохраняем комбинацию (если её нет)
#             combo = hyp.source_combination
#             if not db.save_combination(combo):
#                 print(f"⚠️ Не удалось сохранить комбинацию {combo.id}")
#                 continue
#
#             # 2. Сохраняем гипотезу
#             if self.individual_graph:
#                 self.individual_graph.add_hypothesis(hyp)
#
#             if db.save_hypothesis(hyp):
#                 saved_count += 1
#                 print(f"💾 Гипотеза {hyp.id} сохранена")
#
#         return saved_count
#
#     def save_hypothesis_to_kg(self, hypothesis: Hypothesis) -> bool:
#         """
#         Сохраняет гипотезу как новый узел в ГЗ.
#
#         Args:
#             hypothesis: Гипотеза для сохранения
#
#         Returns:
#             True если сохранение успешно
#         """
#         try:
#             # Создаём новый узел в ГЗ
#             new_node = KnowledgeNode(
#                 id=f"hyp_node_{hypothesis.id[:8]}",
#                 name=f"Модель: {hypothesis.task_description[:30]}",
#                 node_type="hypothesis",
#                 properties=hypothesis.source_combination.properties,
#                 description=hypothesis.description
#             )
#
#             self.global_graph.add_node(new_node)
#
#             # Добавляем связи с исходными узлами
#             for node in hypothesis.source_combination.nodes:
#                 edge = KnowledgeEdge(
#                     id=f"hyp_edge_{new_node.id}_{node.id}",
#                     source_id=new_node.id,
#                     target_id=node.id,
#                     edge_type=EdgeType.HAS_PART,
#                     weight=0.8,
#                     description=f"Содержит {node.name}"
#                 )
#                 self.global_graph.add_edge(edge)
#
#             # Сохраняем в БД
#             from db.knowledge_db import KnowledgeDB
#             db = KnowledgeDB()
#             db.save_node(new_node)
#
#             print(f"✅ Гипотеза сохранена как узел в ГЗ: {new_node.name}")
#             return True
#
#         except Exception as e:
#             print(f"❌ Ошибка сохранения гипотезы в ГЗ: {e}")
#             return False
#
#     def generate_hypotheses(self,
#                             analogies: List[Combination],
#                             required_functions: List[str],
#                             task_description: str = "",  # ← ДОБАВЛЯЕМ
#                             max_hypotheses: int = 5) -> List[Hypothesis]:
#         """
#         Генерирует гипотезы на основе списка аналогов.
#
#         Args:
#             analogies: Список найденных аналогов (Combination)
#             required_functions: Список функциональных свойств
#             task_description: Оригинальный текст задачи (для сохранения)
#             max_hypotheses: Максимальное количество гипотез для возврата
#
#         Returns:
#             Список гипотез (объектов Hypothesis)
#         """
#         hypotheses = []
#
#         for analogy in analogies:
#             # 1. Анализ покрытия
#             covered, missing = self._analyze_coverage(analogy, required_functions)
#
#             # 2. Если все функции покрыты — гипотеза уже есть
#             if not missing:
#                 hyp = self._create_hypothesis_from_analogy(
#                     analogy, required_functions, covered, task_description  # ← ПЕРЕДАЁМ
#                 )
#                 hypotheses.append(hyp)
#                 continue
#
#             # 3. Поиск недостающих узлов
#             missing_nodes = self._find_nodes_for_functions(missing)
#
#             # 4. Генерация комбинаций с добавлением недостающих узлов
#             for combo_nodes in self._generate_node_combinations(missing_nodes, max_add=3):
#                 new_nodes = list(analogy.nodes) + combo_nodes
#                 new_props = self._collect_properties(new_nodes)
#
#                 # Оценка покрытия после добавления
#                 new_covered, new_missing = self._analyze_coverage_by_props(new_props, required_functions)
#
#                 if len(new_covered) > len(covered):
#                     hyp = self._create_hypothesis(
#                         source_analogy=analogy,
#                         nodes=new_nodes,
#                         properties=new_props,
#                         covered_functions=new_covered,
#                         missing_functions=new_missing,
#                         required_functions=required_functions,
#                         added_nodes=combo_nodes,
#                         task_description=task_description  # ← ПЕРЕДАЁМ
#                     )
#                     hypotheses.append(hyp)
#
#         # Сортируем гипотезы по качеству
#         hypotheses.sort(key=lambda h: h.predicted_score, reverse=True)
#         return hypotheses[:max_hypotheses]
#
#     def _analyze_coverage(self, analogy: Combination,
#                           required_functions: List[str]) -> Tuple[Set[str], Set[str]]:
#         """
#         Анализирует покрытие функций узлами аналогии.
#         """
#         analogy_props = set(p.lower() for p in analogy.properties)
#         req_set = set(f.lower() for f in required_functions)
#
#         covered = req_set & analogy_props
#         missing = req_set - covered
#
#         return covered, missing
#
#     def _analyze_coverage_by_props(self, properties: List[str],
#                                    required_functions: List[str]) -> Tuple[Set[str], Set[str]]:
#         """
#         Анализирует покрытие функций набором свойств.
#         """
#         props_set = set(p.lower() for p in properties)
#         req_set = set(f.lower() for f in required_functions)
#
#         covered = req_set & props_set
#         missing = req_set - covered
#
#         return covered, missing
#
#     def _find_nodes_for_functions(self, missing_functions: Set[str]) -> Dict[str, List[KnowledgeNode]]:
#         """
#         Для каждой недостающей функции находит подходящие узлы в ГЗ.
#         """
#         result = {}
#
#         for func in missing_functions:
#             candidates = []
#
#             # 1. Поиск по свойству (точное совпадение)
#             nodes_by_prop = self.global_graph.find_by_properties([func])
#             candidates.extend(nodes_by_prop)
#
#             # 2. Поиск по имени (если функция входит в имя узла)
#             for node in self.global_graph.nodes.values():
#                 if func.lower() in node.name.lower() and node not in candidates:
#                     candidates.append(node)
#
#             # 3. Поиск по синонимам (можно расширить)
#             # Например: 'поднимать груз' → 'лебедка', 'кран'
#             # Пока просто используем найденные
#
#             # Ограничиваем количество кандидатов (берём топ-5)
#             result[func] = candidates[:5]
#
#         return result
#
#     def _generate_node_combinations(self, missing_nodes_dict: Dict[str, List[KnowledgeNode]],
#                                     max_add: int = 3) -> List[List[KnowledgeNode]]:
#         """
#         Генерирует комбинации недостающих узлов (по одному из каждой функции).
#         """
#         # Если нет недостающих функций, возвращаем пустой список
#         if not missing_nodes_dict:
#             return [[]]
#
#         # Собираем все возможные комбинации (по одному узлу для каждой функции)
#         funcs = list(missing_nodes_dict.keys())
#         # Берём функции, у которых есть кандидаты
#         valid_funcs = [f for f in funcs if missing_nodes_dict.get(f)]
#         if not valid_funcs:
#             return []
#
#         # Список списков кандидатов для каждой функции
#         candidate_lists = [missing_nodes_dict[f] for f in valid_funcs]
#
#         # Генерируем все комбинации (берём по одному из каждого списка)
#         combinations_list = []
#         for combo in product(*candidate_lists):
#             nodes_list = list(combo)
#             if len(nodes_list) <= max_add:
#                 combinations_list.append(nodes_list)
#
#         # Если комбинаций слишком много, ограничиваем
#         if len(combinations_list) > 20:
#             combinations_list = combinations_list[:20]
#
#         return combinations_list
#
#     def _collect_properties(self, nodes: List[KnowledgeNode]) -> List[str]:
#         """
#         Собирает все свойства из списка узлов.
#         """
#         props = set()
#         for node in nodes:
#             props.update(node.properties)
#         return list(props)
#
#     def _create_hypothesis_from_analogy(self, analogy: Combination,
#                                         required_functions: List[str],
#                                         covered: Set[str],
#                                         task_description: str = "") -> Hypothesis:
#         """
#         Создаёт гипотезу из аналогии (если все функции покрыты).
#         """
#         total = len(required_functions)
#         score = len(covered) / total if total > 0 else 1.0
#
#         # Используем переданную задачу или заглушку
#         task_desc = task_description if task_description else f"Гипотеза на основе аналогии {analogy.id}"
#
#         hyp = Hypothesis(
#             id=f"hyp_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}",
#             task_description=task_desc,  # ← ИСПОЛЬЗУЕМ
#             source_combination=analogy,
#             modifications=[],
#             description=f"Готовая модель на основе {len(analogy.nodes)} узлов",
#             predicted_score=score,
#             status=HypothesisStatus.PROPOSED,
#             metadata={
#                 'source': 'hypothesis_generator',
#                 'analogy_id': analogy.id,
#                 'covered_functions': list(covered),
#                 'node_count': len(analogy.nodes),
#                 'node_names': [n.name for n in analogy.nodes]
#             }
#         )
#         return hyp
#
#     def _create_hypothesis(self,
#                            source_analogy: Combination,
#                            nodes: List[KnowledgeNode],
#                            properties: List[str],
#                            covered_functions: Set[str],
#                            missing_functions: Set[str],
#                            required_functions: List[str],
#                            added_nodes: List[KnowledgeNode],
#                            task_description: str = "") -> Hypothesis:
#         """
#         Создаёт новую гипотезу с добавленными узлами.
#         """
#         total = len(required_functions)
#         coverage_ratio = len(covered_functions) / total if total > 0 else 1.0
#
#         quality_score = coverage_ratio
#         edge_count = self._count_edges_between_nodes(nodes)
#         if edge_count > 0:
#             quality_score += 0.1 * min(edge_count / len(nodes), 0.5)
#         if len(nodes) > 10:
#             quality_score -= 0.1
#         quality_score = max(0.0, min(1.0, quality_score))
#
#         # Используем переданную задачу или заглушку
#         task_desc = task_description if task_description else f"Гипотеза на основе аналогии {source_analogy.id}"
#
#         hyp = Hypothesis(
#             id=f"hyp_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}",
#             task_description=task_desc,  # ← ИСПОЛЬЗУЕМ
#             source_combination=source_analogy,
#             modifications=[f"add_{n.name}" for n in added_nodes],
#             description=f"Модель из {len(nodes)} узлов, покрывает {len(covered_functions)}/{len(required_functions)} функций",
#             predicted_score=quality_score,
#             status=HypothesisStatus.PROPOSED,
#             metadata={
#                 'source': 'hypothesis_generator',
#                 'analogy_id': source_analogy.id,
#                 'covered_functions': list(covered_functions),
#                 'missing_functions': list(missing_functions),
#                 'added_nodes': [n.name for n in added_nodes],
#                 'node_count': len(nodes),
#                 'node_names': [n.name for n in nodes],
#                 'edge_count': edge_count,
#                 'coverage_ratio': coverage_ratio
#             }
#         )
#         return hyp
#
#     def _count_edges_between_nodes(self, nodes: List[KnowledgeNode]) -> int:
#         """
#         Считает количество связей между узлами в списке.
#         """
#         if len(nodes) < 2:
#             return 0
#
#         node_ids = set(n.id for n in nodes)
#         edge_count = 0
#
#         for edge in self.global_graph.edges.values():
#             if edge.source_id in node_ids and edge.target_id in node_ids:
#                 edge_count += 1
#
#         return edge_count
#
#     def save_hypothesis(self, hypothesis: Hypothesis) -> bool:
#         """
#         Сохраняет гипотезу в ИГЗ.
#         """
#         if not self.individual_graph:
#             return False
#
#         record = {
#             'id': hypothesis.id,
#             'type': 'hypothesis',
#             'source_combination_id': hypothesis.source_combination.id,
#             'description': hypothesis.description,
#             'predicted_score': hypothesis.predicted_score,
#             'metadata': hypothesis.metadata,
#             'created_at': time.time()
#         }
#         self.individual_graph.add_knowledge(record)
#         return True
#
