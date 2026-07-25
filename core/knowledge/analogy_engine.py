# core/knowledge/analogy_engine.py
"""
[ru] Движок аналогий - находит и модифицирует модели знаний.
[ru] Реализует поиск комбинаций узлов для генерации новых знаний.
[en] Analogy engine - finds and modifies knowledge models.
[en] Implements search of node combinations for generating new knowledge.
"""

from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import hashlib
import time
from itertools import combinations
from collections import defaultdict

from core.knowledge.combination import Combination
from core.knowledge.knowledge_node import KnowledgeNode
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from core.knowledge.individual_knowledge_graph import IndividualKnowledgeGraph


class AnalogyEngine:
    """
    [ru] Движок аналогий - находит и модифицирует модели.
    [ru] Использует как ГГЗ, так и ИГЗ.
    [en] Analogy engine - finds and modifies models.
    [en] Uses both the Global Knowledge Graph and the Individual Knowledge Graph.
    """

    def __init__(self, global_graph: GlobalKnowledgeGraph,
                 individual_graph: IndividualKnowledgeGraph):
        self.global_graph = global_graph
        self.individual_graph = individual_graph
        self.analogy_cache = {}

    def find_analogies(self,
                       task_description: str,
                       required_properties: List[str],
                       max_results: int = 10,
                       min_nodes: int = 2,
                       max_nodes: int = 5) -> List[Combination]:
        """
        [ru] Находит аналогии для задачи.
        [ru] 1. Поиск в ИГЗ (быстрый, образный)
        [ru] 2. Построение структурированных аналогов из ГГЗ
        [ru] 3. Комбинация и ранжирование результатов

        [en] Finds analogies for a task.
        [en] 1. Search in the Individual Knowledge Graph (fast, image-based)
        [en] 2. Build structured analogies from the Global Knowledge Graph
        [en] 3. Combine and rank the results
        """
        cache_key = f"{task_description}_{'_'.join(sorted(required_properties))}"
        if cache_key in self.analogy_cache:
            return self.analogy_cache[cache_key][:max_results]

        all_analogies = []

        # [ru] 1. Ищем в ИГЗ (ментальные модели)
        # [en] 1. Search in the Individual Knowledge Graph (mental models)
        mental_analogies = self._search_mental(required_properties)
        all_analogies.extend(mental_analogies)

        # [ru] 2. Строим структурированные аналоги из ГГЗ
        # [en] 2. Build structured analogies from the Global Knowledge Graph
        if self.global_graph:
            from core.knowledge.analogy_builder import AnalogyBuilder
            builder = AnalogyBuilder(self.global_graph, self.individual_graph)
            structured = builder.build_analogy(
                functional_properties=required_properties,
                min_nodes=min_nodes,
                max_nodes=max_nodes
            )
            all_analogies.extend(structured)

        # [ru] 3. Объединяем и ранжируем
        # [en] 3. Merge and rank
        ranked = self._merge_and_rank(all_analogies, required_properties)

        # [ru] Кэшируем
        # [en] Cache the result
        self.analogy_cache[cache_key] = ranked

        return ranked[:max_results]


    def _search_mental(self, required_properties: List[str]) -> List[Combination]:
        """
        [ru] Ищет аналогии в индивидуальном графе знаний (образные модели).
        [en] Searches for analogies in the individual knowledge graph (image-based models).
        """
        analogies = []

        if not self.individual_graph or not hasattr(self.individual_graph, 'find_mental_models_by_properties'):
            return analogies

        found_models = self.individual_graph.find_mental_models_by_properties(required_properties)

        for model_info in found_models[:10]:
            model_data = model_info.get('model_data', {})

            combo = Combination(
                id=f"mental_{model_info['model_id']}",
                nodes=[],
                properties=model_data.get('properties', []),
                metadata={
                    'source': 'individual_graph',
                    'model_id': model_info['model_id'],
                    'model_data': model_data,
                    'match_count': model_info['match_count'],
                    'usage_count': model_info.get('usage_count', 0),
                    'score': model_info['match_count'] / max(len(required_properties), 1)
                }
            )
            analogies.append(combo)

        return analogies

    def _search_combinations(self,
                             required_properties: List[str],
                             min_nodes: int = 2,
                             max_nodes: int = 5) -> List[Combination]:
        """
        [ru] Находит комбинации узлов в ГЗ по требуемым свойствам.
        [ru] Алгоритм:
        [ru] 1. Находим все узлы, содержащие хотя бы одно требуемое свойство
        [ru] 2. Строим комбинации из 2-5 узлов
        [ru] 3. Оцениваем покрытие свойств
        [ru] 4. Ранжируем по полноте покрытия

        [en] Finds node combinations in the Knowledge Graph by required properties.
        [en] Algorithm:
        [en] 1. Find all nodes containing at least one required property
        [en] 2. Build combinations of 2-5 nodes
        [en] 3. Evaluate property coverage
        [en] 4. Rank by coverage completeness
        """
        analogies = []

        if not self.global_graph:
            return analogies

        # [ru] 1. Находим все узлы с нужными свойствами
        # [en] 1. Find all nodes with the required properties
        candidate_nodes = []
        for node in self.global_graph.nodes.values():
            # [ru] Проверяем, есть ли пересечение свойств
            # [en] Check whether there is an intersection of properties
            node_props = set(p.lower() for p in node.properties)
            req_props = set(p.lower() for p in required_properties)
            intersection = node_props & req_props
            if intersection:
                candidate_nodes.append({
                    'node': node,
                    'matched_props': intersection,
                    'match_count': len(intersection)
                })

        if not candidate_nodes:
            return analogies

        # [ru] 2. Сортируем по количеству совпадений
        # [en] 2. Sort by number of matches
        candidate_nodes.sort(key=lambda x: x['match_count'], reverse=True)

        # [ru] 3. Строим комбинации (берем топ-10 узлов для комбинаторики)
        # [en] 3. Build combinations (take top-10 nodes for combinatorics)
        top_nodes = candidate_nodes[:10]
        nodes_list = [item['node'] for item in top_nodes]

        # [ru] 4. Генерируем комбинации размером от min_nodes до max_nodes
        # [en] 4. Generate combinations of size from min_nodes to max_nodes
        seen_combinations = set()

        for size in range(min_nodes, min(max_nodes + 1, len(nodes_list) + 1)):
            for combo_nodes in combinations(nodes_list, size):
                # [ru] Собираем свойства комбинации
                # [en] Collect the properties of the combination
                combo_props = set()
                combo_ids = []
                for node in combo_nodes:
                    combo_props.update(node.properties)
                    combo_ids.append(node.id)

                # [ru] Проверяем покрытие требуемых свойств
                # [en] Check coverage of required properties
                req_set = set(p.lower() for p in required_properties)
                covered = req_set & set(p.lower() for p in combo_props)
                coverage_ratio = len(covered) / len(req_set) if req_set else 1.0

                # [ru] Пропускаем комбинации с низким покрытием
                # [en] Skip combinations with low coverage
                if coverage_ratio < 0.3:
                    continue

                # [ru] Создаём уникальный ID комбинации
                # [en] Create a unique combination ID
                combo_id = f"combo_{hashlib.md5('_'.join(sorted(combo_ids)).encode()).hexdigest()[:8]}"

                # [ru] Проверяем, не было ли такой комбинации
                # [en] Check whether this combination already exists
                if combo_id in seen_combinations:
                    continue
                seen_combinations.add(combo_id)

                # [ru] Создаём комбинацию
                # [en] Create the combination
                combo = Combination(
                    id=combo_id,
                    nodes=list(combo_nodes),
                    properties=list(combo_props),
                    metadata={
                        'source': 'global_graph',
                        'node_count': len(combo_nodes),
                        'coverage_ratio': coverage_ratio,
                        'covered_properties': list(covered),
                        'missing_properties': list(req_set - covered),
                        'node_names': [n.name for n in combo_nodes],
                        'size': len(combo_nodes)
                    }
                )
                analogies.append(combo)

        # [ru] 5. Добавляем одиночные узлы (для случаев, когда один узел уже покрывает всё)
        # [en] 5. Add single nodes (for cases when one node already covers everything)
        for item in candidate_nodes:
            node = item['node']
            req_set = set(p.lower() for p in required_properties)
            node_props = set(p.lower() for p in node.properties)
            covered = req_set & node_props
            coverage_ratio = len(covered) / len(req_set) if req_set else 1.0

            if coverage_ratio >= 0.5:
                combo = Combination(
                    id=f"single_{node.id}",
                    nodes=[node],
                    properties=node.properties.copy(),
                    metadata={
                        'source': 'global_graph',
                        'node_count': 1,
                        'coverage_ratio': coverage_ratio,
                        'covered_properties': list(covered),
                        'missing_properties': list(req_set - covered),
                        'node_names': [node.name],
                        'size': 1
                    }
                )
                analogies.append(combo)

        return analogies

    def _merge_and_rank(self, analogies: List[Combination],
                        required_properties: List[str]) -> List[Combination]:
        """
        [ru] Объединяет и ранжирует аналогии.
        [en] Merges and ranks analogies.
        """
        # [ru] Удаляем дубликаты по ID
        # [en] Remove duplicates by ID
        seen = set()
        unique = []
        for analogy in analogies:
            if analogy.id not in seen:
                seen.add(analogy.id)
                unique.append(analogy)

        # [ru] Вычисляем оценку для каждой аналогии
        # [en] Compute the score for each analogy
        req_set = set(p.lower() for p in required_properties)

        for analogy in unique:
            base_score = 0.0

            # [ru] 1. Покрытие свойств (0-0.6)
            # [en] 1. Property coverage (0-0.6)
            analogy_props = set(p.lower() for p in analogy.properties)
            covered = req_set & analogy_props
            coverage_score = len(covered) / len(req_set) if req_set else 1.0
            base_score += coverage_score * 0.6

            # [ru] 2. Источник (ИГЗ имеет приоритет) (0-0.2)
            # [en] 2. Source (Individual Knowledge Graph has priority) (0-0.2)
            source = analogy.metadata.get('source', '')
            if source == 'individual_graph':
                base_score += 0.2

            # [ru] 3. Количество узлов (0-0.2)
            # [en] 3. Number of nodes (0-0.2)
            node_count = len(analogy.nodes)
            if node_count >= 2:
                base_score += 0.2 * min(node_count / 5, 1.0)

            # [ru] 4. Бонус за связанные узлы (если есть связи между узлами комбинации)
            # [en] 4. Bonus for connected nodes (if there are edges between the combination nodes)
            if node_count >= 2:
                edge_count = self._count_edges_between_nodes(analogy.nodes)
                if edge_count > 0:
                    base_score += 0.1 * min(edge_count / node_count, 1.0)

            # [ru] Сохраняем оценку
            # [en] Save the score
            analogy.metadata['score'] = base_score
            analogy.metadata['coverage_score'] = coverage_score

        # [ru] Сортируем по оценке
        # [en] Sort by score
        unique.sort(key=lambda x: x.metadata.get('score', 0), reverse=True)

        return unique

    def _count_edges_between_nodes(self, nodes: List[KnowledgeNode]) -> int:
        """
        [ru] Считает количество связей между узлами в комбинации.
        [en] Counts the number of edges between nodes in a combination.
        """
        if len(nodes) < 2:
            return 0

        node_ids = set(n.id for n in nodes)
        edge_count = 0

        for edge in self.global_graph.edges.values():
            if edge.source_id in node_ids and edge.target_id in node_ids:
                edge_count += 1

        return edge_count

    def modify_analogy(self, analogy: Combination,
                       modifications: List[str]) -> Combination:
        """
        [ru] Модифицирует аналогию.
        [en] Modifies an analogy.
        """
        modified = analogy.copy()

        for mod in modifications:
            if mod == "replace_part":
                # [ru] Заменяет часть модели другой моделью
                # [en] Replaces a part of the model with another model
                if len(modified.nodes) > 1:
                    replacement = self._find_replacement(modified.nodes[0])
                    if replacement:
                        modified.nodes[0] = replacement
                        modified.metadata['modified'] = True

            elif mod.startswith("add_feature_"):
                # [ru] Добавляет свойство
                # [en] Adds a property
                new_property = mod.replace("add_feature_", "")
                if new_property not in modified.properties:
                    modified.properties.append(new_property)

            elif mod == "combine":
                # [ru] Комбинирует с другой моделью
                # [en] Combines with another model
                other = self._find_combination_partner(modified)
                if other:
                    modified.nodes.append(other)
                    modified.id = f"combined_{modified.id}"
                    modified.metadata['node_count'] = len(modified.nodes)

        return modified

    def _find_replacement(self, node: KnowledgeNode) -> Optional[KnowledgeNode]:
        """
        [ru] Находит замену для узла в ГГЗ.
        [en] Finds a replacement for a node in the Global Knowledge Graph.
        """
        if not self.global_graph:
            return None

        if hasattr(node, 'node_type'):
            candidates = self.global_graph.find_by_type(node.node_type)
            if candidates:
                import random
                return random.choice([n for n in candidates if n.id != node.id])

        return None

    def _find_combination_partner(self, analogy: Combination) -> Optional[KnowledgeNode]:
        """
        [ru] Находит партнёра для комбинирования.
        [en] Finds a partner for combination.
        """
        if not self.global_graph:
            return None

        existing_ids = [n.id for n in analogy.nodes]
        for node in self.global_graph.nodes.values():
            if node.id not in existing_ids:
                return node

        return None

    def clear_cache(self):
        """
        [ru] Очищает кэш аналогий.
        [en] Clears the analogy cache.
        """
        self.analogy_cache.clear()

    def get_combination_statistics(self, combination: Combination) -> Dict[str, Any]:
        """
        [ru] Возвращает статистику по комбинации узлов.
        [en] Returns statistics for a node combination.
        """
        stats = {
            'node_count': len(combination.nodes),
            'property_count': len(combination.properties),
            'nodes': [n.name for n in combination.nodes],
            # [ru] Только первые 10
            # [en] Only the first 10
            'properties': combination.properties[:10],
            'score': combination.metadata.get('score', 0),
            'coverage_ratio': combination.metadata.get('coverage_ratio', 0),
        }
        return stats



# # core/knowledge/analogy_engine.py
# """
# Движок аналогий - находит и модифицирует модели знаний.
# Реализует поиск комбинаций узлов для генерации новых знаний.
# """
#
# from typing import List, Dict, Any, Tuple, Optional, Set
# import numpy as np
# import hashlib
# import time
# from itertools import combinations
# from collections import defaultdict
#
# from core.knowledge.combination import Combination
# from core.knowledge.knowledge_node import KnowledgeNode
# from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
# from core.knowledge.individual_knowledge_graph import IndividualKnowledgeGraph
#
#
# class AnalogyEngine:
#     """
#     Движок аналогий - находит и модифицирует модели.
#     Использует как ГГЗ, так и ИГЗ.
#     """
#
#     def __init__(self, global_graph: GlobalKnowledgeGraph,
#                  individual_graph: IndividualKnowledgeGraph):
#         self.global_graph = global_graph
#         self.individual_graph = individual_graph
#         self.analogy_cache = {}
#
#     def find_analogies(self,
#                        task_description: str,
#                        required_properties: List[str],
#                        max_results: int = 10,
#                        min_nodes: int = 2,
#                        max_nodes: int = 5) -> List[Combination]:
#         """
#         Находит аналогии для задачи.
#
#         1. Поиск в ИГЗ (быстрый, образный)
#         2. Построение структурированных аналогов из ГГЗ
#         3. Комбинация и ранжирование результатов
#         """
#         cache_key = f"{task_description}_{'_'.join(sorted(required_properties))}"
#         if cache_key in self.analogy_cache:
#             return self.analogy_cache[cache_key][:max_results]
#
#         all_analogies = []
#
#         # 1. Ищем в ИГЗ (ментальные модели)
#         mental_analogies = self._search_mental(required_properties)
#         all_analogies.extend(mental_analogies)
#
#         # 2. Строим структурированные аналоги из ГГЗ
#         if self.global_graph:
#             from core.knowledge.analogy_builder import AnalogyBuilder
#             builder = AnalogyBuilder(self.global_graph, self.individual_graph)
#             structured = builder.build_analogy(
#                 functional_properties=required_properties,
#                 min_nodes=min_nodes,
#                 max_nodes=max_nodes
#             )
#             all_analogies.extend(structured)
#
#         # 3. Объединяем и ранжируем
#         ranked = self._merge_and_rank(all_analogies, required_properties)
#
#         # Кэшируем
#         self.analogy_cache[cache_key] = ranked
#
#         return ranked[:max_results]
#
#
#     def _search_mental(self, required_properties: List[str]) -> List[Combination]:
#         """Ищет аналогии в индивидуальном графе знаний (образные модели)."""
#         analogies = []
#
#         if not self.individual_graph or not hasattr(self.individual_graph, 'find_mental_models_by_properties'):
#             return analogies
#
#         found_models = self.individual_graph.find_mental_models_by_properties(required_properties)
#
#         for model_info in found_models[:10]:
#             model_data = model_info.get('model_data', {})
#
#             combo = Combination(
#                 id=f"mental_{model_info['model_id']}",
#                 nodes=[],
#                 properties=model_data.get('properties', []),
#                 metadata={
#                     'source': 'individual_graph',
#                     'model_id': model_info['model_id'],
#                     'model_data': model_data,
#                     'match_count': model_info['match_count'],
#                     'usage_count': model_info.get('usage_count', 0),
#                     'score': model_info['match_count'] / max(len(required_properties), 1)
#                 }
#             )
#             analogies.append(combo)
#
#         return analogies
#
#     def _search_combinations(self,
#                              required_properties: List[str],
#                              min_nodes: int = 2,
#                              max_nodes: int = 5) -> List[Combination]:
#         """
#         Находит комбинации узлов в ГЗ по требуемым свойствам.
#
#         Алгоритм:
#         1. Находим все узлы, содержащие хотя бы одно требуемое свойство
#         2. Строим комбинации из 2-5 узлов
#         3. Оцениваем покрытие свойств
#         4. Ранжируем по полноте покрытия
#         """
#         analogies = []
#
#         if not self.global_graph:
#             return analogies
#
#         # 1. Находим все узлы с нужными свойствами
#         candidate_nodes = []
#         for node in self.global_graph.nodes.values():
#             # Проверяем, есть ли пересечение свойств
#             node_props = set(p.lower() for p in node.properties)
#             req_props = set(p.lower() for p in required_properties)
#             intersection = node_props & req_props
#             if intersection:
#                 candidate_nodes.append({
#                     'node': node,
#                     'matched_props': intersection,
#                     'match_count': len(intersection)
#                 })
#
#         if not candidate_nodes:
#             return analogies
#
#         # 2. Сортируем по количеству совпадений
#         candidate_nodes.sort(key=lambda x: x['match_count'], reverse=True)
#
#         # 3. Строим комбинации (берем топ-10 узлов для комбинаторики)
#         top_nodes = candidate_nodes[:10]
#         nodes_list = [item['node'] for item in top_nodes]
#
#         # 4. Генерируем комбинации размером от min_nodes до max_nodes
#         seen_combinations = set()
#
#         for size in range(min_nodes, min(max_nodes + 1, len(nodes_list) + 1)):
#             for combo_nodes in combinations(nodes_list, size):
#                 # Собираем свойства комбинации
#                 combo_props = set()
#                 combo_ids = []
#                 for node in combo_nodes:
#                     combo_props.update(node.properties)
#                     combo_ids.append(node.id)
#
#                 # Проверяем покрытие требуемых свойств
#                 req_set = set(p.lower() for p in required_properties)
#                 covered = req_set & set(p.lower() for p in combo_props)
#                 coverage_ratio = len(covered) / len(req_set) if req_set else 1.0
#
#                 # Пропускаем комбинации с низким покрытием
#                 if coverage_ratio < 0.3:
#                     continue
#
#                 # Создаём уникальный ID комбинации
#                 combo_id = f"combo_{hashlib.md5('_'.join(sorted(combo_ids)).encode()).hexdigest()[:8]}"
#
#                 # Проверяем, не было ли такой комбинации
#                 if combo_id in seen_combinations:
#                     continue
#                 seen_combinations.add(combo_id)
#
#                 # Создаём комбинацию
#                 combo = Combination(
#                     id=combo_id,
#                     nodes=list(combo_nodes),
#                     properties=list(combo_props),
#                     metadata={
#                         'source': 'global_graph',
#                         'node_count': len(combo_nodes),
#                         'coverage_ratio': coverage_ratio,
#                         'covered_properties': list(covered),
#                         'missing_properties': list(req_set - covered),
#                         'node_names': [n.name for n in combo_nodes],
#                         'size': len(combo_nodes)
#                     }
#                 )
#                 analogies.append(combo)
#
#         # 5. Добавляем одиночные узлы (для случаев, когда один узел уже покрывает всё)
#         for item in candidate_nodes:
#             node = item['node']
#             req_set = set(p.lower() for p in required_properties)
#             node_props = set(p.lower() for p in node.properties)
#             covered = req_set & node_props
#             coverage_ratio = len(covered) / len(req_set) if req_set else 1.0
#
#             if coverage_ratio >= 0.5:
#                 combo = Combination(
#                     id=f"single_{node.id}",
#                     nodes=[node],
#                     properties=node.properties.copy(),
#                     metadata={
#                         'source': 'global_graph',
#                         'node_count': 1,
#                         'coverage_ratio': coverage_ratio,
#                         'covered_properties': list(covered),
#                         'missing_properties': list(req_set - covered),
#                         'node_names': [node.name],
#                         'size': 1
#                     }
#                 )
#                 analogies.append(combo)
#
#         return analogies
#
#     def _merge_and_rank(self, analogies: List[Combination],
#                         required_properties: List[str]) -> List[Combination]:
#         """Объединяет и ранжирует аналогии."""
#         # Удаляем дубликаты по ID
#         seen = set()
#         unique = []
#         for analogy in analogies:
#             if analogy.id not in seen:
#                 seen.add(analogy.id)
#                 unique.append(analogy)
#
#         # Вычисляем оценку для каждой аналогии
#         req_set = set(p.lower() for p in required_properties)
#
#         for analogy in unique:
#             base_score = 0.0
#
#             # 1. Покрытие свойств (0-0.6)
#             analogy_props = set(p.lower() for p in analogy.properties)
#             covered = req_set & analogy_props
#             coverage_score = len(covered) / len(req_set) if req_set else 1.0
#             base_score += coverage_score * 0.6
#
#             # 2. Источник (ИГЗ имеет приоритет) (0-0.2)
#             source = analogy.metadata.get('source', '')
#             if source == 'individual_graph':
#                 base_score += 0.2
#
#             # 3. Количество узлов (0-0.2)
#             node_count = len(analogy.nodes)
#             if node_count >= 2:
#                 base_score += 0.2 * min(node_count / 5, 1.0)
#
#             # 4. Бонус за связанные узлы (если есть связи между узлами комбинации)
#             if node_count >= 2:
#                 edge_count = self._count_edges_between_nodes(analogy.nodes)
#                 if edge_count > 0:
#                     base_score += 0.1 * min(edge_count / node_count, 1.0)
#
#             # Сохраняем оценку
#             analogy.metadata['score'] = base_score
#             analogy.metadata['coverage_score'] = coverage_score
#
#         # Сортируем по оценке
#         unique.sort(key=lambda x: x.metadata.get('score', 0), reverse=True)
#
#         return unique
#
#     def _count_edges_between_nodes(self, nodes: List[KnowledgeNode]) -> int:
#         """Считает количество связей между узлами в комбинации."""
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
#     def modify_analogy(self, analogy: Combination,
#                        modifications: List[str]) -> Combination:
#         """
#         Модифицирует аналогию.
#         """
#         modified = analogy.copy()
#
#         for mod in modifications:
#             if mod == "replace_part":
#                 # Заменяет часть модели другой моделью
#                 if len(modified.nodes) > 1:
#                     replacement = self._find_replacement(modified.nodes[0])
#                     if replacement:
#                         modified.nodes[0] = replacement
#                         modified.metadata['modified'] = True
#
#             elif mod.startswith("add_feature_"):
#                 # Добавляет свойство
#                 new_property = mod.replace("add_feature_", "")
#                 if new_property not in modified.properties:
#                     modified.properties.append(new_property)
#
#             elif mod == "combine":
#                 # Комбинирует с другой моделью
#                 other = self._find_combination_partner(modified)
#                 if other:
#                     modified.nodes.append(other)
#                     modified.id = f"combined_{modified.id}"
#                     modified.metadata['node_count'] = len(modified.nodes)
#
#         return modified
#
#     def _find_replacement(self, node: KnowledgeNode) -> Optional[KnowledgeNode]:
#         """Находит замену для узла в ГГЗ."""
#         if not self.global_graph:
#             return None
#
#         if hasattr(node, 'node_type'):
#             candidates = self.global_graph.find_by_type(node.node_type)
#             if candidates:
#                 import random
#                 return random.choice([n for n in candidates if n.id != node.id])
#
#         return None
#
#     def _find_combination_partner(self, analogy: Combination) -> Optional[KnowledgeNode]:
#         """Находит партнёра для комбинирования."""
#         if not self.global_graph:
#             return None
#
#         existing_ids = [n.id for n in analogy.nodes]
#         for node in self.global_graph.nodes.values():
#             if node.id not in existing_ids:
#                 return node
#
#         return None
#
#     def clear_cache(self):
#         """Очищает кэш аналогий."""
#         self.analogy_cache.clear()
#
#     def get_combination_statistics(self, combination: Combination) -> Dict[str, Any]:
#         """
#         Возвращает статистику по комбинации узлов.
#         """
#         stats = {
#             'node_count': len(combination.nodes),
#             'property_count': len(combination.properties),
#             'nodes': [n.name for n in combination.nodes],
#             'properties': combination.properties[:10],  # Только первые 10
#             'score': combination.metadata.get('score', 0),
#             'coverage_ratio': combination.metadata.get('coverage_ratio', 0),
#         }
#         return stats
#
