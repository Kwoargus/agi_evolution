# core/knowledge/analogy_builder.py
"""
[ru] Построение аналогов из функциональных свойств.
[en] Building analogies from functional properties.
"""

from typing import List, Dict, Set, Optional, Tuple
from core.knowledge.knowledge_node import KnowledgeNode
from core.knowledge.knowledge_edge import KnowledgeEdge, EdgeType
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from core.knowledge.combination import Combination
from core.knowledge.individual_knowledge_graph import IndividualKnowledgeGraph


class AnalogyBuilder:
    """
    [ru] Строит аналоги (комбинации узлов + связи) на основе функциональных свойств.
    [en] Builds analogies (node combinations + connections) based on functional properties.
    """

    def __init__(self, global_graph: GlobalKnowledgeGraph,
                 individual_graph: Optional[IndividualKnowledgeGraph] = None):
        self.global_graph = global_graph
        self.individual_graph = individual_graph

    def build_analogy(self, functional_properties: List[str],
                      min_nodes: int = 2,
                      max_nodes: int = 5) -> List[Combination]:
        """
        [ru] Строит аналоги на основе функциональных свойств.
        [en] Builds analogies based on functional properties.

        Args:
            functional_properties: [ru] Список функциональных свойств. [en] List of functional properties
            min_nodes: [ru] Минимальное количество узлов в комбинации. [en] Minimum number of nodes in a combination
            max_nodes: [ru] Максимальное количество узлов в комбинации. [en] Maximum number of nodes in a combination

        Returns:
            [ru] Список комбинаций-аналогов, отсортированных по качеству
            [en] List of analogy combinations sorted by quality
        """
        # [ru] 1. Находим узлы по свойствам
        # [en] 1. Find nodes by properties
        candidate_nodes = self._find_nodes_by_functions(functional_properties)

        if not candidate_nodes:
            return []

        # [ru] 2. Строим комбинации узлов
        # [en] 2. Build node combinations
        combinations = self._build_combinations(
            candidate_nodes,
            functional_properties,
            min_nodes,
            max_nodes
        )

        # [ru] 3. Для каждой комбинации определяем структуру и связи
        # [en] 3. For each combination, determine structure and connections
        analogies = []
        for combo in combinations:
            structured = self._structure_analogy(combo, functional_properties)
            if structured:
                analogies.append(structured)

        # [ru] 4. Сортируем по качеству
        # [en] 4. Sort by quality
        analogies.sort(key=lambda x: x.metadata.get('quality_score', 0), reverse=True)

        return analogies

    def _find_nodes_by_functions(self, functions: List[str]) -> List[Dict]:
        """
        [ru] Находит узлы, соответствующие функциональным свойствам.
        [en] Finds nodes matching the functional properties.
        """
        candidates = []
        func_lower = [f.lower() for f in functions]

        for node in self.global_graph.nodes.values():
            matched = []
            node_props = [p.lower() for p in node.properties]
            node_name = node.name.lower()

            for func in func_lower:
                # [ru] Проверяем по имени
                # [en] Check by name
                if func in node_name or node_name in func:
                    matched.append(func)
                    continue

                # [ru] Проверяем по свойствам
                # [en] Check by properties
                for prop in node_props:
                    if func in prop or prop in func:
                        matched.append(func)
                        break

            if matched:
                candidates.append({
                    'node': node,
                    'matched_functions': matched,
                    'match_count': len(matched),
                    'match_ratio': len(matched) / len(functions)
                })

        # [ru] Сортируем по количеству совпадений
        # [en] Sort by number of matches
        candidates.sort(key=lambda x: x['match_ratio'], reverse=True)

        return candidates

    def _build_combinations(self,
                            candidates: List[Dict],
                            functions: List[str],
                            min_nodes: int,
                            max_nodes: int) -> List[Combination]:
        """
        [ru] Строит комбинации узлов.
        [en] Builds node combinations.
        """
        from itertools import combinations
        import hashlib
        import time

        analogies = []
        # [ru] Берём топ-15
        # [en] Take top-15
        nodes = [c['node'] for c in candidates[:15]]

        # [ru] Одиночные узлы
        # [en] Single nodes
        for item in candidates:
            if item['match_ratio'] >= 0.5:
                combo = Combination(
                    id=f"single_{item['node'].id}",
                    nodes=[item['node']],
                    properties=item['node'].properties.copy(),
                    metadata={
                        'source': 'global_graph',
                        'node_count': 1,
                        'matched_functions': item['matched_functions'],
                        'match_ratio': item['match_ratio'],
                        'missing_functions': [f for f in functions if f not in item['matched_functions']]
                    }
                )
                analogies.append(combo)

        # [ru] Комбинации из нескольких узлов
        # [en] Multi-node combinations
        seen = set()
        for size in range(min_nodes, min(max_nodes + 1, len(nodes) + 1)):
            for combo_nodes in combinations(nodes, size):
                # [ru] Собираем свойства
                # [en] Collect properties
                combo_props = set()
                combo_ids = []
                matched_all = set()

                for node in combo_nodes:
                    combo_props.update(node.properties)
                    combo_ids.append(node.id)

                    # [ru] Находим совпадения для этого узла
                    # [en] Find matches for this node
                    for item in candidates:
                        if item['node'].id == node.id:
                            matched_all.update(item['matched_functions'])
                            break

                # [ru] Проверяем покрытие
                # [en] Check coverage
                coverage = len(matched_all) / len(functions) if functions else 0

                if coverage < 0.3:
                    continue

                combo_id = f"combo_{hashlib.md5('_'.join(sorted(combo_ids)).encode()).hexdigest()[:8]}"
                if combo_id in seen:
                    continue
                seen.add(combo_id)

                combo = Combination(
                    id=combo_id,
                    nodes=list(combo_nodes),
                    properties=list(combo_props),
                    metadata={
                        'source': 'global_graph',
                        'node_count': len(combo_nodes),
                        'matched_functions': list(matched_all),
                        'match_ratio': coverage,
                        'missing_functions': [f for f in functions if f not in matched_all],
                        'node_names': [n.name for n in combo_nodes]
                    }
                )
                analogies.append(combo)

        return analogies

    def _structure_analogy(self, combination: Combination,
                           functions: List[str]) -> Optional[Combination]:
        """
        [ru] Определяет структуру аналога: главный узел → части.
        [en] Determines the analogy structure: main node → parts.
        """
        if not combination.nodes:
            return None

        nodes = combination.nodes

        # [ru] 1. Определяем главный узел (тот, у которого больше совпадений)
        # [en] 1. Determine the main node (the one with the most matches)
        main_node = None
        max_match = -1

        for node in nodes:
            # [ru] Проверяем, сколько функциональных свойств покрывает узел
            # [en] Check how many functional properties the node covers
            node_props = [p.lower() for p in node.properties]
            node_name = node.name.lower()
            match_count = 0

            for func in functions:
                func_lower = func.lower()
                if func_lower in node_name or node_name in func_lower:
                    match_count += 1
                for prop in node_props:
                    if func_lower in prop or prop in func_lower:
                        match_count += 1
                        break

            if match_count > max_match:
                max_match = match_count
                main_node = node

        if not main_node:
            main_node = nodes[0]

        # [ru] 2. Определяем части (остальные узлы)
        # [en] 2. Determine parts (remaining nodes)
        parts = [n for n in nodes if n.id != main_node.id]

        # [ru] 3. Проверяем связи между главным узлом и частями
        # [en] 3. Check connections between the main node and parts
        connections = self._get_connections(main_node, parts)

        # [ru] 4. Обновляем метаданные
        # [en] 4. Update metadata
        combination.metadata.update({
            'main_node': main_node.name,
            'main_node_id': main_node.id,
            'parts': [n.name for n in parts],
            'part_ids': [n.id for n in parts],
            'connections': connections,
            'is_structured': True,
            'quality_score': self._calculate_quality(combination, functions, connections)
        })

        return combination

    def _get_connections(self, main_node: KnowledgeNode,
                         parts: List[KnowledgeNode]) -> List[Dict]:
        """
        [ru] Находит связи между главным узлом и частями.
        [en] Finds connections between the main node and parts.
        """
        connections = []
        main_id = main_node.id
        part_ids = [p.id for p in parts]

        for edge in self.global_graph.edges.values():
            if edge.source_id == main_id and edge.target_id in part_ids:
                connections.append({
                    'type': 'source_to_target',
                    'source': edge.source_id,
                    'target': edge.target_id,
                    'edge_type': edge.edge_type.value,
                    'weight': edge.weight,
                    'description': edge.description
                })
            elif edge.target_id == main_id and edge.source_id in part_ids:
                connections.append({
                    'type': 'target_from_source',
                    'source': edge.source_id,
                    'target': edge.target_id,
                    'edge_type': edge.edge_type.value,
                    'weight': edge.weight,
                    'description': edge.description
                })

        # [ru] Если связей нет, добавляем "предполагаемые" связи
        # [en] If there are no connections, add "inferred" connections
        if not connections and parts:
            for part in parts:
                connections.append({
                    'type': 'inferred',
                    'source': main_id,
                    'target': part.id,
                    'edge_type': 'has_part',
                    'weight': 0.5,
                    'description': f'{main_node.name} содержит {part.name} (предполагаемая связь)'
                })

        return connections

    def _calculate_quality(self, combination: Combination,
                           functions: List[str],
                           connections: List[Dict]) -> float:
        """
        [ru] Вычисляет качество аналога.
        [en] Calculates the quality of the analogy.
        """
        score = 0.0

        # [ru] 1. Покрытие функций (0-0.5)
        # [en] 1. Function coverage (0-0.5)
        match_ratio = combination.metadata.get('match_ratio', 0)
        score += match_ratio * 0.5

        # [ru] 2. Наличие структуры (0-0.2)
        # [en] 2. Presence of structure (0-0.2)
        if combination.metadata.get('is_structured', False):
            score += 0.2

        # [ru] 3. Наличие связей (0-0.2)
        # [en] 3. Presence of connections (0-0.2)
        if connections:
            real_connections = sum(1 for c in connections if c.get('type') != 'inferred')
            if real_connections > 0:
                score += 0.15
            # [ru] Бонус за наличие всех связей
            # [en] Bonus for having all connections
            if len(connections) >= len(combination.nodes) - 1:
                score += 0.05

        # [ru] 4. Количество узлов (0-0.1)
        # [en] 4. Number of nodes (0-0.1)
        node_count = len(combination.nodes)
        if node_count >= 3:
            score += 0.1
        elif node_count >= 2:
            score += 0.05

        return min(1.0, score)


# # core/knowledge/analogy_builder.py
# """
# Построение аналогов из функциональных свойств.
# """
#
# from typing import List, Dict, Set, Optional, Tuple
# from core.knowledge.knowledge_node import KnowledgeNode
# from core.knowledge.knowledge_edge import KnowledgeEdge, EdgeType
# from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
# from core.knowledge.combination import Combination
# from core.knowledge.individual_knowledge_graph import IndividualKnowledgeGraph
#
#
# class AnalogyBuilder:
#     """
#     Строит аналоги (комбинации узлов + связи) на основе функциональных свойств.
#     """
#
#     def __init__(self, global_graph: GlobalKnowledgeGraph,
#                  individual_graph: Optional[IndividualKnowledgeGraph] = None):
#         self.global_graph = global_graph
#         self.individual_graph = individual_graph
#
#     def build_analogy(self, functional_properties: List[str],
#                       min_nodes: int = 2,
#                       max_nodes: int = 5) -> List[Combination]:
#         """
#         Строит аналоги на основе функциональных свойств.
#
#         Args:
#             functional_properties: Список функциональных свойств
#             min_nodes: Минимальное количество узлов в комбинации
#             max_nodes: Максимальное количество узлов в комбинации
#
#         Returns:
#             Список комбинаций-аналогов, отсортированных по качеству
#         """
#         # 1. Находим узлы по свойствам
#         candidate_nodes = self._find_nodes_by_functions(functional_properties)
#
#         if not candidate_nodes:
#             return []
#
#         # 2. Строим комбинации узлов
#         combinations = self._build_combinations(
#             candidate_nodes,
#             functional_properties,
#             min_nodes,
#             max_nodes
#         )
#
#         # 3. Для каждой комбинации определяем структуру и связи
#         analogies = []
#         for combo in combinations:
#             structured = self._structure_analogy(combo, functional_properties)
#             if structured:
#                 analogies.append(structured)
#
#         # 4. Сортируем по качеству
#         analogies.sort(key=lambda x: x.metadata.get('quality_score', 0), reverse=True)
#
#         return analogies
#
#     def _find_nodes_by_functions(self, functions: List[str]) -> List[Dict]:
#         """
#         Находит узлы, соответствующие функциональным свойствам.
#         """
#         candidates = []
#         func_lower = [f.lower() for f in functions]
#
#         for node in self.global_graph.nodes.values():
#             matched = []
#             node_props = [p.lower() for p in node.properties]
#             node_name = node.name.lower()
#
#             for func in func_lower:
#                 # Проверяем по имени
#                 if func in node_name or node_name in func:
#                     matched.append(func)
#                     continue
#
#                 # Проверяем по свойствам
#                 for prop in node_props:
#                     if func in prop or prop in func:
#                         matched.append(func)
#                         break
#
#             if matched:
#                 candidates.append({
#                     'node': node,
#                     'matched_functions': matched,
#                     'match_count': len(matched),
#                     'match_ratio': len(matched) / len(functions)
#                 })
#
#         # Сортируем по количеству совпадений
#         candidates.sort(key=lambda x: x['match_ratio'], reverse=True)
#
#         return candidates
#
#     def _build_combinations(self,
#                             candidates: List[Dict],
#                             functions: List[str],
#                             min_nodes: int,
#                             max_nodes: int) -> List[Combination]:
#         """
#         Строит комбинации узлов.
#         """
#         from itertools import combinations
#         import hashlib
#         import time
#
#         analogies = []
#         nodes = [c['node'] for c in candidates[:15]]  # Берём топ-15
#
#         # Одиночные узлы
#         for item in candidates:
#             if item['match_ratio'] >= 0.5:
#                 combo = Combination(
#                     id=f"single_{item['node'].id}",
#                     nodes=[item['node']],
#                     properties=item['node'].properties.copy(),
#                     metadata={
#                         'source': 'global_graph',
#                         'node_count': 1,
#                         'matched_functions': item['matched_functions'],
#                         'match_ratio': item['match_ratio'],
#                         'missing_functions': [f for f in functions if f not in item['matched_functions']]
#                     }
#                 )
#                 analogies.append(combo)
#
#         # Комбинации из нескольких узлов
#         seen = set()
#         for size in range(min_nodes, min(max_nodes + 1, len(nodes) + 1)):
#             for combo_nodes in combinations(nodes, size):
#                 # Собираем свойства
#                 combo_props = set()
#                 combo_ids = []
#                 matched_all = set()
#
#                 for node in combo_nodes:
#                     combo_props.update(node.properties)
#                     combo_ids.append(node.id)
#
#                     # Находим совпадения для этого узла
#                     for item in candidates:
#                         if item['node'].id == node.id:
#                             matched_all.update(item['matched_functions'])
#                             break
#
#                 # Проверяем покрытие
#                 coverage = len(matched_all) / len(functions) if functions else 0
#
#                 if coverage < 0.3:
#                     continue
#
#                 combo_id = f"combo_{hashlib.md5('_'.join(sorted(combo_ids)).encode()).hexdigest()[:8]}"
#                 if combo_id in seen:
#                     continue
#                 seen.add(combo_id)
#
#                 combo = Combination(
#                     id=combo_id,
#                     nodes=list(combo_nodes),
#                     properties=list(combo_props),
#                     metadata={
#                         'source': 'global_graph',
#                         'node_count': len(combo_nodes),
#                         'matched_functions': list(matched_all),
#                         'match_ratio': coverage,
#                         'missing_functions': [f for f in functions if f not in matched_all],
#                         'node_names': [n.name for n in combo_nodes]
#                     }
#                 )
#                 analogies.append(combo)
#
#         return analogies
#
#     def _structure_analogy(self, combination: Combination,
#                            functions: List[str]) -> Optional[Combination]:
#         """
#         Определяет структуру аналога: главный узел → части.
#         """
#         if not combination.nodes:
#             return None
#
#         nodes = combination.nodes
#
#         # 1. Определяем главный узел (тот, у которого больше совпадений)
#         main_node = None
#         max_match = -1
#
#         for node in nodes:
#             # Проверяем, сколько функциональных свойств покрывает узел
#             node_props = [p.lower() for p in node.properties]
#             node_name = node.name.lower()
#             match_count = 0
#
#             for func in functions:
#                 func_lower = func.lower()
#                 if func_lower in node_name or node_name in func_lower:
#                     match_count += 1
#                 for prop in node_props:
#                     if func_lower in prop or prop in func_lower:
#                         match_count += 1
#                         break
#
#             if match_count > max_match:
#                 max_match = match_count
#                 main_node = node
#
#         if not main_node:
#             main_node = nodes[0]
#
#         # 2. Определяем части (остальные узлы)
#         parts = [n for n in nodes if n.id != main_node.id]
#
#         # 3. Проверяем связи между главным узлом и частями
#         connections = self._get_connections(main_node, parts)
#
#         # 4. Обновляем метаданные
#         combination.metadata.update({
#             'main_node': main_node.name,
#             'main_node_id': main_node.id,
#             'parts': [n.name for n in parts],
#             'part_ids': [n.id for n in parts],
#             'connections': connections,
#             'is_structured': True,
#             'quality_score': self._calculate_quality(combination, functions, connections)
#         })
#
#         return combination
#
#     def _get_connections(self, main_node: KnowledgeNode,
#                          parts: List[KnowledgeNode]) -> List[Dict]:
#         """
#         Находит связи между главным узлом и частями.
#         """
#         connections = []
#         main_id = main_node.id
#         part_ids = [p.id for p in parts]
#
#         for edge in self.global_graph.edges.values():
#             if edge.source_id == main_id and edge.target_id in part_ids:
#                 connections.append({
#                     'type': 'source_to_target',
#                     'source': edge.source_id,
#                     'target': edge.target_id,
#                     'edge_type': edge.edge_type.value,
#                     'weight': edge.weight,
#                     'description': edge.description
#                 })
#             elif edge.target_id == main_id and edge.source_id in part_ids:
#                 connections.append({
#                     'type': 'target_from_source',
#                     'source': edge.source_id,
#                     'target': edge.target_id,
#                     'edge_type': edge.edge_type.value,
#                     'weight': edge.weight,
#                     'description': edge.description
#                 })
#
#         # Если связей нет, добавляем "предполагаемые" связи
#         if not connections and parts:
#             for part in parts:
#                 connections.append({
#                     'type': 'inferred',
#                     'source': main_id,
#                     'target': part.id,
#                     'edge_type': 'has_part',
#                     'weight': 0.5,
#                     'description': f'{main_node.name} содержит {part.name} (предполагаемая связь)'
#                 })
#
#         return connections
#
#     def _calculate_quality(self, combination: Combination,
#                            functions: List[str],
#                            connections: List[Dict]) -> float:
#         """
#         Вычисляет качество аналога.
#         """
#         score = 0.0
#
#         # 1. Покрытие функций (0-0.5)
#         match_ratio = combination.metadata.get('match_ratio', 0)
#         score += match_ratio * 0.5
#
#         # 2. Наличие структуры (0-0.2)
#         if combination.metadata.get('is_structured', False):
#             score += 0.2
#
#         # 3. Наличие связей (0-0.2)
#         if connections:
#             real_connections = sum(1 for c in connections if c.get('type') != 'inferred')
#             if real_connections > 0:
#                 score += 0.15
#             # Бонус за наличие всех связей
#             if len(connections) >= len(combination.nodes) - 1:
#                 score += 0.05
#
#         # 4. Количество узлов (0-0.1)
#         node_count = len(combination.nodes)
#         if node_count >= 3:
#             score += 0.1
#         elif node_count >= 2:
#             score += 0.05
#
#         return min(1.0, score)