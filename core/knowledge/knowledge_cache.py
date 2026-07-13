# core/knowledge/knowledge_cache.py
"""
Кэш для графа знаний (in-memory).
"""

from typing import Dict, List, Optional
import networkx as nx
from .knowledge_node import KnowledgeNode
from .knowledge_edge import KnowledgeEdge


class KnowledgeCache:
    """
    In-memory кэш графа знаний для быстрого обхода.
    """

    def __init__(self):
        # Граф для быстрого обхода
        self.graph = nx.DiGraph()

        # Кэш узлов
        self.nodes: Dict[str, KnowledgeNode] = {}

        # Кэш ребер
        self.edges: Dict[str, KnowledgeEdge] = {}

        # Индексы
        self.property_index: Dict[str, List[str]] = {}
        self.type_index: Dict[str, List[str]] = {}

    def add_node(self, node: KnowledgeNode):
        """Добавляет узел в кэш."""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, **node.metadata)

        # Индексация по свойствам
        for prop in node.properties:
            if prop not in self.property_index:
                self.property_index[prop] = []
            if node.id not in self.property_index[prop]:
                self.property_index[prop].append(node.id)

        # Индексация по типу
        if node.node_type not in self.type_index:
            self.type_index[node.node_type] = []
        if node.id not in self.type_index[node.node_type]:
            self.type_index[node.node_type].append(node.id)

    def add_edge(self, edge: KnowledgeEdge):
        """Добавляет ребро в кэш."""
        self.edges[edge.id] = edge
        self.graph.add_edge(edge.source_id, edge.target_id,
                            weight=edge.weight,
                            edge_type=edge.edge_type.value)

    def find_by_properties(self, properties: List[str]) -> List[KnowledgeNode]:
        """Находит узлы по свойствам."""
        if not properties:
            return list(self.nodes.values())

        # Находим пересечение узлов по свойствам
        result_ids = None
        for prop in properties:
            if prop in self.property_index:
                ids = set(self.property_index[prop])
                if result_ids is None:
                    result_ids = ids
                else:
                    result_ids = result_ids & ids

        if result_ids is None:
            return []

        return [self.nodes[nid] for nid in result_ids if nid in self.nodes]

    def find_path(self, source_id: str, target_id: str, max_depth: int = 5) -> List[List[str]]:
        """
        Находит пути между двумя узлами.
        """
        if source_id not in self.graph or target_id not in self.graph:
            return []

        try:
            paths = list(nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_depth))
            return paths
        except nx.NetworkXNoPath:
            return []

    def get_neighbors(self, node_id: str) -> List[KnowledgeNode]:
        """Возвращает соседние узлы."""
        if node_id not in self.graph:
            return []

        neighbors = []
        for neighbor_id in self.graph.neighbors(node_id):
            if neighbor_id in self.nodes:
                neighbors.append(self.nodes[neighbor_id])

        # Также добавляем предшественников
        for pred_id in self.graph.predecessors(node_id):
            if pred_id in self.nodes and pred_id not in [n.id for n in neighbors]:
                neighbors.append(self.nodes[pred_id])

        return neighbors

    def clear(self):
        """Очищает кэш."""
        self.graph.clear()
        self.nodes.clear()
        self.edges.clear()
        self.property_index.clear()
        self.type_index.clear()