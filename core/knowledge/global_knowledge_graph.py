# core/knowledge/global_knowledge_graph.py
"""
Глобальный граф знаний (ГГЗ).
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
from .knowledge_node import KnowledgeNode
from .knowledge_edge import KnowledgeEdge, EdgeType


class GlobalKnowledgeGraph:
    """
    Глобальный граф знаний - хранит все известные модели.
    """

    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[str, KnowledgeEdge] = {}

        # Индексы для быстрого поиска
        self.property_index: Dict[str, List[str]] = {}
        self.type_index: Dict[str, List[str]] = {}
        self.name_index: Dict[str, str] = {}

    def add_node(self, node: KnowledgeNode):
        """Добавляет узел в граф."""
        self.nodes[node.id] = node
        self.name_index[node.name.lower()] = node.id

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
        """Добавляет ребро в граф."""
        self.edges[edge.id] = edge

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Возвращает узел по ID."""
        return self.nodes.get(node_id)

    def get_node_by_name(self, name: str) -> Optional[KnowledgeNode]:
        """Возвращает узел по имени."""
        node_id = self.name_index.get(name.lower())
        if node_id:
            return self.nodes.get(node_id)
        return None

    def get_neighbors(self, node_id: str) -> List[KnowledgeNode]:
        """Возвращает соседние узлы."""
        neighbors = []
        for edge in self.edges.values():
            if edge.source_id == node_id:
                target = self.nodes.get(edge.target_id)
                if target:
                    neighbors.append(target)
            elif edge.target_id == node_id:
                source = self.nodes.get(edge.source_id)
                if source:
                    neighbors.append(source)
        return neighbors

    def find_by_properties(self, properties: List[str]) -> List[KnowledgeNode]:
        """Находит узлы по свойствам."""
        if not properties:
            return list(self.nodes.values())

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

    def find_by_type(self, node_type: str) -> List[KnowledgeNode]:
        """Находит узлы по типу."""
        ids = self.type_index.get(node_type, [])
        return [self.nodes[nid] for nid in ids if nid in self.nodes]

    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику графа."""
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': {t: len(ids) for t, ids in self.type_index.items()},
            'property_count': len(self.property_index),
        }

    def load_from_db(self):
        """Загружает данные из БД."""
        try:
            from db.knowledge_db import KnowledgeDB
            db = KnowledgeDB()
            nodes = db.load_all_nodes()
            for node in nodes:
                self.add_node(node)
            print(f"✅ Загружено {len(nodes)} узлов из БД")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки из БД: {e}")

    def save_to_db(self):
        """Сохраняет данные в БД."""
        try:
            from db.knowledge_db import KnowledgeDB
            db = KnowledgeDB()
            for node in self.nodes.values():
                db.save_node(node)
            for edge in self.edges.values():
                db.save_edge(edge)
            print(f"✅ Сохранено {len(self.nodes)} узлов и {len(self.edges)} связей в БД")
        except Exception as e:
            print(f"⚠️ Ошибка сохранения в БД: {e}")



