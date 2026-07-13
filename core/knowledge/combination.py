# core/knowledge/combination.py
"""
Модуль для работы с комбинациями моделей знаний.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from .knowledge_node import KnowledgeNode


@dataclass
class Combination:
    """
    Комбинация узлов графа знаний.
    Используется как кандидат на роль "аналогии" или "прототипа".

    Attributes:
        id: Уникальный идентификатор комбинации
        nodes: Список узлов, входящих в комбинацию
        edges: Список связей между узлами (по ID)
        properties: Объединенные свойства всех узлов
        embedding: Векторное представление комбинации
        score: Оценка соответствия задаче (0-1)
        metadata: Дополнительные метаданные
    """

    id: str
    nodes: List[KnowledgeNode] = field(default_factory=list)
    edge_ids: List[str] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Вычисляет объединенные свойства после инициализации."""
        if not self.properties:
            self._update_properties()
        if self.embedding is None:
            self._compute_embedding()

    def _update_properties(self):
        """Обновляет список свойств на основе узлов."""
        all_props = set()
        for node in self.nodes:
            all_props.update(node.properties)
        self.properties = list(all_props)

    def _compute_embedding(self):
        """Вычисляет эмбеддинг комбинации."""
        if not self.nodes:
            self.embedding = np.zeros(128)
            return

        # Усредняем эмбеддинги узлов
        embeddings = [n.embedding for n in self.nodes if n.embedding is not None]
        if embeddings:
            self.embedding = np.mean(embeddings, axis=0)
        else:
            self.embedding = np.zeros(128)

        # Нормализуем
        norm = np.linalg.norm(self.embedding) + 1e-8
        self.embedding = self.embedding / norm

    def add_node(self, node: KnowledgeNode):
        """Добавляет узел в комбинацию."""
        self.nodes.append(node)
        self._update_properties()
        self._compute_embedding()

    def add_edge(self, edge_id: str):
        """Добавляет связь в комбинацию."""
        if edge_id not in self.edge_ids:
            self.edge_ids.append(edge_id)

    def get_property_coverage(self, required_properties: List[str]) -> float:
        """
        Вычисляет покрытие требуемых свойств.

        Returns:
            Доля требуемых свойств, присутствующих в комбинации
        """
        if not required_properties:
            return 1.0

        covered = sum(1 for p in required_properties if p in self.properties)
        return covered / len(required_properties)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь для сериализации."""
        return {
            'id': self.id,
            'node_ids': [n.id for n in self.nodes],
            'edge_ids': self.edge_ids,
            'properties': self.properties,
            'score': self.score,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], node_map: Dict[str, KnowledgeNode]) -> 'Combination':
        """Восстанавливает комбинацию из словаря."""
        nodes = [node_map[nid] for nid in data.get('node_ids', []) if nid in node_map]
        return cls(
            id=data['id'],
            nodes=nodes,
            edge_ids=data.get('edge_ids', []),
            properties=data.get('properties', []),
            score=data.get('score', 0.0),
            metadata=data.get('metadata', {})
        )