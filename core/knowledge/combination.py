# core/knowledge/combination.py
"""
[ru] Модуль для работы с комбинациями узлов.
[en] Module for working with node combinations.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
import hashlib
import time


@dataclass
class Combination:
    """
    [ru] Комбинация узлов графа знаний. Используется для представления аналогий и гипотез.
    [en] Combination of knowledge graph nodes. Used to represent analogies and hypotheses.
    """

    id: str
    nodes: List[Any] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        """
        [ru] Вычисляет эмбеддинг после инициализации.
        [en] Computes the embedding after initialization.
        """
        if self.embedding is None:
            self._compute_embedding()

    def _compute_embedding(self):
        """
        [ru] Вычисляет эмбеддинг комбинации.
        [en] Computes the combination embedding.
        """
        # [ru] Простой эмбеддинг на основе свойств
        # [en] Simple embedding based on properties
        embedding = np.zeros(64)

        for i, prop in enumerate(self.properties[:20]):
            if i < len(embedding):
                embedding[i] = hash(prop) % 100 / 100.0

        # [ru] Добавляем информацию о количестве узлов
        # [en] Add information about the number of nodes
        if len(self.nodes) > 0:
            embedding[-1] = min(len(self.nodes) / 10, 1.0)

        # [ru] Нормализуем
        # [en] Normalize
        norm = np.linalg.norm(embedding) + 1e-8
        self.embedding = embedding / norm

    def copy(self) -> 'Combination':
        """
        [ru] Создаёт копию комбинации.
        [en] Creates a copy of the combination.
        """
        return Combination(
            id=f"copy_{self.id}",
            nodes=self.nodes.copy(),
            properties=self.properties.copy(),
            metadata=self.metadata.copy(),
            embedding=self.embedding.copy() if self.embedding is not None else None,
            created_at=self.created_at
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        [ru] Преобразует в словарь.
        [en] Converts to a dictionary.
        """
        return {
            'id': self.id,
            'node_ids': [n.id if hasattr(n, 'id') else str(n) for n in self.nodes],
            'properties': self.properties,
            'metadata': self.metadata,
            'created_at': self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Combination':
        """
        [ru] Восстанавливает из словаря.
        [en] Restores from a dictionary.
        """
        return cls(
            id=data['id'],
            # [ru] Узлы нужно восстановить отдельно
            # [en] Nodes need to be restored separately
            nodes=[],
            properties=data.get('properties', []),
            metadata=data.get('metadata', {})
        )



# # core/knowledge/combination.py
# """
# Модуль для работы с комбинациями узлов.
# """
#
# from typing import List, Dict, Any, Optional
# from dataclasses import dataclass, field
# import numpy as np
# import hashlib
# import time
#
#
# @dataclass
# class Combination:
#     """
#     Комбинация узлов графа знаний. Используется для представления аналогий и гипотез.
#     """
#
#     id: str
#     nodes: List[Any] = field(default_factory=list)
#     properties: List[str] = field(default_factory=list)
#     metadata: Dict[str, Any] = field(default_factory=dict)
#     embedding: Optional[np.ndarray] = None
#     created_at: float = field(default_factory=time.time)
#
#     def __post_init__(self):
#         """
#         Вычисляет эмбеддинг после инициализации.
#         """
#         if self.embedding is None:
#             self._compute_embedding()
#
#     def _compute_embedding(self):
#         """
#         Вычисляет эмбеддинг комбинации.
#         """
#         # Простой эмбеддинг на основе свойств
#         embedding = np.zeros(64)
#
#         for i, prop in enumerate(self.properties[:20]):
#             if i < len(embedding):
#                 embedding[i] = hash(prop) % 100 / 100.0
#
#         # Добавляем информацию о количестве узлов
#         if len(self.nodes) > 0:
#             embedding[-1] = min(len(self.nodes) / 10, 1.0)
#
#         # Нормализуем
#         norm = np.linalg.norm(embedding) + 1e-8
#         self.embedding = embedding / norm
#
#     def copy(self) -> 'Combination':
#         """
#         Создаёт копию комбинации.
#         """
#         return Combination(
#             id=f"copy_{self.id}",
#             nodes=self.nodes.copy(),
#             properties=self.properties.copy(),
#             metadata=self.metadata.copy(),
#             embedding=self.embedding.copy() if self.embedding is not None else None,
#             created_at=self.created_at
#         )
#
#     def to_dict(self) -> Dict[str, Any]:
#         """
#         Преобразует в словарь.
#         """
#         return {
#             'id': self.id,
#             'node_ids': [n.id if hasattr(n, 'id') else str(n) for n in self.nodes],
#             'properties': self.properties,
#             'metadata': self.metadata,
#             'created_at': self.created_at
#         }
#
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> 'Combination':
#         """
#         Восстанавливает из словаря.
#         """
#         return cls(
#             id=data['id'],
#             nodes=[],  # Узлы нужно восстановить отдельно
#             properties=data.get('properties', []),
#             metadata=data.get('metadata', {})
#         )
#
