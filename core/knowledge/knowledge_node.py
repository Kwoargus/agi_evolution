# core/knowledge/knowledge_node.py
"""
Модуль для работы с узлами графа знаний.
"""

import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from .function import Function


class KnowledgeNode:
    """
    Узел графа знаний - модель знания.
    """

    def __init__(self,
                 id: str,
                 name: str,
                 node_type: str,  # 'object', 'process', 'system', 'theory', 'concept'
                 properties: List[str],
                 description: str = "",
                 embedding: np.ndarray = None):
        self.id = id
        self.name = name
        self.node_type = node_type
        self.properties = properties
        self.description = description
        self.embedding = embedding if embedding is not None else np.random.randn(128)

        # Нормализуем эмбеддинг
        norm = np.linalg.norm(self.embedding) + 1e-8
        self.embedding = self.embedding / norm

        # Связи с другими узлами (по ID)
        self.connections: List[str] = []

        # Функции модели
        self.functions: List[Function] = []

        # Параметры
        self.parameters: Dict[str, float] = {}

        # Метаданные
        self.metadata: Dict[str, Any] = {}

    def has_property(self, property_name: str) -> bool:
        """Проверяет наличие свойства."""
        return property_name in self.properties

    def add_connection(self, node_id: str):
        """Добавляет связь с другим узлом."""
        if node_id not in self.connections:
            self.connections.append(node_id)

    def add_function(self, function: Function):
        """Добавляет функцию."""
        self.functions.append(function)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь для сериализации."""
        return {
            'id': self.id,
            'name': self.name,
            'node_type': self.node_type,
            'properties': self.properties,
            'description': self.description,
            'connections': self.connections,
            'functions': [f.to_dict() for f in self.functions],
            'parameters': self.parameters,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeNode':
        """Восстанавливает узел из словаря."""
        node = cls(
            id=data['id'],
            name=data['name'],
            node_type=data['node_type'],
            properties=data['properties'],
            description=data.get('description', '')
        )
        node.connections = data.get('connections', [])
        node.parameters = data.get('parameters', {})
        node.metadata = data.get('metadata', {})
        return node