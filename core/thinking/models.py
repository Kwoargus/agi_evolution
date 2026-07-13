# core/thinking/models.py
"""
Модели для системы мышления.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import uuid


@dataclass
class MentalModel:
    """
    Ментальная модель - визуально-образное представление знания.

    Attributes:
        id: Уникальный идентификатор модели (UUID)
        name: Название модели
        model_type: Тип модели
        sequence: Последовательность состояний (кадров)
        embedding: Векторное представление модели
        properties: Свойства модели (JSON)
        related_models: Связанные модели
        metadata: Дополнительные метаданные
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    model_type: str = "mental_model"
    sequence: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    related_models: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        """Создает эмбеддинг, если он не задан."""
        if self.embedding is None:
            self.embedding = self._generate_embedding()

    def _generate_embedding(self) -> List[float]:
        """Генерирует простой эмбеддинг для модели."""
        import random
        embedding = [0.0] * 64

        # Используем свойства для создания эмбеддинга
        if isinstance(self.properties, dict):
            for i, (key, value) in enumerate(self.properties.items()):
                if i < len(embedding):
                    embedding[i] = (hash(str(value)) % 100) / 100.0

        # Добавляем немного случайности
        for i in range(len(embedding)):
            if embedding[i] == 0:
                embedding[i] = random.random() * 0.1

        # Нормализуем
        max_val = max(embedding) if embedding else 1
        if max_val > 0:
            embedding = [v / max_val for v in embedding]

        return embedding

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует модель в словарь для сохранения."""
        return {
            'id': self.id,
            'name': self.name,
            'model_type': self.model_type,
            'sequence': self.sequence,
            'embedding': self.embedding,
            'properties': self.properties,
            'related_models': self.related_models,
            'metadata': self.metadata,
            'created_at': self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MentalModel':
        """Восстанавливает модель из словаря."""
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            name=data.get('name', ''),
            model_type=data.get('model_type', 'mental_model'),
            sequence=data.get('sequence', []),
            embedding=data.get('embedding'),
            properties=data.get('properties', {}),
            related_models=data.get('related_models', []),
            metadata=data.get('metadata', {}),
            created_at=data.get('created_at', time.time())
        )


@dataclass
class ModelTemplate:
    """
    Шаблон модели для создания новых ментальных моделей.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    required_properties: List[str] = field(default_factory=list)
    optional_properties: List[str] = field(default_factory=list)
    structure: str = ""
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует шаблон в словарь."""
        return {
            'id': self.id,
            'name': self.name,
            'required_properties': self.required_properties,
            'optional_properties': self.optional_properties,
            'structure': self.structure,
            'examples': self.examples,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelTemplate':
        """Восстанавливает шаблон из словаря."""
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            name=data.get('name', ''),
            required_properties=data.get('required_properties', []),
            optional_properties=data.get('optional_properties', []),
            structure=data.get('structure', ''),
            examples=data.get('examples', []),
            metadata=data.get('metadata', {})
        )

