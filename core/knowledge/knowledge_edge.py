# core/knowledge/knowledge_edge.py
"""
Модуль для работы с ребрами графа знаний.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class EdgeType(Enum):
    """Типы связей между узлами графа знаний."""

    # ============================================================
    # ИЕРАРХИЧЕСКИЕ
    # ============================================================
    INSTANCE_OF = "instance_of"  # Является экземпляром (Биплан → Самолет)
    PART_OF = "part_of"  # Часть целого (Крыло → Самолет)
    HAS_PART = "has_part"  # Имеет часть (Самолет → Крыло)
    CONTAINS = "contains"  # Содержит (Самолет → Двигатель)
    IS_A = "is_a"  # Является (Пружина → Амортизатор)

    # ============================================================
    # ПРИЧИННО-СЛЕДСТВЕННЫЕ
    # ============================================================
    CAUSES = "causes"  # Вызывает (Крыло → Подъемная сила)
    CAUSED_BY = "caused_by"  # Вызвано (Подъемная сила → Крыло)
    DEPENDS_ON = "depends_on"  # Зависит от (Самолет → Подъемная сила)

    # ============================================================
    # ФУНКЦИОНАЛЬНЫЕ
    # ============================================================
    USES = "uses"  # Использует (Змей → Ветер)
    USED_FOR = "used_for"  # Используется для (Крыло → Полёт)
    USED_IN = "used_in"  # Используется в (Алюминий → Самолет)
    CAPABLE_OF = "capable_of"  # Способен (Крыло → Создавать подъемную силу)
    CAN_HAVE = "can_have"  # Может иметь (Змей → Резиномотор)
    STORES = "stores"  # Накапливает (Резиномотор → Энергия)
    CONNECTED_TO = "connected_to"  # Соединен с (Двигатель → Пропеллер)
    SUPPORTS = "supports"  # Поддерживает (Подшипник → Вал)

    # ============================================================
    # СЕМАНТИЧЕСКИЕ
    # ============================================================
    RELATED_TO = "related_to"  # Связан с (Аэродинамика → Крыло)
    SIMILAR_TO = "similar_to"  # Похож на (Крыло чайки → Крыло самолета)
    OPPOSITE_TO = "opposite_to"  # Противоположен (Паровой → Резиномотор)

    # ============================================================
    # ПРОСТРАНСТВЕННЫЕ
    # ============================================================
    AT_LOCATION = "at_location"  # Находится в
    NEAR_TO = "near_to"  # Рядом с


@dataclass
class KnowledgeEdge:
    """
    Ребро графа знаний - связь между двумя узлами.
    """

    id: str
    source_id: str
    target_id: str
    edge_type: EdgeType = EdgeType.RELATED_TO
    weight: float = 0.5
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь."""
        return {
            'id': self.id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'edge_type': self.edge_type.value,
            'weight': self.weight,
            'description': self.description,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEdge':
        """Восстанавливает связь из словаря."""
        return cls(
            id=data['id'],
            source_id=data['source_id'],
            target_id=data['target_id'],
            edge_type=EdgeType(data['edge_type']),
            weight=data.get('weight', 0.5),
            description=data.get('description', ''),
            metadata=data.get('metadata', {})
        )