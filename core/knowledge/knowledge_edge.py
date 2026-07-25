# core/knowledge/knowledge_edge.py
"""
[ru] Модуль для работы с ребрами графа знаний.
[en] Module for working with knowledge graph edges.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class EdgeType(Enum):
    """
    [ru] Типы связей между узлами графа знаний.
    [en] Types of connections between knowledge graph nodes.
    """

    # ============================================================
    # [ru] ИЕРАРХИЧЕСКИЕ
    # [en] HIERARCHICAL
    # ============================================================
    INSTANCE_OF = "instance_of"  # [ru] Является экземпляром (Биплан → Самолет)  [en] Is an instance of (Biplane → Airplane)
    PART_OF = "part_of"  # [ru] Часть целого (Крыло → Самолет)  [en] Part of a whole (Wing → Airplane)
    HAS_PART = "has_part"  # [ru] Имеет часть (Самолет → Крыло)  [en] Has a part (Airplane → Wing)
    CONTAINS = "contains"  # [ru] Содержит (Самолет → Двигатель)  [en] Contains (Airplane → Engine)
    IS_A = "is_a"  # [ru] Является (Пружина → Амортизатор)  [en] Is a (Spring → Shock absorber)

    # ============================================================
    # [ru] ПРИЧИННО-СЛЕДСТВЕННЫЕ
    # [en] CAUSAL
    # ============================================================
    CAUSES = "causes"  # [ru] Вызывает (Крыло → Подъемная сила)  [en] Causes (Wing → Lift)
    CAUSED_BY = "caused_by"  # [ru] Вызвано (Подъемная сила → Крыло)  [en] Caused by (Lift → Wing)
    DEPENDS_ON = "depends_on"  # [ru] Зависит от (Самолет → Подъемная сила)  [en] Depends on (Airplane → Lift)

    # ============================================================
    # [ru] ФУНКЦИОНАЛЬНЫЕ
    # [en] FUNCTIONAL
    # ============================================================
    USES = "uses"  # [ru] Использует (Змей → Ветер)  [en] Uses (Kite → Wind)
    USED_FOR = "used_for"  # [ru] Используется для (Крыло → Полёт)  [en] Is used for (Wing → Flight)
    USED_IN = "used_in"  # [ru] Используется в (Алюминий → Самолет)  [en] Is used in (Aluminum → Airplane)
    CAPABLE_OF = "capable_of"  # [ru] Способен (Крыло → Создавать подъемную силу)  [en] Is capable of (Wing → Creating lift)
    CAN_HAVE = "can_have"  # [ru] Может иметь (Змей → Резиномотор)  [en] Can have (Kite → Rubber motor)
    STORES = "stores"  # [ru] Накапливает (Резиномотор → Энергия)  [en] Accumulates (Rubber motor → Energy)
    CONNECTED_TO = "connected_to"  # [ru] Соединен с (Двигатель → Пропеллер)  [en] Connected to (Engine → Propeller)
    SUPPORTS = "supports"  # [ru] Поддерживает (Подшипник → Вал)  [en] Supports (Bearing → Shaft)

    # ============================================================
    # [ru] СЕМАНТИЧЕСКИЕ
    # [en] SEMANTIC
    # ============================================================
    RELATED_TO = "related_to"  # [ru] Связан с (Аэродинамика → Крыло)  [en] Related to (Aerodynamics → Wing)
    SIMILAR_TO = "similar_to"  # [ru] Похож на (Крыло чайки → Крыло самолета)  [en] Similar to (Seagull wing → Airplane wing)
    OPPOSITE_TO = "opposite_to"  # [ru] Противоположен (Паровой → Резиномотор)  [en] Opposite to (Steam → Rubber motor)

    # ============================================================
    # [ru] ПРОСТРАНСТВЕННЫЕ
    # [en] SPATIAL
    # ============================================================
    AT_LOCATION = "at_location"  # [ru] Находится в  [en] Located in/at
    NEAR_TO = "near_to"  # [ru] Рядом с  [en] Near


@dataclass
class KnowledgeEdge:
    """
    [ru] Ребро графа знаний - связь между двумя узлами.
    [en] Knowledge graph edge - a connection between two nodes.
    """

    id: str
    source_id: str
    target_id: str
    edge_type: EdgeType = EdgeType.RELATED_TO
    weight: float = 0.5
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        [ru] Преобразует в словарь.
        [en] Converts to a dictionary.
        """
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
        """
        [ru] Восстанавливает связь из словаря.
        [en] Restores the edge from a dictionary.
        """
        return cls(
            id=data['id'],
            source_id=data['source_id'],
            target_id=data['target_id'],
            edge_type=EdgeType(data['edge_type']),
            weight=data.get('weight', 0.5),
            description=data.get('description', ''),
            metadata=data.get('metadata', {})
        )



# # core/knowledge/knowledge_edge.py
# """
# Модуль для работы с ребрами графа знаний.
# """
#
# from typing import Dict, Any, Optional
# from dataclasses import dataclass, field
# from enum import Enum
#
#
# class EdgeType(Enum):
#     """
#     Типы связей между узлами графа знаний.
#     """
#
#     # ============================================================
#     # ИЕРАРХИЧЕСКИЕ
#     # ============================================================
#     INSTANCE_OF = "instance_of"  # Является экземпляром (Биплан → Самолет)
#     PART_OF = "part_of"  # Часть целого (Крыло → Самолет)
#     HAS_PART = "has_part"  # Имеет часть (Самолет → Крыло)
#     CONTAINS = "contains"  # Содержит (Самолет → Двигатель)
#     IS_A = "is_a"  # Является (Пружина → Амортизатор)
#
#     # ============================================================
#     # ПРИЧИННО-СЛЕДСТВЕННЫЕ
#     # ============================================================
#     CAUSES = "causes"  # Вызывает (Крыло → Подъемная сила)
#     CAUSED_BY = "caused_by"  # Вызвано (Подъемная сила → Крыло)
#     DEPENDS_ON = "depends_on"  # Зависит от (Самолет → Подъемная сила)
#
#     # ============================================================
#     # ФУНКЦИОНАЛЬНЫЕ
#     # ============================================================
#     USES = "uses"  # Использует (Змей → Ветер)
#     USED_FOR = "used_for"  # Используется для (Крыло → Полёт)
#     USED_IN = "used_in"  # Используется в (Алюминий → Самолет)
#     CAPABLE_OF = "capable_of"  # Способен (Крыло → Создавать подъемную силу)
#     CAN_HAVE = "can_have"  # Может иметь (Змей → Резиномотор)
#     STORES = "stores"  # Накапливает (Резиномотор → Энергия)
#     CONNECTED_TO = "connected_to"  # Соединен с (Двигатель → Пропеллер)
#     SUPPORTS = "supports"  # Поддерживает (Подшипник → Вал)
#
#     # ============================================================
#     # СЕМАНТИЧЕСКИЕ
#     # ============================================================
#     RELATED_TO = "related_to"  # Связан с (Аэродинамика → Крыло)
#     SIMILAR_TO = "similar_to"  # Похож на (Крыло чайки → Крыло самолета)
#     OPPOSITE_TO = "opposite_to"  # Противоположен (Паровой → Резиномотор)
#
#     # ============================================================
#     # ПРОСТРАНСТВЕННЫЕ
#     # ============================================================
#     AT_LOCATION = "at_location"  # Находится в
#     NEAR_TO = "near_to"  # Рядом с
#
#
# @dataclass
# class KnowledgeEdge:
#     """
#     Ребро графа знаний - связь между двумя узлами.
#     """
#
#     id: str
#     source_id: str
#     target_id: str
#     edge_type: EdgeType = EdgeType.RELATED_TO
#     weight: float = 0.5
#     description: str = ""
#     metadata: Dict[str, Any] = field(default_factory=dict)
#
#     def to_dict(self) -> Dict[str, Any]:
#         """Преобразует в словарь."""
#         return {
#             'id': self.id,
#             'source_id': self.source_id,
#             'target_id': self.target_id,
#             'edge_type': self.edge_type.value,
#             'weight': self.weight,
#             'description': self.description,
#             'metadata': self.metadata
#         }
#
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEdge':
#         """Восстанавливает связь из словаря."""
#         return cls(
#             id=data['id'],
#             source_id=data['source_id'],
#             target_id=data['target_id'],
#             edge_type=EdgeType(data['edge_type']),
#             weight=data.get('weight', 0.5),
#             description=data.get('description', ''),
#             metadata=data.get('metadata', {})
#         )