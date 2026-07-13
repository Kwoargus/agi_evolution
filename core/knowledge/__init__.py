# core/knowledge/__init__.py
"""
Модуль для работы с графом знаний и мышлением.
"""

from .knowledge_node import KnowledgeNode
from .knowledge_edge import KnowledgeEdge, EdgeType
from .global_knowledge_graph import GlobalKnowledgeGraph
from .individual_knowledge_graph import IndividualKnowledgeGraph
from .mental_model import MentalModel
from .function import Function

__all__ = [
    'KnowledgeNode',
    'KnowledgeEdge',
    'EdgeType',
    'GlobalKnowledgeGraph',
    'IndividualKnowledgeGraph',
    'MentalModel',
    'Function',
]
