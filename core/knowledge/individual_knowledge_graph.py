# core/knowledge/individual_knowledge_graph.py
"""
Индивидуальный граф знаний бота.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import time


@dataclass
class IndividualKnowledgeGraph:
    """
    Индивидуальный граф знаний - хранит модели, созданные ботом.
    """

    knowledge: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def add_knowledge(self, record: Dict[str, Any]):
        """Добавляет запись в ИГЗ."""
        record['added_at'] = time.time()
        self.knowledge.append(record)

    def get_by_type(self, knowledge_type: str) -> List[Dict[str, Any]]:
        """Возвращает записи по типу."""
        return [k for k in self.knowledge if k.get('type') == knowledge_type]

    def get_by_id(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """Возвращает запись по ID."""
        for k in self.knowledge:
            if k.get('id') == knowledge_id:
                return k
        return None

    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Возвращает последние записи."""
        sorted_knowledge = sorted(self.knowledge, key=lambda x: x.get('added_at', 0), reverse=True)
        return sorted_knowledge[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику ИГЗ."""
        return {
            'total_knowledge': len(self.knowledge),
            'types': {k.get('type', 'unknown'): len([x for x in self.knowledge if x.get('type') == k.get('type')])
                      for k in self.knowledge}
        }

