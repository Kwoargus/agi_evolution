# core/knowledge/individual_knowledge_graph.py
"""
Индивидуальный граф знаний бота.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import time
import uuid


@dataclass
class IndividualKnowledgeGraph:
    """
    Индивидуальный граф знаний - хранит модели, созданные ботом.
    """

    # Существующие поля
    knowledge: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    # НОВЫЕ ПОЛЯ
    bot_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    mental_models: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # id → модель
    experiences: List[Dict[str, Any]] = field(default_factory=list)
    last_synced: float = field(default_factory=time.time)

    # ============================================================
    # СУЩЕСТВУЮЩИЕ МЕТОДЫ (оставляем как есть)
    # ============================================================

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
            'total_mental_models': len(self.mental_models),
            'total_experiences': len(self.experiences),
            'types': {k.get('type', 'unknown'): len([x for x in self.knowledge if x.get('type') == k.get('type')])
                      for k in self.knowledge}
        }

    # ============================================================
    # НОВЫЕ МЕТОДЫ ДЛЯ МЕНТАЛЬНЫХ МОДЕЛЕЙ
    # ============================================================

    def add_mental_model(self, model_id: str, model_data: Dict[str, Any]) -> None:
        """
        Добавляет ментальную модель в ИГЗ.

        Args:
            model_id: ID модели
            model_data: Данные модели (name, properties, nodes, task, и т.д.)
        """
        self.mental_models[model_id] = {
            'data': model_data,
            'created_at': time.time(),
            'usage_count': 0
        }

    def get_mental_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Возвращает ментальную модель по ID и увеличивает счётчик использования."""
        model = self.mental_models.get(model_id)
        if model:
            model['usage_count'] += 1
        return model

    def get_all_mental_models(self) -> List[Dict[str, Any]]:
        """Возвращает все ментальные модели."""
        return list(self.mental_models.values())

    def find_mental_models_by_properties(self, properties: List[str]) -> List[Dict[str, Any]]:
        """
        Находит ментальные модели по свойствам.

        Args:
            properties: Список свойств для поиска

        Returns:
            Список моделей с указанием количества совпадений
        """
        results = []

        for model_id, model_entry in self.mental_models.items():
            model_data = model_entry.get('data', {})
            model_props = model_data.get('properties', [])

            # Находим пересечение свойств
            matches = [p for p in properties if p in model_props]

            if matches:
                results.append({
                    'model_id': model_id,
                    'model_data': model_data,
                    'matches': matches,
                    'match_count': len(matches),
                    'usage_count': model_entry.get('usage_count', 0)
                })

        # Сортируем по количеству совпадений и популярности
        results.sort(key=lambda x: (x['match_count'], x['usage_count']), reverse=True)

        return results

    # ============================================================
    # НОВЫЕ МЕТОДЫ ДЛЯ ОПЫТА
    # ============================================================

    def add_experience(self, experience: Dict[str, Any]) -> None:
        """Добавляет опыт взаимодействия."""
        experience['timestamp'] = time.time()
        self.experiences.append(experience)

        # Ограничиваем размер
        if len(self.experiences) > 1000:
            self.experiences = self.experiences[-1000:]

    def get_recent_experiences(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Возвращает последние опыты."""
        sorted_exp = sorted(self.experiences, key=lambda x: x.get('timestamp', 0), reverse=True)
        return sorted_exp[:limit]

    # ============================================================
    # НОВЫЕ МЕТОДЫ ДЛЯ СИНХРОНИЗАЦИИ
    # ============================================================

    def sync_with_global(self, global_graph, node_ids: List[str]) -> Dict[str, bool]:
        """
        Синхронизирует узлы с глобальным графом знаний.

        Args:
            global_graph: Глобальный граф знаний
            node_ids: Список ID узлов для синхронизации

        Returns:
            Словарь {node_id: success}
        """
        results = {}

        for node_id in node_ids:
            try:
                # Получаем узел из глобального графа
                node = global_graph.get_node(node_id)
                if not node:
                    results[node_id] = False
                    continue

                # Сохраняем в ИГЗ
                self.knowledge.append({
                    'id': node_id,
                    'type': 'synced_node',
                    'name': node.name,
                    'node_type': node.node_type,
                    'properties': node.properties,
                    'synced_at': time.time()
                })

                results[node_id] = True
            except Exception as e:
                results[node_id] = False
                print(f"⚠️ Ошибка синхронизации узла {node_id}: {e}")

        self.last_synced = time.time()
        return results

    def sync_mental_model_to_global(self, global_graph, model_id: str) -> bool:
        """
        Синхронизирует ментальную модель с глобальным графом знаний.

        Args:
            global_graph: Глобальный граф знаний
            model_id: ID ментальной модели

        Returns:
            True если синхронизация успешна
        """
        model_entry = self.mental_models.get(model_id)
        if not model_entry:
            return False

        model_data = model_entry.get('data', {})

        try:
            from core.knowledge.knowledge_node import KnowledgeNode

            # Создаём узел в глобальном графе
            node = KnowledgeNode(
                id=model_id,
                name=model_data.get('name', f"mental_model_{model_id}"),
                node_type="mental_model",
                properties=model_data.get('properties', []),
                description=f"Ментальная модель из ИГЗ: {model_data.get('task', '')[:100]}"
            )

            # Добавляем в глобальный граф
            global_graph.add_node(node)

            return True
        except Exception as e:
            print(f"⚠️ Ошибка синхронизации ментальной модели {model_id}: {e}")
            return False

    # ============================================================
    # НОВЫЕ МЕТОДЫ ДЛЯ СОХРАНЕНИЯ/ЗАГРУЗКИ
    # ============================================================

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь для сохранения."""
        return {
            'bot_id': self.bot_id,
            'knowledge': self.knowledge,
            'mental_models': self.mental_models,
            'experiences': self.experiences[-100:],  # Только последние 100
            'created_at': self.created_at,
            'last_synced': self.last_synced
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndividualKnowledgeGraph':
        """Восстанавливает из словаря."""
        graph = cls(bot_id=data.get('bot_id', str(uuid.uuid4())[:8]))
        graph.knowledge = data.get('knowledge', [])
        graph.mental_models = data.get('mental_models', {})
        graph.experiences = data.get('experiences', [])
        graph.created_at = data.get('created_at', time.time())
        graph.last_synced = data.get('last_synced', time.time())
        return graph


# @dataclass
# class IndividualKnowledgeGraph:
#     """
#     Индивидуальный граф знаний - хранит модели, созданные ботом.
#     """
#
#     knowledge: List[Dict[str, Any]] = field(default_factory=list)
#     created_at: float = field(default_factory=time.time)
#
#     def add_knowledge(self, record: Dict[str, Any]):
#         """Добавляет запись в ИГЗ."""
#         record['added_at'] = time.time()
#         self.knowledge.append(record)
#
#     def get_by_type(self, knowledge_type: str) -> List[Dict[str, Any]]:
#         """Возвращает записи по типу."""
#         return [k for k in self.knowledge if k.get('type') == knowledge_type]
#
#     def get_by_id(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
#         """Возвращает запись по ID."""
#         for k in self.knowledge:
#             if k.get('id') == knowledge_id:
#                 return k
#         return None
#
#     def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
#         """Возвращает последние записи."""
#         sorted_knowledge = sorted(self.knowledge, key=lambda x: x.get('added_at', 0), reverse=True)
#         return sorted_knowledge[:limit]
#
#     def get_statistics(self) -> Dict[str, Any]:
#         """Возвращает статистику ИГЗ."""
#         return {
#             'total_knowledge': len(self.knowledge),
#             'types': {k.get('type', 'unknown'): len([x for x in self.knowledge if x.get('type') == k.get('type')])
#                       for k in self.knowledge}
#         }

