# core/knowledge/individual_knowledge_graph.py
"""
[ru] Индивидуальный граф знаний бота.
[en] Bot's individual knowledge graph.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import time
import uuid


@dataclass
class IndividualKnowledgeGraph:
    """
    [ru] Индивидуальный граф знаний - хранит модели, созданные ботом.
    [en] Individual knowledge graph - stores models created by the bot.
    """

    # [ru] Существующие поля
    # [en] Existing fields
    knowledge: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    # [ru] НОВЫЕ ПОЛЯ
    # [en] NEW FIELDS
    bot_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    mental_models: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # [ru] id → модель  [en] id → model
    experiences: List[Dict[str, Any]] = field(default_factory=list)
    last_synced: float = field(default_factory=time.time)

    # ============================================================
    # [ru] СУЩЕСТВУЮЩИЕ МЕТОДЫ (оставляем как есть)
    # [en] EXISTING METHODS (leave as is)
    # ============================================================

    def add_knowledge(self, record: Dict[str, Any]):
        """
        [ru] Добавляет запись в ИГЗ.
        [en] Adds a record to the Individual Knowledge Graph.
        """
        record['added_at'] = time.time()
        self.knowledge.append(record)

    def get_by_type(self, knowledge_type: str) -> List[Dict[str, Any]]:
        """
        [ru] Возвращает записи по типу.
        [en] Returns records by type.
        """
        return [k for k in self.knowledge if k.get('type') == knowledge_type]

    def get_by_id(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """
        [ru] Возвращает запись по ID.
        [en] Returns a record by ID.
        """
        for k in self.knowledge:
            if k.get('id') == knowledge_id:
                return k
        return None

    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        [ru] Возвращает последние записи.
        [en] Returns the most recent records.
        """
        sorted_knowledge = sorted(self.knowledge, key=lambda x: x.get('added_at', 0), reverse=True)
        return sorted_knowledge[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """
        [ru] Возвращает статистику ИГЗ.
        [en] Returns Individual Knowledge Graph statistics.
        """
        return {
            'total_knowledge': len(self.knowledge),
            'total_mental_models': len(self.mental_models),
            'total_experiences': len(self.experiences),
            'types': {k.get('type', 'unknown'): len([x for x in self.knowledge if x.get('type') == k.get('type')])
                      for k in self.knowledge}
        }

    # ============================================================
    # [ru] НОВЫЕ МЕТОДЫ ДЛЯ МЕНТАЛЬНЫХ МОДЕЛЕЙ
    # [en] NEW METHODS FOR MENTAL MODELS
    # ============================================================

    def add_mental_model(self, model_id: str, model_data: Dict[str, Any]) -> None:
        """
        [ru] Добавляет ментальную модель в ИГЗ.
        Args:
            [ru] model_id: ID модели
            [ru] model_data: Данные модели (name, properties, nodes, task, и т.д.)


        [en] Adds a mental model to the Individual Knowledge Graph.
        Args:
            [en] model_id: Model ID
            [en] model_data: Model data (name, properties, nodes, task, etc.)
        """
        self.mental_models[model_id] = {
            'data': model_data,
            'created_at': time.time(),
            'usage_count': 0
        }

    def get_mental_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        [ru] Возвращает ментальную модель по ID и увеличивает счётчик использования.
        [en] Returns a mental model by ID and increments the usage counter.
        """
        model = self.mental_models.get(model_id)
        if model:
            model['usage_count'] += 1
        return model

    def get_all_mental_models(self) -> List[Dict[str, Any]]:
        """
        [ru] Возвращает все ментальные модели.
        [en] Returns all mental models.
        """
        return list(self.mental_models.values())

    def find_mental_models_by_properties(self, properties: List[str]) -> List[Dict[str, Any]]:
        """
        [ru] Находит ментальные модели по свойствам.
        Args:
            [ru] properties: Список свойств для поиска
        Returns:
            [ru] Список моделей с указанием количества совпадений

        [en] Finds mental models by properties.
        Args:
            [en] properties: List of properties to search for
        Returns:
            [en] List of models with the number of matches indicated
        """
        results = []

        for model_id, model_entry in self.mental_models.items():
            model_data = model_entry.get('data', {})
            model_props = model_data.get('properties', [])

            # [ru] Находим пересечение свойств
            # [en] Find the intersection of properties
            matches = [p for p in properties if p in model_props]

            if matches:
                results.append({
                    'model_id': model_id,
                    'model_data': model_data,
                    'matches': matches,
                    'match_count': len(matches),
                    'usage_count': model_entry.get('usage_count', 0)
                })

        # [ru] Сортируем по количеству совпадений и популярности
        # [en] Sort by number of matches and popularity
        results.sort(key=lambda x: (x['match_count'], x['usage_count']), reverse=True)

        return results

    # ============================================================
    # [ru] НОВЫЕ МЕТОДЫ ДЛЯ ОПЫТА
    # [en] NEW METHODS FOR EXPERIENCES
    # ============================================================

    def add_experience(self, experience: Dict[str, Any]) -> None:
        """
        [ru] Добавляет опыт взаимодействия.
        [en] Adds an interaction experience.
        """
        experience['timestamp'] = time.time()
        self.experiences.append(experience)

        # [ru] Ограничиваем размер
        # [en] Limit the size
        if len(self.experiences) > 1000:
            self.experiences = self.experiences[-1000:]

    def get_recent_experiences(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        [ru] Возвращает последние опыты.
        [en] Returns the most recent experiences.
        """
        sorted_exp = sorted(self.experiences, key=lambda x: x.get('timestamp', 0), reverse=True)
        return sorted_exp[:limit]

    # ============================================================
    # [ru] НОВЫЕ МЕТОДЫ ДЛЯ СИНХРОНИЗАЦИИ
    # [en] NEW METHODS FOR SYNCHRONIZATION
    # ============================================================

    def sync_with_global(self, global_graph, node_ids: List[str]) -> Dict[str, bool]:
        """
        [ru] Синхронизирует узлы с глобальным графом знаний.
        Args:
            [ru] global_graph: Глобальный граф знаний
            [ru] node_ids: Список ID узлов для синхронизации
        Returns:
            [ru] Словарь {node_id: success}

        [en] Synchronizes nodes with the global knowledge graph.
        Args:
            [en] global_graph: Global knowledge graph
            [en] node_ids: List of node IDs to synchronize
        Returns:
            [en] Dictionary {node_id: success}
        """
        results = {}

        for node_id in node_ids:
            try:
                # [ru] Получаем узел из глобального графа
                # [en] Get the node from the global graph
                node = global_graph.get_node(node_id)
                if not node:
                    results[node_id] = False
                    continue

                # [ru] Сохраняем в ИГЗ
                # [en] Save to the Individual Knowledge Graph
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
                print(f"[ru] Ошибка синхронизации узла {node_id}: {e}")
                print(f"[en] Node synchronization error {node_id}: {e}")

        self.last_synced = time.time()
        return results

    def sync_mental_model_to_global(self, global_graph, model_id: str) -> bool:
        """
        [ru] Синхронизирует ментальную модель с глобальным графом знаний.
        Args:
            [ru] global_graph: Глобальный граф знаний
            [ru] model_id: ID ментальной модели
        Returns:
            [ru] True если синхронизация успешна

        [en] Synchronizes a mental model with the global knowledge graph.
        Args:
            [en] global_graph: Global knowledge graph
            [en] model_id: Mental model ID
        Returns:
            [en] True if synchronization is successful
        """
        model_entry = self.mental_models.get(model_id)
        if not model_entry:
            return False

        model_data = model_entry.get('data', {})

        try:
            from core.knowledge.knowledge_node import KnowledgeNode

            # [ru] Создаём узел в глобальном графе
            # [en] Create a node in the global graph
            node = KnowledgeNode(
                id=model_id,
                name=model_data.get('name', f"mental_model_{model_id}"),
                node_type="mental_model",
                properties=model_data.get('properties', []),
                description=f"Ментальная модель из ИГЗ: {model_data.get('task', '')[:100]}"
            )

            # [ru] Добавляем в глобальный граф
            # [en] Add to the global graph
            global_graph.add_node(node)

            return True
        except Exception as e:
            print(f"⚠️ Ошибка синхронизации ментальной модели {model_id}: {e}")
            return False

    # ============================================================
    # [ru] НОВЫЕ МЕТОДЫ ДЛЯ СОХРАНЕНИЯ/ЗАГРУЗКИ
    # [en] NEW METHODS FOR SAVE/LOAD
    # ============================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        [ru] Преобразует в словарь для сохранения.
        [en] Converts to a dictionary for saving.
        """
        return {
            'bot_id': self.bot_id,
            'knowledge': self.knowledge,
            'mental_models': self.mental_models,
            # [ru] Только последние 100
            # [en] Only the last 100
            'experiences': self.experiences[-100:],
            'created_at': self.created_at,
            'last_synced': self.last_synced
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndividualKnowledgeGraph':
        """
        [ru] Восстанавливает из словаря.
        [en] Restores from a dictionary.
        """
        graph = cls(bot_id=data.get('bot_id', str(uuid.uuid4())[:8]))
        graph.knowledge = data.get('knowledge', [])
        graph.mental_models = data.get('mental_models', {})
        graph.experiences = data.get('experiences', [])
        graph.created_at = data.get('created_at', time.time())
        graph.last_synced = data.get('last_synced', time.time())
        return graph

    def add_hypothesis(self, hypothesis) -> None:
        """
        [ru] Добавляет гипотезу в ИГЗ.
        [en] Adds a hypothesis to the Individual Knowledge Graph.
        """
        record = {
            'id': hypothesis.id,
            'type': 'hypothesis',
            'task_description': hypothesis.task_description,
            'source_combination_id': hypothesis.source_combination.id,
            'modifications': hypothesis.modifications,
            'description': hypothesis.description,
            'predicted_score': hypothesis.predicted_score,
            'actual_score': hypothesis.actual_score,
            'status': hypothesis.status.value,
            'metadata': hypothesis.metadata,
            'created_at': getattr(hypothesis, 'created_at', time.time()),
            'added_at': time.time()
        }
        self.knowledge.append(record)


