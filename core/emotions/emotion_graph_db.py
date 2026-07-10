# core/emotions/emotion_graph_db.py
"""
Расширение EmotionGraph с поддержкой БД.
"""

from core.emotions.emotion_graph import EmotionGraph
from db.emotion_db import EmotionDB
from typing import Optional
from core.emotions.emotion_base import EmotionalEvent, EmotionalResponse, EmotionType

class EmotionGraphDB(EmotionGraph):
    """
    Биграф с поддержкой сохранения в БД.
    """

    def __init__(self, db: Optional[EmotionDB] = None, load_from_db: bool = True):
        # Инициализируем родительский класс
        super().__init__()

        self.db = db or EmotionDB()

        if load_from_db:
            self._load_from_db()

    def _load_from_db(self):
        """Загружает данные из БД."""
        print("📥 Загрузка биграфа из БД...")

        graph_data = self.db.load_full_graph()

        # Загружаем события
        for event in graph_data['events']:
            # Используем метод родительского класса для добавления события
            self._add_event_internal(event)

        # Загружаем эмоции
        for emotion in graph_data['emotions']:
            # Используем метод родительского класса для добавления эмоции
            self._add_emotion_internal(emotion)

        # Загружаем причинные связи
        for link in graph_data['causal_links'].values():
            self.causal_links[link.id] = link
            self.event_graph.add_edge(link.source_id, link.target_id,
                                      weight=link.weight, delay=link.delay)
            self._add_to_index(self._event_outgoing, link.source_id, link.id)
            self._add_to_index(self._event_incoming, link.target_id, link.id)

        # Загружаем цепочки эмоций
        for link in graph_data['emotion_chain_links'].values():
            self.emotion_chain_links[link.id] = link
            self.emotion_graph.add_edge(link.source_id, link.target_id,
                                        weight=link.weight)
            self._add_to_index(self._emotion_outgoing, link.source_id, link.id)
            self._add_to_index(self._emotion_incoming, link.target_id, link.id)

        # Загружаем связи событие→эмоция
        for link in graph_data['event_emotion_links'].values():
            self.event_emotion_links[link.id] = link
            self.event_to_emotion.add_edge(link.source_id, link.target_id,
                                           probability=link.probability)
            self._add_to_index(self._event_outgoing, link.source_id, link.id)
            self._add_to_index(self._emotion_incoming, link.target_id, link.id)

        # Загружаем связи эмоция→событие
        for link in graph_data['emotion_event_links'].values():
            self.emotion_event_links[link.id] = link
            self.emotion_to_event.add_edge(link.source_id, link.target_id,
                                           probability=link.probability)
            self._add_to_index(self._emotion_outgoing, link.source_id, link.id)
            self._add_to_index(self._event_incoming, link.target_id, link.id)

        print(f"✅ Биграф загружен из БД")

    def _add_event_internal(self, event: EmotionalEvent):
        """Внутренний метод добавления события (без сохранения в БД)."""
        self.events[event.id] = event
        self.event_graph.add_node(event.id, embedding=event.embedding)
        self.event_embeddings.append(event.embedding)

    def _add_emotion_internal(self, emotion: EmotionalResponse):
        """Внутренний метод добавления эмоции (без сохранения в БД)."""
        # Получаем строковое представление типа эмоции
        emotion_type_str = emotion.emotion_type.value if hasattr(emotion.emotion_type, 'value') else str(emotion.emotion_type)

        self.emotions[emotion_type_str] = emotion
        self.emotion_graph.add_node(emotion_type_str,
                                    embedding=emotion.embedding)
        self.emotion_embeddings.append(emotion.embedding)

    def _save_to_db(self):
        """Сохраняет текущее состояние в БД."""
        print("💾 Сохранение биграфа в БД...")

        # Сохраняем события
        for event in self.events.values():
            self.db.save_event(event)

        # Сохраняем эмоции
        for emotion in self.emotions.values():
            self.db.save_emotion(emotion)

        # Сохраняем связи
        for link in self.causal_links.values():
            self.db.save_causal_link(link)
        for link in self.emotion_chain_links.values():
            self.db.save_emotion_chain_link(link)
        for link in self.event_emotion_links.values():
            self.db.save_event_emotion_link(link)
        for link in self.emotion_event_links.values():
            self.db.save_emotion_event_link(link)

        print(f"✅ Биграф сохранен в БД")

    def save(self):
        """Сохраняет текущее состояние в БД."""
        self._save_to_db()

    def add_event(self, event: EmotionalEvent):
        """Добавляет событие и сохраняет в БД."""
        self._add_event_internal(event)
        self.db.save_event(event)

    def add_emotion(self, emotion: EmotionalResponse):
        """Добавляет эмоцию и сохраняет в БД."""
        self._add_emotion_internal(emotion)
        self.db.save_emotion(emotion)

# class EmotionGraphDB(EmotionGraph):
#     """
#     Биграф с поддержкой сохранения в БД.
#     """
#
#     def __init__(self, db: Optional[EmotionDB] = None, load_from_db: bool = True):
#         super().__init__()
#
#         self.db = db or EmotionDB()
#
#         if load_from_db:
#             self._load_from_db()
#         else:
#             self._save_to_db()
#
#     def _load_from_db(self):
#         """Загружает данные из БД."""
#         print("📥 Загрузка биграфа из БД...")
#
#         graph_data = self.db.load_full_graph()
#
#         # Загружаем события
#         for event in graph_data['events']:
#             self.add_event(event)
#
#         # Загружаем эмоции
#         for emotion in graph_data['emotions']:
#             self.add_emotion(emotion)
#
#         # Загружаем причинные связи
#         for link in graph_data['causal_links'].values():
#             self.causal_links[link.id] = link
#             self.event_graph.add_edge(link.source_id, link.target_id,
#                                       weight=link.weight, delay=link.delay)
#             self._add_to_index(self._event_outgoing, link.source_id, link.id)
#             self._add_to_index(self._event_incoming, link.target_id, link.id)
#
#         # Загружаем цепочки эмоций
#         for link in graph_data['emotion_chain_links'].values():
#             self.emotion_chain_links[link.id] = link
#             self.emotion_graph.add_edge(link.source_id, link.target_id,
#                                         weight=link.weight)
#             self._add_to_index(self._emotion_outgoing, link.source_id, link.id)
#             self._add_to_index(self._emotion_incoming, link.target_id, link.id)
#
#         # Загружаем связи событие→эмоция
#         for link in graph_data['event_emotion_links'].values():
#             self.event_emotion_links[link.id] = link
#             self.event_to_emotion.add_edge(link.source_id, link.target_id,
#                                            probability=link.probability)
#             self._add_to_index(self._event_outgoing, link.source_id, link.id)
#             self._add_to_index(self._emotion_incoming, link.target_id, link.id)
#
#         # Загружаем связи эмоция→событие
#         for link in graph_data['emotion_event_links'].values():
#             self.emotion_event_links[link.id] = link
#             self.emotion_to_event.add_edge(link.source_id, link.target_id,
#                                            probability=link.probability)
#             self._add_to_index(self._emotion_outgoing, link.source_id, link.id)
#             self._add_to_index(self._event_incoming, link.target_id, link.id)
#
#         print(f"✅ Биграф загружен из БД")
#
#     def _save_to_db(self):
#         """Сохраняет текущее состояние в БД."""
#         print("💾 Сохранение биграфа в БД...")
#
#         # Сохраняем события
#         for event in self.events.values():
#             self.db.save_event(event)
#
#         # Сохраняем эмоции
#         for emotion in self.emotions.values():
#             self.db.save_emotion(emotion)
#
#         # Сохраняем связи
#         for link in self.causal_links.values():
#             self.db.save_causal_link(link)
#         for link in self.emotion_chain_links.values():
#             self.db.save_emotion_chain_link(link)
#         for link in self.event_emotion_links.values():
#             self.db.save_event_emotion_link(link)
#         for link in self.emotion_event_links.values():
#             self.db.save_emotion_event_link(link)
#
#         print(f"✅ Биграф сохранен в БД")
#
#     def save(self):
#         """Сохраняет текущее состояние в БД."""
#         self._save_to_db()
#
#     def add_event(self, event: EmotionalEvent):
#         """Добавляет событие и сохраняет в БД."""
#         super().add_event(event)
#         self.db.save_event(event)
#
#     def add_emotion(self, emotion: EmotionalResponse):
#         """Добавляет эмоцию и сохраняет в БД."""
#         super().add_emotion(emotion)
#         self.db.save_emotion(emotion)