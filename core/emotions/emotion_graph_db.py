# core/emotions/emotion_graph_db.py
"""
[ru] Расширение EmotionGraph с поддержкой БД.
[en] EmotionGraph extension with database support.
"""

from core.emotions.emotion_graph import EmotionGraph
from db.emotion_db import EmotionDB
from typing import Optional
from core.emotions.emotion_base import EmotionalEvent, EmotionalResponse, EmotionType

class EmotionGraphDB(EmotionGraph):
    """
    [ru] Биграф с поддержкой сохранения в БД.
    [en] Bi`graph with support for saving to a database.
    """

    def __init__(self, db: Optional[EmotionDB] = None, load_from_db: bool = True):
        # [ru]  Инициализируем родительский класс
        # [en]  Initialize the parent class
        super().__init__()

        self.db = db or EmotionDB()

        if load_from_db:
            self._load_from_db()

    def _load_from_db(self):
        """
        [ru] Загружает данные из БД.
        [en] Loads data from the database.
        """
        print("📥 [ru] Загрузка биграфа из БД...")
        print("📥 [en] Loading bi`graph from database...")

        graph_data = self.db.load_full_graph()

        # [ru] Загружаем события
        # [en] Loading events
        for event in graph_data['events']:
            # Используем метод родительского класса для добавления события
            self._add_event_internal(event)

        # [ru] Загружаем эмоции
        # [en] Loading emotions
        for emotion in graph_data['emotions']:
            # [ru] Используем метод родительского класса для добавления эмоции
            # [en] We use the parent class method to add an emotion
            self._add_emotion_internal(emotion)

        # [ru] Загружаем причинные связи
        # [en] Loading causal relationships
        for link in graph_data['causal_links'].values():
            self.causal_links[link.id] = link
            self.event_graph.add_edge(link.source_id, link.target_id,
                                      weight=link.weight, delay=link.delay)
            self._add_to_index(self._event_outgoing, link.source_id, link.id)
            self._add_to_index(self._event_incoming, link.target_id, link.id)

        # [ru] Загружаем цепочки эмоций
        # [en] Loading emotion chains
        for link in graph_data['emotion_chain_links'].values():
            self.emotion_chain_links[link.id] = link
            self.emotion_graph.add_edge(link.source_id, link.target_id,
                                        weight=link.weight)
            self._add_to_index(self._emotion_outgoing, link.source_id, link.id)
            self._add_to_index(self._emotion_incoming, link.target_id, link.id)

        # [ru] Загружаем связи событие→эмоция
        # [en] Loading event→emotion links
        for link in graph_data['event_emotion_links'].values():
            self.event_emotion_links[link.id] = link
            self.event_to_emotion.add_edge(link.source_id, link.target_id,
                                           probability=link.probability)
            self._add_to_index(self._event_outgoing, link.source_id, link.id)
            self._add_to_index(self._emotion_incoming, link.target_id, link.id)

        # [ru] Загружаем связи эмоция→событие
        # [en] Loading emotion→event connections
        for link in graph_data['emotion_event_links'].values():
            self.emotion_event_links[link.id] = link
            self.emotion_to_event.add_edge(link.source_id, link.target_id,
                                           probability=link.probability)
            self._add_to_index(self._emotion_outgoing, link.source_id, link.id)
            self._add_to_index(self._event_incoming, link.target_id, link.id)

        print(f"✅ [ru] Биграф загружен из БД")
        print(f"✅ [en] Bi`graph loaded from the database")

    def _add_event_internal(self, event: EmotionalEvent):
        """
        [ru] Внутренний метод добавления события (без сохранения в БД).
        [en] Internal method for adding an event (without saving to the database).
        """
        self.events[event.id] = event
        self.event_graph.add_node(event.id, embedding=event.embedding)
        self.event_embeddings.append(event.embedding)

    def _add_emotion_internal(self, emotion: EmotionalResponse):
        """
        [ru] Внутренний метод добавления эмоции (без сохранения в БД).
        [en] Internal method for adding an emotion (without saving to the database).
        """
        # [ru] Получаем строковое представление типа эмоции
        # [en] We get a string representation of the emotion type
        emotion_type_str = emotion.emotion_type.value if hasattr(emotion.emotion_type, 'value') else str(emotion.emotion_type)

        self.emotions[emotion_type_str] = emotion
        self.emotion_graph.add_node(emotion_type_str,
                                    embedding=emotion.embedding)
        self.emotion_embeddings.append(emotion.embedding)

    def _save_to_db(self):
        """
        [ru] Сохраняет текущее состояние в БД.
        [en] Saves the current state in the database.
        """
        print("💾 [ru] Сохранение биграфа в БД...")
        print("💾 [en] Saving the bi`graph to the database...")

        # [ru] Сохраняем события
        # [en] Saving events
        for event in self.events.values():
            self.db.save_event(event)

        # [ru] Сохраняем эмоции
        # [en] Saving emotions
        for emotion in self.emotions.values():
            self.db.save_emotion(emotion)

        # [ru] Сохраняем связи
        # [en] Saving connections
        for link in self.causal_links.values():
            self.db.save_causal_link(link)
        for link in self.emotion_chain_links.values():
            self.db.save_emotion_chain_link(link)
        for link in self.event_emotion_links.values():
            self.db.save_event_emotion_link(link)
        for link in self.emotion_event_links.values():
            self.db.save_emotion_event_link(link)

        print(f"✅ [ru] Биграф сохранен в БД")
        print(f"✅ [en] The bi`graph is saved in the database")

    def save(self):
        """
        [ru] Сохраняет текущее состояние в БД.
        [en] Saves the current state in the database.
        """
        self._save_to_db()

    def add_event(self, event: EmotionalEvent):
        """
        [ru] Добавляет событие и сохраняет в БД.
        [en] Adds an event and saves it to the database.
        """
        self._add_event_internal(event)
        self.db.save_event(event)

    def add_emotion(self, emotion: EmotionalResponse):
        """
        [ru] Добавляет эмоцию и сохраняет в БД.
        [en] Adds an emotion and saves it in the database.
        """
        self._add_emotion_internal(emotion)
        self.db.save_emotion(emotion)
