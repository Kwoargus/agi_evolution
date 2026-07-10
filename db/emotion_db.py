# db/emotion_db.py - обновленный метод сохранения
import psycopg2
import psycopg2.extras
import json
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime
from core.emotions.emotion_base import EmotionalEvent, EmotionalResponse, EmotionType
from core.emotions.links import (
    CausalLink, EmotionChainLink, EventEmotionLink, EmotionEventLink,
    BaseLink
)

from core.emotions.links import (
    CausalLink, EmotionChainLink, EventEmotionLink, EmotionEventLink,
    BaseLink
)


def save_emotion(self, emotion: EmotionalResponse) -> bool:
    """Сохраняет эмоциональную реакцию в БД."""
    try:
        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute(f"""
            INSERT INTO {self.schema}.emotion_responses 
            (id, emotion_type, intensity, valence, arousal, context, source, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                emotion_type = EXCLUDED.emotion_type,
                intensity = EXCLUDED.intensity,
                valence = EXCLUDED.valence,
                arousal = EXCLUDED.arousal,
                context = EXCLUDED.context,
                source = EXCLUDED.source,
                embedding = EXCLUDED.embedding,
                updated_at = CURRENT_TIMESTAMP
        """, (
            emotion.id,
            emotion.emotion_type.value,
            emotion.intensity,
            emotion.valence,
            emotion.arousal,
            json.dumps(emotion.context),
            emotion.source,
            emotion.embedding.tolist() if hasattr(emotion.embedding, 'tolist') else emotion.embedding
        ))

        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Ошибка сохранения эмоции {emotion.id}: {e}")
        return False






class EmotionDB:
    """
    Класс для сохранения и загрузки биграфа эмоций в/из БД.
    """

    def __init__(self, host='localhost', port=5432,
                 dbname='postgres', user='postgres', password='postgres'):
        self.conn_params = {
            'host': host,
            'port': port,
            'dbname': dbname,
            'user': user,
            'password': password
        }
        self.schema = 'agi_evolution'

    def _get_connection(self):
        """Возвращает соединение с БД."""
        return psycopg2.connect(**self.conn_params)

    # ============================================================
    # СОХРАНЕНИЕ
    # ============================================================

    def save_event(self, event: EmotionalEvent) -> bool:
        """Сохраняет событие в БД."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute(f"""
                INSERT INTO {self.schema}.emotion_events
                (id, description, timestamp, context, participants, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    description = EXCLUDED.description,
                    timestamp = EXCLUDED.timestamp,
                    context = EXCLUDED.context,
                    participants = EXCLUDED.participants,
                    embedding = EXCLUDED.embedding
            """, (
                event.id,
                event.description,
                event.timestamp,
                json.dumps(event.context),
                event.participants,
                event.embedding.tolist() if hasattr(event.embedding, 'tolist') else event.embedding
            ))

            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения события {event.id}: {e}")
            return False

    def save_emotion(self, emotion: EmotionalResponse) -> bool:
        """Сохраняет эмоциональную реакцию в БД."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute(f"""
                INSERT INTO {self.schema}.emotion_responses
                (id, emotion_type, intensity, valence, arousal, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    emotion_type = EXCLUDED.emotion_type,
                    intensity = EXCLUDED.intensity,
                    valence = EXCLUDED.valence,
                    arousal = EXCLUDED.arousal,
                    embedding = EXCLUDED.embedding
            """, (
                emotion.emotion_type.value,
                emotion.emotion_type.value,
                emotion.intensity,
                emotion.valence,
                emotion.arousal,
                emotion.embedding.tolist() if hasattr(emotion.embedding, 'tolist') else emotion.embedding
            ))

            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения эмоции {emotion.emotion_type.value}: {e}")
            return False

    def save_causal_link(self, link: CausalLink) -> bool:
        """Сохраняет причинно-следственную связь."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute(f"""
                INSERT INTO {self.schema}.causal_links
                (id, source_id, target_id, weight, delay, probability,
                 usage_count, success_count, metadata, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    weight = EXCLUDED.weight,
                    delay = EXCLUDED.delay,
                    probability = EXCLUDED.probability,
                    usage_count = EXCLUDED.usage_count,
                    success_count = EXCLUDED.success_count,
                    metadata = EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at
            """, (
                link.id,
                link.source_id,
                link.target_id,
                link.weight,
                link.delay,
                link.probability,
                link.usage_count,
                link.success_count,
                json.dumps(link.metadata),
                link.updated_at
            ))

            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения связи {link.id}: {e}")
            return False

    def save_emotion_chain_link(self, link: EmotionChainLink) -> bool:
        """Сохраняет цепочку эмоций."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute(f"""
                INSERT INTO {self.schema}.emotion_chain_links
                (id, source_id, target_id, weight, intensity_amplification,
                 threshold, usage_count, success_count, metadata, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    weight = EXCLUDED.weight,
                    intensity_amplification = EXCLUDED.intensity_amplification,
                    threshold = EXCLUDED.threshold,
                    usage_count = EXCLUDED.usage_count,
                    success_count = EXCLUDED.success_count,
                    metadata = EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at
            """, (
                link.id,
                link.source_id,
                link.target_id,
                link.weight,
                link.intensity_amplification,
                link.threshold,
                link.usage_count,
                link.success_count,
                json.dumps(link.metadata),
                link.updated_at
            ))

            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения цепочки эмоций {link.id}: {e}")
            return False

    def save_event_emotion_link(self, link: EventEmotionLink) -> bool:
        """Сохраняет связь событие→эмоция."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute(f"""
                INSERT INTO {self.schema}.event_emotion_links
                (id, source_id, target_id, weight, probability,
                 intensity_factor, valence_shift, conditions,
                 usage_count, success_count, metadata, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    weight = EXCLUDED.weight,
                    probability = EXCLUDED.probability,
                    intensity_factor = EXCLUDED.intensity_factor,
                    valence_shift = EXCLUDED.valence_shift,
                    conditions = EXCLUDED.conditions,
                    usage_count = EXCLUDED.usage_count,
                    success_count = EXCLUDED.success_count,
                    metadata = EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at
            """, (
                link.id,
                link.source_id,
                link.target_id,
                link.weight,
                link.probability,
                link.intensity_factor,
                link.valence_shift,
                json.dumps(link.conditions),
                link.usage_count,
                link.success_count,
                json.dumps(link.metadata),
                link.updated_at
            ))

            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения связи событие→эмоция {link.id}: {e}")
            return False

    def save_emotion_event_link(self, link: EmotionEventLink) -> bool:
        """Сохраняет связь эмоция→событие."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute(f"""
                INSERT INTO {self.schema}.emotion_event_links
                (id, source_id, target_id, weight, probability,
                 action_urgency, action_duration,
                 usage_count, success_count, metadata, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    weight = EXCLUDED.weight,
                    probability = EXCLUDED.probability,
                    action_urgency = EXCLUDED.action_urgency,
                    action_duration = EXCLUDED.action_duration,
                    usage_count = EXCLUDED.usage_count,
                    success_count = EXCLUDED.success_count,
                    metadata = EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at
            """, (
                link.id,
                link.source_id,
                link.target_id,
                link.weight,
                link.probability,
                link.action_urgency,
                link.action_duration,
                link.usage_count,
                link.success_count,
                json.dumps(link.metadata),
                link.updated_at
            ))

            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения связи эмоция→событие {link.id}: {e}")
            return False

    # ============================================================
    # ЗАГРУЗКА
    # ============================================================

    def load_events(self) -> List[EmotionalEvent]:
        """Загружает все события из БД."""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute(f"""
                SELECT id, description, timestamp, context, participants, embedding
                FROM {self.schema}.emotion_events
            """)

            rows = cur.fetchall()
            events = []
            for row in rows:
                event = EmotionalEvent(
                    id=row['id'],
                    description=row['description'],
                    timestamp=row['timestamp'],
                    context=row['context'] if row['context'] else {},
                    participants=row['participants'] if row['participants'] else [],
                    embedding=np.array(row['embedding']) if row['embedding'] else np.zeros(128)
                )
                events.append(event)

            cur.close()
            conn.close()
            return events
        except Exception as e:
            print(f"❌ Ошибка загрузки событий: {e}")
            return []

    def load_emotions(self) -> List[EmotionalResponse]:
        """Загружает все эмоции из БД."""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute(f"""
                SELECT id, emotion_type, intensity, valence, arousal, context, source, embedding
                FROM {self.schema}.emotion_responses
            """)

            rows = cur.fetchall()
            emotions = []
            for row in rows:
                # Создаем объект EmotionalResponse
                emotion = EmotionalResponse(
                    id=row['id'],
                    emotion_type=EmotionType(row['emotion_type']),
                    intensity=row['intensity'] or 0.3,
                    valence=row['valence'] or 0.0,
                    arousal=row['arousal'] or 0.0,
                    context=row['context'] if row['context'] else {},
                    source=row['source'] or 'inherited',
                    embedding=np.array(row['embedding']) if row['embedding'] else np.zeros(64)
                )
                emotions.append(emotion)

            cur.close()
            conn.close()
            return emotions
        except Exception as e:
            print(f"❌ Ошибка загрузки эмоций: {e}")
            return []

    # def load_emotions(self) -> List[EmotionalResponse]:
    #     """Загружает все эмоции из БД."""
    #     try:
    #         conn = self._get_connection()
    #         cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    #
    #         cur.execute(f"""
    #             SELECT id, emotion_type, intensity, valence, arousal, embedding
    #             FROM {self.schema}.emotion_responses
    #         """)
    #
    #         rows = cur.fetchall()
    #         emotions = []
    #         for row in rows:
    #             emotion = EmotionalResponse(
    #                 emotion_type=EmotionType(row['emotion_type']),
    #                 intensity=row['intensity'],
    #                 valence=row['valence'],
    #                 arousal=row['arousal'],
    #                 embedding=np.array(row['embedding']) if row['embedding'] else np.zeros(64)
    #             )
    #             # Используем id из БД как ключ
    #             emotions.append(emotion)
    #
    #         cur.close()
    #         conn.close()
    #         return emotions
    #     except Exception as e:
    #         print(f"❌ Ошибка загрузки эмоций: {e}")
    #         return []

    def load_causal_links(self) -> Dict[str, CausalLink]:
        """Загружает причинно-следственные связи."""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute(f"""
                SELECT id, source_id, target_id, weight, delay, probability,
                       usage_count, success_count, metadata
                FROM {self.schema}.causal_links
            """)

            rows = cur.fetchall()
            links = {}
            for row in rows:
                link = CausalLink(
                    id=row['id'],
                    source_id=row['source_id'],
                    target_id=row['target_id'],
                    weight=row['weight'],
                    delay=row['delay'],
                    probability=row['probability'],
                    metadata=row['metadata'] if row['metadata'] else {}
                )
                link.usage_count = row['usage_count'] or 0
                link.success_count = row['success_count'] or 0
                links[link.id] = link

            cur.close()
            conn.close()
            return links
        except Exception as e:
            print(f"❌ Ошибка загрузки причинных связей: {e}")
            return {}

    def load_emotion_chain_links(self) -> Dict[str, EmotionChainLink]:
        """Загружает цепочки эмоций."""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute(f"""
                SELECT id, source_id, target_id, weight, intensity_amplification,
                       threshold, usage_count, success_count, metadata
                FROM {self.schema}.emotion_chain_links
            """)

            rows = cur.fetchall()
            links = {}
            for row in rows:
                link = EmotionChainLink(
                    id=row['id'],
                    source_id=row['source_id'],
                    target_id=row['target_id'],
                    weight=row['weight'],
                    intensity_amplification=row['intensity_amplification'],
                    threshold=row['threshold'],
                    metadata=row['metadata'] if row['metadata'] else {}
                )
                link.usage_count = row['usage_count'] or 0
                link.success_count = row['success_count'] or 0
                links[link.id] = link

            cur.close()
            conn.close()
            return links
        except Exception as e:
            print(f"❌ Ошибка загрузки цепочек эмоций: {e}")
            return {}

    def load_event_emotion_links(self) -> Dict[str, EventEmotionLink]:
        """Загружает связи событие→эмоция."""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute(f"""
                SELECT id, source_id, target_id, weight, probability,
                       intensity_factor, valence_shift, conditions,
                       usage_count, success_count, metadata
                FROM {self.schema}.event_emotion_links
            """)

            rows = cur.fetchall()
            links = {}
            for row in rows:
                link = EventEmotionLink(
                    id=row['id'],
                    source_id=row['source_id'],
                    target_id=row['target_id'],
                    weight=row['weight'],
                    probability=row['probability'],
                    intensity_factor=row['intensity_factor'],
                    valence_shift=row['valence_shift'] or 0.0,
                    conditions=row['conditions'] if row['conditions'] else [],
                    metadata=row['metadata'] if row['metadata'] else {}
                )
                link.usage_count = row['usage_count'] or 0
                link.success_count = row['success_count'] or 0
                links[link.id] = link

            cur.close()
            conn.close()
            return links
        except Exception as e:
            print(f"❌ Ошибка загрузки связей событие→эмоция: {e}")
            return {}

    def load_emotion_event_links(self) -> Dict[str, EmotionEventLink]:
        """Загружает связи эмоция→событие."""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute(f"""
                SELECT id, source_id, target_id, weight, probability,
                       action_urgency, action_duration,
                       usage_count, success_count, metadata
                FROM {self.schema}.emotion_event_links
            """)

            rows = cur.fetchall()
            links = {}
            for row in rows:
                link = EmotionEventLink(
                    id=row['id'],
                    source_id=row['source_id'],
                    target_id=row['target_id'],
                    weight=row['weight'],
                    probability=row['probability'],
                    action_urgency=row['action_urgency'],
                    action_duration=row['action_duration'] or 1.0,
                    metadata=row['metadata'] if row['metadata'] else {}
                )
                link.usage_count = row['usage_count'] or 0
                link.success_count = row['success_count'] or 0
                links[link.id] = link

            cur.close()
            conn.close()
            return links
        except Exception as e:
            print(f"❌ Ошибка загрузки связей эмоция→событие: {e}")
            return {}

    # ============================================================
    # ПОЛНАЯ ЗАГРУЗКА БИГРАФА
    # ============================================================

    def load_full_graph(self) -> Dict[str, Any]:
        """
        Загружает полный биграф из БД.
        Возвращает словарь с данными для восстановления EmotionGraph.
        """
        print("🔄 Загрузка биграфа из БД...")

        graph_data = {
            'events': self.load_events(),
            'emotions': self.load_emotions(),
            'causal_links': self.load_causal_links(),
            'emotion_chain_links': self.load_emotion_chain_links(),
            'event_emotion_links': self.load_event_emotion_links(),
            'emotion_event_links': self.load_emotion_event_links()
        }

        print(f"✅ Загружено: {len(graph_data['events'])} событий, "
              f"{len(graph_data['emotions'])} эмоций, "
              f"{len(graph_data['causal_links'])} причинных связей, "
              f"{len(graph_data['emotion_chain_links'])} цепочек эмоций, "
              f"{len(graph_data['event_emotion_links'])} связей событие→эмоция, "
              f"{len(graph_data['emotion_event_links'])} связей эмоция→событие")

        return graph_data


    # def load_full_graph(self) -> Dict[str, Any]:
    #     """
    #     Загружает полный биграф из БД.
    #     Возвращает словарь с данными для восстановления EmotionGraph.
    #     """
    #     print("🔄 Загрузка биграфа из БД...")
    #
    #     graph_data = {
    #         'events': self.load_events(),
    #         'emotions': self.load_emotions(),
    #         'causal_links': self.load_causal_links(),
    #         'emotion_chain_links': self.load_emotion_chain_links(),
    #         'event_emotion_links': self.load_event_emotion_links(),
    #         'emotion_event_links': self.load_emotion_event_links()
    #     }
    #
    #     print(f"✅ Загружено: {len(graph_data['events'])} событий, "
    #           f"{len(graph_data['emotions'])} эмоций, "
    #           f"{len(graph_data['causal_links'])} причинных связей, "
    #           f"{len(graph_data['emotion_chain_links'])} цепочек эмоций, "
    #           f"{len(graph_data['event_emotion_links'])} связей событие→эмоция, "
    #           f"{len(graph_data['emotion_event_links'])} связей эмоция→событие")
    #
    #     return graph_data