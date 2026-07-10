# scripts/populate_all_tables.py
"""
Быстрое заполнение всех таблиц тестовыми данными.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.emotions.emotion_system import EmotionSystem
from core.emotions.emotion_graph_db import EmotionGraphDB
from db.emotion_db import EmotionDB


def populate_all():
    print("Заполнение всех таблиц...")

    # 1. Инициализация эмоциональной системы
    emotion_system = EmotionSystem()
    emotion_system.engine.graph._save_to_db()

    # 2. Создание тестовых событий
    graph = EmotionGraphDB(load_from_db=False)

    # Добавляем тестовые события
    from core.emotions.emotion_base import EmotionalEvent, EmotionalResponse, EmotionType
    import numpy as np

    test_events = [
        ("danger", "Опасная ситуация"),
        ("success", "Успех"),
        ("failure", "Неудача"),
        ("injustice", "Несправедливость")
    ]

    for event_id, description in test_events:
        event = EmotionalEvent(
            id=event_id,
            description=description,
            timestamp=0,
            embedding=np.random.randn(128)
        )
        graph.add_event(event)

    # Добавляем связи
    graph.add_event_emotion_link("danger", EmotionType.FEAR, 0.8)
    graph.add_event_emotion_link("success", EmotionType.JOY, 0.7)
    graph.add_event_emotion_link("failure", EmotionType.SADNESS, 0.6)
    graph.add_event_emotion_link("injustice", EmotionType.ANGER, 0.7)

    # Сохраняем
    graph.save()

    print("✅ Все таблицы заполнены тестовыми данными!")


if __name__ == "__main__":
    populate_all()