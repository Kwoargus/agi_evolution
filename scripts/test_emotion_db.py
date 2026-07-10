# scripts/test_emotion_db.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.emotions.emotion_graph_db import EmotionGraphDB
from core.emotions.emotion_base import EmotionalEvent, EmotionalResponse, EmotionType
from db.emotion_db import EmotionDB


def test_emotion_db():
    """
    Тестирует сохранение и загрузку биграфа из БД.
    """
    print("=== Тестирование сохранения биграфа в БД ===\n")

    # 1. Создаем биграф с загрузкой из БД
    print("1. Создание биграфа с загрузкой из БД...")
    graph = EmotionGraphDB(load_from_db=True)

    # 2. Добавляем новое событие
    print("\n2. Добавление нового события...")
    new_event = EmotionalEvent(
        id="test_event_1",
        description="Тестовое событие",
        timestamp=0,
        context={"test": True},
        participants=["test"],
        embedding=np.random.randn(128)
    )
    graph.add_event(new_event)

    # 3. Добавляем новую эмоцию
    print("\n3. Добавление новой эмоции...")
    new_emotion = EmotionalResponse(
        emotion_type=EmotionType.JOY,
        intensity=0.7,
        valence=0.8,
        arousal=0.6,
        embedding=np.random.randn(64)
    )
    graph.add_emotion(new_emotion)

    # 4. Добавляем связь
    print("\n4. Добавление связи событие→эмоция...")
    graph.add_event_emotion_link(
        "test_event_1",
        EmotionType.JOY,
        probability=0.7,
        intensity_factor=1.2
    )

    # 5. Сохраняем в БД
    print("\n5. Сохранение в БД...")
    graph.save()

    # 6. Создаем новый биграф и загружаем
    print("\n6. Создание нового биграфа с загрузкой из БД...")
    graph2 = EmotionGraphDB(load_from_db=True)

    # 7. Проверяем данные
    print("\n7. Проверка загруженных данных...")
    print(f"  Событий: {len(graph2.events)}")
    print(f"  Эмоций: {len(graph2.emotions)}")
    print(f"  Причинных связей: {len(graph2.causal_links)}")
    print(f"  Цепочек эмоций: {len(graph2.emotion_chain_links)}")
    print(f"  Связей событие→эмоция: {len(graph2.event_emotion_links)}")
    print(f"  Связей эмоция→событие: {len(graph2.emotion_event_links)}")

    # 8. Проверяем наличие тестового события
    if "test_event_1" in graph2.events:
        print(f"\n✅ Тестовое событие найдено: {graph2.events['test_event_1'].description}")
    else:
        print("\n❌ Тестовое событие не найдено")

    print("\n✅ Тестирование завершено!")


if __name__ == "__main__":
    test_emotion_db()