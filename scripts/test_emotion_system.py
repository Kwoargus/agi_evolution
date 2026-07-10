# scripts/test_emotion_system.py (исправленный)
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.emotions.emotion_system import EmotionSystem
from core.emotions.emotion_base import EmotionalEvent, EmotionType
from typing import List, Dict, Optional, Tuple, Any, Union


def test_emotion_system():
    """
    Тестирует полную эмоциональную подсистему.
    """
    print("=== Тестирование эмоциональной подсистемы ===\n")

    # 1. Инициализация
    emotion_system = EmotionSystem()
    print("✅ Система создана\n")

    # 2. Создаем тестовые сенсорные данные
    print("2. Тестирование обработки сенсорных данных...")
    sensory_data = {
        'vision': np.random.randn(64),
        'sound': np.random.randn(32),
        'smell': np.random.randn(32),
        'context': {'situation': 'test'},
        'participants': ['agent', 'observer']
    }

    responses = emotion_system.process_sensory_input(sensory_data)
    print(f"  Сгенерировано {len(responses)} эмоциональных реакций")
    for r in responses:
        print(f"    - {r.emotion_type.value}: интенсивность {r.intensity:.3f}")

    # 3. Проверка текущего состояния
    print("\n3. Текущее эмоциональное состояние:")
    state = emotion_system.get_emotional_state()
    print(f"  Доминирующая эмоция: {state['dominant_emotion']}")
    print(f"  Интенсивность: {state['intensity']:.3f}")
    print(f"  Валентность: {state['valence']:.3f}")

    # 4. Трассировка цепочки (с проверкой наличия истории)
    print("\n4. Трассировка эмоциональной цепочки:")
    if emotion_system.emotion_history:
        chain = emotion_system.trace_emotional_chain()
        for step in chain[:5]:
            print(f"  {step}")
    else:
        print("  Нет истории эмоций для трассировки")

    # 5. Интуиция
    print("\n5. Интуитивные инсайты:")
    if emotion_system.insight_history:
        for insight in emotion_system.insight_history[-3:]:
            print(f"  {insight['explanation']} (уверенность: {insight['confidence']:.2f})")
    else:
        print("  Нет интуитивных инсайтов")

    # 6. Ментальные модели
    print("\n6. Создание ментальных моделей:")
    model1 = emotion_system.models.create_model(
        name="идеальный стол",
        model_type="object",
        attributes={'прочность': 0.9, 'красота': 0.8, 'цена': 0.5}
    )
    model2 = emotion_system.models.create_model(
        name="функциональный стол",
        model_type="object",
        attributes={'прочность': 0.7, 'красота': 0.5, 'цена': 0.9}
    )

    comparison = emotion_system.compare_mental_models("идеальный стол", "функциональный стол")
    if 'error' not in comparison:
        print(f"  Сходство моделей: {comparison['similarity']:.3f}")
        print(f"  Синхронизация: {comparison['synchronization_score']:.3f}")
        if comparison.get('differences'):
            print(f"  Различия: {comparison['differences']}")
    else:
        print(f"  Ошибка: {comparison['error']}")

    # 7. Синхронизация моделей
    print("\n7. Синхронизация моделей:")
    merged = emotion_system.synchronize_mental_models("идеальный стол", "функциональный стол")
    if merged:
        print(f"  Создана объединенная модель: {merged.name}")
        print(f"  Атрибуты: {merged.attributes}")
    else:
        print("  Синхронизация не выполнена")

    print("\n✅ Тестирование эмоциональной подсистемы завершено!")


if __name__ == "__main__":
    test_emotion_system()