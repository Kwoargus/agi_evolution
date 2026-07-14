# core/run_emotion_test.py - ТЕСТОВЫЙ СКРИПТ

import numpy as np
import time
from core.emotions.emotion_system import EmotionSystem

system = EmotionSystem()

# Тест 1: Создание эмоции
from core.emotions.emotion_base import EmotionalEvent, EmotionalResponse, EmotionType

event = EmotionalEvent(
    id="test_event",
    description="Тестовое событие",
    timestamp=time.time(),
    context={"type": "success"},
    participants=["bot"],
    embedding=np.random.randn(128)
)

responses = system.process_sensory_input({
    'vision': np.random.randn(8),
    'sound': np.random.randn(8),
    'smell': np.random.randn(8),
    'context': {'type': 'success'}
})

print(f"Эмоциональные реакции: {responses}")
print(f"Состояние: {system.get_emotional_state()}")