# scripts/emotion_scenarios.py
"""
Реалистичные сценарии для тестирования эмоциональной системы.
"""

import numpy as np
from core.emotions.emotion_base import EmotionalEvent, EmotionalResponse, EmotionType


class EmotionScenarioGenerator:
    """
    Генерирует реалистичные сценарии с разными эмоциональными реакциями.
    """

    def __init__(self):
        # Предопределенные эмбеддинги для разных типов событий
        self.event_embeddings = {
            'danger': self._create_embedding([0.9, 0.8, 0.1, 0.0, 0.0]),  # Страх
            'success': self._create_embedding([0.1, 0.0, 0.9, 0.8, 0.0]),  # Радость
            'failure': self._create_embedding([0.0, 0.9, 0.8, 0.1, 0.0]),  # Печаль
            'injustice': self._create_embedding([0.8, 0.1, 0.0, 0.0, 0.9]),  # Гнев
            'surprise': self._create_embedding([0.1, 0.9, 0.1, 0.0, 0.0]),  # Удивление
            'love': self._create_embedding([0.1, 0.0, 0.1, 0.9, 0.8]),  # Любовь
            'betrayal': self._create_embedding([0.9, 0.1, 0.0, 0.8, 0.1]),  # Обида
            'threat': self._create_embedding([0.8, 0.9, 0.1, 0.0, 0.1]),  # Страх
            'achievement': self._create_embedding([0.1, 0.0, 0.8, 0.9, 0.1]),  # Гордость
            'loss': self._create_embedding([0.0, 0.9, 0.1, 0.0, 0.8]),  # Печаль
        }

        # Связи событие → эмоция
        self.event_emotion_map = {
            'danger': EmotionType.FEAR,
            'success': EmotionType.JOY,
            'failure': EmotionType.SADNESS,
            'injustice': EmotionType.ANGER,
            'surprise': EmotionType.SURPRISE,
            'love': EmotionType.LOVE,
            'betrayal': EmotionType.RESENTMENT,
            'threat': EmotionType.FEAR,
            'achievement': EmotionType.JOY,
            'loss': EmotionType.SADNESS,
        }

        # Сложность реакции (сколько эмоций вызывается)
        self.event_complexity = {
            'danger': 2,  # Страх + Удивление
            'success': 2,  # Радость + Гордость
            'failure': 2,  # Печаль + Разочарование
            'injustice': 3,  # Гнев + Обида + Разочарование
            'surprise': 1,  # Удивление
            'love': 2,  # Любовь + Доверие
            'betrayal': 3,  # Обида + Гнев + Печаль
            'threat': 2,  # Страх + Тревога
            'achievement': 2,  # Радость + Гордость
            'loss': 2,  # Печаль + Тоска
        }

        # Интенсивность эмоций
        self.event_intensity = {
            'danger': 0.9,
            'success': 0.8,
            'failure': 0.7,
            'injustice': 0.8,
            'surprise': 0.6,
            'love': 0.7,
            'betrayal': 0.9,
            'threat': 0.8,
            'achievement': 0.7,
            'loss': 0.8,
        }

    def _create_embedding(self, pattern: list) -> np.ndarray:
        """Создает эмбеддинг из паттерна."""
        embedding = np.zeros(128)
        embedding[:len(pattern)] = pattern
        # Добавляем небольшой шум
        embedding += np.random.randn(128) * 0.05
        # Нормализуем
        norm = np.linalg.norm(embedding) + 1e-8
        return embedding / norm

    def generate_scenario(self, event_type: str) -> dict:
        """
        Генерирует сценарий с заданным типом события.
        """
        if event_type not in self.event_embeddings:
            event_type = 'success'

        return {
            'type': event_type,
            'embedding': self.event_embeddings[event_type],
            'expected_emotion': self.event_emotion_map[event_type],
            'complexity': self.event_complexity[event_type],
            'intensity': self.event_intensity[event_type],
            'description': self._get_description(event_type),
            'context': self._get_context(event_type),
            'participants': self._get_participants(event_type)
        }

    def _get_description(self, event_type: str) -> str:
        """Возвращает описание события."""
        descriptions = {
            'danger': 'Опасная ситуация: рядом хищник!',
            'success': 'Успешно достигнута цель!',
            'failure': 'Не удалось достичь цели.',
            'injustice': 'Несправедливое отношение!',
            'surprise': 'Неожиданное событие!',
            'love': 'Встреча с близким существом.',
            'betrayal': 'Предательство со стороны друга.',
            'threat': 'Угроза жизни или здоровью.',
            'achievement': 'Достижение важной цели.',
            'loss': 'Потеря чего-то важного.'
        }
        return descriptions.get(event_type, 'Событие')

    def _get_context(self, event_type: str) -> dict:
        """Возвращает контекст события."""
        contexts = {
            'danger': {'type': 'danger', 'level': 'high'},
            'success': {'type': 'achievement', 'level': 'high'},
            'failure': {'type': 'achievement', 'level': 'low'},
            'injustice': {'type': 'social', 'level': 'high'},
            'surprise': {'type': 'unexpected', 'level': 'medium'},
            'love': {'type': 'social', 'level': 'high'},
            'betrayal': {'type': 'social', 'level': 'very_high'},
            'threat': {'type': 'danger', 'level': 'critical'},
            'achievement': {'type': 'achievement', 'level': 'high'},
            'loss': {'type': 'personal', 'level': 'high'}
        }
        return contexts.get(event_type, {'type': 'unknown'})

    def _get_participants(self, event_type: str) -> list:
        """Возвращает участников события."""
        participants = {
            'danger': ['bot', 'predator'],
            'success': ['bot'],
            'failure': ['bot'],
            'injustice': ['bot', 'other'],
            'surprise': ['bot'],
            'love': ['bot', 'other'],
            'betrayal': ['bot', 'friend'],
            'threat': ['bot', 'danger'],
            'achievement': ['bot'],
            'loss': ['bot']
        }
        return participants.get(event_type, ['bot'])

    def generate_sequence(self, length: int = 50) -> list:
        """
        Генерирует последовательность событий для эволюции.
        """
        # Создаем реалистичную последовательность
        sequence = []

        # Базовые события, которые должны быть
        base_events = ['success', 'failure', 'danger', 'injustice', 'love']

        for i in range(length):
            if i < len(base_events):
                event_type = base_events[i]
            else:
                # Случайные события с весами (некоторые чаще)
                weights = {
                    'success': 0.2,
                    'failure': 0.15,
                    'danger': 0.15,
                    'injustice': 0.1,
                    'surprise': 0.1,
                    'love': 0.1,
                    'betrayal': 0.05,
                    'threat': 0.05,
                    'achievement': 0.05,
                    'loss': 0.05
                }
                event_type = np.random.choice(
                    list(weights.keys()),
                    p=list(weights.values())
                )

            sequence.append(self.generate_scenario(event_type))

        return sequence