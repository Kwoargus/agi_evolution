# core/emotions/links.py
"""
[ru] Классы связей для биграфа событий/эмоций. Каждая связь — это полноценный объект с атрибутами и методами.
[en] Relationship classes for the event/emotion bigraph. Each edge is a fully-fledged object with attributes and methods.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np
from datetime import datetime
import json


@dataclass
class BaseLink:
    """
    [ru] Базовый класс для всех связей в биграфе.
    [ru] Base class for all edges in a bigraph.
    """
    id: str
    source_id: str
    target_id: str
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    success_count: int = 0

    # Дополнительные метаданные
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_weight(self, delta: float, max_weight: float = 1.0):
        """
        [ru] Обновляет вес связи.
        [en] Updates the weight of an edge.
        """
        self.weight = min(max_weight, self.weight + delta)
        self.updated_at = datetime.now()

    def record_usage(self, success: bool = True):
        """
        [ru]  Записывает использование связи.
        [en] Records edge usage.
        """
        self.usage_count += 1
        if success:
            self.success_count += 1
        self.updated_at = datetime.now()

    def get_success_rate(self) -> float:
        """
        [ru] Возвращает процент успешных использований.
        [en] Returns the percentage of successful uses.
        """
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count

    def get_confidence(self) -> float:
        """

        [ru] Вычисляет уверенность в связи на основе истории использования.
        [en] Calculates edge confidence based on usage history.
        """
        if self.usage_count < 5:
            return 0.3  # Низкая уверенность для новых связей

        base_confidence = self.get_success_rate()
        # Увеличиваем уверенность с количеством использований
        confidence_boost = min(0.3, self.usage_count / 100)
        return min(1.0, base_confidence + confidence_boost)

    def to_dict(self) -> Dict:
        """
        [ru] Сериализация связи.
        [en] Edge serialization.
        """
        return {
            'id': self.id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'weight': self.weight,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'usage_count': self.usage_count,
            'success_count': self.success_count,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'BaseLink':
        """
        [ru] Десериализация связи.
        [en] Deserialization of the edge.
        """
        link = cls(
            id=data['id'],
            source_id=data['source_id'],
            target_id=data['target_id'],
            weight=data['weight'],
            metadata=data.get('metadata', {})
        )
        link.usage_count = data.get('usage_count', 0)
        link.success_count = data.get('success_count', 0)
        if 'created_at' in data:
            link.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            link.updated_at = datetime.fromisoformat(data['updated_at'])
        return link


@dataclass
class CausalLink(BaseLink):
    """
    [ru] Причинно-следственная связь между событиями. event1 → event2 (event1 вызывает event2)
    [en] Cause and effect relationship between events: event1 → event2 (event1 causes event2)
    """
    delay: float = 0.0  # Временная задержка между событиями
    probability: float = 0.5  # Вероятность наступления следствия
    context_dependent: bool = False  # Зависит ли от контекста

    # [ru] Контекстные модификаторы
    # [en] Context modifiers
    context_modifiers: Dict[str, float] = field(default_factory=dict)

    def get_modified_probability(self, context: Dict[str, Any]) -> float:
        """
        Вычисляет вероятность с учетом контекста.
        [ru] Вычисляет вероятность с учетом контекста.
        [en] Calculates probability given context.
        """
        if not self.context_dependent:
            return self.probability

        prob = self.probability
        for key, modifier in self.context_modifiers.items():
            if key in context:
                prob *= modifier

        return min(1.0, max(0.0, prob))

    def update_from_experience(self, occurred: bool, context: Dict = None):
        """
        [ru] Обновляет связь на основе опыта.
        [en] Updates the edge based on experience.
        """
        self.record_usage(occurred)

        # Обновляем вероятность
        # [ru]
        # [en]
        if occurred:
            self.probability = min(1.0, self.probability + 0.05)
        else:
            self.probability = max(0.0, self.probability - 0.05)

        # [ru] Обновляем вес
        # [en] Updating the weight
        if occurred:
            self.update_weight(0.1)
        else:
            self.update_weight(-0.05)

        self.updated_at = datetime.now()

    def predict(self, context: Dict = None) -> float:
        """
        [ru] Предсказывает вероятность наступления следствия.
        [en] Predicts the likelihood of an investigation occurring.
        """
        if context:
            return self.get_modified_probability(context)
        return self.probability


@dataclass
class EmotionChainLink(BaseLink):
    """
    [ru] Связь между эмоциями: эмоция1 → эмоция2. Например: Обида → Ненависть
    [en] The relationship between emotions: emotion 1 → emotion 2. For example: Resentment → Hatred
    """
    intensity_amplification: float = 1.0  # Усиление/ослабление интенсивности
    threshold: float = 0.3  # Минимальная интенсивность для перехода

    def get_transition_intensity(self, source_intensity: float) -> float:
        """
        [ru] Вычисляет интенсивность целевой эмоции.
        [en] Calculates the intensity of the target emotion.
        """
        if source_intensity < self.threshold:
            return 0.0

        return min(1.0, source_intensity * self.intensity_amplification)

    def can_transition(self, source_intensity: float) -> bool:
        """
        [ru] Проверяет, возможен ли переход.
        [en] Checks if the transition is possible.
        """
        return source_intensity >= self.threshold

    def update_from_experience(self, source_intensity: float,
                               target_intensity: float,
                               success: bool):
        """
        [ru] Обновляет связь на основе опыта перехода.
        [en] Updates the connection based on the transition experience.
        """
        self.record_usage(success)

        if success:
            # Успешный переход: усиливаем связь
            self.weight = min(1.0, self.weight + 0.05)
            self.intensity_amplification = min(
                1.5,
                self.intensity_amplification + 0.02
            )
            self.threshold = max(0.1, self.threshold - 0.01)
        else:
            # Неудачный переход: ослабляем связь
            self.weight = max(0.1, self.weight - 0.05)
            self.intensity_amplification = max(
                0.5,
                self.intensity_amplification - 0.02
            )
            self.threshold = min(0.5, self.threshold + 0.01)

        self.updated_at = datetime.now()


@dataclass
class EventEmotionLink(BaseLink):
    """
    [ru] Связь между событием и эмоцией. обытие → Эмоциональная реакция
    [en] The connection between an event and an emotion. Event → Emotional reaction
    """
    probability: float = 0.5
    intensity_factor: float = 1.0  # Множитель интенсивности
    valence_shift: float = 0.0  # Сдвиг валентности

    # Условия для активации
    conditions: List[Dict[str, Any]] = field(default_factory=list)

    def get_response_intensity(self, event_intensity: float) -> float:
        """
        [ru] Вычисляет интенсивность эмоциональной реакции.
        [en] Calculates the intensity of an emotional response.
        """
        return min(1.0, event_intensity * self.intensity_factor)

    def get_valence(self, base_valence: float) -> float:
        """
        [ru] Вычисляет итоговую валентность.
        [en] Calculates the final valence.
        """
        return min(1.0, max(-1.0, base_valence + self.valence_shift))

    def is_triggered(self, context: Dict[str, Any]) -> bool:
        """
        [ru] Проверяет, активируется ли связь в данном контексте.
        [en] Checks whether the connection is activated in the given context.
        """
        if not self.conditions:
            return True

        for condition in self.conditions:
            key = condition.get('key')
            expected = condition.get('value')
            operator = condition.get('operator', 'eq')

            if key not in context:
                return False

            actual = context[key]

            if operator == 'eq':
                if actual != expected:
                    return False
            elif operator == 'gt':
                if actual <= expected:
                    return False
            elif operator == 'lt':
                if actual >= expected:
                    return False
            elif operator == 'range':
                if not (expected[0] <= actual <= expected[1]):
                    return False

        return True

    def update_from_experience(self, triggered: bool,
                               intensity: float,
                               success: bool):
        """
        [ru] Обновляет связь на основе опыта.
        [en] Updates the connection based on experience.
        """
        self.record_usage(success)

        if triggered:
            # [ru] Связь сработала
            # [en] The connection worked.
            self.probability = min(1.0, self.probability + 0.05)

            # [ru] Корректируем интенсивность
            # [en] Adjusting the intensity
            if success:
                self.intensity_factor = min(
                    1.5,
                    self.intensity_factor + 0.02
                )
            else:
                self.intensity_factor = max(
                    0.5,
                    self.intensity_factor - 0.02
                )
        else:
            # [ru] Связь не сработала
            # [en] The connection didn't work.
            self.probability = max(0.0, self.probability - 0.05)

        self.updated_at = datetime.now()


@dataclass
class EmotionEventLink(BaseLink):
    """
    [ru] Связь между эмоцией и действием (событием). Эмоция → Действие (Событие)
    [en] The relationship between emotion and action (event). Emotion → Action (Event)
    """
    probability: float = 0.5
    action_urgency: float = 0.5  # Срочность действия
    action_duration: float = 1.0  # Продолжительность действия

    def get_urgency(self, emotion_intensity: float) -> float:
        """
        [ru] Вычисляет срочность действия на основе интенсивности эмоции.
        [en] Calculates the urgency of action based on the intensity of emotion.
        """
        return min(1.0, emotion_intensity * self.action_urgency)

    def is_action_triggered(self, emotion_intensity: float) -> bool:
        """
        [ru] Проверяет, достаточно ли сильна эмоция для действия.
        [en] Checks if the emotion is strong enough to trigger action.
        """
        return emotion_intensity >= 0.3

    def update_from_experience(self, emotion_intensity: float,
                               action_taken: bool,
                               success: bool):
        """
        [ru] Обновляет связь на основе опыта.
        [en] Updates the connection based on experience.
        """
        self.record_usage(success)

        if action_taken:
            self.probability = min(1.0, self.probability + 0.05)

            if success:
                self.action_urgency = min(1.0, self.action_urgency + 0.02)
            else:
                self.action_urgency = max(0.1, self.action_urgency - 0.02)
        else:
            self.probability = max(0.0, self.probability - 0.05)

        self.updated_at = datetime.now()


# [ru] Фабрика для создания связей
# [en] Factory for creating connections
class LinkFactory:
    """
    [ru] Создает связи разных типов.
    [en] Creates connections of different types.
    """

    @staticmethod
    def create_causal_link(source_id: str, target_id: str,
                           weight: float = 1.0,
                           delay: float = 0.0,
                           probability: float = 0.5) -> CausalLink:
        return CausalLink(
            id=f"causal_{source_id}_{target_id}",
            source_id=source_id,
            target_id=target_id,
            weight=weight,
            delay=delay,
            probability=probability
        )

    @staticmethod
    def create_emotion_chain_link(source_id: str, target_id: str,
                                  weight: float = 1.0,
                                  intensity_amplification: float = 1.0,
                                  threshold: float = 0.3) -> EmotionChainLink:
        return EmotionChainLink(
            id=f"emotion_chain_{source_id}_{target_id}",
            source_id=source_id,
            target_id=target_id,
            weight=weight,
            intensity_amplification=intensity_amplification,
            threshold=threshold
        )

    @staticmethod
    def create_event_emotion_link(source_id: str, target_id: str,
                                  weight: float = 1.0,
                                  probability: float = 0.5,
                                  intensity_factor: float = 1.0) -> EventEmotionLink:
        return EventEmotionLink(
            id=f"event_emotion_{source_id}_{target_id}",
            source_id=source_id,
            target_id=target_id,
            weight=weight,
            probability=probability,
            intensity_factor=intensity_factor
        )

    @staticmethod
    def create_emotion_event_link(source_id: str, target_id: str,
                                  weight: float = 1.0,
                                  probability: float = 0.5,
                                  action_urgency: float = 0.5) -> EmotionEventLink:
        return EmotionEventLink(
            id=f"emotion_event_{source_id}_{target_id}",
            source_id=source_id,
            target_id=target_id,
            weight=weight,
            probability=probability,
            action_urgency=action_urgency
        )