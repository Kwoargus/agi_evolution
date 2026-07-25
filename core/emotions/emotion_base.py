# core/emotions/emotion_base.py
import uuid
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field  # <-- ДОБАВЛЯЕМ field
from typing import Optional, Dict, Any
import numpy as np
from enum import Enum
import time


class EmotionType(Enum):
    """[ru] Типы эмоций."""
    """[en] Types of emotions."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    LOVE = "love"
    OPTIMISM = "optimism"
    AWE = "awe"
    CONTEMPT = "contempt"
    RESENTMENT = "resentment"
    HATRED = "hatred"
    GUILT = "guilt"
    SHAME = "shame"
    EMPATHY = "empathy"

# class EmotionType(Enum):
#     """[ru] Базовые типы эмоций."""
#     """[en] Basic types of emotions."""
#     JOY = "joy"  # Радость
#     SADNESS = "sadness"  # Печаль
#     ANGER = "anger"  # Гнев
#     FEAR = "fear"  # Страх
#     SURPRISE = "surprise"  # Удивление
#     DISGUST = "disgust"  # Отвращение
#     TRUST = "trust"  # Доверие
#     ANTICIPATION = "anticipation"  # Ожидание
#
#     # [ru] Составные эмоции
#     # [en] Compound emotions
#     LOVE = "love"  # Радость + Доверие
#     OPTIMISM = "optimism"  # Радость + Ожидание
#     AWE = "awe"  # Страх + Удивление
#     CONTEMPT = "contempt"  # Гнев + Отвращение
#
#     # [ru] Сложные эмоции
#     # [en] Complex emotions
#     RESENTMENT = "resentment"  # Обида
#     HATRED = "hatred"  # Ненависть
#     GUILT = "guilt"  # Вина
#     SHAME = "shame"  # Стыд
#     EMPATHY = "empathy"  # Эмпатия


@dataclass
class EmotionalResponse:
    """
    [ru] Эмоциональная реакция с UUID.
    [en] Emotional reaction with UUID.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    emotion_type: EmotionType = EmotionType.TRUST
    intensity: float = 0.3
    valence: float = 0.0
    arousal: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    source: str = "inherited"  # 'inherited', 'learned', 'generated'
    embedding: Optional[np.ndarray] = None
    trigger_event_id: Optional[str] = None
    chain_id: Optional[str] = None

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = np.zeros(64)

    def to_dict(self) -> Dict:
        """[ru] Преобразует в словарь для БД."""
        """[en] Converts to a dictionary for the database."""
        return {
            'id': self.id,
            'emotion_type': self.emotion_type.value,
            'intensity': self.intensity,
            'valence': self.valence,
            'arousal': self.arousal,
            'context': self.context,
            'source': self.source,
            'embedding': self.embedding.tolist() if hasattr(self.embedding, 'tolist') else self.embedding,
            'trigger_event_id': self.trigger_event_id,
            'chain_id': self.chain_id
        }


@dataclass
class EmotionalEvent:
    """
    [ru] Событие, вызывающее эмоциональную реакцию.
    [en] An event that evokes an emotional response.
    """
    id: str
    description: str
    timestamp: float
    context: Dict[str, any]
    participants: List[str]

    # [ru] Векторное представление события
    # [en] Vector representation of an event
    embedding: np.ndarray  # 128-dim


@dataclass
class EmotionalResponse:
    """

    [ru] Эмоциональная реакция (ЭР).
    [en] Emotional reaction (ER).
    """
    emotion_type: EmotionType
    intensity: float  # 0.0 - 1.0
    valence: float  # -1.0 (негативная) до +1.0 (позитивная)
    arousal: float  # 0.0 (спокойная) до 1.0 (возбужденная)


    # [ru] Векторное представление эмоциональной реакции
    # [en] Vector representation of emotional response

    embedding: np.ndarray  # 64-dim


    # [ru] Причина возникновения
    # [en] Cause of occurrence
    trigger_event_id: Optional[str] = None
    trigger_emotion_id: Optional[str] = None


    # [ru] Цепочка предшествующих эмоциональных реакцйи
    # [en] Chain of preceding emotional reactions
    chain_id: Optional[str] = None


@dataclass
class MentalModel:
    """
    [ru] Ментальная модель объекта/ситуации/процесса. ЕДИНЫЙ КЛАСС для всей системы.
    [en] Mental model of an object/situation/process. A SINGLE CLASS for the entire system.
    """
    id: str
    name: str
    type: str  # 'object', 'situation', 'process', 'social'


    # [ru] Векторное представление модели
    # [en] Vector representation of the model
    embedding: np.ndarray  # 256-dim


    # [ru] Атрибуты модели
    # [en] Model attributes
    attributes: Dict[str, float]


    # [ru] Связи с другими моделями
    # [en] Links to other models
    related_models: List[str]


    # [ru] Прогностические свойства
    # [en] Prognostic properties
    predictions: List[Dict[str, any]]


    # [ru] Метаданные
    # [en] Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


