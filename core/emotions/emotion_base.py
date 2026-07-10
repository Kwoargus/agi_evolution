# core/emotions/emotion_base.py
import uuid
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field  # <-- ДОБАВЛЯЕМ field
from typing import Optional, Dict, Any
import numpy as np
from enum import Enum


class EmotionType(Enum):
    """Типы эмоций."""
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
#     """Базовые типы эмоций."""
#     JOY = "joy"  # Радость
#     SADNESS = "sadness"  # Печаль
#     ANGER = "anger"  # Гнев
#     FEAR = "fear"  # Страх
#     SURPRISE = "surprise"  # Удивление
#     DISGUST = "disgust"  # Отвращение
#     TRUST = "trust"  # Доверие
#     ANTICIPATION = "anticipation"  # Ожидание
#
#     # Составные эмоции
#     LOVE = "love"  # Радость + Доверие
#     OPTIMISM = "optimism"  # Радость + Ожидание
#     AWE = "awe"  # Страх + Удивление
#     CONTEMPT = "contempt"  # Гнев + Отвращение
#
#     # Сложные эмоции
#     RESENTMENT = "resentment"  # Обида
#     HATRED = "hatred"  # Ненависть
#     GUILT = "guilt"  # Вина
#     SHAME = "shame"  # Стыд
#     EMPATHY = "empathy"  # Эмпатия


@dataclass
class EmotionalResponse:
    """
    Эмоциональная реакция с UUID.
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
        """Преобразует в словарь для БД."""
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
    Событие, вызывающее эмоциональную реакцию.
    """
    id: str
    description: str
    timestamp: float
    context: Dict[str, any]
    participants: List[str]

    # Векторное представление события
    embedding: np.ndarray  # 128-dim


@dataclass
class EmotionalResponse:
    """
    Эмоциональная реакция (ЭР).
    """
    emotion_type: EmotionType
    intensity: float  # 0.0 - 1.0
    valence: float  # -1.0 (негативная) до +1.0 (позитивная)
    arousal: float  # 0.0 (спокойная) до 1.0 (возбужденная)

    # Векторное представление ЭР
    embedding: np.ndarray  # 64-dim

    # Причина возникновения
    trigger_event_id: Optional[str] = None
    trigger_emotion_id: Optional[str] = None

    # Цепочка предшествующих ЭР
    chain_id: Optional[str] = None


@dataclass
class MentalModel:
    """
    Ментальная модель объекта/ситуации/процесса.
    """
    id: str
    name: str
    type: str  # 'object', 'situation', 'process', 'social'

    # Векторное представление модели
    embedding: np.ndarray  # 256-dim

    # Атрибуты модели
    attributes: Dict[str, float]

    # Связи с другими моделями
    related_models: List[str]

    # Прогностические свойства
    predictions: List[Dict[str, any]]