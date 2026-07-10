# core/instinct_system.py
import numpy as np
from dataclasses import dataclass
import torch
from typing import List, Dict, Optional, Tuple, Any, Union

# Добавляем импорт MultimodalEncoder
from sensors.multimodal_sensor import MultimodalEncoder
from typing import List, Dict, Optional, Tuple, Any, Union


@dataclass
class SensoryInput:
    vision: np.ndarray  # векторное представление того, что видит бот
    sound: np.ndarray   # векторное представление того, что слышит бот
    smell: np.ndarray   # векторное представление того, что чувствует бот
    position: Tuple[float, float]  # позиция объекта


@dataclass
class InstinctAction:
    action_type: str  # 'run_away', 'approach', 'investigate', 'ignore'
    priority: float
    target_position: Optional[Tuple[float, float]] = None
    confidence: float = 0.0


class InstinctSystem:
    """
    Система инстинктов с мультимодальным восприятием.
    """

    def __init__(self, multimodal_encoder: MultimodalEncoder):
        self.encoder = multimodal_encoder
        self.instinct_patterns: List[Dict] = []
        self.instinct_gan: Optional['InstinctGAN'] = None

    def perceive(self, sensory_input: SensoryInput) -> np.ndarray:
        """
        Преобразует сенсорные данные в единый вектор.
        """
        vision_tensor = torch.FloatTensor(sensory_input.vision).unsqueeze(0)
        sound_tensor = torch.FloatTensor(sensory_input.sound).unsqueeze(0)
        smell_tensor = torch.FloatTensor(sensory_input.smell).unsqueeze(0)

        with torch.no_grad():
            combined = self.encoder(vision_tensor, sound_tensor, smell_tensor)

        return combined.numpy().flatten()

    def get_best_action(self, sensory_input: SensoryInput) -> Optional[InstinctAction]:
        """
        Находит лучшее действие на основе восприятия.
        """
        perception_vector = self.perceive(sensory_input)

        best_action = None
        best_priority = -1.0

        for pattern in self.instinct_patterns:
            # Вычисляем схожесть с паттерном
            similarity = self._compute_similarity(
                perception_vector,
                pattern['combined_pattern']
            )

            if similarity > pattern.get('threshold', 0.7):
                priority = similarity * pattern.get('priority', 1.0)
                if priority > best_priority:
                    best_priority = priority
                    best_action = InstinctAction(
                        action_type=pattern['action_type'],
                        priority=priority,
                        confidence=similarity
                    )

        return best_action

    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Вычисляет косинусное сходство."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))