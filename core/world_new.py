# core/world_new.py
import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import Enum


class ObjectType(Enum):
    EXPLOSION = "explosion"
    FIRE = "fire"
    FLOOD = "flood"
    FOOD = "food"
    PREDATOR = "predator"
    SHELTER = "shelter"


class DangerousObject:
    """Опасный объект с мультимодальными характеристиками."""

    def __init__(self, x: float, z: float, obj_type: ObjectType,
                 vision_signature: str, sound_signature: str, smell_signature: str,
                 danger_level: float = 0.5):

        self.x = x
        self.z = z
        self.type = obj_type
        self.danger_level = danger_level

        # Векторные представления для распознавания
        self.vision_vector = self._generate_vision_vector(vision_signature)
        self.sound_vector = self._generate_sound_vector(sound_signature)
        self.smell_vector = self._generate_smell_vector(smell_signature)

        self.active = True
        self.lifetime = 100  # количество шагов до исчезновения

    def _generate_vision_vector(self, signature: str) -> np.ndarray:
        """Генерирует векторное представление визуального образа."""
        # В реальном приложении здесь была бы CNN для обработки изображений
        base = np.random.randn(64) * 0.1
        # Кодируем сигнатуру
        signatures = {
            'bright_flash': np.array([1.0, 0.0, 0.0, 0.0]),
            'red_glow': np.array([0.0, 1.0, 0.0, 0.0]),
            'dark_shape': np.array([0.0, 0.0, 1.0, 0.0]),
            'moving_shape': np.array([0.0, 0.0, 0.0, 1.0]),
        }
        if signature in signatures:
            base[:4] = signatures[signature]
        return base

    def _generate_sound_vector(self, signature: str) -> np.ndarray:
        """Генерирует векторное представление звука."""
        base = np.random.randn(32) * 0.1
        signatures = {
            'loud_crash': np.array([1.0, 0.0, 0.0, 0.0]),
            'hiss': np.array([0.0, 1.0, 0.0, 0.0]),
            'roar': np.array([0.0, 0.0, 1.0, 0.0]),
            'crackle': np.array([0.0, 0.0, 0.0, 1.0]),
        }
        if signature in signatures:
            base[:4] = signatures[signature]
        return base

    def _generate_smell_vector(self, signature: str) -> np.ndarray:
        """Генерирует векторное представление запаха."""
        base = np.random.randn(16) * 0.1
        signatures = {
            'smoke': np.array([1.0, 0.0, 0.0, 0.0]),
            'food_smell': np.array([0.0, 1.0, 0.0, 0.0]),
            'predator_smell': np.array([0.0, 0.0, 1.0, 0.0]),
            'water': np.array([0.0, 0.0, 0.0, 1.0]),
        }
        if signature in signatures:
            base[:4] = signatures[signature]
        return base

    def get_sensory_data(self) -> Dict:
        """Возвращает сенсорные данные объекта."""
        return {
            'vision': self.vision_vector,
            'sound': self.sound_vector,
            'smell': self.smell_vector,
            'position': (self.x, self.z),
            'danger_level': self.danger_level,
            'type': self.type.value
        }


class NewWorld:
    """Новый мир с разнообразными объектами."""

    def __init__(self, size: int = 30):
        self.size = size
        self.objects: List[DangerousObject] = []
        self._populate_world()

    def _populate_world(self):
        """Заполняет мир объектами."""
        # Опасные объекты
        dangerous_objects = [
            (ObjectType.EXPLOSION, 'bright_flash', 'loud_crash', 'smoke', 0.9),
            (ObjectType.FIRE, 'red_glow', 'crackle', 'smoke', 0.7),
            (ObjectType.FLOOD, 'dark_shape', 'hiss', 'water', 0.6),
            (ObjectType.PREDATOR, 'moving_shape', 'roar', 'predator_smell', 0.8),
        ]

        for obj_type, vision, sound, smell, danger in dangerous_objects:
            for _ in range(3):  # по 3 объекта каждого типа
                x = np.random.randint(-self.size // 2, self.size // 2)
                z = np.random.randint(-self.size // 2, self.size // 2)
                obj = DangerousObject(x, z, obj_type, vision, sound, smell, danger)
                self.objects.append(obj)

        # Полезные объекты (еда)
        for _ in range(5):
            x = np.random.randint(-self.size // 2, self.size // 2)
            z = np.random.randint(-self.size // 2, self.size // 2)
            obj = DangerousObject(x, z, ObjectType.FOOD,
                                  'bright_shape', 'sizzle', 'food_smell', 0.0)
            self.objects.append(obj)

    def get_objects_in_range(self, x: float, z: float, radius: float = 10.0) -> List[DangerousObject]:
        """Возвращает объекты в радиусе."""
        result = []
        for obj in self.objects:
            dist = np.sqrt((obj.x - x) ** 2 + (obj.z - z) ** 2)
            if dist <= radius and obj.active:
                result.append(obj)
        return result