# core/world_new.py (исправленный)
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
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

        # Округляем координаты до узлов сетки (шаг 2)
        self.x = round(x / 2) * 2
        self.z = round(z / 2) * 2
        self.type = obj_type
        self.danger_level = danger_level

        # Векторные представления для распознавания
        self.vision_vector = self._generate_vision_vector(vision_signature)
        self.sound_vector = self._generate_sound_vector(sound_signature)
        self.smell_vector = self._generate_smell_vector(smell_signature)

        self.active = True
        self.lifetime = 100  # количество шагов до исчезновения
        self.collected = False  # флаг для еды (собрана или нет)

    def _generate_vision_vector(self, signature: str) -> np.ndarray:
        """Генерирует векторное представление визуального образа."""
        base = np.random.randn(64) * 0.1
        signatures = {
            'bright_flash': np.array([1.0, 0.0, 0.0, 0.0]),
            'red_glow': np.array([0.0, 1.0, 0.0, 0.0]),
            'dark_shape': np.array([0.0, 0.0, 1.0, 0.0]),
            'moving_shape': np.array([0.0, 0.0, 0.0, 1.0]),
            'bright_shape': np.array([1.0, 0.5, 0.0, 0.0]),  # для еды
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
            'sizzle': np.array([0.5, 0.5, 0.0, 0.0]),  # для еды
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
            'type': self.type.value,
            'active': self.active,
            'collected': self.collected
        }

    def collect(self) -> bool:
        """Собирает объект (для еды)."""
        if self.type == ObjectType.FOOD and not self.collected:
            self.collected = True
            self.active = False
            return True
        return False


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
            for _ in range(3):
                # Генерируем координаты с шагом 2
                x = np.random.randint(-self.size // 2, self.size // 2)
                x = round(x / 2) * 2
                z = np.random.randint(-self.size // 2, self.size // 2)
                z = round(z / 2) * 2
                obj = DangerousObject(x, z, obj_type, vision, sound, smell, danger)
                self.objects.append(obj)

        # Полезные объекты (еда)
        food_positions = [
            (-8, 8), (8, -8), (-6, -6), (6, 6),
            (0, 10), (10, 0), (-10, 0), (0, -10)
        ]
        for x, z in food_positions[:5]:
            obj = DangerousObject(x, z, ObjectType.FOOD,
                                  'bright_shape', 'sizzle', 'food_smell', 0.0)
            self.objects.append(obj)

    def get_objects_in_range(self, x: float, z: float, radius: float = 10.0) -> List[DangerousObject]:
        """Возвращает активные объекты в радиусе."""
        result = []
        for obj in self.objects:
            if not obj.active:
                continue
            dist = np.sqrt((obj.x - x) ** 2 + (obj.z - z) ** 2)
            if dist <= radius:
                result.append(obj)
        return result

    def get_object_at(self, x: float, z: float) -> Optional[DangerousObject]:
        """Возвращает объект в указанной позиции."""
        for obj in self.objects:
            if not obj.active:
                continue
            # Проверяем, находится ли бот в той же клетке
            if abs(obj.x - x) < 1.0 and abs(obj.z - z) < 1.0:
                return obj
        return None

    def collect_food(self, x: float, z: float) -> bool:
        """Собирает еду в указанной позиции."""
        obj = self.get_object_at(x, z)
        if obj and obj.type == ObjectType.FOOD:
            return obj.collect()
        return False

    def get_active_objects(self) -> List[DangerousObject]:
        """Возвращает список активных объектов."""
        return [obj for obj in self.objects if obj.active]

    def reset(self):
        """Сбрасывает мир (восстанавливает все объекты)."""
        for obj in self.objects:
            obj.active = True
            obj.collected = False



# # core/world_new.py
# import numpy as np
# from typing import List, Dict, Optional, Tuple, Any, Union
# from enum import Enum
#
#
# class ObjectType(Enum):
#     EXPLOSION = "explosion"
#     FIRE = "fire"
#     FLOOD = "flood"
#     FOOD = "food"
#     PREDATOR = "predator"
#     SHELTER = "shelter"
#
#
# class DangerousObject:
#     """Опасный объект с мультимодальными характеристиками."""
#
#     def __init__(self, x: float, z: float, obj_type: ObjectType,
#                  vision_signature: str, sound_signature: str, smell_signature: str,
#                  danger_level: float = 0.5):
#
#         self.x = x
#         self.z = z
#         self.type = obj_type
#         self.danger_level = danger_level
#
#         # Векторные представления для распознавания
#         self.vision_vector = self._generate_vision_vector(vision_signature)
#         self.sound_vector = self._generate_sound_vector(sound_signature)
#         self.smell_vector = self._generate_smell_vector(smell_signature)
#
#         self.active = True
#         self.lifetime = 100  # количество шагов до исчезновения
#
#     def _generate_vision_vector(self, signature: str) -> np.ndarray:
#         """Генерирует векторное представление визуального образа."""
#         # В реальном приложении здесь была бы CNN для обработки изображений
#         base = np.random.randn(64) * 0.1
#         # Кодируем сигнатуру
#         signatures = {
#             'bright_flash': np.array([1.0, 0.0, 0.0, 0.0]),
#             'red_glow': np.array([0.0, 1.0, 0.0, 0.0]),
#             'dark_shape': np.array([0.0, 0.0, 1.0, 0.0]),
#             'moving_shape': np.array([0.0, 0.0, 0.0, 1.0]),
#         }
#         if signature in signatures:
#             base[:4] = signatures[signature]
#         return base
#
#     def _generate_sound_vector(self, signature: str) -> np.ndarray:
#         """Генерирует векторное представление звука."""
#         base = np.random.randn(32) * 0.1
#         signatures = {
#             'loud_crash': np.array([1.0, 0.0, 0.0, 0.0]),
#             'hiss': np.array([0.0, 1.0, 0.0, 0.0]),
#             'roar': np.array([0.0, 0.0, 1.0, 0.0]),
#             'crackle': np.array([0.0, 0.0, 0.0, 1.0]),
#         }
#         if signature in signatures:
#             base[:4] = signatures[signature]
#         return base
#
#     def _generate_smell_vector(self, signature: str) -> np.ndarray:
#         """Генерирует векторное представление запаха."""
#         base = np.random.randn(16) * 0.1
#         signatures = {
#             'smoke': np.array([1.0, 0.0, 0.0, 0.0]),
#             'food_smell': np.array([0.0, 1.0, 0.0, 0.0]),
#             'predator_smell': np.array([0.0, 0.0, 1.0, 0.0]),
#             'water': np.array([0.0, 0.0, 0.0, 1.0]),
#         }
#         if signature in signatures:
#             base[:4] = signatures[signature]
#         return base
#
#     def get_sensory_data(self) -> Dict:
#         """Возвращает сенсорные данные объекта."""
#         return {
#             'vision': self.vision_vector,
#             'sound': self.sound_vector,
#             'smell': self.smell_vector,
#             'position': (self.x, self.z),
#             'danger_level': self.danger_level,
#             'type': self.type.value
#         }
#
#
# class NewWorld:
#     """Новый мир с разнообразными объектами."""
#
#     def __init__(self, size: int = 30):
#         self.size = size
#         self.objects: List[DangerousObject] = []
#         self._populate_world()
#
#     def _populate_world(self):
#         """Заполняет мир объектами."""
#         # Опасные объекты
#         dangerous_objects = [
#             (ObjectType.EXPLOSION, 'bright_flash', 'loud_crash', 'smoke', 0.9),
#             (ObjectType.FIRE, 'red_glow', 'crackle', 'smoke', 0.7),
#             (ObjectType.FLOOD, 'dark_shape', 'hiss', 'water', 0.6),
#             (ObjectType.PREDATOR, 'moving_shape', 'roar', 'predator_smell', 0.8),
#         ]
#
#         for obj_type, vision, sound, smell, danger in dangerous_objects:
#             for _ in range(3):  # по 3 объекта каждого типа
#                 x = np.random.randint(-self.size // 2, self.size // 2)
#                 z = np.random.randint(-self.size // 2, self.size // 2)
#                 obj = DangerousObject(x, z, obj_type, vision, sound, smell, danger)
#                 self.objects.append(obj)
#
#         # Полезные объекты (еда)
#         for _ in range(5):
#             x = np.random.randint(-self.size // 2, self.size // 2)
#             z = np.random.randint(-self.size // 2, self.size // 2)
#             obj = DangerousObject(x, z, ObjectType.FOOD,
#                                   'bright_shape', 'sizzle', 'food_smell', 0.0)
#             self.objects.append(obj)
#
#     def get_objects_in_range(self, x: float, z: float, radius: float = 10.0) -> List[DangerousObject]:
#         """Возвращает объекты в радиусе."""
#         result = []
#         for obj in self.objects:
#             dist = np.sqrt((obj.x - x) ** 2 + (obj.z - z) ** 2)
#             if dist <= radius and obj.active:
#                 result.append(obj)
#         return result