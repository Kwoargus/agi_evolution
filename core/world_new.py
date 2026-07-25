# core/world.py
import pygame
import math
import random
from core.objects import GameObject, Predator, Food, Explosion
from typing import List, Dict, Optional, Tuple, Any, Union

class World:
    def __init__(self, width=1200, height=800, world_size=20, cell_size=40, objects_config=None):
        self.width = width
        self.height = height
        self.world_size = world_size
        self.cell_size = cell_size
        self.objects = []
        # [ru] для взрывов (если будут)
        # [en] for explosions (if any)
        self.explosions = []

        # [ru] Параметры камеры (для визуализации)
        # [en] Camera parameters (for visualization)
        self.cam_x = 0.0
        self.cam_z = 0.0
        self.yaw = 45.0
        self.pitch = 30.0
        self.zoom = 1.0

        # [ru] Цвета
        # [en] Colors
        self.COLOR_BG = (30, 30, 30)
        self.COLOR_GRID = (200, 200, 200)
        self.COLOR_MARKER = (255, 100, 100)

        # [ru] Для работы с экраном
        # [en] For working with the screen
        self.screen = None

        # [ru] список взрывов
        # [en] list of explosions
        self.explosions = []
        self.explosion_timer = 10.0
        self.bot = None

        # [ru] Если задана конфигурация объектов, создаём их
        # [en] If object configuration is provided, create them
        if objects_config:
            for obj_data in objects_config:
                obj_type = obj_data.get('type')
                x = obj_data.get('x')
                z = obj_data.get('z')
                if obj_type == 'fire':
                    self.objects.append(GameObject(x, z, obj_type='fire', temperature=800))
                elif obj_type == 'food':
                    self.objects.append(Food(x, z, name='apple', obj_type='food', smell='food_smell'))
                elif obj_type == 'predator':
                    self.objects.append(Predator(x, z, name='wolf', obj_type='predator'))
                # [ru] можно добавить другие типы
                # [en] can add other types

    def reset(self, bot_start_pos=(0, 0)):
        """
        [ru] Сбрасывает мир в начальное состояние (для нового эпизода).
        [en] Resets the world to its initial state (for a new episode).
        """
        # [ru] Пока просто ничего не делаем
        # [en] For now, just do nothing
        pass

    def get_state(self, bot):
        """
        [ru] Возвращает состояние мира для бота.
        [en] Returns the world state for the bot.
        [ru] РАЗМЕРНОСТЬ ВСЕГДА 21: [x, z] + 6 объектов × 3 + 1 резерв
        [en] DIMENSION IS ALWAYS 21: [x, z] + 6 objects × 3 + 1 reserve
        """
        half = self.world_size / 2.0

        # [ru] Если бот не передан, возвращаем нулевое состояние
        # [en] If bot is not provided, return zero state
        if bot is None:
            return [0.0] * 21

        state = [bot.x / half, bot.z / half]  # 2 значения

        # [ru] Берем ровно 6 объектов из мира (или меньше)
        # [en] Take exactly 6 objects from the world (or fewer)
        objects_list = list(self.objects)[:6]

        # [ru] Добавляем информацию о каждом объекте
        # [en] Add information about each object
        for obj in objects_list:
            if obj is None:
                state.extend([0.0, 0.0, 0.0])
            else:
                # [ru] Кодируем тип объекта числом
                # [en] Encode object type as a number
                if hasattr(obj, 'type'):
                    # [ru] Если type - строка, преобразуем в число
                    # [en] If type is a string, convert to number
                    if isinstance(obj.type, str):
                        if obj.type == 'food' or obj.type == 'Food':
                            type_code = 2.0
                        elif obj.type == 'predator' or obj.type == 'Predator':
                            type_code = 3.0
                        elif obj.type == 'fire' or obj.type == 'Fire':
                            type_code = 1.0
                        else:
                            type_code = 4.0  # [ru] неизвестный тип  [en] unknown type
                    else:
                        type_code = float(obj.type)
                elif isinstance(obj, Food):
                    type_code = 2.0
                elif isinstance(obj, Predator):
                    type_code = 3.0
                elif isinstance(obj, GameObject):
                    if hasattr(obj, 'type') and obj.type == 'fire':
                        type_code = 1.0
                    else:
                        type_code = 4.0
                else:
                    type_code = 0.0

                dx = (obj.x - bot.x) / half
                dz = (obj.z - bot.z) / half
                state.extend([float(type_code), float(dx), float(dz)])

        # [ru] Если объектов меньше 6, дополняем нулями
        # [en] If fewer than 6 objects, pad with zeros
        while len(objects_list) < 6:
            state.extend([0.0, 0.0, 0.0])
            objects_list.append(None)

        # [ru] У нас должно быть 2 + 6*3 = 20 значений. Добавляем 1 резервный ноль для размерности 21
        # [en] We should have 2 + 6*3 = 20 values. Add 1 reserve zero for dimension 21
        state.append(0.0)

        # [ru] Гарантируем ровно 21
        # [en] Guarantee exactly 21
        while len(state) < 21:
            state.append(0.0)
        state = state[:21]

        return state


    def get_scale(self):
        return self.cell_size * self.zoom

    def is_within_world(self, x, z):
        half = self.world_size // 2
        return -half <= x <= half and -half <= z <= half

    def get_object_at(self, x, z):
        for obj in self.objects:
            if hasattr(obj, 'size'):
                half = obj.size * 0.5
                if abs(x - obj.x) < half and abs(z - obj.z) < half:
                    return obj
            elif hasattr(obj, 'body_w') and hasattr(obj, 'body_d'):
                half_w = obj.body_w / 2
                half_d = obj.body_d / 2
                if abs(x - obj.x) < half_w and abs(z - obj.z) < half_d:
                    return obj
            elif hasattr(obj, 'radius'):
                if abs(x - obj.x) < obj.radius and abs(z - obj.z) < obj.radius:
                    return obj
        return None

    def add_object(self, obj):
        self.objects.append(obj)

    def remove_object(self, obj):
        if obj in self.objects:
            self.objects.remove(obj)
            return True
        return False

    # [ru] ---------- Визуализация ----------
    # [en] ---------- Visualization ----------
    def world_to_screen(self, wx, wz, wy=0):
        dx = wx - self.cam_x
        dz = wz - self.cam_z

        rad_yaw = math.radians(self.yaw)
        cos_y = math.cos(rad_yaw)
        sin_y = math.sin(rad_yaw)
        dx_rot = dx * cos_y + dz * sin_y
        dz_rot = -dx * sin_y + dz * cos_y

        rad_pitch = math.radians(self.pitch)
        sin_p = math.sin(rad_pitch)
        cos_p = math.cos(rad_pitch)

        x_proj = dx_rot
        y_proj = -dz_rot * sin_p - wy * cos_p

        scale = self.cell_size * self.zoom
        screen_x = self.width // 2 + x_proj * scale
        screen_y = self.height // 2 + y_proj * scale
        return int(screen_x), int(screen_y)

    def draw_grid(self, screen):
        half = self.world_size // 2
        for z in range(-half, half + 1, 2):
            x1, y1 = self.world_to_screen(-half, z, 0)
            x2, y2 = self.world_to_screen(half, z, 0)
            pygame.draw.line(screen, self.COLOR_GRID, (x1, y1), (x2, y2), 1)
        for x in range(-half, half + 1, 2):
            x1, y1 = self.world_to_screen(x, -half, 0)
            x2, y2 = self.world_to_screen(x, half, 0)
            pygame.draw.line(screen, self.COLOR_GRID, (x1, y1), (x2, y2), 1)

        for x in range(-half, half + 1, 2):
            for z in range(-half, half + 1, 2):
                if (x, z) == (0, 0):
                    continue
                px, py = self.world_to_screen(x, z, 0)
                pygame.draw.circle(screen, self.COLOR_MARKER, (px, py), 4)
        cx, cy = self.world_to_screen(0, 0, 0)
        pygame.draw.circle(screen, (255, 255, 0), (cx, cy), 6)

    def draw_ui(self, screen, bot=None):
        font = pygame.font.Font(None, 24)
        lines = [
            f"AGI Evolution",
            f"Cam: ({self.cam_x:.1f}, {self.cam_z:.1f})  Yaw: {self.yaw:.1f}°  Pitch: {self.pitch:.1f}°  Zoom: {self.zoom:.2f}",
            f"Objects: {len(self.objects)}",
            "Arrows: move | A/D: rotate | W/S: tilt | Scroll: zoom | Space: reset | Esc: exit"
        ]
        if bot:
            lines.append(f"Bot pos: ({bot.x:.1f}, {bot.z:.1f})  Steps: {len(bot.visited_nodes)-1}")
        y = 10
        for line in lines:
            text = font.render(line, True, (255, 255, 255))
            screen.blit(text, (10, y))
            y += 25

    def draw(self, screen, bot=None):
        screen.fill(self.COLOR_BG)
        self.draw_grid(screen)

        for obj in self.objects:
            obj.draw(screen, self.world_to_screen)

        # [ru] Отрисовка взрывов (если есть)
        # [en] Draw explosions (if any)
        for expl in self.explosions:
            expl.draw(screen, self.world_to_screen)

        if bot:
            bot.draw(screen, self.world_to_screen, self.get_scale())

        self.draw_ui(screen, bot)
        pygame.display.flip()

    def trigger_explosion(self):
        print("[ru] Взрыв создан!")
        print("[en] Explosion created!")
        corners = [(-8, -8), (-8, 8), (8, -8), (8, 8)]
        x, z = random.choice(corners)
        print(f"[ru] Взрыв в точке ({x}, {z})")  # [ru] отладка
        print(f"[en] Explosion at point ({x}, {z})")  # [en] debugging
        explosion = Explosion(x, z)
        self.explosions.append(explosion)
        if self.bot:
            print(f"[Explosion] triggered at ({x}, {z}), notifying bot")  # [ru] <-- исправлено  [en] <-- fixed
            self.bot.notify('explosion', {
                'sound': 'loud crash',
                'vision': 'bright_flash',
                'position': (x, z)
            })

    # [ru] ---------- Запуск (для визуализации) ----------
    # [en] ---------- Run (for visualization) ----------
    def run(self, bot=None):

        # [ru] сохраняем ссылку
        # [en] save reference
        self.bot = bot

        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("AGI Evolution")
        clock = pygame.time.Clock()
        self.screen = screen

        running = True
        while running:
            # [ru] Обработка событий (камера, выход)
            # [en] Event processing (camera, exit)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_SPACE:
                        self.cam_x, self.cam_z = 0.0, 0.0
                        self.yaw = 45.0
                        self.pitch = 30.0
                        self.zoom = 1.0
                if event.type == pygame.MOUSEWHEEL:
                    self.zoom += event.y * 0.1
                    self.zoom = max(0.2, min(3.0, self.zoom))

            # [ru] Управление камерой (клавиши)
            # [en] Camera control (keys)
            keys = pygame.key.get_pressed()
            speed = 0.3 / self.zoom
            if keys[pygame.K_UP]:
                self.cam_x += math.sin(math.radians(self.yaw)) * speed
                self.cam_z += math.cos(math.radians(self.yaw)) * speed
            if keys[pygame.K_DOWN]:
                self.cam_x -= math.sin(math.radians(self.yaw)) * speed
                self.cam_z -= math.cos(math.radians(self.yaw)) * speed
            if keys[pygame.K_LEFT]:
                self.cam_x -= math.cos(math.radians(self.yaw)) * speed
                self.cam_z += math.sin(math.radians(self.yaw)) * speed
            if keys[pygame.K_RIGHT]:
                self.cam_x += math.cos(math.radians(self.yaw)) * speed
                self.cam_z -= math.sin(math.radians(self.yaw)) * speed
            if keys[pygame.K_a]:
                self.yaw -= 2.0
            if keys[pygame.K_d]:
                self.yaw += 2.0
            if keys[pygame.K_w]:
                self.pitch += 1.0
            if keys[pygame.K_s]:
                self.pitch -= 1.0
            self.pitch = max(10.0, min(85.0, self.pitch))

            # [ru] Обновление бота (если есть)
            # [en] Update bot (if exists)
            if bot:
                bot.update(self)

            # [ru] Отрисовка
            # [en] Drawing
            self.draw(screen, bot)
            clock.tick(60)

            # [ru] Обновление таймера взрыва
            # [en] Update explosion timer
            if self.explosion_timer > 0:
                self.explosion_timer -= 1 / 60
                if self.explosion_timer <= 0:
                    self.trigger_explosion()

            # [ru] Обновление взрывов
            # [en] Update explosions
            for expl in self.explosions[:]:
                expl.update(1 / 60)
                if not expl.active:
                    self.explosions.remove(expl)

        pygame.quit()



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
#     """
#     Опасный объект с мультимодальными характеристиками.
#     """
#
#     def __init__(self, x: float, z: float, obj_type: ObjectType,
#                  vision_signature: str, sound_signature: str, smell_signature: str,
#                  danger_level: float = 0.5):
#
#         # Округляем координаты до узлов сетки (шаг 2)
#         self.x = round(x / 2) * 2
#         self.z = round(z / 2) * 2
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
#         self.collected = False  # флаг для еды (собрана или нет)
#
#     def _generate_vision_vector(self, signature: str) -> np.ndarray:
#         """
#         Генерирует векторное представление визуального образа.
#         """
#         base = np.random.randn(64) * 0.1
#         signatures = {
#             'bright_flash': np.array([1.0, 0.0, 0.0, 0.0]),
#             'red_glow': np.array([0.0, 1.0, 0.0, 0.0]),
#             'dark_shape': np.array([0.0, 0.0, 1.0, 0.0]),
#             'moving_shape': np.array([0.0, 0.0, 0.0, 1.0]),
#             'bright_shape': np.array([1.0, 0.5, 0.0, 0.0]),  # для еды
#         }
#         if signature in signatures:
#             base[:4] = signatures[signature]
#         return base
#
#     def _generate_sound_vector(self, signature: str) -> np.ndarray:
#         """
#         Генерирует векторное представление звука.
#         """
#         base = np.random.randn(32) * 0.1
#         signatures = {
#             'loud_crash': np.array([1.0, 0.0, 0.0, 0.0]),
#             'hiss': np.array([0.0, 1.0, 0.0, 0.0]),
#             'roar': np.array([0.0, 0.0, 1.0, 0.0]),
#             'crackle': np.array([0.0, 0.0, 0.0, 1.0]),
#             'sizzle': np.array([0.5, 0.5, 0.0, 0.0]),  # для еды
#         }
#         if signature in signatures:
#             base[:4] = signatures[signature]
#         return base
#
#     def _generate_smell_vector(self, signature: str) -> np.ndarray:
#         """
#         Генерирует векторное представление запаха.
#         """
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
#         """
#         Возвращает сенсорные данные объекта.
#         """
#         return {
#             'vision': self.vision_vector,
#             'sound': self.sound_vector,
#             'smell': self.smell_vector,
#             'position': (self.x, self.z),
#             'danger_level': self.danger_level,
#             'type': self.type.value,
#             'active': self.active,
#             'collected': self.collected
#         }
#
#     def collect(self) -> bool:
#         """
#         Собирает объект (для еды).
#         """
#         if self.type == ObjectType.FOOD and not self.collected:
#             self.collected = True
#             self.active = False
#             return True
#         return False
#
#
# class NewWorld:
#     """
#     Новый мир с разнообразными объектами.
#     """
#
#     def __init__(self, size: int = 30):
#         self.size = size
#         self.objects: List[DangerousObject] = []
#         self._populate_world()
#
#     def _populate_world(self):
#         """
#         Заполняет мир объектами.
#         """
#         # Опасные объекты
#         dangerous_objects = [
#             (ObjectType.EXPLOSION, 'bright_flash', 'loud_crash', 'smoke', 0.9),
#             (ObjectType.FIRE, 'red_glow', 'crackle', 'smoke', 0.7),
#             (ObjectType.FLOOD, 'dark_shape', 'hiss', 'water', 0.6),
#             (ObjectType.PREDATOR, 'moving_shape', 'roar', 'predator_smell', 0.8),
#         ]
#
#         for obj_type, vision, sound, smell, danger in dangerous_objects:
#             for _ in range(3):
#                 # Генерируем координаты с шагом 2
#                 x = np.random.randint(-self.size // 2, self.size // 2)
#                 x = round(x / 2) * 2
#                 z = np.random.randint(-self.size // 2, self.size // 2)
#                 z = round(z / 2) * 2
#                 obj = DangerousObject(x, z, obj_type, vision, sound, smell, danger)
#                 self.objects.append(obj)
#
#         # Полезные объекты (еда)
#         food_positions = [
#             (-8, 8), (8, -8), (-6, -6), (6, 6),
#             (0, 10), (10, 0), (-10, 0), (0, -10)
#         ]
#         for x, z in food_positions[:5]:
#             obj = DangerousObject(x, z, ObjectType.FOOD,
#                                   'bright_shape', 'sizzle', 'food_smell', 0.0)
#             self.objects.append(obj)
#
#     def get_objects_in_range(self, x: float, z: float, radius: float = 10.0) -> List[DangerousObject]:
#         """
#         Возвращает активные объекты в радиусе.
#         """
#         result = []
#         for obj in self.objects:
#             if not obj.active:
#                 continue
#             dist = np.sqrt((obj.x - x) ** 2 + (obj.z - z) ** 2)
#             if dist <= radius:
#                 result.append(obj)
#         return result
#
#     def get_object_at(self, x: float, z: float) -> Optional[DangerousObject]:
#         """
#         Возвращает объект в указанной позиции.
#         """
#         for obj in self.objects:
#             if not obj.active:
#                 continue
#             # Проверяем, находится ли бот в той же клетке
#             if abs(obj.x - x) < 1.0 and abs(obj.z - z) < 1.0:
#                 return obj
#         return None
#
#     def collect_food(self, x: float, z: float) -> bool:
#         """
#         Собирает еду в указанной позиции.
#         """
#         obj = self.get_object_at(x, z)
#         if obj and obj.type == ObjectType.FOOD:
#             return obj.collect()
#         return False
#
#     def get_active_objects(self) -> List[DangerousObject]:
#         """
#         Возвращает список активных объектов.
#         """
#         return [obj for obj in self.objects if obj.active]
#
#     def reset(self):
#         """
#         Сбрасывает мир (восстанавливает все объекты).
#         """
#         for obj in self.objects:
#             obj.active = True
#             obj.collected = False
#
#
#
# # # core/world_new.py
# # import numpy as np
# # from typing import List, Dict, Optional, Tuple, Any, Union
# # from enum import Enum
# #
# #
# # class ObjectType(Enum):
# #     EXPLOSION = "explosion"
# #     FIRE = "fire"
# #     FLOOD = "flood"
# #     FOOD = "food"
# #     PREDATOR = "predator"
# #     SHELTER = "shelter"
# #
# #
# # class DangerousObject:
# #     """Опасный объект с мультимодальными характеристиками."""
# #
# #     def __init__(self, x: float, z: float, obj_type: ObjectType,
# #                  vision_signature: str, sound_signature: str, smell_signature: str,
# #                  danger_level: float = 0.5):
# #
# #         self.x = x
# #         self.z = z
# #         self.type = obj_type
# #         self.danger_level = danger_level
# #
# #         # Векторные представления для распознавания
# #         self.vision_vector = self._generate_vision_vector(vision_signature)
# #         self.sound_vector = self._generate_sound_vector(sound_signature)
# #         self.smell_vector = self._generate_smell_vector(smell_signature)
# #
# #         self.active = True
# #         self.lifetime = 100  # количество шагов до исчезновения
# #
# #     def _generate_vision_vector(self, signature: str) -> np.ndarray:
# #         """Генерирует векторное представление визуального образа."""
# #         # В реальном приложении здесь была бы CNN для обработки изображений
# #         base = np.random.randn(64) * 0.1
# #         # Кодируем сигнатуру
# #         signatures = {
# #             'bright_flash': np.array([1.0, 0.0, 0.0, 0.0]),
# #             'red_glow': np.array([0.0, 1.0, 0.0, 0.0]),
# #             'dark_shape': np.array([0.0, 0.0, 1.0, 0.0]),
# #             'moving_shape': np.array([0.0, 0.0, 0.0, 1.0]),
# #         }
# #         if signature in signatures:
# #             base[:4] = signatures[signature]
# #         return base
# #
# #     def _generate_sound_vector(self, signature: str) -> np.ndarray:
# #         """Генерирует векторное представление звука."""
# #         base = np.random.randn(32) * 0.1
# #         signatures = {
# #             'loud_crash': np.array([1.0, 0.0, 0.0, 0.0]),
# #             'hiss': np.array([0.0, 1.0, 0.0, 0.0]),
# #             'roar': np.array([0.0, 0.0, 1.0, 0.0]),
# #             'crackle': np.array([0.0, 0.0, 0.0, 1.0]),
# #         }
# #         if signature in signatures:
# #             base[:4] = signatures[signature]
# #         return base
# #
# #     def _generate_smell_vector(self, signature: str) -> np.ndarray:
# #         """Генерирует векторное представление запаха."""
# #         base = np.random.randn(16) * 0.1
# #         signatures = {
# #             'smoke': np.array([1.0, 0.0, 0.0, 0.0]),
# #             'food_smell': np.array([0.0, 1.0, 0.0, 0.0]),
# #             'predator_smell': np.array([0.0, 0.0, 1.0, 0.0]),
# #             'water': np.array([0.0, 0.0, 0.0, 1.0]),
# #         }
# #         if signature in signatures:
# #             base[:4] = signatures[signature]
# #         return base
# #
# #     def get_sensory_data(self) -> Dict:
# #         """Возвращает сенсорные данные объекта."""
# #         return {
# #             'vision': self.vision_vector,
# #             'sound': self.sound_vector,
# #             'smell': self.smell_vector,
# #             'position': (self.x, self.z),
# #             'danger_level': self.danger_level,
# #             'type': self.type.value
# #         }
# #
# #
# # class NewWorld:
# #     """Новый мир с разнообразными объектами."""
# #
# #     def __init__(self, size: int = 30):
# #         self.size = size
# #         self.objects: List[DangerousObject] = []
# #         self._populate_world()
# #
# #     def _populate_world(self):
# #         """Заполняет мир объектами."""
# #         # Опасные объекты
# #         dangerous_objects = [
# #             (ObjectType.EXPLOSION, 'bright_flash', 'loud_crash', 'smoke', 0.9),
# #             (ObjectType.FIRE, 'red_glow', 'crackle', 'smoke', 0.7),
# #             (ObjectType.FLOOD, 'dark_shape', 'hiss', 'water', 0.6),
# #             (ObjectType.PREDATOR, 'moving_shape', 'roar', 'predator_smell', 0.8),
# #         ]
# #
# #         for obj_type, vision, sound, smell, danger in dangerous_objects:
# #             for _ in range(3):  # по 3 объекта каждого типа
# #                 x = np.random.randint(-self.size // 2, self.size // 2)
# #                 z = np.random.randint(-self.size // 2, self.size // 2)
# #                 obj = DangerousObject(x, z, obj_type, vision, sound, smell, danger)
# #                 self.objects.append(obj)
# #
# #         # Полезные объекты (еда)
# #         for _ in range(5):
# #             x = np.random.randint(-self.size // 2, self.size // 2)
# #             z = np.random.randint(-self.size // 2, self.size // 2)
# #             obj = DangerousObject(x, z, ObjectType.FOOD,
# #                                   'bright_shape', 'sizzle', 'food_smell', 0.0)
# #             self.objects.append(obj)
# #
# #     def get_objects_in_range(self, x: float, z: float, radius: float = 10.0) -> List[DangerousObject]:
# #         """Возвращает объекты в радиусе."""
# #         result = []
# #         for obj in self.objects:
# #             dist = np.sqrt((obj.x - x) ** 2 + (obj.z - z) ** 2)
# #             if dist <= radius and obj.active:
# #                 result.append(obj)
# #         return result