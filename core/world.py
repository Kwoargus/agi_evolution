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
        self.explosions = []          # для взрывов (если будут)

        # Параметры камеры (для визуализации)
        self.cam_x = 0.0
        self.cam_z = 0.0
        self.yaw = 45.0
        self.pitch = 30.0
        self.zoom = 1.0

        # Цвета
        self.COLOR_BG = (30, 30, 30)
        self.COLOR_GRID = (200, 200, 200)
        self.COLOR_MARKER = (255, 100, 100)

        # Для работы с экраном
        self.screen = None

        self.explosions = []  # список взрывов
        self.explosion_timer = 10.0
        self.bot = None

        # Если задана конфигурация объектов, создаём их
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
                # можно добавить другие типы

    def reset(self, bot_start_pos=(0, 0)):
        """Сбрасывает мир в начальное состояние (для нового эпизода)."""
        # Пока просто ничего не делаем
        pass

    def get_state(self, bot):
        """
        Возвращает состояние мира для бота.
        РАЗМЕРНОСТЬ ВСЕГДА 21: [x, z] + 6 объектов × 3 + 1 резерв
        """
        half = self.world_size / 2.0

        # Если бот не передан, возвращаем нулевое состояние
        if bot is None:
            return [0.0] * 21

        state = [bot.x / half, bot.z / half]  # 2 значения

        # Берем ровно 6 объектов из мира (или меньше)
        objects_list = list(self.objects)[:6]

        # Добавляем информацию о каждом объекте
        for obj in objects_list:
            if obj is None:
                state.extend([0.0, 0.0, 0.0])
            else:
                # Кодируем тип объекта числом
                if hasattr(obj, 'type'):
                    # Если type - строка, преобразуем в число
                    if isinstance(obj.type, str):
                        if obj.type == 'food' or obj.type == 'Food':
                            type_code = 2.0
                        elif obj.type == 'predator' or obj.type == 'Predator':
                            type_code = 3.0
                        elif obj.type == 'fire' or obj.type == 'Fire':
                            type_code = 1.0
                        else:
                            type_code = 4.0  # неизвестный тип
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

        # Если объектов меньше 6, дополняем нулями
        while len(objects_list) < 6:
            state.extend([0.0, 0.0, 0.0])
            objects_list.append(None)

        # У нас должно быть 2 + 6*3 = 20 значений
        # Добавляем 1 резервный ноль для размерности 21
        state.append(0.0)

        # Гарантируем ровно 21
        while len(state) < 21:
            state.append(0.0)
        state = state[:21]

        return state

    # def get_state(self, bot):
    #     """
    #     Возвращает состояние мира для бота.
    #     Размерность ФИКСИРОВАННАЯ: всегда 21 (8 + 13 объектов).
    #     """
    #     half = self.world_size / 2.0
    #     state = [bot.x / half, bot.z / half]  # 2 значения
    #
    #     # Фиксируем список объектов для консистентности
    #     # Берем до 6 объектов (2 + 6*3 = 20, плюс резерв)
    #     max_objects = 6
    #     objects_list = list(self.objects)[:max_objects]
    #
    #     # Дополняем до max_objects пустыми объектами
    #     while len(objects_list) < max_objects:
    #         objects_list.append(None)
    #
    #     for obj in objects_list[:max_objects]:
    #         if obj is None:
    #             # Пустой объект: тип 0, позиция 0, 0
    #             state.extend([0.0, 0.0, 0.0])
    #         else:
    #             # Кодируем тип объекта числом
    #             if hasattr(obj, 'type'):
    #                 type_code = obj.type
    #             elif isinstance(obj, GameObject):
    #                 type_code = 1  # огонь/костёр
    #             elif isinstance(obj, Food):
    #                 type_code = 2  # еда
    #             elif isinstance(obj, Predator):
    #                 type_code = 3  # хищник
    #             else:
    #                 type_code = 0  # неизвестный тип
    #
    #             # Нормализуем относительное положение
    #             dx = (obj.x - bot.x) / half
    #             dz = (obj.z - bot.z) / half
    #
    #             state.extend([float(type_code), float(dx), float(dz)])
    #
    #     # Гарантируем размерность 2 + max_objects * 3 = 2 + 6*3 = 20
    #     # Добавляем ещё один резервный объект для 21
    #     state.extend([0.0, 0.0, 0.0])  # +3 = 23, но оставим 21
    #
    #     # Обрезаем до 21
    #     state = state[:21]
    #
    #     # Если меньше 21, дополняем нулями
    #     while len(state) < 21:
    #         state.append(0.0)
    #
    #     return state

    # def get_state(self, bot):
    #     half = self.world_size / 2.0  # половина размера мира (10)
    #     state = [bot.x / half, bot.z / half]  # позиция бота нормализована в [-1, 1]
    #
    #     for obj in self.objects:
    #         # Кодируем тип объекта числом
    #         if isinstance(obj, GameObject):
    #             type_code = 1  # огонь/костёр
    #         elif isinstance(obj, Food):
    #             type_code = 2  # еда
    #         elif isinstance(obj, Predator):
    #             type_code = 3  # хищник
    #         else:
    #             type_code = 0  # неизвестный тип
    #
    #         # Нормализуем относительное положение объекта относительно бота
    #         dx = (obj.x - bot.x) / half
    #         dz = (obj.z - bot.z) / half
    #
    #         state.extend([type_code, dx, dz])
    #         # print("state: ", state)
    #     return state

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

    # ---------- Визуализация ----------
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

        # Отрисовка взрывов (если есть)
        for expl in self.explosions:
            expl.draw(screen, self.world_to_screen)

        if bot:
            bot.draw(screen, self.world_to_screen, self.get_scale())

        self.draw_ui(screen, bot)
        pygame.display.flip()

    def trigger_explosion(self):
        print("Взрыв создан!")
        corners = [(-8, -8), (-8, 8), (8, -8), (8, 8)]
        x, z = random.choice(corners)
        print(f"Взрыв в точке ({x}, {z})")  # отладка
        explosion = Explosion(x, z)
        self.explosions.append(explosion)
        if self.bot:
            print(f"[Explosion] triggered at ({x}, {z}), notifying bot")  # <-- исправлено
            self.bot.notify('explosion', {
                'sound': 'loud crash',
                'vision': 'bright_flash',
                'position': (x, z)
            })

    # ---------- Запуск (для визуализации) ----------
    def run(self, bot=None):

        self.bot = bot  # сохраняем ссылку

        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("AGI Evolution")
        clock = pygame.time.Clock()
        self.screen = screen

        running = True
        while running:
            # Обработка событий (камера, выход)
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

            # Управление камерой (клавиши)
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

            # Обновление бота (если есть)
            if bot:
                bot.update(self)

            # Отрисовка
            self.draw(screen, bot)
            clock.tick(60)

            # Обновление таймера взрыва
            if self.explosion_timer > 0:
                self.explosion_timer -= 1 / 60
                if self.explosion_timer <= 0:
                    self.trigger_explosion()

            # Обновление взрывов
            for expl in self.explosions[:]:
                expl.update(1 / 60)
                if not expl.active:
                    self.explosions.remove(expl)

        pygame.quit()