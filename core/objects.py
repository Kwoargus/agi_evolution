import pygame
import math

class GameObject:
    def __init__(self, x, z, obj_type="fire", temperature=800, size=1.0):
        self.x = x
        self.z = z
        self.type = obj_type
        self.temperature = temperature
        self.size = size

    def get(self, params):
        result = {}
        for p in params:
            if hasattr(self, p):
                result[p] = getattr(self, p)
        return result

    def draw(self, screen, world_to_screen_func):
        half = self.size * 0.5
        base_corners = [
            (self.x - half, self.z - half),
            (self.x + half, self.z - half),
            (self.x + half, self.z + half),
            (self.x - half, self.z + half)
        ]
        top_world = (self.x, self.z, self.size * 1.2)

        base_screen = [world_to_screen_func(x, z, 0) for (x, z) in base_corners]
        top_screen = world_to_screen_func(*top_world)

        color = (200, 50, 50)
        for i in range(4):
            j = (i+1) % 4
            triangle = [base_screen[i], base_screen[j], top_screen]
            pygame.draw.polygon(screen, color, triangle)
        pygame.draw.polygon(screen, (150, 30, 30), base_screen)


class Predator:
    def __init__(self, x, z, name="wolf", obj_type="predator", smell="predator_smell", sound="predator_roar"):
        self.x = x
        self.z = z
        self.name = name
        self.type = obj_type
        self.smell = smell
        self.sound = sound
        self.body_w = 1.0
        self.body_d = 2.0
        self.body_h = 0.8
        self.head_w = 0.6
        self.head_d = 0.8
        self.head_h = 0.6

    def get(self, params):
        result = {}
        for p in params:
            if hasattr(self, p):
                result[p] = getattr(self, p)
        return result

    def draw(self, screen, world_to_screen_func):
        cx, cz = self.x, self.z
        corners = [
            (cx - self.body_w/2, cz - self.body_d/2),
            (cx + self.body_w/2, cz - self.body_d/2),
            (cx + self.body_w/2, cz + self.body_d/2),
            (cx - self.body_w/2, cz + self.body_d/2)
        ]
        base_points = [world_to_screen_func(x, z, 0) for (x, z) in corners]
        top_points = [world_to_screen_func(x, z, self.body_h) for (x, z) in corners]

        color_body = (150, 150, 150)
        pygame.draw.polygon(screen, color_body, top_points)
        for i in range(4):
            j = (i+1) % 4
            pts = [base_points[i], base_points[j], top_points[j], top_points[i]]
            pygame.draw.polygon(screen, (100, 100, 100), pts)
        pygame.draw.polygon(screen, (80, 80, 80), base_points)

        head_offset_z = self.body_d/2 + self.head_d/2 - 0.1
        head_cx = cx
        head_cz = cz + self.body_d/2 + self.head_d/2 - 0.1

        head_corners = [
            (head_cx - self.head_w/2, head_cz - self.head_d/2),
            (head_cx + self.head_w/2, head_cz - self.head_d/2),
            (head_cx + self.head_w/2, head_cz + self.head_d/2),
            (head_cx - self.head_w/2, head_cz + self.head_d/2)
        ]
        head_base = [world_to_screen_func(x, z, self.body_h - 0.1) for (x, z) in head_corners]
        head_top = [world_to_screen_func(x, z, self.body_h + self.head_h - 0.1) for (x, z) in head_corners]

        color_head = (180, 150, 120)
        pygame.draw.polygon(screen, color_head, head_top)
        for i in range(4):
            j = (i+1) % 4
            pts = [head_base[i], head_base[j], head_top[j], head_top[i]]
            pygame.draw.polygon(screen, (140, 110, 80), pts)

        eye_y = self.body_h + self.head_h * 0.7
        eye_offset_x = self.head_w * 0.3
        eye_offset_z = self.head_d * 0.3
        for side in [-1, 1]:
            ex = head_cx + side * eye_offset_x
            ez = head_cz + eye_offset_z
            eye_screen = world_to_screen_func(ex, ez, eye_y)
            pygame.draw.circle(screen, (255, 255, 0), eye_screen, 3)


class Food:
    def __init__(self, x, z, name="apple", obj_type="food", smell="food_smell"):
        self.x = x
        self.z = z
        self.name = name
        self.type = obj_type
        self.smell = smell
        self.radius = 0.4
        self.leaf_size = 0.6

    def get(self, params):
        result = {}
        for p in params:
            if hasattr(self, p):
                result[p] = getattr(self, p)
        return result

    def draw(self, screen, world_to_screen_func):
        # Яблоко (жёлтая сфера)
        center_screen = world_to_screen_func(self.x, self.z, self.radius)
        # Радиус в пикселях
        rad_px = int(self.radius * 40)  # приблизительно
        if rad_px > 1:
            pygame.draw.circle(screen, (255, 255, 0), center_screen, rad_px)
            # Листочек (зелёный треугольник)
            leaf_top = world_to_screen_func(self.x-0.4, self.z+0.15, self.radius + self.leaf_size)
            leaf_left = world_to_screen_func(self.x-0.4 - self.leaf_size, self.z+0.3, self.radius + self.leaf_size*0.5)
            leaf_right = world_to_screen_func(self.x-0.6 + self.leaf_size, self.z+0.05, self.radius + self.leaf_size*0.5)
            pygame.draw.polygon(screen, (0, 200, 0), [leaf_top, leaf_left, leaf_right])


class Explosion:
    def __init__(self, x, z):
        self.x = x
        self.z = z
        self.lifetime = 3.0  # секунд
        self.age = 0.0
        self.phase = 0  # 0-белый, 1-жёлтый, 2-красный
        self.rotation = 0
        self.scale = 1.0
        self.active = True
        self.color_sequence = [(255, 255, 255), (255, 255, 0), (255, 0, 0)]
        self.size = 2.0

    def update(self, dt):
        self.age += dt
        if self.age > self.lifetime:
            self.active = False
            return
        # Меняем цвет каждую секунду
        phase_duration = 1.0
        phase_index = int(self.age // phase_duration)
        if phase_index >= len(self.color_sequence):
            phase_index = len(self.color_sequence) - 1
        self.phase = phase_index
        # Поворот на 120° и увеличение в 2 раза при смене фазы
        # (упрощённо: увеличиваем масштаб пропорционально времени)
        self.scale = 1.0 + (self.age / self.lifetime) * 2.0
        self.rotation = 120 * (self.age // 1.0)  # градусов

    def draw(self, screen, world_to_screen_func):
        if not self.active:
            return
        # Рисуем пирамиду (перевёрнутая остриём вниз)
        half = self.size * 0.5 * self.scale
        base_corners = [
            (self.x - half, self.z - half),
            (self.x + half, self.z - half),
            (self.x + half, self.z + half),
            (self.x - half, self.z + half)
        ]
        # Основание на уровне земли, вершина внизу
        top_world = (self.x, self.z, -self.size * 0.8 * self.scale)  # остриё вниз

        base_screen = [world_to_screen_func(x, z, 0) for (x, z) in base_corners]
        top_screen = world_to_screen_func(*top_world)

        color = self.color_sequence[self.phase]
        # Четыре треугольные грани
        for i in range(4):
            j = (i+1) % 4
            triangle = [base_screen[i], base_screen[j], top_screen]
            pygame.draw.polygon(screen, color, triangle)

        # Рисуем молнию (несколько зигзагов)
        # Упрощённо: нарисуем несколько линий от вершины к основанию с рывками
        lightning_color = (200, 200, 200)
        for i in range(4):
            x1, y1 = base_screen[i]
            x2, y2 = base_screen[(i+1)%4]
            # Несколько случайных зигзагов (захардкодим)
            for step in range(0, 10, 2):
                t = step / 10.0
                x_mid = (x1 + x2) / 2 + (y1 - y2) * 0.2 * (1 - t)
                y_mid = (y1 + y2) / 2 + (x2 - x1) * 0.2 * (1 - t)
                pygame.draw.line(screen, lightning_color, (x1, y1), (x_mid, y_mid), 2)
                pygame.draw.line(screen, lightning_color, (x_mid, y_mid), (x2, y2), 2)