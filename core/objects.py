# core/objects.py

import pygame
import math
import numpy as np


class GameObject:
    def __init__(self, x, z, obj_type="fire", temperature=800, size=1.0):
        self.x = round(x / 2) * 2
        self.z = round(z / 2) * 2
        self.type = obj_type
        self.temperature = temperature
        self.size = size
        self.active = True

    def get(self, params):
        result = {}
        for p in params:
            if hasattr(self, p):
                result[p] = getattr(self, p)
        return result

    def draw(self, screen, world_to_screen_func):
        if not self.active:
            return
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
        self.x = round(x / 2) * 2
        self.z = round(z / 2) * 2
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
        self.active = True

    def get(self, params):
        result = {}
        for p in params:
            if hasattr(self, p):
                result[p] = getattr(self, p)
        return result

    def draw(self, screen, world_to_screen_func):
        if not self.active:
            return
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
        self.x = round(x / 2) * 2
        self.z = round(z / 2) * 2
        self.name = name
        self.type = obj_type
        self.smell = smell
        self.radius = 0.4
        self.leaf_size = 0.6
        self.active = True

    def get(self, params):
        result = {}
        for p in params:
            if hasattr(self, p):
                result[p] = getattr(self, p)
        return result

    def draw(self, screen, world_to_screen_func):
        if not self.active:
            return
        center_screen = world_to_screen_func(self.x, self.z, self.radius)
        rad_px = int(self.radius * 40)
        if rad_px > 1:
            pygame.draw.circle(screen, (255, 255, 0), center_screen, rad_px)
            leaf_top = world_to_screen_func(self.x-0.4, self.z+0.15, self.radius + self.leaf_size)
            leaf_left = world_to_screen_func(self.x-0.4 - self.leaf_size, self.z+0.3, self.radius + self.leaf_size*0.5)
            leaf_right = world_to_screen_func(self.x-0.6 + self.leaf_size, self.z+0.05, self.radius + self.leaf_size*0.5)
            pygame.draw.polygon(screen, (0, 200, 0), [leaf_top, leaf_left, leaf_right])

class Explosion:
    def __init__(self, x, z):
        self.x = round(x / 2) * 2
        self.z = round(z / 2) * 2
        self.lifetime = 40  # 40 кадров = 2 секунды при 20 FPS
        self.age = 0
        self.active = True
        self.radius = 0.2
        self.max_radius = 2.5  # Немного больше
        self.color = (255, 200, 50)
        self.damage_dealt = False

    def update(self, dt=1.0):
        self.age += dt
        if self.age > self.lifetime:
            self.active = False
        progress = self.age / self.lifetime
        # Быстрое расширение (50% времени) и быстрое затухание
        if progress < 0.5:
            self.radius = 0.2 + (self.max_radius - 0.2) * (progress / 0.5)
        else:
            self.radius = self.max_radius * (1 - (progress - 0.5) / 0.5)
        self.radius = max(0.1, self.radius)

    def draw(self, screen, world_to_screen_func):
        if not self.active:
            return

        center = world_to_screen_func(self.x, self.z, 0)
        radius_px = int(self.radius * 40)

        # Внешние кольца - яркие и быстрые
        for i in range(3):
            r = radius_px - i * 6
            if r > 0:
                color = (255, 200 - i * 50, 50 - i * 20)
                pygame.draw.circle(screen, color, center, r, max(1, 3 - i))

        # Внутренняя вспышка
        inner_radius = max(4, radius_px // 2)
        pygame.draw.circle(screen, (255, 255, 220), center, inner_radius)

        # Яркое ядро
        core_radius = max(3, inner_radius // 2)
        pygame.draw.circle(screen, (255, 255, 255), center, core_radius)

        # Лучи
        for angle in range(0, 360, 20):
            rad = math.radians(angle)
            end_x = center[0] + int(radius_px * math.cos(rad))
            end_y = center[1] + int(radius_px * math.sin(rad))
            pygame.draw.line(screen, (255, 220, 50), center, (end_x, end_y), 2)

        # Искры
        num_sparks = max(0, 12 - int(self.age / 3))
        for _ in range(num_sparks):
            angle = np.random.uniform(0, 2 * math.pi)
            dist = np.random.uniform(0.3, 0.9) * radius_px
            spark_x = center[0] + int(dist * math.cos(angle))
            spark_y = center[1] + int(dist * math.sin(angle))
            spark_color = (255, 200 - np.random.randint(0, 150), 50 - np.random.randint(0, 50))
            pygame.draw.circle(screen, spark_color, (spark_x, spark_y), np.random.randint(1, 3))