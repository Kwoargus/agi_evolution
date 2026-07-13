# scripts/social_scenario.py (полностью исправленный)
"""
Сценарий "Выживание трёх ботов"
Демонстрация социального взаимодействия и эмоций.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import numpy as np
import math
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

from core.world import World
from core.individual import Individual
from core.genome import Genome
from core.emotions.emotion_system import EmotionSystem
from core.emotions.emotion_base import EmotionalEvent, EmotionalResponse, EmotionType
from core.objects import Food, Predator, GameObject, Explosion  # Добавляем Explosion


@dataclass
class BotStats:
    """Статистика бота."""
    name: str
    color: Tuple[int, int, int]
    strength: int = 10
    health: int = 100
    karma: float = 0.5
    food_collected: int = 0
    survived: bool = True

    @property
    def score(self) -> float:
        """Итоговый счет."""
        return (self.strength + self.health) * self.karma


class SocialBot(Individual):
    """
    Бот с социальными параметрами.
    """

    def __init__(self, x: float, z: float, name: str, color: Tuple[int, int, int],
                 strength: int = 10, health: int = 100, karma: float = 0.5):
        super().__init__(x, z)
        self.bot_name = name
        self.bot_color = color
        self.stats = BotStats(
            name=name,
            color=color,
            strength=strength,
            health=health,
            karma=karma
        )
        self.emotion_system = EmotionSystem()
        self.emotion_history = []
        self.last_action = None

        # Настройки внешности для отрисовки
        self.body_w = 0.8
        self.body_d = 0.5
        self.body_h = 1.2
        self.head_r = 0.3

    def collect_food(self) -> bool:
        """Собирает еду."""
        self.stats.food_collected += 1
        self.stats.strength += 1
        print(f"🍎 {self.bot_name} собрал еду! Сила: {self.stats.strength}")
        return True

    def take_damage(self, damage: int) -> bool:
        """Получает урон."""
        self.stats.health -= damage
        if self.stats.health <= 0:
            self.stats.health = 0
            self.stats.survived = False
            print(f"💀 {self.bot_name} погиб!")
            return False
        print(f"⚔️ {self.bot_name} получил {damage} урона. Здоровье: {self.stats.health}")
        return True

    def help_bot(self, other: 'SocialBot') -> bool:
        """Помогает другому боту."""
        if other.stats.health < 50 and self.stats.strength > 5:
            # Делится силой
            self.stats.strength -= 2
            other.stats.health += 10
            self.stats.karma = min(1.0, self.stats.karma + 0.05)
            print(f"🤝 {self.bot_name} помог {other.bot_name}! Карма: {self.stats.karma:.2f}")
            return True
        return False

    def compete(self, other: 'SocialBot') -> bool:
        """Конкурирует с другим ботом."""
        if self.stats.strength > other.stats.strength:
            # Отбирает еду
            if other.stats.food_collected > 0:
                other.stats.food_collected -= 1
                self.stats.food_collected += 1
                self.stats.karma = max(0.0, self.stats.karma - 0.05)
                other.stats.karma = max(0.0, other.stats.karma - 0.05)
                print(f"⚡ {self.bot_name} отобрал еду у {other.bot_name}! Карма: {self.stats.karma:.2f}")
                return True
        return False

    def _clamp_color(self, value: int) -> int:
        """Ограничивает значение цвета в диапазоне 0-255."""
        return max(0, min(255, int(value)))

    def draw(self, screen, world_to_screen_func, scale):
        """Отрисовывает бота с его цветом."""
        cx, cz = self.x, self.z
        ang_rad = math.radians(self.angle)

        def rotate_point(lx, lz):
            rx = lx * math.cos(ang_rad) - lz * math.sin(ang_rad)
            rz = lx * math.sin(ang_rad) + lz * math.cos(ang_rad)
            return cx + rx, cz + rz

        corners_local = [
            (-self.body_w / 2, -self.body_d / 2),
            (self.body_w / 2, -self.body_d / 2),
            (self.body_w / 2, self.body_d / 2),
            (-self.body_w / 2, self.body_d / 2)
        ]
        corners_world = [rotate_point(lx, lz) for (lx, lz) in corners_local]
        base_points = [world_to_screen_func(wx, wz, 0) for (wx, wz) in corners_world]
        top_points = [world_to_screen_func(wx, wz, self.body_h) for (wx, wz) in corners_world]

        # Основной цвет - цвет бота
        color_body = self.bot_color
        color_dark = tuple(self._clamp_color(c - 50) for c in self.bot_color)

        pygame.draw.polygon(screen, color_body, top_points)
        for i in range(4):
            j = (i + 1) % 4
            pts = [base_points[i], base_points[j], top_points[j], top_points[i]]
            pygame.draw.polygon(screen, color_dark, pts)
        pygame.draw.polygon(screen, color_dark, base_points)

        # Голова
        head_offset_z = 0.2
        head_x, head_z = rotate_point(0, head_offset_z)
        head_y = self.body_h + self.head_r * 0.8
        head_screen = world_to_screen_func(head_x, head_z, head_y)
        rad_px = int(self.head_r * scale)
        if rad_px > 1:
            pygame.draw.circle(screen, (255, 200, 150), head_screen, rad_px)

        # Имя над ботом
        font = pygame.font.Font(None, 16)
        name_text = font.render(self.bot_name, True, (255, 255, 255))
        name_pos = world_to_screen_func(cx, cz, self.body_h + 0.5)
        screen.blit(name_text, (name_pos[0] - 20, name_pos[1] - 10))

        # Полоска здоровья
        health_pos = world_to_screen_func(cx, cz, self.body_h + 0.2)
        bar_width = 30
        bar_height = 4
        health_ratio = max(0, min(1.0, self.stats.health / 100.0))

        # Фон полоски
        pygame.draw.rect(screen, (50, 50, 50),
                         (health_pos[0] - bar_width // 2, health_pos[1], bar_width, bar_height))

        # Цвет полоски: зеленый -> желтый -> красный
        red = self._clamp_color(255 * (1 - health_ratio))
        green = self._clamp_color(255 * health_ratio)
        health_color = (red, green, 0)

        # Полоска здоровья
        pygame.draw.rect(screen, health_color,
                         (health_pos[0] - bar_width // 2, health_pos[1],
                          int(bar_width * health_ratio), bar_height))


class SocialScenario:
    """
    Сценарий социального взаимодействия трёх ботов.
    """

    def __init__(self, world_size: int = 20, cell_size: int = 40):
        self.world_size = world_size
        self.cell_size = cell_size

        # Создаем мир
        self.world = World(width=1200, height=800, world_size=world_size, cell_size=cell_size)

        # Создаем трёх ботов с разными параметрами
        self.bots = self._create_bots()

        # Объекты в мире
        self._populate_world()

        # История для визуализации
        self.history = []
        self.step_count = 0
        self.max_steps = 500

        # Настройки Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((1600, 900))
        pygame.display.set_caption("Сценарий: Выживание трёх ботов")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 16)
        self.big_font = pygame.font.Font(None, 28)
        # Список активных взрывов
        self.explosions = []

        print("=" * 60)
        print("СЦЕНАРИЙ: ВЫЖИВАНИЕ ТРЁХ БОТОВ")
        print("=" * 60)
        for bot in self.bots:
            print(f"  {bot.bot_name}: Сила={bot.stats.strength}, Здоровье={bot.stats.health}, Карма={bot.stats.karma:.2f}")
        print("=" * 60)

    def _create_bots(self) -> List[SocialBot]:
        """Создает трёх ботов с разными параметрами."""
        bots = [
            SocialBot(
                x=-4, z=-4,
                name="Евро",
                color=(245, 235, 220),  # Бело-бежевый (очень светлый)
                strength=10,
                health=100,
                karma=0.5
            ),
            SocialBot(
                x=4, z=4,
                name="Азио",
                color=(210, 180, 150),  # Светло-коричневый
                strength=8,
                health=120,
                karma=0.5
            ),
            SocialBot(
                x=0, z=-6,
                name="Афро",
                color=(160, 120, 80),  # Коричневый
                strength=12,
                health=80,
                karma=0.5
            )
        ]
        return bots

    def _get_safe_direction(self, bot: SocialBot, from_x: float, from_z: float) -> Tuple[float, float]:
        """
        Находит безопасное направление для убегания.
        Избегает углов и опасных объектов.
        """
        half = self.world_size // 2
        escape_speed = 2.0

        # Направление от источника опасности
        dist = math.sqrt((bot.x - from_x) ** 2 + (bot.z - from_z) ** 2)
        if dist > 0.1:
            dx = (bot.x - from_x) / dist
            dz = (bot.z - from_z) / dist
        else:
            dx, dz = 0, 1

        # Проверяем, не ведет ли направление в угол
        # Если бот в углу или рядом с ним - меняем направление
        corner_distance = 2.0

        # Проверяем все 4 угла
        corners = [(-half, -half), (-half, half), (half, -half), (half, half)]
        for cx, cz in corners:
            dist_to_corner = math.sqrt((bot.x - cx) ** 2 + (bot.z - cz) ** 2)
            if dist_to_corner < corner_distance:
                # Бежим в противоположный угол
                dx = -cx / half
                dz = -cz / half
                break

        # Проверяем, нет ли опасных объектов на пути
        danger_radius = 2.0
        for obj in self.world.objects:
            if not hasattr(obj, 'active') or not obj.active:
                continue
            if hasattr(obj, 'type') and obj.type in ['predator', 'fire']:
                # Проверяем, не ведет ли путь к объекту
                future_x = bot.x + dx * 2
                future_z = bot.z + dz * 2
                dist_to_obj = math.sqrt((future_x - obj.x) ** 2 + (future_z - obj.z) ** 2)
                if dist_to_obj < danger_radius:
                    # Меняем направление
                    dx = (bot.x - obj.x) / (dist_to_obj + 0.1)
                    dz = (bot.z - obj.z) / (dist_to_obj + 0.1)

        # Нормализуем и умножаем на скорость
        norm = math.sqrt(dx ** 2 + dz ** 2)
        if norm > 0:
            dx = dx / norm * escape_speed
            dz = dz / norm * escape_speed

        return dx, dz

    def _populate_world(self):
        """Заполняет мир объектами."""
        # Очищаем мир
        self.world.objects = []
        self.explosions = []  # Очищаем взрывы

        # Еда (яблоки) - 12 штук, разбросаны по всему миру
        food_positions = [
            (-8, 0), (8, 0), (0, -8), (0, 8),
            (-5, 5), (5, -5), (-5, -5), (5, 5),
            (-9, 3), (9, -3), (-3, 9), (3, -9)
        ]
        for x, z in food_positions:
            food = Food(x, z, name="apple", obj_type="food", smell="food_smell")
            food.active = True
            self.world.add_object(food)

        # Волк - ОДИН, в безопасном месте (не в углу)
        predator_positions = [(-3, 8)]  # Только один волк
        for x, z in predator_positions:
            predator = Predator(x, z, name="wolf", obj_type="predator", smell="predator_smell")
            predator.active = True
            self.world.add_object(predator)

        # Огонь - ОДИН, в безопасном месте (не в углу)
        fire_positions = [(3, -8)]  # Только один костёр
        for x, z in fire_positions:
            fire = GameObject(x, z, obj_type="fire", temperature=800, size=1.0)
            fire.active = True
            self.world.add_object(fire)

    def _get_nearby_objects(self, x: float, z: float, radius: float = 2.0) -> List:
        """Возвращает объекты рядом с позицией."""
        nearby = []
        for obj in self.world.objects:
            if hasattr(obj, 'active') and not obj.active:
                continue
            dist = np.sqrt((obj.x - x) ** 2 + (obj.z - z) ** 2)
            if dist < radius:
                nearby.append((obj, dist))
        return nearby

    def _process_bot_emotion(self, bot: SocialBot) -> Optional[str]:
        """Обрабатывает эмоции бота на основе окружения."""
        nearby = self._get_nearby_objects(bot.x, bot.z, radius=2.0)

        for obj, dist in nearby:
            if hasattr(obj, 'type'):
                if obj.type == 'food':
                    # Находит еду - радость
                    bot.collect_food()
                    obj.active = False
                    return 'joy'
                elif obj.type == 'predator':
                    # Встретил хищника - страх
                    bot.take_damage(10)
                    return 'fear'
                elif obj.type == 'fire':
                    # Подошел к огню - страх
                    bot.take_damage(5)
                    return 'fear'

        # Проверяем других ботов
        for other in self.bots:
            if other is bot or not other.stats.survived:
                continue
            dist = np.sqrt((other.x - bot.x) ** 2 + (other.z - bot.z) ** 2)
            if dist < 2.0:
                # Взаимодействие с другим ботом
                if bot.stats.food_collected > other.stats.food_collected + 2:
                    # Бот собрал больше еды - делится или конкурирует
                    if bot.stats.karma > 0.6:
                        # Дружелюбный бот - помогает
                        bot.help_bot(other)
                        return 'love'
                    else:
                        # Конкурентный бот - отбирает
                        bot.compete(other)
                        return 'anger'

        return None

    def _move_bot(self, bot: SocialBot, dx: float, dz: float):
        """Двигает бота."""
        new_x = bot.x + dx
        new_z = bot.z + dz
        half = self.world_size // 2
        if -half <= new_x <= half and -half <= new_z <= half:
            bot.x = new_x
            bot.z = new_z
            return True
        return False

    def _get_bot_action(self, bot: SocialBot) -> Tuple[float, float]:
        """Определяет действие бота на основе его состояния."""
        dx, dz = 0, 0
        normal_speed = 0.15

        # ============================================================
        # 1. СНАЧАЛА ПРОВЕРЯЕМ ВЗРЫВЫ - ВЫСШИЙ ПРИОРИТЕТ!
        # ============================================================

        # Находим ближайший активный взрыв
        nearest_expl = None
        nearest_dist = float('inf')

        for expl in self.explosions:
            if not expl.active:
                continue
            dist = math.sqrt((bot.x - expl.x) ** 2 + (bot.z - expl.z) ** 2)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_expl = expl

        # Если есть хоть один взрыв - УБЕГАЕМ В БЕЗОПАСНОЕ МЕСТО!
        if nearest_expl is not None:
            # Находим безопасное направление
            dx, dz = self._get_safe_direction(bot, nearest_expl.x, nearest_expl.z)

            # Наносим урон, если взрыв очень близко
            if nearest_dist < 2.0 and not nearest_expl.damage_dealt:
                bot.take_damage(30)
                nearest_expl.damage_dealt = True
                print(f"💥 {bot.bot_name} получил урон от взрыва на расстоянии {nearest_dist:.1f}!")

            # Добавляем небольшую случайность
            dx += random.uniform(-0.05, 0.05)
            dz += random.uniform(-0.05, 0.05)

            return dx, dz

        # ============================================================
        # 2. ЕСЛИ НЕТ ВЗРЫВОВ - ПРОВЕРЯЕМ ОПАСНЫЕ ОБЪЕКТЫ
        # ============================================================
        danger_nearby = self._get_nearby_objects(bot.x, bot.z, radius=2.5)
        for obj, dist in danger_nearby:
            if hasattr(obj, 'type') and obj.type in ['predator', 'fire']:
                if dist > 0.1:
                    # Убегаем от опасного объекта
                    dx = (bot.x - obj.x) / dist * normal_speed * 2.0
                    dz = (bot.z - obj.z) / dist * normal_speed * 2.0
                    dx += random.uniform(-0.02, 0.02)
                    dz += random.uniform(-0.02, 0.02)
                return dx, dz

        # ============================================================
        # 3. ИЩЕМ БЛИЖАЙШУЮ ЕДУ
        # ============================================================
        best_food = None
        best_dist = float('inf')
        for obj in self.world.objects:
            if hasattr(obj, 'type') and obj.type == 'food':
                if hasattr(obj, 'active') and not obj.active:
                    continue
                dist = np.sqrt((obj.x - bot.x) ** 2 + (obj.z - bot.z) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_food = obj

        if best_food and best_dist > 0.1:
            dx = (best_food.x - bot.x) / best_dist * normal_speed
            dz = (best_food.z - bot.z) / best_dist * normal_speed

        return dx, dz


    # def _get_bot_action(self, bot: SocialBot) -> Tuple[float, float]:
    #     """Определяет действие бота на основе его состояния."""
    #     dx, dz = 0, 0
    #     normal_speed = 0.15
    #     escape_speed = 2.0  # Очень быстро!
    #
    #     # ============================================================
    #     # 1. СНАЧАЛА ПРОВЕРЯЕМ ВЗРЫВЫ - ВЫСШИЙ ПРИОРИТЕТ!
    #     #    Бот убегает от ЛЮБОГО взрыва в мире
    #     # ============================================================
    #
    #     # Находим ближайший активный взрыв
    #     nearest_expl = None
    #     nearest_dist = float('inf')
    #
    #     for expl in self.explosions:
    #         if not expl.active:
    #             continue
    #         dist = math.sqrt((bot.x - expl.x) ** 2 + (bot.z - expl.z) ** 2)
    #         if dist < nearest_dist:
    #             nearest_dist = dist
    #             nearest_expl = expl
    #
    #     # Если есть хоть один взрыв - УБЕГАЕМ!
    #     if nearest_expl is not None:
    #         # Направление ОТ взрыва (в противоположную сторону)
    #         if nearest_dist > 0.1:
    #             # Вектор от взрыва к боту (нормализованный)
    #             dx = (bot.x - nearest_expl.x) / nearest_dist * escape_speed
    #             dz = (bot.z - nearest_expl.z) / nearest_dist * escape_speed
    #         else:
    #             # Если бот в центре взрыва - бежим в случайном направлении
    #             angle = random.uniform(0, 2 * math.pi)
    #             dx = math.cos(angle) * escape_speed
    #             dz = math.sin(angle) * escape_speed
    #
    #         # Наносим урон, если взрыв очень близко
    #         if nearest_dist < 2.0 and not nearest_expl.damage_dealt:
    #             bot.take_damage(30)
    #             nearest_expl.damage_dealt = True
    #             print(f"💥 {bot.bot_name} получил урон от взрыва на расстоянии {nearest_dist:.1f}!")
    #
    #         # Проверяем границы - если бот у стены, разворачиваемся и бежим вдоль стены
    #         half = self.world_size // 2
    #         new_x = bot.x + dx
    #         new_z = bot.z + dz
    #
    #         # Обработка границ
    #         if abs(new_x) > half - 0.5:
    #             dx = -dx * 0.5
    #             # Бежим вдоль стены в случайном направлении
    #             if abs(bot.x) > half - 1:
    #                 dx = -np.sign(bot.x) * escape_speed * 0.8
    #                 dz = np.random.choice([-1, 1]) * escape_speed * 0.8
    #
    #         if abs(new_z) > half - 0.5:
    #             dz = -dz * 0.5
    #             if abs(bot.z) > half - 1:
    #                 dz = -np.sign(bot.z) * escape_speed * 0.8
    #                 dx = np.random.choice([-1, 1]) * escape_speed * 0.8
    #
    #         # Добавляем случайность для разнообразия
    #         dx += random.uniform(-0.05, 0.05)
    #         dz += random.uniform(-0.05, 0.05)
    #
    #         return dx, dz

        # ============================================================
        # 2. ЕСЛИ НЕТ ВЗРЫВОВ - ПРОВЕРЯЕМ ОПАСНЫЕ ОБЪЕКТЫ
        # ============================================================
        danger_nearby = self._get_nearby_objects(bot.x, bot.z, radius=2.5)
        for obj, dist in danger_nearby:
            if hasattr(obj, 'type') and obj.type in ['predator', 'fire']:
                if dist > 0.1:
                    dx = (bot.x - obj.x) / dist * normal_speed * 2.0
                    dz = (bot.z - obj.z) / dist * normal_speed * 2.0
                    dx += random.uniform(-0.02, 0.02)
                    dz += random.uniform(-0.02, 0.02)
                return dx, dz

        # ============================================================
        # 3. ИЩЕМ БЛИЖАЙШУЮ ЕДУ
        # ============================================================
        best_food = None
        best_dist = float('inf')
        for obj in self.world.objects:
            if hasattr(obj, 'type') and obj.type == 'food':
                if hasattr(obj, 'active') and not obj.active:
                    continue
                dist = np.sqrt((obj.x - bot.x) ** 2 + (obj.z - bot.z) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_food = obj

        if best_food and best_dist > 0.1:
            dx = (best_food.x - bot.x) / best_dist * normal_speed
            dz = (best_food.z - bot.z) / best_dist * normal_speed

        return dx, dz



    def _clamp_color(self, value: float) -> int:
        """Ограничивает значение цвета в диапазоне 0-255."""
        return max(0, min(255, int(value)))

    def _draw_stats_panel(self, screen, x_offset: int):
        """Рисует панель статистики ботов."""
        # Фон панели
        panel_rect = pygame.Rect(x_offset, 0, 400, 900)
        pygame.draw.rect(screen, (40, 40, 50), panel_rect)

        # Заголовок
        title = self.font.render("СТАТИСТИКА БОТОВ", True, (255, 255, 255))
        screen.blit(title, (x_offset + 10, 10))

        # Информация о каждом боте
        y = 50
        for i, bot in enumerate(self.bots):
            color = bot.bot_color
            name_color = (255, 255, 255)

            # Имя
            name_text = self.font.render(f"{bot.bot_name}", True, name_color)
            screen.blit(name_text, (x_offset + 10, y))

            # Полоска здоровья
            y += 25
            health_text = self.small_font.render(f"Здоровье: {bot.stats.health}", True, (200, 200, 200))
            screen.blit(health_text, (x_offset + 10, y))

            bar_rect = pygame.Rect(x_offset + 120, y + 2, 150, 12)
            pygame.draw.rect(screen, (50, 50, 50), bar_rect)
            health_ratio = max(0, min(1.0, bot.stats.health / 100.0))
            fill_width = int(150 * health_ratio)
            red = self._clamp_color(255 * (1 - health_ratio))
            green = self._clamp_color(255 * health_ratio)
            pygame.draw.rect(screen, (red, green, 0),
                             (x_offset + 120, y + 2, fill_width, 12))

            # Сила
            y += 20
            strength_text = self.small_font.render(f"Сила: {bot.stats.strength}", True, (200, 200, 200))
            screen.blit(strength_text, (x_offset + 10, y))

            # Еда
            y += 20
            food_text = self.small_font.render(f"🍎: {bot.stats.food_collected}", True, (200, 200, 200))
            screen.blit(food_text, (x_offset + 10, y))

            # Карма
            y += 20
            if bot.stats.karma > 0.6:
                karma_color = (100, 200, 100)
            elif bot.stats.karma > 0.3:
                karma_color = (255, 200, 0)
            else:
                karma_color = (255, 100, 100)
            karma_text = self.small_font.render(f"Карма: {bot.stats.karma:.2f}", True, karma_color)
            screen.blit(karma_text, (x_offset + 10, y))

            # Статус
            y += 20
            status = "✅ Жив" if bot.stats.survived else "💀 Погиб"
            status_color = (100, 200, 100) if bot.stats.survived else (200, 100, 100)
            status_text = self.small_font.render(status, True, status_color)
            screen.blit(status_text, (x_offset + 10, y))

            # Разделитель
            y += 30
            pygame.draw.line(screen, (60, 60, 70), (x_offset + 10, y), (x_offset + 390, y), 1)
            y += 10

        # Итоговый счет
        y += 20
        score_text = self.font.render("ИТОГОВЫЙ СЧЕТ", True, (255, 255, 255))
        screen.blit(score_text, (x_offset + 10, y))

        y += 30
        sorted_bots = sorted(self.bots, key=lambda b: b.stats.score, reverse=True)
        for i, bot in enumerate(sorted_bots):
            color = bot.bot_color
            name_text = self.font.render(f"{i + 1}. {bot.bot_name}: {bot.stats.score:.1f}", True, color)
            screen.blit(name_text, (x_offset + 10, y + i * 25))

        # Шаг
        y += 100
        step_text = self.small_font.render(f"Шаг: {self.step_count}/{self.max_steps}", True, (200, 200, 200))
        screen.blit(step_text, (x_offset + 10, y))

        # Легенда объектов
        y += 30
        legend_text = self.small_font.render("Объекты:", True, (200, 200, 200))
        screen.blit(legend_text, (x_offset + 10, y))

        objects_info = [
            ("🍎 Еда", (100, 200, 100)),
            ("🐺 Волк", (200, 100, 100)),
            ("🔥 Огонь", (255, 150, 0)),
        ]
        for i, (name, color) in enumerate(objects_info):
            y_pos = y + 25 + i * 20
            color_rect = pygame.Rect(x_offset + 10, y_pos, 12, 12)
            pygame.draw.rect(screen, color, color_rect)
            text = self.small_font.render(name, True, (200, 200, 200))
            screen.blit(text, (x_offset + 30, y_pos))

        # Подсказка
        y = 850
        help_text = self.small_font.render("Пробел - пауза | Esc - выход", True, (150, 150, 150))
        screen.blit(help_text, (x_offset + 10, y))

    def run(self):
        """Запускает сценарий."""
        running = True
        paused = False
        panel_offset = 1200

        # Счетчик шагов до взрыва
        explosion_timer = 200
        explosion_warning = 5  # За 5 шагов до взрыва боты начинают беспокоиться

        while running:
            # Обработка событий
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                        print(f"{'⏸️ Пауза' if paused else '▶️ Продолжение'}")

            if not paused and self.step_count < self.max_steps:
                # Обновляем каждого бота
                for bot in self.bots:
                    if not bot.stats.survived:
                        continue

                    # Определяем действие
                    dx, dz = self._get_bot_action(bot)
                    self._move_bot(bot, dx * 0.5, dz * 0.5)

                    # Обрабатываем эмоции
                    emotion = self._process_bot_emotion(bot)
                    if emotion:
                        print(f"🧠 {bot.bot_name}: {emotion}")

                    # Обновляем бота (рефлексы, инстинкты)
                    bot.update(self.world)

                    # Предупреждение о приближающемся взрыве
                if self.step_count % explosion_timer > explosion_timer - explosion_warning:
                    # Боты начинают беспокоиться и искать укрытие
                    for bot in self.bots:
                        if bot.stats.survived:
                            # Небольшое случайное движение в сторону центра
                            dx = random.uniform(-0.05, 0.05)
                            dz = random.uniform(-0.05, 0.05)
                            self._move_bot(bot, dx, dz)

                # Создаем взрыв
                if self.step_count % explosion_timer == 0 and self.step_count > 0:
                    x = random.randint(-6, 6)
                    z = random.randint(-6, 6)

                    # Проверяем, чтобы взрыв был не слишком близко к ботам
                    too_close = True
                    attempts = 0
                    while too_close and attempts < 20:
                        too_close = False
                        for bot in self.bots:
                            if bot.stats.survived:
                                dist = np.sqrt((x - bot.x) ** 2 + (z - bot.z) ** 2)
                                if dist < 3.0:
                                    too_close = True
                                    x = random.randint(-6, 6)
                                    z = random.randint(-6, 6)
                                    attempts += 1
                                    break

                    print(f"\n💥 ВЗРЫВ в ({x}, {z})!")
                    print(f"   Активных взрывов: {len(self.explosions)}")

                    explosion = Explosion(x, z)
                    self.explosions.append(explosion)
                    print(f"   Взрыв создан, всего взрывов: {len(self.explosions)}")

                    # Выводим расстояния до ботов
                    for b in self.bots:
                        if b.stats.survived:
                            dist = np.sqrt((x - b.x) ** 2 + (z - b.z) ** 2)
                            print(f"     {b.bot_name}: ({round(b.x, 1)}, {round(b.z, 1)}) - расстояние {round(dist, 1)}")

                self.step_count += 1

            # Проверяем, все ли боты погибли
            if all(not b.stats.survived for b in self.bots):
                print("\n💀 Все боты погибли!")
                running = False

            # Обновляем взрывы
            for expl in self.explosions[:]:
                expl.update(1.0)
                if not expl.active:
                    self.explosions.remove(expl)

            # Отрисовка
            self.screen.fill((30, 30, 30))
            self.world.draw(self.screen, None)

            # Рисуем взрывы поверх мира
            for expl in self.explosions:
                expl.draw(self.screen, self.world.world_to_screen)

            # Рисуем ботов поверх мира
            for bot in self.bots:
                if bot.stats.survived:
                    bot.draw(self.screen, self.world.world_to_screen, self.world.get_scale())

            self._draw_stats_panel(self.screen, panel_offset)

            # Индикатор взрыва (текстовое предупреждение)
            if self.step_count % explosion_timer > explosion_timer - 40:
                flash_text = self.big_font.render("💥 ВНИМАНИЕ! ВЗРЫВ!", True, (255, 100, 100))
                self.screen.blit(flash_text, (10, 10))

            pygame.display.flip()
            self.clock.tick(20)

        # Вывод результатов
        self._show_results()
        pygame.quit()

    def _show_results(self):
        """Показывает результаты сценария."""
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ СЦЕНАРИЯ")
        print("=" * 60)

        sorted_bots = sorted(self.bots, key=lambda b: b.stats.score, reverse=True)
        for i, bot in enumerate(sorted_bots):
            status = "✅ ЖИВ" if bot.stats.survived else "💀 ПОГИБ"
            print(f"{i + 1}. {bot.bot_name}:")
            print(f"   Сила: {bot.stats.strength}, Здоровье: {bot.stats.health}, Карма: {bot.stats.karma:.2f}")
            print(f"   🍎: {bot.stats.food_collected}, Статус: {status}")
            print(f"   ИТОГОВЫЙ СЧЕТ: {bot.stats.score:.1f}")
            print()

        winner = sorted_bots[0]
        print(f"🏆 ПОБЕДИТЕЛЬ: {winner.bot_name} с результатом {winner.stats.score:.1f}")
        print("=" * 60)


def main():
    scenario = SocialScenario(world_size=20, cell_size=40)
    scenario.run()


if __name__ == "__main__":
    main()



