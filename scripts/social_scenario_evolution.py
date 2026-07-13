# scripts/social_scenario_evolution.py
"""
Сценарий "Выживание трёх ботов" с эволюцией стратегий (GAN + GA).
Боты эволюционируют поколениями, улучшая свои стратегии выживания.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

from core.world import World
from core.individual import Individual
from core.genome import Genome
from core.emotions.emotion_system import EmotionSystem
from core.emotions.emotion_base import EmotionalEvent, EmotionalResponse, EmotionType
from core.objects import Food, Predator, GameObject, Explosion


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
        return (self.strength + self.health) * self.karma


class SocialBot(Individual):
    """Бот с социальными параметрами и стратегией."""

    def __init__(self, x: float, z: float, name: str, color: Tuple[int, int, int],
                 strength: int = 10, health: int = 100, karma: float = 0.5):
        super().__init__(x, z)
        self.bot_name = name
        self.bot_color = color
        self.stats = BotStats(
            name=name, color=color, strength=strength,
            health=health, karma=karma
        )
        self.emotion_system = EmotionSystem()
        self.strategy = None  # Будет установлен позднее

        self.body_w = 0.8
        self.body_d = 0.5
        self.body_h = 1.2
        self.head_r = 0.3

    def collect_food(self) -> bool:
        self.stats.food_collected += 1
        self.stats.strength += 1
        return True

    def take_damage(self, damage: int) -> bool:
        self.stats.health -= damage
        if self.stats.health <= 0:
            self.stats.health = 0
            self.stats.survived = False
            return False
        return True

    def _clamp_color(self, value: int) -> int:
        return max(0, min(255, int(value)))

    def draw(self, screen, world_to_screen_func, scale):
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

        color_body = self.bot_color
        color_dark = tuple(self._clamp_color(c - 50) for c in self.bot_color)

        pygame.draw.polygon(screen, color_body, top_points)
        for i in range(4):
            j = (i + 1) % 4
            pts = [base_points[i], base_points[j], top_points[j], top_points[i]]
            pygame.draw.polygon(screen, color_dark, pts)
        pygame.draw.polygon(screen, color_dark, base_points)

        head_offset_z = 0.2
        head_x, head_z = rotate_point(0, head_offset_z)
        head_y = self.body_h + self.head_r * 0.8
        head_screen = world_to_screen_func(head_x, head_z, head_y)
        rad_px = int(self.head_r * scale)
        if rad_px > 1:
            pygame.draw.circle(screen, (255, 200, 150), head_screen, rad_px)

        font = pygame.font.Font(None, 16)
        name_text = font.render(self.bot_name, True, (255, 255, 255))
        name_pos = world_to_screen_func(cx, cz, self.body_h + 0.5)
        screen.blit(name_text, (name_pos[0] - 20, name_pos[1] - 10))

        health_pos = world_to_screen_func(cx, cz, self.body_h + 0.2)
        bar_width = 30
        bar_height = 4
        health_ratio = max(0, min(1.0, self.stats.health / 100.0))

        pygame.draw.rect(screen, (50, 50, 50),
                         (health_pos[0] - bar_width // 2, health_pos[1], bar_width, bar_height))

        red = self._clamp_color(255 * (1 - health_ratio))
        green = self._clamp_color(255 * health_ratio)
        health_color = (red, green, 0)

        pygame.draw.rect(screen, health_color,
                         (health_pos[0] - bar_width // 2, health_pos[1],
                          int(bar_width * health_ratio), bar_height))


class StrategyGenerator(nn.Module):
    """Генератор стратегий (GAN)."""

    def __init__(self, latent_dim=16, strategy_dim=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.strategy_dim = strategy_dim

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, strategy_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)


class StrategyDiscriminator(nn.Module):
    """Дискриминатор стратегий (GAN)."""

    def __init__(self, strategy_dim=6):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(strategy_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, s):
        return self.model(s)


class StrategyGAN:
    """GAN для генерации стратегий поведения."""

    def __init__(self, latent_dim=16, strategy_dim=6, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        self.strategy_dim = strategy_dim

        self.generator = StrategyGenerator(latent_dim, strategy_dim).to(self.device)
        self.discriminator = StrategyDiscriminator(strategy_dim).to(self.device)

        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=0.001)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()

        self.g_losses = []
        self.d_losses = []

    def train(self, real_strategies, epochs=100):
        """Обучает GAN на реальных стратегиях."""
        if len(real_strategies) == 0:
            return

        # Преобразуем стратегии в тензоры
        real_tensors = torch.FloatTensor(real_strategies).to(self.device)
        dataset_size = len(real_tensors)

        for epoch in range(epochs):
            # Перемешиваем данные
            indices = torch.randperm(dataset_size)
            shuffled = real_tensors[indices]

            batch_size = min(8, dataset_size)

            for i in range(0, dataset_size, batch_size):
                batch = shuffled[i:i + batch_size]
                batch_size_actual = batch.size(0)

                if batch_size_actual < 2:
                    continue

                real_labels = torch.ones(batch_size_actual, 1).to(self.device)
                fake_labels = torch.zeros(batch_size_actual, 1).to(self.device)

                # --- Тренируем дискриминатор ---
                self.optimizer_d.zero_grad()

                real_output = self.discriminator(batch)
                d_loss_real = self.criterion(real_output, real_labels)

                z = torch.randn(batch_size_actual, self.latent_dim).to(self.device)
                fake_strategies = self.generator(z)
                fake_output = self.discriminator(fake_strategies.detach())
                d_loss_fake = self.criterion(fake_output, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.optimizer_d.step()

                # --- Тренируем генератор ---
                self.optimizer_g.zero_grad()

                z = torch.randn(batch_size_actual, self.latent_dim).to(self.device)
                fake_strategies = self.generator(z)
                fake_output = self.discriminator(fake_strategies)

                g_loss = self.criterion(fake_output, real_labels)
                g_loss.backward()
                self.optimizer_g.step()

                self.g_losses.append(g_loss.item())
                self.d_losses.append(d_loss.item())

    def generate_strategies(self, n=5) -> List[Dict]:
        """Генерирует новые стратегии."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim).to(self.device)
            strategies_tensor = self.generator(z).cpu().numpy()
        self.generator.train()

        # Преобразуем в словари
        strategies = []
        for s in strategies_tensor:
            strategies.append({
                'food_weight': 0.5 + s[0] * 1.5,  # 0.5 - 2.0
                'danger_weight': 0.5 + s[1] * 2.0,  # 0.5 - 2.5
                'social_weight': s[2],  # 0.0 - 1.0
                'exploration_bias': s[3],  # 0.0 - 1.0
                'karma_threshold': 0.2 + s[4] * 0.6,  # 0.2 - 0.8
                'aggressiveness': s[5]  # 0.0 - 1.0
            })
        return strategies


class BotEvolution:
    """Генетический алгоритм для эволюции стратегий."""

    def __init__(self, population_size=10):
        self.population_size = population_size
        self.population = []
        self.fitness_history = []
        self.generation = 0

        self._init_population()

    def _init_population(self):
        """Создает начальную популяцию."""
        for _ in range(self.population_size):
            strategy = {
                'food_weight': random.uniform(0.5, 1.5),
                'danger_weight': random.uniform(0.5, 2.0),
                'social_weight': random.uniform(0.0, 1.0),
                'exploration_bias': random.uniform(0.0, 1.0),
                'karma_threshold': random.uniform(0.2, 0.8),
                'aggressiveness': random.uniform(0.0, 1.0)
            }
            self.population.append(strategy)

    def evaluate_fitness(self, bots_results: List[BotStats]) -> List[float]:
        """Оценивает фитнес каждой стратегии."""
        fitnesses = []
        for i, stats in enumerate(bots_results):
            if i < len(self.population):
                # Фитнес = (сила + здоровье) * карма
                fitness = stats.score
                fitnesses.append(fitness)
            else:
                fitnesses.append(0.0)
        return fitnesses

    def evolve(self, fitnesses: List[float], gan_strategies: List[Dict] = None) -> List[Dict]:
        """
        Выполняет один шаг эволюции.
        """
        self.generation += 1

        # 1. Сортируем по фитнесу
        combined = list(zip(self.population, fitnesses))
        combined.sort(key=lambda x: x[1], reverse=True)

        # 2. Сохраняем историю лучшего фитнеса
        if combined:
            self.fitness_history.append(combined[0][1])

        # 3. Отбор элиты (лучшие 30%)
        elite_count = max(1, int(self.population_size * 0.3))
        elite = [s for s, f in combined[:elite_count]]

        # 4. Добавляем GAN-стратегии
        new_population = elite.copy()
        if gan_strategies:
            new_population.extend(gan_strategies[:3])

        # 5. Заполняем оставшиеся места потомками
        while len(new_population) < self.population_size:
            # Выбираем родителей из элиты
            parent1 = random.choice(elite)
            parent2 = random.choice(elite)

            # Кроссовер
            child = {}
            for key in parent1.keys():
                if random.random() < 0.5:
                    child[key] = parent1[key]
                else:
                    child[key] = parent2[key]

            # Мутация
            for key in child.keys():
                if random.random() < 0.15:
                    mutation_strength = random.uniform(-0.15, 0.15)
                    child[key] = max(0, min(1 if key != 'food_weight' and key != 'danger_weight' else 3,
                                            child[key] + mutation_strength))
                    if key == 'food_weight':
                        child[key] = max(0.3, min(2.0, child[key]))
                    elif key == 'danger_weight':
                        child[key] = max(0.3, min(2.5, child[key]))
                    elif key == 'karma_threshold':
                        child[key] = max(0.1, min(0.9, child[key]))

            new_population.append(child)

        self.population = new_population[:self.population_size]
        return self.population


class SocialScenarioEvolution:
    """
    Сценарий с эволюцией стратегий через GAN + GA.
    """

    def __init__(self, world_size: int = 20, cell_size: int = 40,
                 population_size: int = 3, generations: int = 5):
        self.world_size = world_size
        self.cell_size = cell_size
        self.population_size = population_size
        self.max_generations = generations

        # Создаем мир
        self.world = World(width=1200, height=800, world_size=world_size, cell_size=cell_size)

        # Создаем GAN и GA
        self.gan = StrategyGAN(latent_dim=16, strategy_dim=6)
        self.evolution = BotEvolution(population_size=population_size * 3)

        # История эволюции
        self.generation_history = []
        self.best_fitness_history = []

        # Создаем ботов
        self.bots = []
        self._create_bots()
        self._populate_world()

        # Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((1600, 900))
        pygame.display.set_caption("Эволюция стратегий ботов (GAN + GA)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 16)
        self.big_font = pygame.font.Font(None, 28)
        self.explosions = []

        self.step_count = 0
        self.max_steps = 300
        self.generation_count = 0
        self.running = True

        print("=" * 60)
        print("ЭВОЛЮЦИЯ СТРАТЕГИЙ БОТОВ (GAN + GA)")
        print("=" * 60)
        print(f"Популяция: {population_size * 3} стратегий")
        print(f"Поколений: {generations}")
        print("=" * 60)

    def _create_bots(self):
        """Создает ботов с разными стратегиями."""
        colors = [
            (245, 235, 220),  # Евро - бело-бежевый
            (210, 180, 150),  # Азио - светло-коричневый
            (160, 120, 80)  # Афро - коричневый
        ]
        names = ["Евро", "Азио", "Афро"]

        self.bots = []
        for i in range(self.population_size):
            # Берем стратегию из популяции
            if i < len(self.evolution.population):
                strategy = self.evolution.population[i]
            else:
                strategy = self.evolution.population[0] if self.evolution.population else None

            bot = SocialBot(
                x=random.randint(-5, 5),
                z=random.randint(-5, 5),
                name=names[i % len(names)],
                color=colors[i % len(colors)],
                strength=10 + random.randint(-2, 2),
                health=100 + random.randint(-10, 10),
                karma=0.5
            )
            bot.strategy = strategy
            self.bots.append(bot)

    def _populate_world(self):
        """Заполняет мир объектами."""
        self.world.objects = []
        self.explosions = []

        # Еда
        food_positions = [
            (-8, 0), (8, 0), (0, -8), (0, 8),
            (-5, 5), (5, -5), (-5, -5), (5, 5),
            (-9, 3), (9, -3), (-3, 9), (3, -9)
        ]
        for x, z in food_positions:
            food = Food(x, z, name="apple", obj_type="food", smell="food_smell")
            food.active = True
            self.world.add_object(food)

        # Волк
        predator = Predator(-3, 8, name="wolf", obj_type="predator", smell="predator_smell")
        predator.active = True
        self.world.add_object(predator)

        # Костёр
        fire = GameObject(3, -8, obj_type="fire", temperature=800, size=1.0)
        fire.active = True
        self.world.add_object(fire)

    def _get_nearby_objects(self, x: float, z: float, radius: float = 2.0) -> List:
        nearby = []
        for obj in self.world.objects:
            if hasattr(obj, 'active') and not obj.active:
                continue
            dist = np.sqrt((obj.x - x) ** 2 + (obj.z - z) ** 2)
            if dist < radius:
                nearby.append((obj, dist))
        return nearby

    def _get_bot_action(self, bot: SocialBot) -> Tuple[float, float]:
        """Определяет действие бота на основе его стратегии."""
        dx, dz = 0, 0

        if bot.strategy is None:
            # Стратегия по умолчанию
            food_weight = 1.0
            danger_weight = 1.0
            social_weight = 0.5
            exploration_bias = 0.5
        else:
            s = bot.strategy
            food_weight = s.get('food_weight', 1.0)
            danger_weight = s.get('danger_weight', 1.0)
            social_weight = s.get('social_weight', 0.5)
            exploration_bias = s.get('exploration_bias', 0.5)

        # 1. Проверяем взрывы
        for expl in self.explosions:
            if not expl.active:
                continue
            dist = math.sqrt((bot.x - expl.x) ** 2 + (bot.z - expl.z) ** 2)
            if dist < 5.0:
                speed = 2.0 * danger_weight
                if dist > 0.1:
                    dx = (bot.x - expl.x) / dist * speed
                    dz = (bot.z - expl.z) / dist * speed
                return dx, dz

        # 2. Проверяем опасные объекты
        danger_nearby = self._get_nearby_objects(bot.x, bot.z, radius=3.0)
        for obj, dist in danger_nearby:
            if hasattr(obj, 'type') and obj.type in ['predator', 'fire']:
                if dist > 0.1:
                    speed = 0.3 * danger_weight
                    dx = (bot.x - obj.x) / dist * speed
                    dz = (bot.z - obj.z) / dist * speed
                return dx, dz

        # 3. Ищем еду
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
            speed = 0.15 * food_weight
            dx = (best_food.x - bot.x) / best_dist * speed
            dz = (best_food.z - bot.z) / best_dist * speed

        # Добавляем исследовательское поведение
        if random.random() < exploration_bias * 0.1:
            dx += random.uniform(-0.05, 0.05)
            dz += random.uniform(-0.05, 0.05)

        return dx, dz

    def _process_bot_emotion(self, bot: SocialBot) -> Optional[str]:
        """Обрабатывает эмоции бота."""
        nearby = self._get_nearby_objects(bot.x, bot.z, radius=2.0)

        for obj, dist in nearby:
            if hasattr(obj, 'type'):
                if obj.type == 'food' and obj.active:
                    bot.collect_food()
                    obj.active = False
                    return 'joy'
                elif obj.type == 'predator':
                    bot.take_damage(10)
                    return 'fear'
                elif obj.type == 'fire':
                    bot.take_damage(5)
                    return 'fear'

        # Социальное взаимодействие
        for other in self.bots:
            if other is bot or not other.stats.survived:
                continue
            dist = np.sqrt((other.x - bot.x) ** 2 + (other.z - bot.z) ** 2)
            if dist < 2.0:
                if bot.stats.food_collected > other.stats.food_collected + 2:
                    if bot.strategy and bot.strategy.get('karma_threshold', 0.5) > 0.6:
                        # Помощь
                        if other.stats.health < 50 and bot.stats.strength > 5:
                            bot.stats.strength -= 2
                            other.stats.health += 10
                            bot.stats.karma = min(1.0, bot.stats.karma + 0.05)
                            return 'love'
                    else:
                        # Конкуренция
                        if other.stats.food_collected > 0:
                            other.stats.food_collected -= 1
                            bot.stats.food_collected += 1
                            bot.stats.karma = max(0.0, bot.stats.karma - 0.05)
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

    def _draw_stats_panel(self, screen, x_offset: int):
        """Рисует панель статистики."""
        panel_rect = pygame.Rect(x_offset, 0, 400, 900)
        pygame.draw.rect(screen, (40, 40, 50), panel_rect)

        title = self.font.render("ЭВОЛЮЦИЯ СТРАТЕГИЙ", True, (255, 255, 255))
        screen.blit(title, (x_offset + 10, 10))

        # Информация о поколении
        gen_text = self.font.render(f"Поколение: {self.generation_count}", True, (200, 200, 200))
        screen.blit(gen_text, (x_offset + 10, 40))

        if self.evolution.fitness_history:
            best_fitness = self.evolution.fitness_history[-1] if self.evolution.fitness_history else 0
            fit_text = self.font.render(f"Лучший фитнес: {best_fitness:.1f}", True, (100, 255, 100))
            screen.blit(fit_text, (x_offset + 10, 65))

        # Стратегии ботов
        y = 100
        for i, bot in enumerate(self.bots):
            if not bot.stats.survived:
                continue

            color = bot.bot_color
            name_text = self.small_font.render(f"{bot.bot_name}", True, color)
            screen.blit(name_text, (x_offset + 10, y))
            y += 20

            # Параметры стратегии
            if bot.strategy:
                s = bot.strategy
                strat_text = self.small_font.render(
                    f"Еда:{s.get('food_weight', 0):.2f} Опас:{s.get('danger_weight', 0):.2f} Карма:{s.get('karma_threshold', 0):.2f}",
                    True, (200, 200, 200)
                )
                screen.blit(strat_text, (x_offset + 10, y))
                y += 18

            # Статистика
            stats_text = self.small_font.render(
                f"Сила:{bot.stats.strength} Здоровье:{bot.stats.health} Карма:{bot.stats.karma:.2f} 🍎:{bot.stats.food_collected}",
                True, (200, 200, 200)
            )
            screen.blit(stats_text, (x_offset + 10, y))
            y += 25

        # История фитнеса
        y = 450
        hist_text = self.small_font.render("История фитнеса:", True, (200, 200, 200))
        screen.blit(hist_text, (x_offset + 10, y))
        y += 20

        if self.evolution.fitness_history:
            # Показываем последние 10 поколений
            history = self.evolution.fitness_history[-10:]
            for i, f in enumerate(history):
                bar_width = int(f * 1.5)
                bar_rect = pygame.Rect(x_offset + 10, y + i * 12, min(bar_width, 380), 10)
                pygame.draw.rect(screen, (100, 200, 100), bar_rect)
                text = self.small_font.render(f"{f:.1f}", True, (255, 255, 255))
                screen.blit(text, (x_offset + 100, y + i * 12))

        # Подсказка
        y = 850
        help_text = self.small_font.render("Пробел - пауза | Esc - выход", True, (150, 150, 150))
        screen.blit(help_text, (x_offset + 10, y))

    def _show_generation_results(self):
        """Показывает результаты поколения."""
        print("\n" + "=" * 60)
        print(f"ПОКОЛЕНИЕ {self.generation_count} - РЕЗУЛЬТАТЫ")
        print("=" * 60)

        sorted_bots = sorted(self.bots, key=lambda b: b.stats.score, reverse=True)
        for i, bot in enumerate(sorted_bots):
            status = "✅" if bot.stats.survived else "💀"
            print(f"{i + 1}. {status} {bot.bot_name}:")
            print(f"   Сила: {bot.stats.strength}, Здоровье: {bot.stats.health}, Карма: {bot.stats.karma:.2f}")
            print(f"   🍎: {bot.stats.food_collected}, Счет: {bot.stats.score:.1f}")
            if bot.strategy:
                s = bot.strategy
                print(f"   Стратегия: Еда={s.get('food_weight', 0):.2f}, Опас={s.get('danger_weight', 0):.2f}")
            print()

        if sorted_bots:
            winner = sorted_bots[0]
            print(f"🏆 ПОБЕДИТЕЛЬ: {winner.bot_name} с результатом {winner.stats.score:.1f}")

        if self.evolution.fitness_history:
            print(f"📈 Лучший фитнес: {self.evolution.fitness_history[-1]:.1f}")
        print("=" * 60)

    def run_generation(self):
        """Запускает одно поколение."""
        # Сбрасываем ботов
        for bot in self.bots:
            bot.x = random.randint(-5, 5)
            bot.z = random.randint(-5, 5)
            bot.stats.health = 100 + random.randint(-10, 10)
            bot.stats.strength = 10 + random.randint(-2, 2)
            bot.stats.food_collected = 0
            bot.stats.survived = True
            bot.stats.karma = 0.5

        self._populate_world()
        self.explosions = []
        self.step_count = 0

        explosion_timer = 150
        paused = False

        print(f"\n🚀 Запуск поколения {self.generation_count + 1}...")

        while self.step_count < self.max_steps:
            # Обработка событий
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        return
                    if event.key == pygame.K_SPACE:
                        paused = not paused

            if paused:
                continue

            # Обновляем ботов
            for bot in self.bots:
                if not bot.stats.survived:
                    continue

                dx, dz = self._get_bot_action(bot)
                self._move_bot(bot, dx, dz)

                emotion = self._process_bot_emotion(bot)
                if emotion:
                    pass  # Можно логировать эмоции

                bot.update(self.world)

            # Взрывы
            if self.step_count % explosion_timer == 0 and self.step_count > 0:
                x = random.randint(-6, 6)
                z = random.randint(-6, 6)
                for bot in self.bots:
                    if abs(bot.x - x) < 2 and abs(bot.z - z) < 2:
                        x = random.randint(-6, 6)
                        z = random.randint(-6, 6)
                        break
                explosion = Explosion(x, z)
                self.explosions.append(explosion)

            # Обновляем взрывы
            for expl in self.explosions[:]:
                expl.update(1.0)
                if not expl.active:
                    self.explosions.remove(expl)

            # Отрисовка
            self.screen.fill((30, 30, 30))
            self.world.draw(self.screen, None)

            for expl in self.explosions:
                expl.draw(self.screen, self.world.world_to_screen)

            for bot in self.bots:
                if bot.stats.survived:
                    bot.draw(self.screen, self.world.world_to_screen, self.world.get_scale())

            self._draw_stats_panel(self.screen, 1200)

            pygame.display.flip()
            self.clock.tick(20)

            self.step_count += 1

        # Собираем результаты
        fitnesses = self.evolution.evaluate_fitness([b.stats for b in self.bots])

        # Показываем результаты
        self._show_generation_results()

        # Эволюционируем
        self.generation_count += 1

        if self.generation_count < self.max_generations:
            # Генерируем новые стратегии через GAN
            best_strategies = []
            for bot in sorted(self.bots, key=lambda b: b.stats.score, reverse=True)[:3]:
                if bot.strategy:
                    best_strategies.append(bot.strategy)

            if best_strategies:
                # Обучаем GAN на лучших стратегиях
                strategy_vectors = []
                for s in best_strategies:
                    vec = [
                        (s.get('food_weight', 1.0) - 0.5) / 1.5,
                        (s.get('danger_weight', 1.0) - 0.5) / 2.0,
                        s.get('social_weight', 0.5),
                        s.get('exploration_bias', 0.5),
                        (s.get('karma_threshold', 0.5) - 0.2) / 0.6,
                        s.get('aggressiveness', 0.5)
                    ]
                    strategy_vectors.append(vec)

                if strategy_vectors:
                    self.gan.train(strategy_vectors, epochs=50)
                    gan_strategies = self.gan.generate_strategies(n=3)
                else:
                    gan_strategies = None
            else:
                gan_strategies = None

            # Эволюционируем популяцию
            self.evolution.evolve(fitnesses, gan_strategies)

            # Обновляем стратегии ботов
            for i, bot in enumerate(self.bots):
                if i < len(self.evolution.population):
                    bot.strategy = self.evolution.population[i]

    def run(self):
        """Запускает полную эволюцию."""
        for gen in range(self.max_generations):
            if not self.running:
                break
            self.run_generation()

        print("\n" + "=" * 60)
        print("ЭВОЛЮЦИЯ ЗАВЕРШЕНА!")
        print("=" * 60)
        print(f"Всего поколений: {self.generation_count}")
        if self.evolution.fitness_history:
            print(f"Лучший фитнес: {max(self.evolution.fitness_history):.1f}")
            print(f"Средний фитнес: {sum(self.evolution.fitness_history) / len(self.evolution.fitness_history):.1f}")
        print("=" * 60)

        # Показываем финальные стратегии
        print("\n🏆 ЛУЧШИЕ СТРАТЕГИИ:")
        sorted_bots = sorted(self.bots, key=lambda b: b.stats.score, reverse=True)
        for i, bot in enumerate(sorted_bots[:3]):
            if bot.strategy:
                s = bot.strategy
                print(f"{i + 1}. {bot.bot_name}:")
                print(f"   Еда: {s.get('food_weight', 0):.2f}")
                print(f"   Опасность: {s.get('danger_weight', 0):.2f}")
                print(f"   Социальность: {s.get('social_weight', 0):.2f}")
                print(f"   Исследование: {s.get('exploration_bias', 0):.2f}")
                print(f"   Порог кармы: {s.get('karma_threshold', 0):.2f}")

        pygame.quit()


def main():
    scenario = SocialScenarioEvolution(
        world_size=20,
        cell_size=40,
        population_size=3,
        generations=5
    )
    scenario.run()


if __name__ == "__main__":
    main()