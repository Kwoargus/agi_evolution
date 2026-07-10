# scripts/visualize_bot_evolution.py
"""
Визуализация эволюции бота: сравнение поведения до и после эволюции.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
from typing import List, Dict, Any
import random
import time

from core.world import World
from core.individual import Individual
from core.genome import Genome
from core.objects import Food, Predator, GameObject
from core.evolution import EvolutionEngine
from db.connector import load_best_genome, load_reflex_rules, load_instinct_patterns


class BotComparator:
    """
    Сравнивает поведение ботов до и после эволюции.
    """

    def __init__(self, world_size=20):
        self.world_size = world_size
        self.world = World(width=800, height=600, world_size=world_size, cell_size=40)
        self._setup_world()

        # Загружаем правила
        self.reflex_rules = load_reflex_rules() or [
            {'sense_type': 'smell', 'signal_type': 'food_smell', 'action': 'grab'},
            {'sense_type': 'smell', 'signal_type': 'predator_smell', 'action': 'avoid'},
        ]
        self.instinct_patterns = load_instinct_patterns() or [
            {'pattern': {'signals': {'sound': 'loud crash'}}, 'action': {'action': 'run_away'}}
        ]

        # Создаем ботов
        self.base_bot = None  # Без эволюции
        self.evolved_bot = None  # После эволюции

        # История траекторий
        self.base_trajectory = []
        self.evolved_trajectory = []

        # Статистика
        self.stats = {
            'base': {'nodes': 0, 'food': 0, 'survival': 0, 'actions': []},
            'evolved': {'nodes': 0, 'food': 0, 'survival': 0, 'actions': []}
        }

    def _setup_world(self):
        """Настраивает мир с объектами."""
        # Очищаем мир
        self.world.objects = []

        # Добавляем объекты
        food_positions = [(-6, 6), (6, -6), (-6, -6), (6, 6), (0, 8), (8, 0), (-8, 0), (0, -8)]
        for x, z in food_positions[:4]:
            self.world.add_object(Food(x, z, name="apple", obj_type="food", smell="food_smell"))

        predator_positions = [(7, 7), (-7, -7), (7, -7), (-7, 7)]
        for x, z in predator_positions[:2]:
            self.world.add_object(Predator(x, z, name="wolf", obj_type="predator", smell="predator_smell"))

        # Огонь
        fire_positions = [(-4, -4), (4, 4)]
        for x, z in fire_positions:
            self.world.add_object(GameObject(x, z, obj_type="fire", temperature=800, size=1.0))

    def create_base_bot(self):
        """Создает базового бота (без эволюции)."""
        genome = Genome({
            'move_delay': 5,
            'step_size': 2.0,
            'reflex_priorities': {'grab': 0.5, 'avoid': 0.5},
            'instinct_priorities': {'run_away': 0.5},
            'exploration_bias': 0.5,
            'max_steps': 500
        })

        self.base_bot = Individual(
            x=0, z=0,
            genome=genome,
            reflex_rules=self.reflex_rules,
            instinct_patterns=self.instinct_patterns,
            move_delay=5
        )
        return self.base_bot

    def create_evolved_bot(self):
        """Создает эволюционировавшего бота (загружает лучший геном из БД)."""
        # Пытаемся загрузить лучший геном из БД
        best_genome_data = load_best_genome()

        if best_genome_data:
            genome = Genome.from_dict(best_genome_data)
            self.evolved_bot = Individual(
                x=0, z=0,
                genome=genome,
                reflex_rules=self.reflex_rules,
                instinct_patterns=self.instinct_patterns
            )
        else:
            # Если нет сохраненного генома, создаем улучшенного бота
            print("⚠️ Лучший геном не найден. Создаем улучшенного бота...")
            genome = Genome({
                'move_delay': 2,  # Быстрее
                'step_size': 2.0,
                'reflex_priorities': {'grab': 0.8, 'avoid': 0.7},
                'instinct_priorities': {'run_away': 0.9},
                'exploration_bias': 0.8,
                'max_steps': 500
            })
            self.evolved_bot = Individual(
                x=0, z=0,
                genome=genome,
                reflex_rules=self.reflex_rules,
                instinct_patterns=self.instinct_patterns
            )

        return self.evolved_bot

    def run_bot(self, bot: Individual, max_steps: int = 300,
                track_actions: bool = False) -> Dict[str, Any]:
        """
        Запускает бота в мире и собирает статистику.
        """
        # Сбрасываем мир и бота
        self._setup_world()
        bot.reset()
        bot.x = 0
        bot.z = 0

        trajectory = []
        actions = []
        food_collected = 0

        for step in range(max_steps):
            if not bot.alive:
                break

            # Запоминаем позицию
            trajectory.append((bot.x, bot.z))

            # Сохраняем действие (если нужно)
            if track_actions:
                # Простое логирование действий
                if hasattr(bot, 'last_action'):
                    actions.append(bot.last_action)

            # Обновляем бота
            bot.update(self.world)

            # Считаем собранную еду
            if hasattr(bot, 'food_collected'):
                food_collected = bot.food_collected

        return {
            'trajectory': trajectory,
            'nodes_visited': len(bot.visited_nodes),
            'food_collected': food_collected,
            'alive': bot.alive,
            'steps': step + 1,
            'actions': actions
        }

    def compare_bots(self, max_steps: int = 300) -> Dict[str, Any]:
        """
        Сравнивает поведение базового и эволюционировавшего бота.
        """
        print("🔄 Сравнение поведения ботов...")

        # Запускаем базового бота
        print("  🐣 Запуск базового бота...")
        base_result = self.run_bot(self.base_bot, max_steps, track_actions=True)

        # Запускаем эволюционировавшего бота
        print("  🚀 Запуск эволюционировавшего бота...")
        evolved_result = self.run_bot(self.evolved_bot, max_steps, track_actions=True)

        # Собираем статистику
        comparison = {
            'base': {
                'trajectory': base_result['trajectory'],
                'nodes_visited': base_result['nodes_visited'],
                'food_collected': base_result['food_collected'],
                'alive': base_result['alive'],
                'steps': base_result['steps']
            },
            'evolved': {
                'trajectory': evolved_result['trajectory'],
                'nodes_visited': evolved_result['nodes_visited'],
                'food_collected': evolved_result['food_collected'],
                'alive': evolved_result['alive'],
                'steps': evolved_result['steps']
            },
            'improvements': {
                'nodes': evolved_result['nodes_visited'] - base_result['nodes_visited'],
                'food': evolved_result['food_collected'] - base_result['food_collected'],
                'survival': evolved_result['alive'] - base_result['alive'],
                'steps': evolved_result['steps'] - base_result['steps']
            }
        }

        return comparison

    def visualize_comparison(self, comparison: Dict[str, Any]):
        """
        Визуализирует сравнение ботов.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Сравнение поведения: Базовый бот vs Эволюционировавший бот', fontsize=16)

        # 1. Траектории движения
        ax1 = axes[0, 0]
        self._plot_trajectory(ax1, comparison['base']['trajectory'],
                              'Базовый бот', color='red', alpha=0.7)
        ax1.set_title('Траектория базового бота')
        ax1.set_xlim(-10, 10)
        ax1.set_ylim(-10, 10)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        self._plot_trajectory(ax2, comparison['evolved']['trajectory'],
                              'Эволюционировавший бот', color='blue', alpha=0.7)
        ax2.set_title('Траектория эволюционировавшего бота')
        ax2.set_xlim(-10, 10)
        ax2.set_ylim(-10, 10)
        ax2.grid(True, alpha=0.3)

        # 3. Сравнение траекторий
        ax3 = axes[0, 2]
        self._plot_trajectory(ax3, comparison['base']['trajectory'],
                              'Базовый', color='red', alpha=0.5, linewidth=1)
        self._plot_trajectory(ax3, comparison['evolved']['trajectory'],
                              'Эволюционировавший', color='blue', alpha=0.5, linewidth=2)
        ax3.set_title('Сравнение траекторий')
        ax3.set_xlim(-10, 10)
        ax3.set_ylim(-10, 10)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Статистика
        ax4 = axes[1, 0]
        self._plot_stats(ax4, comparison)

        # 5. Улучшения
        ax5 = axes[1, 1]
        self._plot_improvements(ax5, comparison)

        # 6. Тепловая карта посещений
        ax6 = axes[1, 2]
        self._plot_heatmap(ax6, comparison)

        plt.tight_layout()
        plt.show()

    def _plot_trajectory(self, ax, trajectory, label, color='blue', alpha=0.7, linewidth=2):
        """Рисует траекторию бота."""
        if not trajectory:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)
            return

        x = [p[0] for p in trajectory]
        z = [p[1] for p in trajectory]

        ax.plot(x, z, color=color, alpha=alpha, linewidth=linewidth, label=label)
        ax.scatter(x[0], z[0], color='green', s=100, marker='o', label='Старт')
        ax.scatter(x[-1], z[-1], color='red', s=100, marker='x', label='Финиш')

    def _plot_stats(self, ax, comparison):
        """Отображает статистику."""
        base = comparison['base']
        evolved = comparison['evolved']

        metrics = ['Посещено узлов', 'Собрано еды', 'Выжил', 'Шагов']
        base_values = [base['nodes_visited'], base['food_collected'],
                       1 if base['alive'] else 0, base['steps']]
        evolved_values = [evolved['nodes_visited'], evolved['food_collected'],
                          1 if evolved['alive'] else 0, evolved['steps']]

        x = np.arange(len(metrics))
        width = 0.35

        ax.bar(x - width / 2, base_values, width, label='Базовый', color='red', alpha=0.7)
        ax.bar(x + width / 2, evolved_values, width, label='Эволюционировавший', color='blue', alpha=0.7)

        ax.set_title('Сравнение метрик')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Добавляем значения на столбцы
        for i, (b, e) in enumerate(zip(base_values, evolved_values)):
            ax.text(i - width / 2, b + 0.5, str(b), ha='center', va='bottom', fontsize=9)
            ax.text(i + width / 2, e + 0.5, str(e), ha='center', va='bottom', fontsize=9)

    def _plot_improvements(self, ax, comparison):
        """Отображает улучшения."""
        improvements = comparison['improvements']

        metrics = ['Узлы', 'Еда', 'Выживание', 'Шаги']
        values = [improvements['nodes'], improvements['food'],
                  improvements['survival'] * 100, improvements['steps']]

        colors = ['green' if v > 0 else 'red' for v in values]

        ax.bar(metrics, values, color=colors, alpha=0.7)
        ax.set_title('Улучшения (Эволюционировавший - Базовый)')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)

        # Добавляем значения
        for i, v in enumerate(values):
            ax.text(i, v + (0.5 if v >= 0 else -1), f'{v:.1f}',
                    ha='center', va='bottom' if v >= 0 else 'top', fontsize=10, fontweight='bold')

    def _plot_heatmap(self, ax, comparison):
        """Рисует тепловую карту посещений."""
        # Создаем сетку
        grid_size = 20
        grid = np.zeros((grid_size, grid_size))

        # Заполняем из траекторий
        for traj in [comparison['base']['trajectory'], comparison['evolved']['trajectory']]:
            for x, z in traj:
                grid_x = int((x + 10) / 20 * grid_size) % grid_size
                grid_z = int((z + 10) / 20 * grid_size) % grid_size
                grid[grid_x, grid_z] += 1

        # Показываем разницу (эволюционировавший - базовый)
        im = ax.imshow(grid.T, cmap='hot', interpolation='nearest', origin='lower')
        ax.set_title('Тепловая карта посещений')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        plt.colorbar(im, ax=ax, label='Посещений')

    def run_full_comparison(self):
        """Запускает полное сравнение и визуализацию."""
        print("=" * 60)
        print("СРАВНЕНИЕ БОТОВ: ДО И ПОСЛЕ ЭВОЛЮЦИИ")
        print("=" * 60)

        # Создаем ботов
        self.create_base_bot()
        self.create_evolved_bot()

        # Сравниваем
        comparison = self.compare_bots(max_steps=300)

        # Выводим результаты
        print("\n📊 РЕЗУЛЬТАТЫ СРАВНЕНИЯ:")
        print("-" * 40)
        print(f"  Показатель          | Базовый | Эволюционировавший | Улучшение")
        print("-" * 40)
        print(f"  Посещено узлов      | {comparison['base']['nodes_visited']:6}   | {comparison['evolved']['nodes_visited']:6}          | +{comparison['improvements']['nodes']}")
        print(f"  Собрано еды         | {comparison['base']['food_collected']:6}   | {comparison['evolved']['food_collected']:6}          | +{comparison['improvements']['food']}")
        print(f"  Выжил               | {comparison['base']['alive']:6}   | {comparison['evolved']['alive']:6}          | +{comparison['improvements']['survival']}")
        print(f"  Шагов сделано       | {comparison['base']['steps']:6}   | {comparison['evolved']['steps']:6}          | +{comparison['improvements']['steps']}")
        print("-" * 40)

        # Визуализируем
        self.visualize_comparison(comparison)

        return comparison


def run_evolution_and_compare():
    """
    Запускает эволюцию, а затем сравнивает ботов.
    """
    print("🚀 Запуск эволюции и сравнение ботов...")

    # 1. Запускаем эволюцию (если нужно)
    world = World(width=800, height=600, world_size=20, cell_size=40)

    # Добавляем объекты
    from core.objects import GameObject, Predator, Food
    world.add_object(GameObject(-5, -5, obj_type="fire", temperature=800, size=1.0))
    world.add_object(Predator(5, 5, name="wolf", obj_type="predator", smell="predator_smell"))
    world.add_object(Food(-6, 6, name="apple", obj_type="food", smell="food_smell"))
    world.add_object(Food(6, -6, name="apple", obj_type="food", smell="food_smell"))

    # Создаем движок эволюции
    engine = EvolutionEngine(
        world=world,
        population_size=15,
        generations=20,  # Небольшое количество для демонстрации
        steps_per_episode=300,
        elite_count=3,
        mutation_rate=0.15,
        use_gan=True,
        gan_training_epochs=3
    )

    # Запускаем эволюцию
    print("🔄 Запуск эволюции...")
    history = engine.run(save_to_db=True)
    print(f"✅ Эволюция завершена. Лучший фитнес: {max(history):.2f}")

    # 2. Сравниваем ботов
    comparator = BotComparator()
    comparison = comparator.run_full_comparison()

    return comparison


if __name__ == "__main__":
    # Запускаем сравнение
    comparison = run_evolution_and_compare()