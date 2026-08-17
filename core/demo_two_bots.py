# core/demo_two_bots.py

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.world import World
from core.individual import Individual
from core.genome import Genome
import random

def create_bot_with_emotions(x, z, emotions_enabled, health=100, food_reserve=0):
    """Создаёт бота с заданными параметрами."""
    genome = Genome({
        'move_delay': 10,
        'step_size': 1.0,
        'reflex_rules': [],
        'instinct_patterns': [],
        'max_steps': 1000
    })
    bot = Individual(x=x, z=z, genome=genome)
    bot.health = health
    bot.food_reserve = food_reserve
    bot.emotions_enabled = emotions_enabled
    bot.alive = True
    bot.food_collected = 0
    bot.total_reward = 0.0
    # Устанавливаем более медленное движение
    bot.step_size = 1.0
    bot.move_delay = 10
    return bot

def main():
    # 1. Создаём мир с большим количеством объектов
    objects_config = [
        # Еда (много)
        # {'type': 'food', 'x': -6, 'z': -6},
        # {'type': 'food', 'x': -6, 'z': 6},
        # {'type': 'food', 'x': 6, 'z': -6},
        # {'type': 'food', 'x': 6, 'z': 6},
        # {'type': 'food', 'x': 0, 'z': 10},
        # {'type': 'food', 'x': 10, 'z': 0},
        # {'type': 'food', 'x': -10, 'z': 0},
        # {'type': 'food', 'x': 0, 'z': -10},
        # {'type': 'food', 'x': 12, 'z': 12},
        # {'type': 'food', 'x': -12, 'z': -12},
        # {'type': 'food', 'x': 10, 'z': 4},
        # {'type': 'food', 'x': 6, 'z': 10},
        # {'type': 'food', 'x': -10, 'z': -5},
        # {'type': 'food', 'x': -6, 'z': -10},
        # Хищники
        {'type': 'predator', 'x': 8, 'z': 6},
        {'type': 'predator', 'x': -8, 'z': -8},
        {'type': 'predator', 'x': 10, 'z': -8},
        {'type': 'predator', 'x': -8, 'z': 4},
    ]

    world = World(
        width=1200,
        height=800,
        world_size=25,
        cell_size=40,
        objects_config=objects_config
    )

    # Устанавливаем таймер взрыва на 5 секунд
    world.explosion_timer = 5.0

    # 2. Создаём двух ботов
    # Бот 1: с эмоциями, здоровый, имеет запас еды
    bot1 = create_bot_with_emotions(
        x=0, z=0,
        emotions_enabled=True,
        health=100,
        food_reserve=5
    )
    # Бот 2: без эмоций, раненый, без еды
    bot2 = create_bot_with_emotions(
        x=8, z=8,
        emotions_enabled=False,
        health=10,  # ранен, чтобы дать время на сбор еды
        food_reserve=0
    )
    # Бот 3: без эмоций, раненый, без еды
    bot3 = create_bot_with_emotions(
        x=6, z=6,
        emotions_enabled=False,
        health=10,  # ранен, чтобы дать время на сбор еды
        food_reserve=0
    )
    # Бот 4: без эмоций, раненый, без еды
    bot4 = create_bot_with_emotions(
        x=-8, z=-88,
        emotions_enabled=False,
        health=10,  # ранен, чтобы дать время на сбор еды
        food_reserve=0
    )
    # Бот 5: без эмоций, раненый, без еды
    bot5 = create_bot_with_emotions(
        x=-4, z=8,
        emotions_enabled=False,
        health=10,  # ранен, чтобы дать время на сбор еды
        food_reserve=0
    )
    # Бот 6: без эмоций, раненый, без еды
    bot6 = create_bot_with_emotions(
        x=8, z=-6,
        emotions_enabled=False,
        health=10,  # ранен, чтобы дать время на сбор еды
        food_reserve=0
    )
    # Бот 7: без эмоций, раненый, без еды
    bot7 = create_bot_with_emotions(
        x=-2, z=4,
        emotions_enabled=False,
        health=10,  # ранен, чтобы дать время на сбор еды
        food_reserve=0
    )
    # Бот 8: без эмоций, раненый, без еды
    bot8 = create_bot_with_emotions(
        x=2, z=-6,
        emotions_enabled=False,
        health=10,  # ранен, чтобы дать время на сбор еды
        food_reserve=0
    )

    # 3. Добавляем ботов в мир
    world.bots = [bot1, bot2, bot3, bot4, bot5, bot6, bot7, bot8]

    # 4. Запускаем симуляцию с FPS=30 для замедления
    print("Запуск демонстрации сочувствия...")
    print("Бот1 (с эмоциями): здоров, имеет еду")
    print("Бот2 (без эмоций): ранен, без еды")
    print("Ожидаемое поведение: Бот1 подойдёт и поделится едой.")
    world.run(bots=[bot1, bot2, bot3, bot4, bot5, bot6, bot7, bot8], fps=10)

if __name__ == "__main__":
    main()



# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#
# from core.world import World
# from core.individual import Individual
# from core.genome import Genome
# import random
#
# def create_bot_with_emotions(x, z, emotions_enabled, health=100, food_reserve=0):
#     """Создаёт бота с заданными параметрами."""
#     genome = Genome({
#         'move_delay': 25,
#         'step_size': 1.0,
#         'reflex_rules': [],
#         'instinct_patterns': [],
#         'max_steps': 1000
#     })
#     bot = Individual(x=x, z=z, genome=genome)
#     bot.health = health
#     bot.food_reserve = food_reserve
#     bot.emotions_enabled = emotions_enabled
#     # Добавляем недостающие атрибуты для совместимости
#     bot.alive = True
#     bot.food_collected = 0
#     bot.total_reward = 0.0
#     return bot
#
# def main():
#     # 1. Создаём мир с объектами (еда и т.д.)
#     objects_config = [
#         {'type': 'predator', 'x': 12, 'z': 12},
#         {'type': 'predator', 'x': -12, 'z': -12},
#         {'type': 'predator', 'x': 0, 'z': 10},
#         {'type': 'predator', 'x': -8, 'z': -8},
#         {'type': 'food', 'x': 2, 'z': 2},
#         {'type': 'food', 'x': -2, 'z': 4},
#         {'type': 'food', 'x': 6, 'z': -3},
#         {'type': 'food', 'x': -5, 'z': -5},
#         {'type': 'food', 'x': 0, 'z': -6},
#         {'type': 'food', 'x': -8, 'z': 8},
#         {'type': 'food', 'x': 8, 'z': -8},
#         {'type': 'food', 'x': 10, 'z': 10},
#         {'type': 'food', 'x': -10, 'z': -10},
#         {'type': 'food', 'x': 4, 'z': -4},
#         {'type': 'food', 'x': 10, 'z': 10},
#         {'type': 'food', 'x': 12, 'z': 12},
#         {'type': 'food', 'x': 14, 'z': 14},
#         # Добавьте другие объекты по желанию
#     ]
#     world = World(
#         width=1200,
#         height=800,
#         world_size=30,  # поменьше, чтобы боты быстрее встретились
#         cell_size=40,
#         objects_config=objects_config
#     )
#
#     # 2. Создаём двух ботов
#     # Бот 1: с эмоциями, здоровый, имеет запас еды
#     bot1 = create_bot_with_emotions(
#         x=0, z=0,
#         emotions_enabled=True,
#         health=100,
#         food_reserve=5  # у него есть еда для передачи
#     )
#     # Бот 2: без эмоций, раненый, без еды
#     bot2 = create_bot_with_emotions(
#         x=8, z=8,
#         emotions_enabled=False,
#         health=10,   # критическое состояние
#         food_reserve=0
#     )
#
#     # 3. Добавляем ботов в мир
#     world.bots = [bot1, bot2]
#
#     # 4. Запускаем симуляцию
#     print("Запуск демонстрации сочувствия...")
#     print("Бот1 (с эмоциями): здоров, имеет еду")
#     print("Бот2 (без эмоций): ранен, без еды")
#     print("Ожидаемое поведение: Бот1 подойдёт и поделится едой.")
#     world.run(bots=[bot1, bot2])
#
# if __name__ == "__main__":
#     main()