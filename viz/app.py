# viz/app.py
import json
from core.world import World
from core.individual import Individual
from core.objects import GameObject, Predator, Food
from core.genome import Genome
from db.connector import load_reflex_rules, load_instinct_patterns, load_best_genome
from typing import List, Dict, Optional, Tuple, Any, Union

if __name__ == "__main__":
    # Загружаем правила из БД (они нужны для работы рефлексов и инстинктов) если в БД пусто, то берём дефолтный хардкод
    # reflex_rules = load_reflex_rules()
    reflex_rules = load_reflex_rules() or [
        {'sense_type': 'smell', 'signal_type': 'food_smell', 'action': 'grab'},
        {'sense_type': 'smell', 'signal_type': 'predator_smell', 'action': 'avoid'},
    ]
    # instinct_patterns = load_instinct_patterns()
    # Загружаем паттерны из БД (они нужны для работы инстинктов) если в БД пусто, то берём дефолтный хардкод
    instinct_patterns = load_instinct_patterns() or [
        {'pattern': {'signals': {'sound': 'loud crash'}}, 'action': {'action': 'run_away'}}
    ]
    print(f"Загружено {len(reflex_rules)} правил рефлексов")
    print(f"Загружено {len(instinct_patterns)} паттернов инстинктов")

    # Создаём мир
    world = World(width=1200, height=800, world_size=20, cell_size=40)

    # Пытаемся загрузить лучший геном из БД
    best_genome_data = load_best_genome()
    # Создаём бота – правила передаём всегда
    if best_genome_data:
        genome = Genome.from_dict(best_genome_data)
        bot = Individual(x=0, z=0, genome=genome,
                         reflex_rules=reflex_rules,
                         instinct_patterns=instinct_patterns)
    else:
        bot = Individual(x=0, z=0,
                         reflex_rules=reflex_rules,
                         instinct_patterns=instinct_patterns,
                         move_delay=5)

    # Добавляем объекты в мир (костры, волк, яблоки)
    fire_positions = [(-4, -4), (6, -2), (0, 8), (-8, -8)]
    for (x, z) in fire_positions:
        fire = GameObject(x, z, obj_type="fire", temperature=800, size=1.0)
        world.add_object(fire)

    wolf = Predator(x=8, z=6, name="wolf", obj_type="predator", smell="predator_smell", sound="predator_roar")
    world.add_object(wolf)

    apple1 = Food(x=-6, z=4, name="apple", obj_type="food", smell="food_smell")
    apple2 = Food(x=8, z=-6, name="apple", obj_type="food", smell="food_smell")
    world.add_object(apple1)
    world.add_object(apple2)

    print(bot.genome.params)

    # Запуск визуализации
    world.run(bot)

