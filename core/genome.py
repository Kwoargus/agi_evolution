# core/genome.py
import copy
import random
from typing import List, Dict, Optional, Tuple, Any, Union

class Genome:
    def __init__(self, params=None):
        if params is None:
            # [ru] Значения по умолчанию
            # [en] Default values
            params = {
                'move_delay': 5,
                'step_size': 2.0,
                'reflex_thresholds': {},  # {'food_smell': 0.2, 'predator_smell': 0.1, ...}
                'instinct_priorities': {}, # {'run_away': 1.0, ...}
                'max_steps': 500,
                'step_size': 2.0,  # длина шага (влияет на манёвренность)
                'speed': 1.0,  # множитель скорости (задержка)
                'reflex_priorities': {  # веса для каждого действия
                    'grab': 1.0,
                    'move_on': 0.8,
                    'avoid': 0.5
                },
                'instinct_priorities': {  # веса для инстинктов
                    'run_away': 1.0
                },
                'exploration_bias': 0.5,  # склонность к исследованию новых узлов
                'exploration_rate': 0.0   # параметры, влияющие на исследование
            }
        self.params = params

    def get(self, key, default=None):
        return self.params.get(key, default)

    def set(self, key, value):
        self.params[key] = value

    def mutate(self, mutation_rate=0.1):
        """
        [ru] Мутирует параметры генома.
        [en] Mutates genome parameters.
        """
        new_params = copy.deepcopy(self.params)
        for key, value in new_params.items():
            if isinstance(value, (int, float)):
                if random.random() < mutation_rate:
                    if isinstance(value, int):
                        delta = random.randint(-1, 1) * max(1, abs(value) // 2)
                        new_params[key] = max(0, value + delta)  # неотрицательные
                    else:
                        delta = random.uniform(-0.2, 0.2) * abs(value) if value != 0 else random.uniform(-0.5, 0.5)
                        new_params[key] = max(0.0, value + delta)
            elif isinstance(value, dict):
                # [ru] Для словарей мутируем каждый элемент
                # [en] For dictionaries, we mutate each element
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        if random.random() < mutation_rate:
                            if isinstance(sub_value, int):
                                sub_value = max(0, sub_value + random.randint(-1, 1))
                            else:
                                sub_value = max(0.0, sub_value + random.uniform(-0.1, 0.1))
                            new_params[key][sub_key] = sub_value
        return Genome(new_params)

    def crossover(self, other):
        """
        [ru] Скрещивание двух геномов (усреднение или случайный выбор).
        [en] Crossing two genomes (averaging or random selection).
        """
        child_params = {}
        for key in self.params:
            if isinstance(self.params[key], (int, float)):
                # [ru] Усреднение
                # [en] Averaging
                child_params[key] = (self.params[key] + other.params[key]) / 2
                if isinstance(self.params[key], int):
                    child_params[key] = int(round(child_params[key]))
            elif isinstance(self.params[key], dict):
                # [ru] Для словарей — усреднение значений
                # [en] For dictionaries - averaging of values
                child_dict = {}
                all_keys = set(self.params[key].keys()) | set(other.params[key].keys())
                for k in all_keys:
                    v1 = self.params[key].get(k)
                    v2 = other.params[key].get(k)
                    if v1 is None:
                        child_dict[k] = v2
                    elif v2 is None:
                        child_dict[k] = v1
                    elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                        child_dict[k] = (v1 + v2) / 2
                        if isinstance(v1, int):
                            child_dict[k] = int(round(child_dict[k]))
                    else:
                        child_dict[k] = v1 if random.random() < 0.5 else v2
                child_params[key] = child_dict
            else:
                # [ru] Остальное — случайный выбор
                # [en] The rest is random selection
                child_params[key] = self.params[key] if random.random() < 0.5 else other.params[key]
        return Genome(child_params)

    def to_dict(self):
        return self.params

    @classmethod
    def from_dict(cls, data):
        return cls(data)