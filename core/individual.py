# core/individual.py
import pygame
import math
from core.genome import Genome
from models.reflex_module import ReflexModule
from models.instinct_module import InstinctModule
from core.base_strategy import Perception, ActionSuggestion
import random
from collections import deque
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
from core.emotions.emotion_system import EmotionSystem
from core.emotions.emotion_base import EmotionType
from core.objects import Food, Predator


class Individual:
    def __init__(self, x=0, z=0, angle=0, move_delay=10, reflex_rules=None, instinct_patterns=None, genome=None, max_buffer_size=10000):
        self.x = x
        self.z = z
        self.angle = angle
        self.body_w = 0.8
        self.body_d = 0.5
        self.body_h = 1.2
        self.head_r = 0.3

        if genome is None:
            genome = Genome({
                'move_delay': move_delay,
                'step_size': 1.0,
                'reflex_rules': reflex_rules if reflex_rules else [],
                'instinct_patterns': instinct_patterns if instinct_patterns else [],
                'max_steps': 2000
            })
        self.genome = genome

        self.fitness = 0.0
        self.alive = True
        self.step_size = 1.0
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.dir_index = 0

        self.visited_nodes = [(self.x, self.z)]
        self.visited_edges = []
        self.visited_edges_set = set()
        self.frame_counter = 0
        self.move_delay = genome.get('move_delay', 10)
        self.max_steps = genome.get('max_steps', 2000)

        self.reflex_module = ReflexModule(
            reflex_rules if reflex_rules is not None else genome.get('reflex_rules', [])
        )
        self.instinct_module = InstinctModule(
            instinct_patterns if instinct_patterns is not None else genome.get('instinct_patterns', [])
        )

        self.nearby_object = None
        self.nearby_params = {}
        self.runaway_target = None
        self.awaiting_steps = 0
        self.moving = True
        self.max_buffer_size = max_buffer_size
        self.memory_buffer = deque(maxlen=self.max_buffer_size)
        self.runaway_attempts = 0

        self.food_collected = 0
        self.total_reward = 0.0
        self.reflex_stats = {}
        self.instinct_stats = {}
        self._has_reflex_action = False
        self._has_instinct_action = False

        self.emotion_system = EmotionSystem()
        self.emotions_enabled = False
        self.health = 100
        self.food_reserve = 0
        self.current_emotions = []

    def setInform(self, obj):
        self.nearby_object = obj
        self.nearby_params = obj.get(['type', 'temperature', 'smell', 'sound', 'name'])

    def _add_edge(self, node1, node2):
        if node1 > node2:
            node1, node2 = node2, node1
        self.visited_edges_set.add((node1, node2))

    def _move_one_step(self, world):
        dx, dz = self.directions[self.dir_index]
        next_x = self.x + dx * self.step_size
        next_z = self.z + dz * self.step_size
        if world.is_within_world(next_x, next_z) and world.get_object_at(next_x, next_z) is None:
            self.visited_edges.append(((self.x, self.z), (next_x, next_z)))
            self._add_edge((self.x, self.z), (next_x, next_z))
            self.x, self.z = next_x, next_z
            self.visited_nodes.append((self.x, self.z))
            return True
        return False

    def _turn_right(self):
        self.dir_index = (self.dir_index + 1) % 4

    def execute_action(self, action, world, state):
        if action == 'move_on':
            self.dir_index = (self.dir_index + 2) % 4
            for _ in range(2):
                if not self._move_one_step(world):
                    break
        elif action == 'grab':
            self._grab_object(world, state)
        elif action == 'avoid':
            self._turn_right()
        else:
            print(f"Неизвестное действие: {action}")

    def _grab_object(self, world, state):
        if self.nearby_object is None:
            return False
        target_x = self.nearby_object.x
        target_z = self.nearby_object.z
        dx = target_x - self.x
        dz = target_z - self.z
        if abs(dx) > self.step_size or abs(dz) > self.step_size:
            return False
        world.remove_object(self.nearby_object)
        self.visited_edges.append(((self.x, self.z), (target_x, target_z)))
        self._add_edge((self.x, self.z), (target_x, target_z))
        self.x, self.z = target_x, target_z
        self.visited_nodes.append((self.x, self.z))
        self.food_collected += 1
        if self.health < 100:
            self.health = min(100, self.health + 10)
        else:
            self.food_reserve += 1
        reward = 10.0
        next_state = world.get_state(self)
        self.add_experience(state, 4, reward, next_state)
        self.total_reward += reward
        self.nearby_object = None
        self.nearby_params = {}
        return True

    def notify(self, event_type, data):
        if event_type == 'explosion':
            perception = Perception({
                'sound': data.get('sound'),
                'vision': data.get('vision'),
                'position': data.get('position')
            })
            suggestion = self.instinct_module.get_best_action(perception)
            if suggestion:
                self.execute_instinct(suggestion.action_id, data.get('position'))

    def execute_instinct(self, action_id, target_pos):
        if action_id == 'run_away':
            dx = self.x - target_pos[0]
            dz = self.z - target_pos[1]
            if abs(dx) >= abs(dz):
                dir_vec = (1, 0) if dx >= 0 else (-1, 0)
            else:
                dir_vec = (0, 1) if dz >= 0 else (0, -1)
            self.runaway_target = dir_vec
            self.awaiting_steps = 0
            self.moving = True

    def _update_runaway(self, world):
        dx, dz = self.runaway_target
        next_x = self.x + dx * self.step_size
        next_z = self.z + dz * self.step_size
        if not world.is_within_world(next_x, next_z):
            if self.awaiting_steps == 0:
                self.awaiting_steps = 200
            else:
                self.awaiting_steps -= 1
                if self.awaiting_steps <= 0:
                    self.runaway_target = None
                    self.moving = True
                    self.frame_counter = 0
            return
        if world.get_object_at(next_x, next_z) is None:
            self.runaway_attempts = 0
            self.visited_edges.append(((self.x, self.z), (next_x, next_z)))
            self._add_edge((self.x, self.z), (next_x, next_z))
            self.x, self.z = next_x, next_z
            self.visited_nodes.append((self.x, self.z))
            self.awaiting_steps = 0
            return
        self.runaway_attempts += 1
        if self.runaway_attempts > 5:
            if self.awaiting_steps == 0:
                self.awaiting_steps = 200
            else:
                self.awaiting_steps -= 1
                if self.awaiting_steps <= 0:
                    self.runaway_target = None
                    self.moving = True
                    self.frame_counter = 0
            return
        left = (dz, -dx)
        right = (-dz, dx)
        for (ndx, ndz) in [left, right]:
            nnext_x = self.x + ndx * self.step_size
            nnext_z = self.z + ndz * self.step_size
            if world.is_within_world(nnext_x, nnext_z) and world.get_object_at(nnext_x, nnext_z) is None:
                self.runaway_target = (ndx, ndz)
                self.runaway_attempts = 0
                self.visited_edges.append(((self.x, self.z), (nnext_x, nnext_z)))
                self._add_edge((self.x, self.z), (nnext_x, nnext_z))
                self.x, self.z = nnext_x, nnext_z
                self.visited_nodes.append((self.x, self.z))
                self.awaiting_steps = 0
                return
        if self.awaiting_steps == 0:
            self.awaiting_steps = 200
        else:
            self.awaiting_steps -= 1
            if self.awaiting_steps <= 0:
                self.runaway_target = None
                self.moving = True
                self.frame_counter = 0

    def _perceive_others(self, world):
        others = []
        if not hasattr(world, 'bots'):
            return others
        for other in world.bots:
            if other is not self and other.alive:
                dx = other.x - self.x
                dz = other.z - self.z
                dist = math.hypot(dx, dz)
                if dist < 10.0:
                    others.append({
                        'bot': other,
                        'distance': dist,
                        'health': other.health,
                        'food_reserve': other.food_reserve
                    })
        return others

    def _get_vision(self, world):
        vision = np.zeros(8)
        if self.nearby_object:
            vision[0] = 1.0
        return vision

    def _get_sound(self, world):
        return np.zeros(8)

    def _get_smell(self, world):
        smell = np.zeros(8)
        if self.nearby_params and 'smell' in self.nearby_params:
            smell[0] = 1.0
        return smell

    def _get_sensory_data(self, world) -> Dict:
        # Определяем описание события
        event_desc = "неизвестное событие"
        if self.nearby_object:
            if isinstance(self.nearby_object, Food):
                event_desc = "встреча с едой"
            elif isinstance(self.nearby_object, Predator):
                event_desc = "встреча с хищником"
            else:
                event_desc = self.nearby_params.get('type', 'неизвестный объект')
        else:
            # Проверяем, есть ли раненый бот поблизости
            other_bots = self._perceive_others(world)
            for bot_info in other_bots:
                if bot_info['health'] < 30 and bot_info['distance'] < 5.0:
                    event_desc = "встреча с раненым соплеменником"
                    break

        return {
            'vision': self._get_vision(world),
            'sound': self._get_sound(world),
            'smell': self._get_smell(world),
            'position': (self.x, self.z),
            'context': {
                'nearby_object': self.nearby_params,
                'visited_count': len(self.visited_nodes),
                'food_collected': self.food_collected,
                'other_bots': self._perceive_others(world),
                'event_description': event_desc
            }
        }

    # def _get_sensory_data(self, world) -> Dict:
    #     # Определяем описание события на основе nearby_object
    #     event_desc = "неизвестное событие"
    #     if self.nearby_object:
    #         if isinstance(self.nearby_object, Food):
    #             event_desc = "встреча с едой"
    #         elif isinstance(self.nearby_object, Predator):
    #             event_desc = "встреча с хищником"
    #         else:
    #             event_desc = self.nearby_params.get('type', 'неизвестный объект')
    #     # Если рядом есть раненый бот, меняем описание
    #     other_bots = self._perceive_others(world)
    #     for bot_info in other_bots:
    #         if bot_info['health'] < 30 and bot_info['distance'] < 3.0:
    #             event_desc = "встреча с раненым соплеменником"
    #             break
    #
    #     return {
    #         'vision': self._get_vision(world),
    #         'sound': self._get_sound(world),
    #         'smell': self._get_smell(world),
    #         'position': (self.x, self.z),
    #         'context': {
    #             'nearby_object': self.nearby_params,
    #             'visited_count': len(self.visited_nodes),
    #             'food_collected': self.food_collected,
    #             'other_bots': other_bots,
    #             'event_description': event_desc  # <-- добавлено
    #         }
    #     }

    # def _get_sensory_data(self, world):
    #     return {
    #         'vision': self._get_vision(world),
    #         'sound': self._get_sound(world),
    #         'smell': self._get_smell(world),
    #         'position': (self.x, self.z),
    #         'context': {
    #             'nearby_object': self.nearby_params,
    #             'visited_count': len(self.visited_nodes),
    #             'food_collected': self.food_collected,
    #             'other_bots': self._perceive_others(world)
    #         }
    #     }

    def share_food(self, target_bot, amount=1):
        if self.food_reserve >= amount and target_bot.alive:
            self.food_reserve -= amount
            target_bot.food_reserve += amount
            target_bot.health = min(100, target_bot.health + 10)
            print(f"[ru] Бот поделился едой с {target_bot}")
            print(f"[en] Bot shared food with {target_bot}")
            return True
        return False

    def update(self, world):
        # ----- 1. Эмоции -----
        if self.emotions_enabled:
            sensory_data = self._get_sensory_data(world)
            self.current_emotions = self.emotion_system.process_sensory_input(sensory_data)

        if self.emotions_enabled:
            # self.current_emotions = self.emotion_system.process_sensory_input(sensory_data)
            if self.current_emotions:
                print(f"[DEBUG] Bot {id(self)} emotions: {[e.emotion_type.value for e in self.current_emotions]}")

        # ----- 2.1. Эмпатия (активный поиск раненого) -----
        if self.emotions_enabled and self.food_reserve > 0:
            other_bots = self._perceive_others(world)
            for info in other_bots:
                if info['health'] < 30 and info['distance'] < 5.0:
                    # Если есть эмпатия или просто по умолчанию (если эмоция не сработала)
                    if self.current_emotions:
                        sympathy = next((e for e in self.current_emotions if e.emotion_type == EmotionType.EMPATHY), None)
                        if sympathy and sympathy.intensity > 0.5:
                            # Делимся едой, если рядом
                            if info['distance'] < 3.0:
                                if self.share_food(info['bot'], 1):
                                    sympathy.intensity *= 0.5
                                    break
                            else:
                                # Двигаемся к раненому боту
                                dx = info['bot'].x - self.x
                                dz = info['bot'].z - self.z
                                if abs(dx) > abs(dz):
                                    target_dir = (1 if dx > 0 else -1, 0)
                                else:
                                    target_dir = (0, 1 if dz > 0 else -1)
                                current_dir = self.directions[self.dir_index]
                                if current_dir != target_dir:
                                    for i, d in enumerate(self.directions):
                                        if d == target_dir:
                                            diff = (i - self.dir_index) % 4
                                            if diff <= 2:
                                                self.dir_index = (self.dir_index + diff) % 4
                                            else:
                                                self.dir_index = (self.dir_index - (4 - diff)) % 4
                                            break
                                self._move_one_step(world)
                                return  # выходим, чтобы не делать другие действия
                    else:
                        # Если эмоций нет, но мы видим раненого, всё равно помогаем (для демонстрации)
                        if info['distance'] < 3.0 and self.food_reserve > 0:
                            self.share_food(info['bot'], 1)
                            break

        # ----- 2.2. Эмпатия (делиться едой) -----
        if self.emotions_enabled and self.current_emotions:
            sympathy = next((e for e in self.current_emotions if e.emotion_type == EmotionType.EMPATHY), None)
            if sympathy and sympathy.intensity > 0.4 and self.food_reserve > 0:
                other_bots = self._perceive_others(world)
                for info in other_bots:
                    if info['health'] < 30 and info['distance'] < 3.0:
                        if self.share_food(info['bot'], 1):
                            sympathy.intensity *= 0.5
                            break

        # ----- 3. Инстинкты (убегание) -----
        if self.runaway_target:
            if self._is_safe(world):
                self.runaway_target = None
            else:
                self._update_runaway(world)
                self._has_instinct_action = True
                return

        # ----- 4. Избегание хищников (приоритет выше поиска еды) -----
        if not self.runaway_target:
            nearest_predator = None
            min_dist_pred = float('inf')
            for obj in world.objects:
                if isinstance(obj, Predator) and obj.active:
                    dx = obj.x - self.x
                    dz = obj.z - self.z
                    dist = math.hypot(dx, dz)
                    if dist < min_dist_pred:
                        min_dist_pred = dist
                        nearest_predator = obj

            if nearest_predator and min_dist_pred < 4.0:
                dx = self.x - nearest_predator.x
                dz = self.z - nearest_predator.z
                if abs(dx) > abs(dz):
                    target_dir = (1 if dx > 0 else -1, 0)
                else:
                    target_dir = (0, 1 if dz > 0 else -1)

                current_dir = self.directions[self.dir_index]
                if current_dir != target_dir:
                    for i, d in enumerate(self.directions):
                        if d == target_dir:
                            diff = (i - self.dir_index) % 4
                            if diff <= 2:
                                self.dir_index = (self.dir_index + diff) % 4
                            else:
                                self.dir_index = (self.dir_index - (4 - diff)) % 4
                            break

                if self._move_one_step(world):
                    # Проверяем столкновение с хищником
                    for obj in world.objects:
                        if isinstance(obj, Predator) and obj.active:
                            dx = obj.x - self.x
                            dz = obj.z - self.z
                            if abs(dx) < self.step_size and abs(dz) < self.step_size:
                                self.health -= 10
                                if self.health <= 0:
                                    self.alive = False
                                    print(f"[ru] Бот {self} убит хищником!")
                                    print(f"[en] Bot {self} killed by predator!")
                                if dx != 0 or dz != 0:
                                    self.x -= dx * 0.5
                                    self.z -= dz * 0.5
                                break
                    return

        # ----- 5. Активный поиск еды -----
        if not self.runaway_target:
            nearest_food = None
            min_dist = float('inf')
            for obj in world.objects:
                if isinstance(obj, Food) and obj.active:
                    dx = obj.x - self.x
                    dz = obj.z - self.z
                    dist = math.hypot(dx, dz)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_food = obj

            if nearest_food and min_dist < 8.0:
                # Проверяем, нет ли хищника на пути к еде
                path_blocked = False
                for obj in world.objects:
                    if isinstance(obj, Predator) and obj.active:
                        dx = obj.x - nearest_food.x
                        dz = obj.z - nearest_food.z
                        if abs(dx) < 3.0 and abs(dz) < 3.0:
                            path_blocked = True
                            break
                if not path_blocked:
                    dx = nearest_food.x - self.x
                    dz = nearest_food.z - self.z
                    if abs(dx) > abs(dz):
                        target_dir = (1 if dx > 0 else -1, 0)
                    else:
                        target_dir = (0, 1 if dz > 0 else -1)

                    current_dir = self.directions[self.dir_index]
                    if current_dir != target_dir:
                        for i, d in enumerate(self.directions):
                            if d == target_dir:
                                diff = (i - self.dir_index) % 4
                                if diff <= 2:
                                    self.dir_index = (self.dir_index + diff) % 4
                                else:
                                    self.dir_index = (self.dir_index - (4 - diff)) % 4
                                break

                    self._move_one_step(world)
                    for (dx_ch, dz_ch) in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        check_x = self.x + dx_ch * self.step_size
                        check_z = self.z + dz_ch * self.step_size
                        obj = world.get_object_at(check_x, check_z)
                        if obj and isinstance(obj, Food) and obj.active:
                            self.setInform(obj)
                            state = world.get_state(self)
                            self._grab_object(world, state)
                            break
                    return

        # ----- 6. Взаимодействие с хищниками (если уже рядом) -----
        for obj in world.objects:
            if isinstance(obj, Predator) and obj.active:
                dx = obj.x - self.x
                dz = obj.z - self.z
                dist = math.hypot(dx, dz)
                if dist < self.step_size + 0.5:  # если очень близко
                    self.health -= 5
                    if self.health <= 0:
                        self.alive = False
                        print(f"[ru] Бот {self} убит хищником!")
                        print(f"[en] Bot {self} killed by predator!")
                    # Отталкиваем от хищника
                    if dist > 0.1:
                        self.x -= dx / dist * self.step_size * 0.5
                        self.z -= dz / dist * self.step_size * 0.5
                    else:
                        # Если стоим на месте хищника, отодвигаемся случайно
                        self.x += random.choice([-1, 1]) * self.step_size
                        self.z += random.choice([-1, 1]) * self.step_size
                    break



        # for obj in world.objects:
        #     if isinstance(obj, Predator) and obj.active:
        #         dx = obj.x - self.x
        #         dz = obj.z - self.z
        #         if abs(dx) < self.step_size and abs(dz) < self.step_size:
        #             self.health -= 5
        #             if self.health <= 0:
        #                 self.alive = False
        #                 print(f"[ru] Бот {self} убит хищником!")
        #                 print(f"[en] Bot {self} killed by predator!")
        #             if dx != 0 or dz != 0:
        #                 self.x -= dx * 0.5
        #                 self.z -= dz * 0.5
        #             break

        # ----- 7. Рефлексы -----
        state = world.get_state(self)
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for (dx, dz) in dirs:
            check_x = self.x + dx * self.step_size
            check_z = self.z + dz * self.step_size
            obj = world.get_object_at(check_x, check_z)
            if obj:
                self.setInform(obj)
                perception = Perception(self.nearby_params.copy())
                thresholds = self.genome.get('reflex_thresholds', {})
                suggestion = self.reflex_module.get_best_action(perception, thresholds)
                if suggestion:
                    self.execute_action(suggestion.action_id, world, state)
                    self._has_reflex_action = True
                    self._record_reflex_outcome(suggestion.action_id, True)
                    next_state = world.get_state(self)
                    self.add_experience(state, suggestion.action_id, 1.0, next_state)
                    return

        # ----- 8. Исследование -----
        self._explore(world)

    def _is_safe(self, world):
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for (dx, dz) in dirs:
            check_x = self.x + dx * self.step_size * 3
            check_z = self.z + dz * self.step_size * 3
            obj = world.get_object_at(check_x, check_z)
            if obj and hasattr(obj, 'danger_level') and obj.danger_level > 0.5:
                return False
        return True

    def _record_reflex_outcome(self, action_id, success):
        if action_id not in self.reflex_stats:
            self.reflex_stats[action_id] = {'success': 0, 'total': 0}
        self.reflex_stats[action_id]['total'] += 1
        if success:
            self.reflex_stats[action_id]['success'] += 1

    def _record_instinct_outcome(self, pattern_id, success):
        if pattern_id not in self.instinct_stats:
            self.instinct_stats[pattern_id] = {'success': 0, 'total': 0}
        self.instinct_stats[pattern_id]['total'] += 1
        if success:
            self.instinct_stats[pattern_id]['success'] += 1

    def get_reflex_success_rate(self, action_id):
        stats = self.reflex_stats.get(action_id, {'success': 0, 'total': 0})
        if stats['total'] == 0:
            return 0.0
        return stats['success'] / stats['total']

    def _explore(self, world):
        state = world.get_state(self)
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        candidates = []
        fallback = []
        for (dx, dz) in dirs:
            next_x = self.x + dx * self.step_size
            next_z = self.z + dz * self.step_size
            if not world.is_within_world(next_x, next_z):
                continue
            obj = world.get_object_at(next_x, next_z)
            if obj is not None:
                if isinstance(obj, Predator):
                    continue
                continue
            node1 = (self.x, self.z)
            node2 = (next_x, next_z)
            if node1 > node2:
                node1, node2 = node2, node1
            if (node1, node2) not in self.visited_edges_set:
                candidates.append((dx, dz))
            else:
                fallback.append((dx, dz))
        if candidates:
            dx, dz = random.choice(candidates)
            reward_step = 1.0
            action_idx = dirs.index((dx, dz))
        elif fallback:
            dx, dz = random.choice(fallback)
            reward_step = -0.1
            action_idx = dirs.index((dx, dz))
        else:
            self.alive = False
            reward_step = -1.0
            next_state = world.get_state(self)
            self.add_experience(state, 0, reward_step, next_state)
            self.total_reward += reward_step
            return
        next_x = self.x + dx * self.step_size
        next_z = self.z + dz * self.step_size
        self.visited_edges.append(((self.x, self.z), (next_x, next_z)))
        self._add_edge((self.x, self.z), (next_x, next_z))
        self.x, self.z = next_x, next_z
        self.visited_nodes.append((self.x, self.z))
        next_state = world.get_state(self)
        self.add_experience(state, action_idx, reward_step, next_state)
        self.total_reward += reward_step
        for (dx_check, dz_check) in dirs:
            check_x = self.x + dx_check * self.step_size
            check_z = self.z + dz_check * self.step_size
            obj = world.get_object_at(check_x, check_z)
            if obj:
                self.setInform(obj)
                perception = Perception(self.nearby_params.copy())
                thresholds = self.genome.get('reflex_thresholds', {})
                suggestion = self.reflex_module.get_best_action(perception, thresholds)
                if suggestion:
                    self.execute_action(suggestion.action_id, world, state)
                    self._has_reflex_action = True
                break
        if len(self.visited_nodes) > self.max_steps:
            self.alive = False

    def calculate_fitness(self):
        visited_score = len(self.visited_nodes) * 1.0
        food_score = self.food_collected * 10.0
        survival_score = 50.0 if self.alive else 0.0
        unique_nodes = len(set(self.visited_nodes))
        repeat_penalty = (len(self.visited_nodes) - unique_nodes) * 0.5
        diversity_bonus = unique_nodes * 0.1
        return max(0, visited_score + food_score + survival_score - repeat_penalty + diversity_bonus)

    def draw_path(self, screen, world_to_screen_func):
        if not self.visited_edges:
            return
        for (x1, z1), (x2, z2) in self.visited_edges:
            p1 = world_to_screen_func(x1, z1, 0.1)
            p2 = world_to_screen_func(x2, z2, 0.1)
            pygame.draw.line(screen, (0, 255, 0), p1, p2, 3)

    def draw(self, screen, world_to_screen_func, scale):
        self.draw_path(screen, world_to_screen_func)
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
        pygame.draw.polygon(screen, (0, 150, 200), top_points)
        for i in range(4):
            j = (i + 1) % 4
            pts = [base_points[i], base_points[j], top_points[j], top_points[i]]
            pygame.draw.polygon(screen, (0, 100, 150), pts)
        pygame.draw.polygon(screen, (0, 80, 120), base_points)
        head_offset_z = 0.2
        head_x, head_z = rotate_point(0, head_offset_z)
        head_y = self.body_h + self.head_r * 0.8
        head_screen = world_to_screen_func(head_x, head_z, head_y)
        rad_px = int(self.head_r * scale)
        if rad_px > 1:
            pygame.draw.circle(screen, (255, 200, 150), head_screen, rad_px)
        if self.emotions_enabled and self.current_emotions:
            dominant = max(self.current_emotions, key=lambda x: x.intensity)
            pos = world_to_screen_func(self.x, self.z, self.body_h + 1.0)
            font = pygame.font.Font(None, 24)
            text = font.render(dominant.emotion_type.value[:3].upper(), True, (255, 255, 0))
            screen.blit(text, (pos[0] - 20, pos[1] - 30))

    def reset(self, start_x=0, start_z=0):
        self.x = start_x
        self.z = start_z
        self.visited_nodes = [(self.x, self.z)]
        self.visited_edges = []
        self.visited_edges_set = set()
        self.frame_counter = 0
        self.runaway_target = None
        self.awaiting_steps = 0
        self.moving = True
        self.alive = True
        self.nearby_object = None
        self.nearby_params = {}
        self.fitness = 0.0
        self.dir_index = 0

    def evaluate(self, world, max_steps=500):
        self.reset()
        for _ in range(max_steps):
            if not self.alive:
                break
            self.update(world)
        self.fitness = self.calculate_fitness()
        return self.fitness

    def add_experience(self, state, action, reward, next_state):
        self.memory_buffer.append((state, action, reward, next_state))
        if len(self.memory_buffer) > self.max_buffer_size:
            self.memory_buffer.pop(0)

    def get_experiences(self):
        data = list(self.memory_buffer)
        self.memory_buffer.clear()
        return data





# # core/individual.py
# import pygame
# import math
# from core.genome import Genome
# from models.reflex_module import ReflexModule
# from models.instinct_module import InstinctModule
# from core.base_strategy import Perception, ActionSuggestion
# import random
# from collections import deque
# from typing import List, Dict, Optional, Tuple, Any, Union
# import numpy as np
# from core.emotions.emotion_system import EmotionSystem
# from core.emotions.emotion_base import EmotionType
# from core.objects import Food, Predator
#
# class Individual:
#     def __init__(self, x=0, z=0, angle=0, move_delay=10, reflex_rules=None, instinct_patterns=None, genome=None,  max_buffer_size=10000):
#         self.x = x
#         self.z = z
#         self.angle = angle
#         self.body_w = 0.8
#         self.body_d = 0.5
#         self.body_h = 1.2
#         self.head_r = 0.3
#
#         if genome is None:
#             genome = Genome({
#                 'move_delay': move_delay,
#                 'step_size': 1.0,
#                 'reflex_rules': reflex_rules if reflex_rules else [],
#                 'instinct_patterns': instinct_patterns if instinct_patterns else [],
#                 'max_steps': 2000
#             })
#         self.genome = genome
#
#         self.fitness = 0.0
#         self.alive = True
#         self.step_size = 1.0  # уменьшено для медленного движения
#         self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
#         self.dir_index = 0
#
#         self.visited_nodes = [(self.x, self.z)]
#         self.visited_edges = []
#         self.visited_edges_set = set()
#         self.frame_counter = 0
#         self.move_delay = genome.get('move_delay', 10)
#         self.max_steps = genome.get('max_steps', 2000)
#
#         self.reflex_module = ReflexModule(
#             reflex_rules if reflex_rules is not None else genome.get('reflex_rules', [])
#         )
#         self.instinct_module = InstinctModule(
#             instinct_patterns if instinct_patterns is not None else genome.get('instinct_patterns', [])
#         )
#
#         self.nearby_object = None
#         self.nearby_params = {}
#         self.runaway_target = None
#         self.awaiting_steps = 0
#         self.moving = True
#         self.max_buffer_size = max_buffer_size
#         self.memory_buffer = deque(maxlen=self.max_buffer_size)
#         self.runaway_attempts = 0
#
#         self.food_collected = 0
#         self.total_reward = 0.0
#         self.reflex_stats = {}
#         self.instinct_stats = {}
#         self._has_reflex_action = False
#         self._has_instinct_action = False
#
#         # Эмоциональная система
#         self.emotion_system = EmotionSystem()
#         self.emotions_enabled = False
#         self.health = 100
#         self.food_reserve = 0
#         self.current_emotions = []
#
#     # ---------- Вспомогательные методы (без изменений) ----------
#     def setInform(self, obj):
#         self.nearby_object = obj
#         self.nearby_params = obj.get(['type', 'temperature', 'smell', 'sound', 'name'])
#
#     def _add_edge(self, node1, node2):
#         if node1 > node2:
#             node1, node2 = node2, node1
#         self.visited_edges_set.add((node1, node2))
#
#     def _move_one_step(self, world):
#         dx, dz = self.directions[self.dir_index]
#         next_x = self.x + dx * self.step_size
#         next_z = self.z + dz * self.step_size
#         # Проверяем, что клетка в пределах мира и не занята объектом
#         # (включая хищников)
#         if world.is_within_world(next_x, next_z) and world.get_object_at(next_x, next_z) is None:
#             self.visited_edges.append(((self.x, self.z), (next_x, next_z)))
#             self._add_edge((self.x, self.z), (next_x, next_z))
#             self.x, self.z = next_x, next_z
#             self.visited_nodes.append((self.x, self.z))
#             return True
#         return False
#
#     # def _move_one_step(self, world):
#     #     dx, dz = self.directions[self.dir_index]
#     #     next_x = self.x + dx * self.step_size
#     #     next_z = self.z + dz * self.step_size
#     #     if world.is_within_world(next_x, next_z) and world.get_object_at(next_x, next_z) is None:
#     #         self.visited_edges.append(((self.x, self.z), (next_x, next_z)))
#     #         self._add_edge((self.x, self.z), (next_x, next_z))
#     #         self.x, self.z = next_x, next_z
#     #         self.visited_nodes.append((self.x, self.z))
#     #         return True
#     #     return False
#
#     def _turn_right(self):
#         self.dir_index = (self.dir_index + 1) % 4
#
#     def execute_action(self, action, world, state):
#         if action == 'move_on':
#             self.dir_index = (self.dir_index + 2) % 4
#             for _ in range(2):
#                 if not self._move_one_step(world):
#                     break
#         elif action == 'grab':
#             self._grab_object(world, state)
#         elif action == 'avoid':
#             self._turn_right()
#         else:
#             print(f"Неизвестное действие: {action}")
#
#     def _grab_object(self, world, state):
#         if self.nearby_object is None:
#             return False
#         target_x = self.nearby_object.x
#         target_z = self.nearby_object.z
#         dx = target_x - self.x
#         dz = target_z - self.z
#         if abs(dx) > self.step_size or abs(dz) > self.step_size:
#             return False
#         world.remove_object(self.nearby_object)
#         self.visited_edges.append(((self.x, self.z), (target_x, target_z)))
#         self._add_edge((self.x, self.z), (target_x, target_z))
#         self.x, self.z = target_x, target_z
#         self.visited_nodes.append((self.x, self.z))
#
#         self.food_collected += 1
#         if self.health < 100:
#             self.health = min(100, self.health + 10)  # восстанавливаем здоровье, но не выше 100
#         else:
#             self.food_reserve += 1  # если здоровье уже 100, пополняем запас
#
#         # self.food_collected += 1
#         # self.food_reserve += 1
#
#         reward = 10.0
#         next_state = world.get_state(self)
#         self.add_experience(state, 4, reward, next_state)
#         self.total_reward += reward
#         self.nearby_object = None
#         self.nearby_params = {}
#         return True
#
#     def notify(self, event_type, data):
#         if event_type == 'explosion':
#             perception = Perception({
#                 'sound': data.get('sound'),
#                 'vision': data.get('vision'),
#                 'position': data.get('position')
#             })
#             suggestion = self.instinct_module.get_best_action(perception)
#             if suggestion:
#                 self.execute_instinct(suggestion.action_id, data.get('position'))
#
#     def execute_instinct(self, action_id, target_pos):
#         if action_id == 'run_away':
#             dx = self.x - target_pos[0]
#             dz = self.z - target_pos[1]
#             if abs(dx) >= abs(dz):
#                 dir_vec = (1, 0) if dx >= 0 else (-1, 0)
#             else:
#                 dir_vec = (0, 1) if dz >= 0 else (0, -1)
#             self.runaway_target = dir_vec
#             self.awaiting_steps = 0
#             self.moving = True
#
#     def _update_runaway(self, world):
#         dx, dz = self.runaway_target
#         next_x = self.x + dx * self.step_size
#         next_z = self.z + dz * self.step_size
#         if not world.is_within_world(next_x, next_z):
#             if self.awaiting_steps == 0:
#                 self.awaiting_steps = 200
#             else:
#                 self.awaiting_steps -= 1
#                 if self.awaiting_steps <= 0:
#                     self.runaway_target = None
#                     self.moving = True
#                     self.frame_counter = 0
#             return
#         if world.get_object_at(next_x, next_z) is None:
#             self.runaway_attempts = 0
#             self.visited_edges.append(((self.x, self.z), (next_x, next_z)))
#             self._add_edge((self.x, self.z), (next_x, next_z))
#             self.x, self.z = next_x, next_z
#             self.visited_nodes.append((self.x, self.z))
#             self.awaiting_steps = 0
#             return
#         self.runaway_attempts += 1
#         if self.runaway_attempts > 5:
#             if self.awaiting_steps == 0:
#                 self.awaiting_steps = 200
#             else:
#                 self.awaiting_steps -= 1
#                 if self.awaiting_steps <= 0:
#                     self.runaway_target = None
#                     self.moving = True
#                     self.frame_counter = 0
#             return
#         left = (dz, -dx)
#         right = (-dz, dx)
#         for (ndx, ndz) in [left, right]:
#             nnext_x = self.x + ndx * self.step_size
#             nnext_z = self.z + ndz * self.step_size
#             if world.is_within_world(nnext_x, nnext_z) and world.get_object_at(nnext_x, nnext_z) is None:
#                 self.runaway_target = (ndx, ndz)
#                 self.runaway_attempts = 0
#                 self.visited_edges.append(((self.x, self.z), (nnext_x, nnext_z)))
#                 self._add_edge((self.x, self.z), (nnext_x, nnext_z))
#                 self.x, self.z = nnext_x, nnext_z
#                 self.visited_nodes.append((self.x, self.z))
#                 self.awaiting_steps = 0
#                 return
#         if self.awaiting_steps == 0:
#             self.awaiting_steps = 200
#         else:
#             self.awaiting_steps -= 1
#             if self.awaiting_steps <= 0:
#                 self.runaway_target = None
#                 self.moving = True
#                 self.frame_counter = 0
#
#     def _perceive_others(self, world):
#         others = []
#         if not hasattr(world, 'bots'):
#             return others
#         for other in world.bots:
#             if other is not self and other.alive:
#                 dx = other.x - self.x
#                 dz = other.z - self.z
#                 dist = math.hypot(dx, dz)
#                 if dist < 10.0:
#                     others.append({
#                         'bot': other,
#                         'distance': dist,
#                         'health': other.health,
#                         'food_reserve': other.food_reserve
#                     })
#         return others
#
#     def _get_vision(self, world):
#         vision = np.zeros(8)
#         if self.nearby_object:
#             vision[0] = 1.0
#         return vision
#
#     def _get_sound(self, world):
#         return np.zeros(8)
#
#     def _get_smell(self, world):
#         smell = np.zeros(8)
#         if self.nearby_params and 'smell' in self.nearby_params:
#             smell[0] = 1.0
#         return smell
#
#     def _get_sensory_data(self, world):
#         return {
#             'vision': self._get_vision(world),
#             'sound': self._get_sound(world),
#             'smell': self._get_smell(world),
#             'position': (self.x, self.z),
#             'context': {
#                 'nearby_object': self.nearby_params,
#                 'visited_count': len(self.visited_nodes),
#                 'food_collected': self.food_collected,
#                 'other_bots': self._perceive_others(world)
#             }
#         }
#
#     def share_food(self, target_bot, amount=1):
#         if self.food_reserve >= amount and target_bot.alive:
#             self.food_reserve -= amount
#             target_bot.food_reserve += amount
#             target_bot.health = min(100, target_bot.health + 10)
#             print(f"[ru] Бот поделился едой с {target_bot}")
#             print(f"[en] Bot shared food with {target_bot}")
#             return True
#         return False
#
#     # ---------- ОСНОВНОЙ МЕТОД UPDATE ----------
#     def update(self, world):
#         # ----- 1. Эмоции -----
#         if self.emotions_enabled:
#             sensory_data = self._get_sensory_data(world)
#             self.current_emotions = self.emotion_system.process_sensory_input(sensory_data)
#
#         # ----- 2. Эмпатия (делиться едой) -----
#         if self.emotions_enabled and self.current_emotions:
#             sympathy = next((e for e in self.current_emotions if e.emotion_type == EmotionType.EMPATHY), None)
#             if sympathy and sympathy.intensity > 0.6 and self.food_reserve > 0:
#                 other_bots = self._perceive_others(world)
#                 for info in other_bots:
#                     if info['health'] < 30 and info['distance'] < 3.0:
#                         if self.share_food(info['bot'], 1):
#                             sympathy.intensity *= 0.5
#                             break
#
#         # ----- 3. Инстинкты (убегание от взрыва) -----
#         if self.runaway_target:
#             if self._is_safe(world):
#                 self.runaway_target = None
#             else:
#                 self._update_runaway(world)
#                 self._has_instinct_action = True
#                 return
#
#         # ============================================================
#         # [ru] НОВЫЙ БЛОК: ИЗБЕГАНИЕ ХИЩНИКОВ (высокий приоритет)
#         # [en] NEW BLOCK: AVOID PREDATORS (high priority)
#         # ============================================================
#         nearest_predator = None
#         min_pred_dist = float('inf')
#         for obj in world.objects:
#             if isinstance(obj, Predator) and obj.active:
#                 dx = obj.x - self.x
#                 dz = obj.z - self.z
#                 dist = math.hypot(dx, dz)
#                 if dist < min_pred_dist:
#                     min_pred_dist = dist
#                     nearest_predator = obj
#
#         # Если хищник рядом (расстояние < 4.0), убегаем от него
#         if nearest_predator and min_pred_dist < 4.0:
#             dx = self.x - nearest_predator.x
#             dz = self.z - nearest_predator.z
#             # Выбираем направление от хищника
#             if abs(dx) >= abs(dz):
#                 target_dir = (1 if dx > 0 else -1, 0)
#             else:
#                 target_dir = (0, 1 if dz > 0 else -1)
#
#             current_dir = self.directions[self.dir_index]
#             if current_dir != target_dir:
#                 for i, d in enumerate(self.directions):
#                     if d == target_dir:
#                         diff = (i - self.dir_index) % 4
#                         if diff <= 2:
#                             self.dir_index = (self.dir_index + diff) % 4
#                         else:
#                             self.dir_index = (self.dir_index - (4 - diff)) % 4
#                         break
#
#             # Делаем шаг от хищника
#             self._move_one_step(world)
#             # Дополнительно: если хищник слишком близко (расстояние < 1.5), отталкиваем
#             if min_pred_dist < 1.5:
#                 # Отталкиваем бота от хищника
#                 self.x -= dx * 0.3
#                 self.z -= dz * 0.3
#             return  # после ухода от хищника выходим
#
#         # ============================================================
#         # [ru] ВЗАИМОДЕЙСТВИЕ С ХИЩНИКАМИ (потеря здоровья)
#         # [en] INTERACTION WITH PREDATORS (health loss)
#         # ============================================================
#         for obj in world.objects:
#             if isinstance(obj, Predator) and obj.active:
#                 dx = obj.x - self.x
#                 dz = obj.z - self.z
#                 # Увеличиваем радиус обнаружения до 1.5*step_size
#                 if abs(dx) < self.step_size * 1.5 and abs(dz) < self.step_size * 1.5:
#                     self.health -= 5
#                     if self.health <= 0:
#                         self.alive = False
#                         print(f"[ru] Бот {self} убит хищником!")
#                         print(f"[en] Bot {self} killed by predator!")
#                     # Отталкиваем бота от хищника
#                     if dx != 0 or dz != 0:
#                         self.x -= dx * 0.5
#                         self.z -= dz * 0.5
#                     break
#
#         # ----- 3.5. Избегание хищников (приоритет выше поиска еды) -----
#         if not self.runaway_target:
#             # Проверяем, есть ли хищник рядом (в радиусе 3 клеток)
#             nearest_predator = None
#             min_dist_pred = float('inf')
#             for obj in world.objects:
#                 if isinstance(obj, Predator) and obj.active:
#                     dx = obj.x - self.x
#                     dz = obj.z - self.z
#                     dist = math.hypot(dx, dz)
#                     if dist < min_dist_pred:
#                         min_dist_pred = dist
#                         nearest_predator = obj
#
#             if nearest_predator and min_dist_pred < 4.0:
#                 # Убегаем от хищника (противоположное направление)
#                 dx = self.x - nearest_predator.x
#                 dz = self.z - nearest_predator.z
#                 if abs(dx) > abs(dz):
#                     target_dir = (1 if dx > 0 else -1, 0)
#                 else:
#                     target_dir = (0, 1 if dz > 0 else -1)
#
#                 current_dir = self.directions[self.dir_index]
#                 if current_dir != target_dir:
#                     for i, d in enumerate(self.directions):
#                         if d == target_dir:
#                             diff = (i - self.dir_index) % 4
#                             if diff <= 2:
#                                 self.dir_index = (self.dir_index + diff) % 4
#                             else:
#                                 self.dir_index = (self.dir_index - (4 - diff)) % 4
#                             break
#
#                 # Пытаемся сделать шаг в направлении убегания
#                 if self._move_one_step(world):
#                     # Проверяем, не столкнулись ли с хищником
#                     for obj in world.objects:
#                         if isinstance(obj, Predator) and obj.active:
#                             dx = obj.x - self.x
#                             dz = obj.z - self.z
#                             if abs(dx) < self.step_size and abs(dz) < self.step_size:
#                                 self.health -= 10
#                                 if self.health <= 0:
#                                     self.alive = False
#                                     print(f"[ru] Бот {self} убит хищником!")
#                                     print(f"[en] Bot {self} killed by predator!")
#                                 # Отталкиваем
#                                 if dx != 0 or dz != 0:
#                                     self.x -= dx * 0.5
#                                     self.z -= dz * 0.5
#                                 break
#                     return  # выходим, чтобы избежать других действий в этом шаге
#
#         # ----- 4. Активный поиск еды -----
#         if not self.runaway_target:
#             nearest_food = None
#             min_dist = float('inf')
#             for obj in world.objects:
#                 if isinstance(obj, Food) and obj.active:
#                     dx = obj.x - self.x
#                     dz = obj.z - self.z
#                     dist = math.hypot(dx, dz)
#                     if dist < min_dist:
#                         min_dist = dist
#                         nearest_food = obj
#
#             if nearest_food and min_dist < 8.0:
#                 # Двигаемся к еде
#                 dx = nearest_food.x - self.x
#                 dz = nearest_food.z - self.z
#                 if abs(dx) > abs(dz):
#                     target_dir = (1 if dx > 0 else -1, 0)
#                 else:
#                     target_dir = (0, 1 if dz > 0 else -1)
#
#                 current_dir = self.directions[self.dir_index]
#                 if current_dir != target_dir:
#                     for i, d in enumerate(self.directions):
#                         if d == target_dir:
#                             diff = (i - self.dir_index) % 4
#                             if diff <= 2:
#                                 self.dir_index = (self.dir_index + diff) % 4
#                             else:
#                                 self.dir_index = (self.dir_index - (4 - diff)) % 4
#                             break
#
#                 self._move_one_step(world)
#                 # Проверяем, не рядом ли еда (для захвата)
#                 for (dx_ch, dz_ch) in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
#                     check_x = self.x + dx_ch * self.step_size
#                     check_z = self.z + dz_ch * self.step_size
#                     obj = world.get_object_at(check_x, check_z)
#                     if obj and isinstance(obj, Food) and obj.active:
#                         self.setInform(obj)
#                         state = world.get_state(self)
#                         self._grab_object(world, state)
#                         break
#                 return
#
#         # ----- 5. Рефлексы (реакция на объекты рядом) -----
#         state = world.get_state(self)
#         dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
#         for (dx, dz) in dirs:
#             check_x = self.x + dx * self.step_size
#             check_z = self.z + dz * self.step_size
#             obj = world.get_object_at(check_x, check_z)
#             if obj:
#                 self.setInform(obj)
#                 perception = Perception(self.nearby_params.copy())
#                 thresholds = self.genome.get('reflex_thresholds', {})
#                 suggestion = self.reflex_module.get_best_action(perception, thresholds)
#                 if suggestion:
#                     self.execute_action(suggestion.action_id, world, state)
#                     self._has_reflex_action = True
#                     self._record_reflex_outcome(suggestion.action_id, True)
#                     next_state = world.get_state(self)
#                     self.add_experience(state, suggestion.action_id, 1.0, next_state)
#                     return
#
#         # ----- 6. Исследование (если нет действий) -----
#         self._explore(world)
#
#     # def update(self, world):
#     #     # ----- 1. Эмоции -----
#     #     if self.emotions_enabled:
#     #         sensory_data = self._get_sensory_data(world)
#     #         self.current_emotions = self.emotion_system.process_sensory_input(sensory_data)
#     #
#     #     # ----- 2. Эмпатия (делиться едой) -----
#     #     if self.emotions_enabled and self.current_emotions:
#     #         sympathy = next((e for e in self.current_emotions if e.emotion_type == EmotionType.EMPATHY), None)
#     #         if sympathy and sympathy.intensity > 0.6 and self.food_reserve > 0:
#     #             other_bots = self._perceive_others(world)
#     #             for info in other_bots:
#     #                 if info['health'] < 30 and info['distance'] < 3.0:
#     #                     if self.share_food(info['bot'], 1):
#     #                         sympathy.intensity *= 0.5
#     #                         break
#     #
#     #     # ----- 3. Инстинкты (убегание) -----
#     #     if self.runaway_target:
#     #         if self._is_safe(world):
#     #             self.runaway_target = None
#     #         else:
#     #             self._update_runaway(world)
#     #             self._has_instinct_action = True
#     #             return
#     #
#     #     # ----- 4. Активный поиск еды (приоритет выше рефлексов) -----
#     #     if not self.runaway_target:
#     #         nearest_food = None
#     #         min_dist = float('inf')
#     #         for obj in world.objects:
#     #             if isinstance(obj, Food) and obj.active:
#     #                 dx = obj.x - self.x
#     #                 dz = obj.z - self.z
#     #                 dist = math.hypot(dx, dz)
#     #                 if dist < min_dist:
#     #                     min_dist = dist
#     #                     nearest_food = obj
#     #
#     #         if nearest_food and min_dist < 8.0:
#     #             # Двигаемся к еде
#     #             dx = nearest_food.x - self.x
#     #             dz = nearest_food.z - self.z
#     #             if abs(dx) > abs(dz):
#     #                 target_dir = (1 if dx > 0 else -1, 0)
#     #             else:
#     #                 target_dir = (0, 1 if dz > 0 else -1)
#     #
#     #             current_dir = self.directions[self.dir_index]
#     #             if current_dir != target_dir:
#     #                 for i, d in enumerate(self.directions):
#     #                     if d == target_dir:
#     #                         diff = (i - self.dir_index) % 4
#     #                         if diff <= 2:
#     #                             self.dir_index = (self.dir_index + diff) % 4
#     #                         else:
#     #                             self.dir_index = (self.dir_index - (4 - diff)) % 4
#     #                         break
#     #
#     #             self._move_one_step(world)
#     #             # Проверяем, не рядом ли еда (для захвата)
#     #             for (dx_ch, dz_ch) in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
#     #                 check_x = self.x + dx_ch * self.step_size
#     #                 check_z = self.z + dz_ch * self.step_size
#     #                 obj = world.get_object_at(check_x, check_z)
#     #                 if obj and isinstance(obj, Food) and obj.active:
#     #                     self.setInform(obj)
#     #                     state = world.get_state(self)
#     #                     self._grab_object(world, state)
#     #                     break
#     #             return  # выходим, чтобы не делать рефлексы в том же шаге
#     #
#     #     # ----- 5. Взаимодействие с хищниками (потеря здоровья) -----
#     #     for obj in world.objects:
#     #         if isinstance(obj, Predator) and obj.active:
#     #             dx = obj.x - self.x
#     #             dz = obj.z - self.z
#     #             if abs(dx) < self.step_size and abs(dz) < self.step_size:
#     #                 self.health -= 5
#     #                 if self.health <= 0:
#     #                     self.alive = False
#     #                     print(f"[ru] Бот {self} убит хищником!")
#     #                     print(f"[en] Bot {self} killed by predator!")
#     #                 # Отталкиваем
#     #                 if dx != 0 or dz != 0:
#     #                     self.x -= dx * 0.5
#     #                     self.z -= dz * 0.5
#     #                 break
#     #
#     #     # ----- 6. Рефлексы -----
#     #     state = world.get_state(self)
#     #     dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
#     #     for (dx, dz) in dirs:
#     #         check_x = self.x + dx * self.step_size
#     #         check_z = self.z + dz * self.step_size
#     #         obj = world.get_object_at(check_x, check_z)
#     #         if obj:
#     #             self.setInform(obj)
#     #             perception = Perception(self.nearby_params.copy())
#     #             thresholds = self.genome.get('reflex_thresholds', {})
#     #             suggestion = self.reflex_module.get_best_action(perception, thresholds)
#     #             if suggestion:
#     #                 self.execute_action(suggestion.action_id, world, state)
#     #                 self._has_reflex_action = True
#     #                 self._record_reflex_outcome(suggestion.action_id, True)
#     #                 next_state = world.get_state(self)
#     #                 self.add_experience(state, suggestion.action_id, 1.0, next_state)
#     #                 return
#     #
#     #     # ----- 7. Исследование (если нет инстинктов, рефлексов и еды рядом) -----
#     #     self._explore(world)
#
#     # ---------- Остальные методы (без изменений) ----------
#     def _is_safe(self, world):
#         dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
#         for (dx, dz) in dirs:
#             check_x = self.x + dx * self.step_size * 3
#             check_z = self.z + dz * self.step_size * 3
#             obj = world.get_object_at(check_x, check_z)
#             if obj and hasattr(obj, 'danger_level') and obj.danger_level > 0.5:
#                 return False
#         return True
#
#     def _record_reflex_outcome(self, action_id, success):
#         if action_id not in self.reflex_stats:
#             self.reflex_stats[action_id] = {'success': 0, 'total': 0}
#         self.reflex_stats[action_id]['total'] += 1
#         if success:
#             self.reflex_stats[action_id]['success'] += 1
#
#     def _record_instinct_outcome(self, pattern_id, success):
#         if pattern_id not in self.instinct_stats:
#             self.instinct_stats[pattern_id] = {'success': 0, 'total': 0}
#         self.instinct_stats[pattern_id]['total'] += 1
#         if success:
#             self.instinct_stats[pattern_id]['success'] += 1
#
#     def get_reflex_success_rate(self, action_id):
#         stats = self.reflex_stats.get(action_id, {'success': 0, 'total': 0})
#         if stats['total'] == 0:
#             return 0.0
#         return stats['success'] / stats['total']
#
#     def _explore(self, world):
#         state = world.get_state(self)
#         dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
#         candidates = []
#         fallback = []
#         for (dx, dz) in dirs:
#             next_x = self.x + dx * self.step_size
#             next_z = self.z + dz * self.step_size
#             if not world.is_within_world(next_x, next_z):
#                 continue
#
#             obj = world.get_object_at(next_x, next_z)
#             if obj is not None:
#                 if isinstance(obj, Predator):
#                     continue  # не идём на клетку с хищником
#                 continue
#
#             # if world.get_object_at(next_x, next_z) is not None:
#             #     continue
#             node1 = (self.x, self.z)
#             node2 = (next_x, next_z)
#             if node1 > node2:
#                 node1, node2 = node2, node1
#             if (node1, node2) not in self.visited_edges_set:
#                 candidates.append((dx, dz))
#             else:
#                 fallback.append((dx, dz))
#         if candidates:
#             dx, dz = random.choice(candidates)
#             reward_step = 1.0
#             action_idx = dirs.index((dx, dz))
#         elif fallback:
#             dx, dz = random.choice(fallback)
#             reward_step = -0.1
#             action_idx = dirs.index((dx, dz))
#         else:
#             self.alive = False
#             reward_step = -1.0
#             next_state = world.get_state(self)
#             self.add_experience(state, 0, reward_step, next_state)
#             self.total_reward += reward_step
#             return
#         next_x = self.x + dx * self.step_size
#         next_z = self.z + dz * self.step_size
#         self.visited_edges.append(((self.x, self.z), (next_x, next_z)))
#         self._add_edge((self.x, self.z), (next_x, next_z))
#         self.x, self.z = next_x, next_z
#         self.visited_nodes.append((self.x, self.z))
#         next_state = world.get_state(self)
#         self.add_experience(state, action_idx, reward_step, next_state)
#         self.total_reward += reward_step
#         for (dx_check, dz_check) in dirs:
#             check_x = self.x + dx_check * self.step_size
#             check_z = self.z + dz_check * self.step_size
#             obj = world.get_object_at(check_x, check_z)
#             if obj:
#                 self.setInform(obj)
#                 perception = Perception(self.nearby_params.copy())
#                 thresholds = self.genome.get('reflex_thresholds', {})
#                 suggestion = self.reflex_module.get_best_action(perception, thresholds)
#                 if suggestion:
#                     self.execute_action(suggestion.action_id, world, state)
#                     self._has_reflex_action = True
#                 break
#         if len(self.visited_nodes) > self.max_steps:
#             self.alive = False
#
#     def calculate_fitness(self):
#         visited_score = len(self.visited_nodes) * 1.0
#         food_score = self.food_collected * 10.0
#         survival_score = 50.0 if self.alive else 0.0
#         unique_nodes = len(set(self.visited_nodes))
#         repeat_penalty = (len(self.visited_nodes) - unique_nodes) * 0.5
#         diversity_bonus = unique_nodes * 0.1
#         return max(0, visited_score + food_score + survival_score - repeat_penalty + diversity_bonus)
#
#     def draw_path(self, screen, world_to_screen_func):
#         if not self.visited_edges:
#             return
#         for (x1, z1), (x2, z2) in self.visited_edges:
#             p1 = world_to_screen_func(x1, z1, 0.1)
#             p2 = world_to_screen_func(x2, z2, 0.1)
#             pygame.draw.line(screen, (0, 255, 0), p1, p2, 3)
#
#     def draw(self, screen, world_to_screen_func, scale):
#         self.draw_path(screen, world_to_screen_func)
#         cx, cz = self.x, self.z
#         ang_rad = math.radians(self.angle)
#         def rotate_point(lx, lz):
#             rx = lx * math.cos(ang_rad) - lz * math.sin(ang_rad)
#             rz = lx * math.sin(ang_rad) + lz * math.cos(ang_rad)
#             return cx + rx, cz + rz
#         corners_local = [
#             (-self.body_w/2, -self.body_d/2),
#             (self.body_w/2, -self.body_d/2),
#             (self.body_w/2, self.body_d/2),
#             (-self.body_w/2, self.body_d/2)
#         ]
#         corners_world = [rotate_point(lx, lz) for (lx, lz) in corners_local]
#         base_points = [world_to_screen_func(wx, wz, 0) for (wx, wz) in corners_world]
#         top_points = [world_to_screen_func(wx, wz, self.body_h) for (wx, wz) in corners_world]
#         pygame.draw.polygon(screen, (0, 150, 200), top_points)
#         for i in range(4):
#             j = (i+1) % 4
#             pts = [base_points[i], base_points[j], top_points[j], top_points[i]]
#             pygame.draw.polygon(screen, (0, 100, 150), pts)
#         pygame.draw.polygon(screen, (0, 80, 120), base_points)
#         head_offset_z = 0.2
#         head_x, head_z = rotate_point(0, head_offset_z)
#         head_y = self.body_h + self.head_r * 0.8
#         head_screen = world_to_screen_func(head_x, head_z, head_y)
#         rad_px = int(self.head_r * scale)
#         if rad_px > 1:
#             pygame.draw.circle(screen, (255, 200, 150), head_screen, rad_px)
#         if self.emotions_enabled and self.current_emotions:
#             dominant = max(self.current_emotions, key=lambda x: x.intensity)
#             pos = world_to_screen_func(self.x, self.z, self.body_h + 1.0)
#             font = pygame.font.Font(None, 24)
#             text = font.render(dominant.emotion_type.value[:3].upper(), True, (255, 255, 0))
#             screen.blit(text, (pos[0]-20, pos[1]-30))
#
#     def reset(self, start_x=0, start_z=0):
#         self.x = start_x
#         self.z = start_z
#         self.visited_nodes = [(self.x, self.z)]
#         self.visited_edges = []
#         self.visited_edges_set = set()
#         self.frame_counter = 0
#         self.runaway_target = None
#         self.awaiting_steps = 0
#         self.moving = True
#         self.alive = True
#         self.nearby_object = None
#         self.nearby_params = {}
#         self.fitness = 0.0
#         self.dir_index = 0
#
#     def evaluate(self, world, max_steps=500):
#         self.reset()
#         for _ in range(max_steps):
#             if not self.alive:
#                 break
#             self.update(world)
#         self.fitness = self.calculate_fitness()
#         return self.fitness
#
#     def calculate_fitness(self):
#         return len(self.visited_nodes) * 1.0
#
#     def add_experience(self, state, action, reward, next_state):
#         self.memory_buffer.append((state, action, reward, next_state))
#         if len(self.memory_buffer) > self.max_buffer_size:
#             self.memory_buffer.pop(0)
#
#     def get_experiences(self):
#         data = list(self.memory_buffer)
#         self.memory_buffer.clear()
#         return data
#
#
#
#
#
# # # core/individual.py
# # import pygame
# # import math
# # from core.genome import Genome
# # from models.reflex_module import ReflexModule
# # from models.instinct_module import InstinctModule
# # from core.base_strategy import Perception, ActionSuggestion
# # import random
# # from collections import deque
# # from typing import List, Dict, Optional, Tuple, Any, Union
# # import numpy as np
# # from core.emotions.emotion_system import EmotionSystem
# # from emotions.emotion_engine import EmotionType
# #
# # class Individual:
# #     def __init__(self, x=0, z=0, angle=0, move_delay=5, reflex_rules=None, instinct_patterns=None, genome=None,  max_buffer_size=10000):
# #         self.x = x
# #         self.z = z
# #         self.angle = angle
# #         self.body_w = 0.8
# #         self.body_d = 0.5
# #         self.body_h = 1.2
# #         self.head_r = 0.3
# #
# #         # [ru] Если передан genome, используем его, иначе создаём из параметров
# #         # [en] If genome is provided, use it, otherwise create from parameters
# #         if genome is None:
# #             genome = Genome({
# #                 'move_delay': move_delay,
# #                 'step_size': 2.0,
# #                 'reflex_rules': reflex_rules if reflex_rules else [],
# #                 'instinct_patterns': instinct_patterns if instinct_patterns else [],
# #                 'max_steps': 500
# #             })
# #         self.genome = genome
# #
# #         self.fitness = 0.0
# #         self.alive = True
# #         # [ru] всегда 2, игнорируем геном
# #         # [en] always 2, ignore genome
# #         self.step_size = 1.0
# #         # self.step_size = genome.get('step_size', 2.0)
# #         self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
# #         self.dir_index = 0
# #
# #         # [ru] Переменные для движения и сбора данных
# #         # [en] Variables for movement and data collection
# #         self.visited_nodes = [(self.x, self.z)]
# #         self.visited_edges = []
# #         self.visited_edges_set = set()
# #         self.frame_counter = 0
# #         self.move_delay = genome.get('move_delay', 5)
# #         self.max_steps = genome.get('max_steps', 1000)
# #
# #         # [ru] Модули рефлексов и инстинктов
# #         # [en] Reflex and instinct modules
# #         self.reflex_module = ReflexModule(
# #             reflex_rules if reflex_rules is not None else genome.get('reflex_rules', [])
# #         )
# #         self.instinct_module = InstinctModule(
# #             instinct_patterns if instinct_patterns is not None else genome.get('instinct_patterns', [])
# #         )
# #
# #         self.nearby_object = None
# #         self.nearby_params = {}
# #
# #         # [ru] Состояние убегания (для взрыва и т.п.)
# #         # [en] Runaway state (for explosion, etc.)
# #         self.runaway_target = None
# #         self.awaiting_steps = 0
# #         self.moving = True
# #
# #         self.max_buffer_size = max_buffer_size
# #         # [ru] список переходов (state, action, reward, next_state)
# #         # [en] list of transitions (state, action, reward, next_state)
# #         self.memory_buffer = deque(maxlen=self.max_buffer_size)
# #
# #         # self.max_buffer_size = 10000
# #
# #         self.runaway_attempts = 0
# #
# #         # [ru] счётчик собранной еды
# #         # [en] collected food counter
# #         self.food_collected = 0
# #         # [ru] суммарная награда за эпизод
# #         # [en] total reward for episode
# #         self.total_reward = 0.0
# #
# #         # [ru] счётчики для обратной связи {action_id: {'success': 0, 'total': 0}}
# #         # [en] counters for feedback {action_id: {'success': 0, 'total': 0}}
# #         self.reflex_stats = {}
# #         # {pattern_id: {'success': 0, 'total': 0}}
# #         self.instinct_stats = {}
# #
# #         # [ru] флаги для приоритетов
# #         # [en] flags for priorities
# #         self._has_reflex_action = False
# #         self._has_instinct_action = False
# #
# #         # [ru] Создаём эмоциональную подсистему для каждого бота
# #         # [en] We create an emotional subsystem for each bot
# #         self.emotion_system = EmotionSystem()
# #         # По умолчанию эмоции выключены (для демонстрации)
# #         self.emotions_enabled = False
# #
# #         self.health = 100       # здоровье
# #         self.food_reserve = 0   # запас еды
# #
# #         self.current_emotions = []  # для хранения активных эмоций
# #         # self.current_emotions = self.emotion_system.process_sensory_input(sensory_data)  # для хранения активных эмоций
# #         self.emotions_enabled = False  # по умолчанию выключены
# #
# #     def set_emotions_enabled(self, enabled: bool):
# #         self.emotions_enabled = enabled
# #
# #     def share_food(self, target_bot, amount=1):
# #         """
# #         [ru] Передаёт часть запаса еды другому боту.
# #         [en] Transfers part of the food supply to another bot.
# #         """
# #         if self.food_reserve >= amount and target_bot.alive:
# #             self.food_reserve -= amount
# #             target_bot.food_reserve += amount
# #             target_bot.health = min(100, target_bot.health + 10)  # восстанавливаем здоровье
# #             print(f"[ru] {self} поделился едой с {target_bot}")
# #             print(f"[en] {self} shared food with {target_bot}")
# #             return True
# #         return False
# #
# #     def _perceive_others(self, world):
# #         """
# #         [ru] Собирает информацию о других ботах в поле зрения.
# #         [en] Collects information about other bots in the field of view.
# #         """
# #         others = []
# #         for other in world.bots:  # предположим, что world хранит список ботов
# #             if other is not self and other.alive:
# #                 dx = other.x - self.x
# #                 dz = other.z - self.z
# #                 dist = math.hypot(dx, dz)
# #                 if dist < 10.0:  # радиус восприятия
# #                     others.append({
# #                         'bot': other,
# #                         'distance': dist,
# #                         'health': other.health,
# #                         'food_reserve': other.food_reserve
# #                     })
# #         return others
# #
# #     # [ru] ---------- Взаимодействие с объектами ----------
# #     # [en] ---------- Interaction with objects ----------
# #     def setInform(self, obj):
# #         self.nearby_object = obj
# #         self.nearby_params = obj.get(['type', 'temperature', 'smell', 'sound', 'name'])
# #
# #     def _add_edge(self, node1, node2):
# #         if node1 > node2:
# #             node1, node2 = node2, node1
# #         self.visited_edges_set.add((node1, node2))
# #
# #     def _move_one_step(self, world):
# #         dx, dz = self.directions[self.dir_index]
# #         next_x = self.x + dx * self.step_size
# #         next_z = self.z + dz * self.step_size
# #         if world.is_within_world(next_x, next_z) and world.get_object_at(next_x, next_z) is None:
# #             self.visited_edges.append(((self.x, self.z), (next_x, next_z)))
# #             self._add_edge((self.x, self.z), (next_x, next_z))
# #             self.x, self.z = next_x, next_z
# #             self.visited_nodes.append((self.x, self.z))
# #             return True
# #         return False
# #
# #     def _turn_right(self):
# #         self.dir_index = (self.dir_index + 1) % 4
# #
# #     # [ru] ---------- Действия от рефлексов ----------
# #     # [en] ---------- Actions from reflexes ----------
# #     def execute_action(self, action, world, state):
# #         if action == 'move_on':
# #             print("[ru] Рефлекс: move_on! Разворот и уход на 2 клетки.")
# #             print("[en] Reflex: move_on! Turn around and move 2 cells.")
# #             self.dir_index = (self.dir_index + 2) % 4
# #             for _ in range(2):
# #                 if not self._move_one_step(world):
# #                     break
# #         elif action == 'grab':
# #             print("[ru] Рефлекс: grab! Захват еды.")
# #             print("[en] Reflex: grab! Grabbing food.")
# #             # [ru] передаём state
# #             # [en] pass state
# #             self._grab_object(world, state)
# #         elif action == 'avoid':
# #             print("[ru] Рефлекс: avoid! Отворачиваем.")
# #             print("[en] Reflex: avoid! Turning away.")
# #             self._turn_right()
# #         else:
# #             print(f"[ru] Неизвестное действие: {action}")
# #             print(f"[en] Unknown action: {action}")
# #
# #         print(f"[Action] executing {action}")
# #
# #
# #     def _grab_object(self, world, state):
# #         if self.nearby_object is None:
# #             return False
# #         target_x = self.nearby_object.x
# #         target_z = self.nearby_object.z
# #         dx = target_x - self.x
# #         dz = target_z - self.z
# #         if abs(dx) > self.step_size or abs(dz) > self.step_size:
# #             print("[ru] Объект не в соседней клетке")
# #             print("[en] Object is not in adjacent cell")
# #             return False
# #         # [ru] Удаляем объект до перемещения, чтобы сохранить правильное next_state
# #         # [en] Remove object before movement to preserve correct next_state
# #         world.remove_object(self.nearby_object)
# #         # [ru] Сохраняем старое состояние для next_state (после удаления объекта). Но next_state должно быть после перемещения, поэтому сначала перемещаемся
# #         # [en] Save old state for next_state (after object removal).  But next_state should be after movement, so move first
# #         self.visited_edges.append(((self.x, self.z), (target_x, target_z)))
# #         self._add_edge((self.x, self.z), (target_x, target_z))
# #         self.x, self.z = target_x, target_z
# #         self.visited_nodes.append((self.x, self.z))
# #         self.food_collected += 1
# #         self.food_reserve += 1  # добавляем в запас
# #         # [ru] Награда за захват еды
# #         # [en] Reward for grabbing food
# #         reward = 10.0
# #         # [ru] Получаем новое состояние
# #         # [en] Get new state
# #         next_state = world.get_state(self)
# #         # [ru] Добавляем опыт с action_id=4 (специальный код для "grab")
# #         # [en] Add experience with action_id=4 (special code for "grab")
# #         self.add_experience(state, 4, reward, next_state)
# #         self.total_reward += reward
# #         print(
# #             f"[ru] Бот схватил {self.nearby_object.name if hasattr(self.nearby_object, 'name') else 'еду'} и переместился в ({self.x}, {self.z})")
# #         print(
# #             f"[en] Bot grabbed {self.nearby_object.name if hasattr(self.nearby_object, 'name') else 'food'} and moved to ({self.x}, {self.z})")
# #         self.nearby_object = None
# #         self.nearby_params = {}
# #         return True
# #
# #
# #     # [ru] ---------- Инстинкты (взрыв) ----------
# #     # [en] ---------- Instincts (explosion) ----------
# #     def notify(self, event_type, data):
# #         print(f"notify called: event_type={event_type}, data={data}")
# #         if event_type == 'explosion':
# #             perception = Perception({
# #                 'sound': data.get('sound'),
# #                 'vision': data.get('vision'),
# #                 'position': data.get('position')
# #             })
# #             print(f" Perception: {perception}")
# #             suggestion = self.instinct_module.get_best_action(perception)
# #             print(f" Suggestion: {suggestion}")
# #             if suggestion:
# #                 self.execute_instinct(suggestion.action_id, data.get('position'))
# #
# #
# #     def execute_instinct(self, action_id, target_pos):
# #         if action_id == 'run_away':
# #             dx = self.x - target_pos[0]
# #             dz = self.z - target_pos[1]
# #             if abs(dx) >= abs(dz):
# #                 dir_vec = (1, 0) if dx >= 0 else (-1, 0)
# #             else:
# #                 dir_vec = (0, 1) if dz >= 0 else (0, -1)
# #             self.runaway_target = dir_vec
# #             self.awaiting_steps = 0
# #             self.moving = True
# #             print(f"[ru] Убегаем в направлении {dir_vec}")
# #             print(f"[en] Running away in direction {dir_vec}")
# #
# #     def _update_runaway(self, world):
# #         dx, dz = self.runaway_target
# #         next_x = self.x + dx * self.step_size
# #         next_z = self.z + dz * self.step_size
# #
# #         # [ru] Если следующий шаг выходит за границы мира – сразу останавливаемся
# #         # [en] If next step goes beyond world boundaries - stop immediately
# #         if not world.is_within_world(next_x, next_z):
# #             if self.awaiting_steps == 0:
# #                 self.awaiting_steps = 200
# #                 print(f"[ru] Достигнута граница мира, остановка убегания на {self.awaiting_steps} шагов")
# #                 print(f"[en] World boundary reached, stopping runaway for {self.awaiting_steps} steps")
# #             else:
# #                 self.awaiting_steps -= 1
# #                 if self.awaiting_steps <= 0:
# #                     self.runaway_target = None
# #                     self.moving = True
# #                     self.frame_counter = 0
# #                     print("[ru] Возврат к обходу")
# #                     print("[en] Return to exploration")
# #             return
# #
# #         # [ru] Проверяем, можно ли двинуться прямо (в пределах мира и нет объекта)
# #         # [en] Check if can move forward (within world and no object)
# #         if world.get_object_at(next_x, next_z) is None:
# #             # [ru] Успешный шаг – сбрасываем счётчик неудач
# #             # [en] Successful step - reset failure counter
# #             self.runaway_attempts = 0
# #             self.visited_edges.append(((self.x, self.z), (next_x, next_z)))
# #             self._add_edge((self.x, self.z), (next_x, next_z))
# #             self.x, self.z = next_x, next_z
# #             self.visited_nodes.append((self.x, self.z))
# #             self.awaiting_steps = 0
# #             return
# #
# #         # [ru] Если прямо нельзя (в пределах мира, но занято объектом) – увеличиваем счётчик неудач
# #         # [en] If cannot go forward (within world but occupied by object) - increment failure counter
# #         self.runaway_attempts += 1
# #
# #         # [ru] Если уже много неудачных попыток подряд – прекращаем попытки и ждём
# #         # [en] If too many failed attempts in a row - stop trying and wait
# #         if self.runaway_attempts > 5:
# #             if self.awaiting_steps == 0:
# #                 self.awaiting_steps = 200
# #                 print(f"[ru] Достигнут предел попыток, остановка убегания на {self.awaiting_steps} шагов")
# #                 print(f"[en] Attempt limit reached, stopping runaway for {self.awaiting_steps} steps")
# #             else:
# #                 self.awaiting_steps -= 1
# #                 if self.awaiting_steps <= 0:
# #                     self.runaway_target = None
# #                     self.moving = True
# #                     self.frame_counter = 0
# #                     print("[ru] Возврат к обходу")
# #                     print("[en] Return to exploration")
# #             return
# #
# #         # [ru] Пробуем повернуть влево или вправо
# #         # [en] Try to turn left or right
# #         left = (dz, -dx)
# #         right = (-dz, dx)
# #         for (ndx, ndz) in [left, right]:
# #             nnext_x = self.x + ndx * self.step_size
# #             nnext_z = self.z + ndz * self.step_size
# #             # [ru] Проверяем, что поворот ведёт в пределах мира и не занят объектом
# #             # [en] Check if turn leads within world and is not occupied by object
# #             if world.is_within_world(nnext_x, nnext_z) and world.get_object_at(nnext_x, nnext_z) is None:
# #                 # [ru] Обновляем направление, сбрасываем счётчик неудач
# #                 # [en] Update direction, reset failure counter
# #                 self.runaway_target = (ndx, ndz)
# #                 self.runaway_attempts = 0
# #                 self.visited_edges.append(((self.x, self.z), (nnext_x, nnext_z)))
# #                 self._add_edge((self.x, self.z), (nnext_x, nnext_z))
# #                 self.x, self.z = nnext_x, nnext_z
# #                 self.visited_nodes.append((self.x, self.z))
# #                 self.awaiting_steps = 0
# #                 return
# #
# #         # [ru] Если ни один поворот не сработал (например, везде объекты) – ждём
# #         # [en] If no turn worked (e.g., objects everywhere) - wait
# #         if self.awaiting_steps == 0:
# #             self.awaiting_steps = 200
# #             print(f"[ru] Нет доступных направлений, остановка убегания на {self.awaiting_steps} шагов")
# #             print(f"[en] No available directions, stopping runaway for {self.awaiting_steps} steps")
# #         else:
# #             self.awaiting_steps -= 1
# #             if self.awaiting_steps <= 0:
# #                 self.runaway_target = None
# #                 self.moving = True
# #                 self.frame_counter = 0
# #                 print("[ru] Возврат к обходу")
# #                 print("[en] Return to exploration")
# #
# #     # [ru] ---------- Основной цикл обновления ----------
# #     # [en] ---------- Main update loop ----------
# #     def update(self, world):
# #
# #         # [ru] Собираем сенсорные данные для эмоций
# #         # [en] Collecting sensory data for emotions
# #         sensory_data = {
# #             'vision': self._get_vision(world),
# #             'sound': self._get_sound(world),
# #             'smell': self._get_smell(world),
# #             'position': (self.x, self.z),
# #             'context': {
# #                 'nearby_object': self.nearby_params,
# #                 'visited_count': len(self.visited_nodes),
# #                 'food_collected': self.food_collected,
# #                 'other_bots': self._perceive_others(world)
# #             }
# #         }
# #
# #         if self.emotions_enabled:
# #             sensory_data = {
# #                 'vision': self._get_vision(world),
# #                 'sound': self._get_sound(world),
# #                 'smell': self._get_smell(world),
# #                 'position': (self.x, self.z),
# #                 'context': {
# #                     'nearby_object': self.nearby_params,
# #                     'visited_count': len(self.visited_nodes),
# #                     'food_collected': self.food_collected,
# #                     'other_bots': self._perceive_others(world)
# #                 }
# #             }
# #             self.current_emotions = self.emotion_system.process_sensory_input(sensory_data)
# #
# #         # [ru] Если эмоции включены и есть активная эмоция "сочувствие" с достаточной интенсивностью
# #         # [en] If emotions are included and there is an active emotion of "sympathy" with sufficient intensity
# #         if self.emotions_enabled and self.current_emotions:
# #             # Находим эмоцию "сочувствие" (EMPATHY)
# #             sympathy = next((e for e in self.current_emotions if e.emotion_type == EmotionType.EMPATHY), None)
# #             if sympathy and sympathy.intensity > 0.6:  # порог сочувствия
# #                 # Ищем ближайшего раненого бота (здоровье < 30)
# #                 other_bots = self._perceive_others(world)
# #                 for info in other_bots:
# #                     if info['health'] < 30 and info['distance'] < 3.0:  # рядом
# #                         if self.food_reserve > 0:
# #                             self.share_food(info['bot'], 1)
# #                             # После передачи еды, снижаем интенсивность эмоции (чтобы не делиться бесконечно)
# #                             sympathy.intensity *= 0.5
# #                         break
# #
# #         # [ru] Если эмоции включены – обновляем эмоциональное состояние
# #         # [en] If emotions are enabled, update the emotional state
# #         # if self.emotions_enabled:
# #         #     responses = self.emotion_system.process_sensory_input(sensory_data)
# #         #     # Сохраняем активные эмоции для использования в решениях
# #         #     self.current_emotions = responses
# #
# #         # ============================================================
# #         # [ru] УРОВЕНЬ 1: ИНСТИНКТЫ (ВЫСШИЙ ПРИОРИТЕТ - ВЫЖИВАНИЕ)
# #         # [en] LEVEL 1: INSTINCTS (HIGHEST PRIORITY - SURVIVAL)
# #         # ============================================================
# #         if self.runaway_target:
# #             # [ru] Проверяем, достигли ли безопасного расстояния
# #             # [en] Check if safe distance is reached
# #             if self._is_safe(world):
# #                 self.runaway_target = None
# #                 print("[ru] Безопасно, возврат к исследованию")
# #                 print("[en] Safe, return to exploration")
# #             else:
# #                 self._update_runaway(world)
# #                 self._has_instinct_action = True
# #                 return
# #
# #         # Активный поиск еды (если нет активных инстинктов)
# #         if not self.runaway_target:
# #             # Ищем ближайшую еду
# #             nearest_food = None
# #             min_dist = float('inf')
# #             for obj in world.objects:
# #                 if isinstance(obj, Food) and obj.active:
# #                     dx = obj.x - self.x
# #                     dz = obj.z - self.z
# #                     dist = math.hypot(dx, dz)
# #                     if dist < min_dist:
# #                         min_dist = dist
# #                         nearest_food = obj
# #             if nearest_food and min_dist < 5.0:
# #                 # Двигаемся к еде
# #                 dx = nearest_food.x - self.x
# #                 dz = nearest_food.z - self.z
# #                 # Выбираем направление с наибольшей проекцией
# #                 if abs(dx) > abs(dz):
# #                     target_dir = (1 if dx > 0 else -1, 0)
# #                 else:
# #                     target_dir = (0, 1 if dz > 0 else -1)
# #                 # Пытаемся повернуть в нужном направлении
# #                 current_dir = self.directions[self.dir_index]
# #                 if current_dir != target_dir:
# #                     # Пробуем повернуть налево или направо
# #                     left = (current_dir[1], -current_dir[0])
# #                     right = (-current_dir[1], current_dir[0])
# #                     if left == target_dir:
# #                         self.dir_index = (self.dir_index - 1) % 4
# #                     elif right == target_dir:
# #                         self.dir_index = (self.dir_index + 1) % 4
# #                 # Двигаемся
# #                 self._move_one_step(world)
# #                 return
# #
# #         # ============================================================
# #         # [ru] УРОВЕНЬ 2: РЕФЛЕКСЫ (СРЕДНИЙ ПРИОРИТЕТ)
# #         # [en] LEVEL 2: REFLEXES (MEDIUM PRIORITY)
# #         # ============================================================
# #         state = world.get_state(self)
# #         dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
# #
# #         # [ru] Проверяем все соседние клетки
# #         # [en] Check all adjacent cells
# #         for (dx, dz) in dirs:
# #             check_x = self.x + dx * self.step_size
# #             check_z = self.z + dz * self.step_size
# #             obj = world.get_object_at(check_x, check_z)
# #
# #             if obj:
# #                 self.setInform(obj)
# #                 perception = Perception(self.nearby_params.copy())
# #
# #                 # [ru] Получаем пороги из генома
# #                 # [en] Get thresholds from genome
# #                 thresholds = self.genome.get('reflex_thresholds', {})
# #
# #                 suggestion = self.reflex_module.get_best_action(
# #                     perception,
# #                     thresholds
# #                 )
# #
# #                 if suggestion:
# #                     # [ru] ВЫПОЛНЯЕМ РЕФЛЕКС
# #                     # [en] EXECUTE REFLEX
# #                     self.execute_action(suggestion.action_id, world, state)
# #
# #                     # [ru] Сохраняем результат для обратной связи
# #                     # [en] Save result for feedback
# #                     self._has_reflex_action = True
# #                     self._record_reflex_outcome(suggestion.action_id, True)
# #
# #                     # [ru] Сохраняем опыт
# #                     # [en] Save experience
# #                     next_state = world.get_state(self)
# #                     self.add_experience(state, suggestion.action_id, 1.0, next_state)
# #                     # [ru] Рефлекс выполнен, выходим
# #                     # [en] Reflex executed, exit
# #                     return
# #
# #         # ============================================================
# #         # [ru] УРОВЕНЬ 3: ИССЛЕДОВАНИЕ (НИЗШИЙ ПРИОРИТЕТ)
# #         # [en] LEVEL 3: EXPLORATION (LOWEST PRIORITY)
# #         # ============================================================
# #         self._explore(world)
# #
# #         # ============================================================
# #         # [ru] ОБНОВЛЕНИЕ ЭМОЦИОНАЛЬНОЙ СИСТЕМЫ
# #         # [en] EMOTION SYSTEM UPDATE
# #         # ============================================================
# #         if hasattr(self, 'emotion_system') and self.emotion_system:
# #             sensory_data = self._get_sensory_data(world)
# #             self.current_emotions = self.emotion_system.process_sensory_input(sensory_data)
# #
# #         # if hasattr(self, 'emotion_system') and self.emotion_system:
# #         #     sensory_data = self._get_sensory_data(world)
# #         #     self.emotion_system.update(sensory_data)
# #
# #         # ============================================================
# #         # [ru] ЭМОЦИОНАЛЬНОЕ ПОВЕДЕНИЕ: СОЧУВСТВИЕ
# #         # [en] EMOTIONAL BEHAVIOR: EMPATHY
# #         # ============================================================
# #         if self.emotions_enabled and hasattr(self, 'current_emotions') and self.current_emotions:
# #             sympathy = next((e for e in self.current_emotions if e.emotion_type == EmotionType.EMPATHY), None)
# #             if sympathy and sympathy.intensity > 0.6 and self.food_reserve > 0:
# #                 other_bots = self._perceive_others(world)
# #                 for info in other_bots:
# #                     if info['health'] < 30 and info['distance'] < 3.0:
# #                         if self.share_food(info['bot'], 1):
# #                             sympathy.intensity *= 0.5
# #                             break
# #
# #
# #         # if self.emotions_enabled and hasattr(self, 'current_emotions'):
# #         #     # Ищем эмоцию "сочувствие" (EMPATHY)
# #         #     sympathy = next((e for e in self.current_emotions if e.emotion_type == EmotionType.EMPATHY), None)
# #         #     if sympathy and sympathy.intensity > 0.6 and self.food_reserve > 0:
# #         #         # Ищем раненого бота поблизости
# #         #         other_bots = self._perceive_others(world)
# #         #         for info in other_bots:
# #         #             if info['health'] < 30 and info['distance'] < 3.0:
# #         #                 # Делимся едой
# #         #                 if self.share_food(info['bot'], 1):
# #         #                     sympathy.intensity *= 0.5  # снижаем эмоцию после акта
# #         #                     break
# #
# #     def _get_sensory_data(self, world) -> Dict:
# #         """
# #         [ru] Собирает сенсорные данные для эмоциональной системы.
# #         [en] Collects sensory data for the emotion system.
# #         """
# #         sensory_data = {
# #             'vision': self._get_vision(world),
# #             'sound': self._get_sound(world),
# #             'smell': self._get_smell(world),
# #             'position': (self.x, self.z),
# #             'context': {
# #                 'nearby_object': self.nearby_params,
# #                 'visited_count': len(self.visited_nodes),
# #                 'food_collected': self.food_collected
# #             }
# #         }
# #         return sensory_data
# #
# #     def _get_vision(self, world) -> np.ndarray:
# #         """
# #         [ru] Получает визуальные данные.
# #         [en] Gets visual data.
# #         """
# #         # [ru] Простая реализация: смотрим, что рядом
# #         # [en] Simple implementation: look at what's nearby
# #         vision = np.zeros(8)
# #         if self.nearby_object:
# #             vision[0] = 1.0
# #         return vision
# #
# #     def _get_sound(self, world) -> np.ndarray:
# #         """
# #         [ru] Получает звуковые данные.
# #         [en] Gets sound data.
# #         """
# #         sound = np.zeros(8)
# #         # [ru] Простая реализация
# #         # [en] Simple implementation
# #         return sound
# #
# #     def _get_smell(self, world) -> np.ndarray:
# #         """
# #         [ru] Получает данные о запахах.
# #         [en] Gets smell data.
# #         """
# #         smell = np.zeros(8)
# #         if self.nearby_params:
# #             if 'smell' in self.nearby_params:
# #                 smell[0] = 1.0
# #         return smell
# #
# #     def _perceive_others(self, world):
# #         """
# #         [ru] Собирает информацию о других ботах в поле зрения.
# #         [en] Collects information about other bots in the field of view.
# #         """
# #         others = []
# #         if not hasattr(world, 'bots'):
# #             return others
# #         for other in world.bots:
# #             if other is not self and other.alive:
# #                 dx = other.x - self.x
# #                 dz = other.z - self.z
# #                 dist = math.hypot(dx, dz)
# #                 if dist < 10.0:  # радиус восприятия
# #                     others.append({
# #                         'bot': other,
# #                         'distance': dist,
# #                         'health': other.health,
# #                         'food_reserve': other.food_reserve
# #                     })
# #         return others
# #
# #     def _is_safe(self, world) -> bool:
# #         """
# #         [ru] Проверяет, безопасно ли текущее местоположение.
# #         [en] Checks if current location is safe.
# #         """
# #         # [ru] Проверяем, нет ли рядом опасных объектов
# #         # [en] Check if there are dangerous objects nearby
# #         dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
# #         for (dx, dz) in dirs:
# #             check_x = self.x + dx * self.step_size * 3
# #             check_z = self.z + dz * self.step_size * 3
# #             obj = world.get_object_at(check_x, check_z)
# #             if obj and hasattr(obj, 'danger_level') and obj.danger_level > 0.5:
# #                 return False
# #         return True
# #
# #     def _record_reflex_outcome(self, action_id: str, success: bool):
# #         """
# #         [ru] Записывает результат рефлекса для обратной связи.
# #         [en] Records reflex result for feedback.
# #         """
# #         if action_id not in self.reflex_stats:
# #             self.reflex_stats[action_id] = {'success': 0, 'total': 0}
# #         self.reflex_stats[action_id]['total'] += 1
# #         if success:
# #             self.reflex_stats[action_id]['success'] += 1
# #
# #     def _record_instinct_outcome(self, pattern_id: str, success: bool):
# #         """
# #         [ru] Записывает результат инстинкта для обратной связи.
# #         [en] Records instinct result for feedback.
# #         """
# #         if pattern_id not in self.instinct_stats:
# #             self.instinct_stats[pattern_id] = {'success': 0, 'total': 0}
# #         self.instinct_stats[pattern_id]['total'] += 1
# #         if success:
# #             self.instinct_stats[pattern_id]['success'] += 1
# #
# #     def get_reflex_success_rate(self, action_id: str) -> float:
# #         """
# #         [ru] Возвращает процент успешности рефлекса.
# #         [en] Returns reflex success percentage.
# #         """
# #         stats = self.reflex_stats.get(action_id, {'success': 0, 'total': 0})
# #         if stats['total'] == 0:
# #             return 0.0
# #         return stats['success'] / stats['total']
# #
# #     def _explore(self, world):
# #         """
# #         [ru] УРОВЕНЬ 3: ИССЛЕДОВАНИЕ (низший приоритет). Выполняется, если нет активных инстинктов и рефлексов.
# #         [en] LEVEL 3: EXPLORATION (lowest priority). Executed if there are no active instincts and reflexes.
# #         """
# #         # [ru] Получаем состояние до действия
# #         # [en] Get state before action
# #         state = world.get_state(self)
# #
# #         dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
# #         candidates = []
# #         fallback = []
# #
# #         for (dx, dz) in dirs:
# #             next_x = self.x + dx * self.step_size
# #             next_z = self.z + dz * self.step_size
# #
# #             # [ru] Проверяем границы
# #             # [en] Check boundaries
# #             if not world.is_within_world(next_x, next_z):
# #                 continue
# #
# #             # [ru] Проверяем, не занято ли объектом
# #             # [en] Check if not occupied by object
# #             if world.get_object_at(next_x, next_z) is not None:
# #                 continue
# #
# #             node1 = (self.x, self.z)
# #             node2 = (next_x, next_z)
# #             if node1 > node2:
# #                 node1, node2 = node2, node1
# #
# #             # [ru] Предпочитаем непосещённые узлы
# #             # [en] Prefer unvisited nodes
# #             if (node1, node2) not in self.visited_edges_set:
# #                 candidates.append((dx, dz))
# #             else:
# #                 fallback.append((dx, dz))
# #
# #         # [ru] Выбираем направление
# #         # [en] Choose direction
# #         if candidates:
# #             dx, dz = random.choice(candidates)
# #             # [ru] Новый узел
# #             # [en] New node
# #             reward_step = 1.0
# #             action_idx = dirs.index((dx, dz))
# #         elif fallback:
# #             dx, dz = random.choice(fallback)
# #             # [ru] Штраф за повтор
# #             # [en] Penalty for repeat
# #             reward_step = -0.1
# #             action_idx = dirs.index((dx, dz))
# #         else:
# #             # [ru] Нет доступных направлений - бот застрял
# #             # [en] No available directions - bot is stuck
# #             self.alive = False
# #             reward_step = -1.0
# #             next_state = world.get_state(self)
# #             self.add_experience(state, 0, reward_step, next_state)
# #             self.total_reward += reward_step
# #             return
# #
# #         # [ru] Делаем шаг
# #         # [en] Take step
# #         next_x = self.x + dx * self.step_size
# #         next_z = self.z + dz * self.step_size
# #         self.visited_edges.append(((self.x, self.z), (next_x, next_z)))
# #         self._add_edge((self.x, self.z), (next_x, next_z))
# #         self.x, self.z = next_x, next_z
# #         self.visited_nodes.append((self.x, self.z))
# #
# #         # [ru] Получаем новое состояние
# #         # [en] Get new state
# #         next_state = world.get_state(self)
# #
# #         # [ru] Добавляем переход в буфер
# #         # [en] Add transition to buffer
# #         self.add_experience(state, action_idx, reward_step, next_state)
# #         self.total_reward += reward_step
# #
# #         # [ru] Проверяем объекты в соседних клетках (рефлексы)
# #         # [en] Check objects in adjacent cells (reflexes)
# #         for (dx_check, dz_check) in dirs:
# #             check_x = self.x + dx_check * self.step_size
# #             check_z = self.z + dz_check * self.step_size
# #             obj = world.get_object_at(check_x, check_z)
# #             if obj:
# #                 self.setInform(obj)
# #                 perception = Perception(self.nearby_params.copy())
# #                 thresholds = self.genome.get('reflex_thresholds', {})
# #                 suggestion = self.reflex_module.get_best_action(perception, thresholds)
# #                 if suggestion:
# #                     self.execute_action(suggestion.action_id, world, state)
# #                     self._has_reflex_action = True
# #                 break
# #
# #         # [ru] Проверяем лимит шагов
# #         # [en] Check step limit
# #         if len(self.visited_nodes) > self.max_steps:
# #             self.alive = False
# #             print("[ru] Достигнут лимит шагов")
# #             print("[en] Step limit reached")
# #
# #
# #
# #     # [ru] ---------- Фитнес ----------
# #     # [en] ---------- Fitness ----------
# #     def calculate_fitness(self):
# #         """
# #         [ru] Расчёт фитнеса с учётом разных факторов.
# #         [en] Fitness calculation considering various factors.
# #         """
# #         # [ru] 1. Количество посещённых узлов (максимум 500)
# #         # [en] 1. Number of visited nodes (maximum 500)
# #         visited_score = len(self.visited_nodes) * 1.0
# #
# #         # [ru] 2. Бонус за собранную еду
# #         # [en] 2. Bonus for collected food
# #         food_score = self.food_collected * 10.0
# #
# #         # [ru] 3. Бонус за выживание
# #         # [en] 3. Survival bonus
# #         survival_score = 50.0 if self.alive else 0.0
# #
# #         # [ru] 4. Штраф за повторение узлов (чем меньше повторов, тем лучше)
# #         # [en] 4. Penalty for node repetition (fewer repeats = better)
# #         unique_nodes = len(set(self.visited_nodes))
# #         repeat_penalty = (len(self.visited_nodes) - unique_nodes) * 0.5
# #
# #         # [ru] 5. Бонус за разнообразие посещённых узлов
# #         # [en] 5. Bonus for diversity of visited nodes
# #         diversity_bonus = unique_nodes * 0.1
# #
# #         # [ru] Итоговый фитнес
# #         # [en] Final fitness
# #         fitness = visited_score + food_score + survival_score - repeat_penalty + diversity_bonus
# #
# #         return max(0, fitness)
# #
# #
# #     # [ru] ---------- Отрисовка ----------
# #     # [en] ---------- Drawing ----------
# #     def draw_path(self, screen, world_to_screen_func):
# #         if not self.visited_edges:
# #             return
# #         for (x1, z1), (x2, z2) in self.visited_edges:
# #             p1 = world_to_screen_func(x1, z1, 0.1)
# #             p2 = world_to_screen_func(x2, z2, 0.1)
# #             pygame.draw.line(screen, (0, 255, 0), p1, p2, 3)
# #
# #     def draw(self, screen, world_to_screen_func, scale):
# #         self.draw_path(screen, world_to_screen_func)
# #
# #         cx, cz = self.x, self.z
# #         ang_rad = math.radians(self.angle)
# #
# #         def rotate_point(lx, lz):
# #             rx = lx * math.cos(ang_rad) - lz * math.sin(ang_rad)
# #             rz = lx * math.sin(ang_rad) + lz * math.cos(ang_rad)
# #             return cx + rx, cz + rz
# #
# #         corners_local = [
# #             (-self.body_w/2, -self.body_d/2),
# #             (self.body_w/2, -self.body_d/2),
# #             (self.body_w/2, self.body_d/2),
# #             (-self.body_w/2, self.body_d/2)
# #         ]
# #         corners_world = [rotate_point(lx, lz) for (lx, lz) in corners_local]
# #         base_points = [world_to_screen_func(wx, wz, 0) for (wx, wz) in corners_world]
# #         top_points = [world_to_screen_func(wx, wz, self.body_h) for (wx, wz) in corners_world]
# #
# #         pygame.draw.polygon(screen, (0, 150, 200), top_points)
# #         for i in range(4):
# #             j = (i+1) % 4
# #             pts = [base_points[i], base_points[j], top_points[j], top_points[i]]
# #             pygame.draw.polygon(screen, (0, 100, 150), pts)
# #         pygame.draw.polygon(screen, (0, 80, 120), base_points)
# #
# #         head_offset_z = 0.2
# #         head_x, head_z = rotate_point(0, head_offset_z)
# #         head_y = self.body_h + self.head_r * 0.8
# #         head_screen = world_to_screen_func(head_x, head_z, head_y)
# #         rad_px = int(self.head_r * scale)
# #         if rad_px > 1:
# #             pygame.draw.circle(screen, (255, 200, 150), head_screen, rad_px)
# #
# #         # Отображение здоровья и еды
# #         pos = world_to_screen_func(self.x, self.z, self.body_h + 1.5)
# #         font = pygame.font.Font(None, 20)
# #         text = font.render(f"HP:{self.health}  Food:{self.food_reserve}", True, (255, 255, 255))
# #         screen.blit(text, (pos[0] - 30, pos[1] - 10))
# #
# #         # Рисуем иконку эмоции
# #         if self.emotions_enabled and hasattr(self, 'current_emotions') and self.current_emotions:
# #             # Берём доминирующую эмоцию
# #             dominant = max(self.current_emotions, key=lambda x: x.intensity)
# #             pos = world_to_screen_func(self.x, self.z, self.body_h + 1.0)
# #             font = pygame.font.Font(None, 24)
# #             text = font.render(dominant.emotion_type.value[:3].upper(), True, (255, 255, 0))
# #             screen.blit(text, (pos[0]-20, pos[1]-30))
# #
# #     def reset(self, start_x=0, start_z=0):
# #         """
# #         [ru] Сбрасывает состояние бота для нового эпизода.
# #         [en] Resets bot state for new episode.
# #         """
# #         self.x = start_x
# #         self.z = start_z
# #         self.visited_nodes = [(self.x, self.z)]
# #         self.visited_edges = []
# #         self.visited_edges_set = set()
# #         self.frame_counter = 0
# #         self.runaway_target = None
# #         self.awaiting_steps = 0
# #         self.moving = True
# #         self.alive = True
# #         self.nearby_object = None
# #         self.nearby_params = {}
# #         self.fitness = 0.0
# #         self.dir_index = 0
# #
# #     def evaluate(self, world, max_steps=500):
# #         """
# #         [ru] Запускает бота в мире на max_steps шагов и вычисляет фитнес.
# #         [en] Runs bot in world for max_steps and calculates fitness.
# #         """
# #         self.reset()
# #         for _ in range(max_steps):
# #             if not self.alive:
# #                 break
# #             self.update(world)
# #         self.fitness = self.calculate_fitness()
# #         return self.fitness
# #
# #     def calculate_fitness(self):
# #         # [ru] Базовая: количество посещённых узлов (чем больше, тем лучше)
# #         # [en] Basic: number of visited nodes (more = better)
# #         return len(self.visited_nodes) * 1.0
# #
# #     def add_experience(self, state, action, reward, next_state):
# #         """
# #         [ru] Добавляет переход в буфер памяти. Автоматически удаляет старые при переполнении.
# #         [en] Adds transition to memory buffer. Automatically removes old ones when overflow.
# #         """
# #         self.memory_buffer.append((state, action, reward, next_state))
# #
# #         # [ru] Если буфер слишком большой, удаляем старые записи
# #         # [en] If buffer is too large, remove old records
# #         if len(self.memory_buffer) > self.max_buffer_size:
# #             self.memory_buffer.pop(0)
# #
# #     def get_experiences(self):
# #         """
# #         [ru] Возвращает список всех переходов и очищает буфер.
# #         [en] Returns list of all transitions and clears buffer.
# #         """
# #         data = list(self.memory_buffer)
# #         self.memory_buffer.clear()
# #         return data
# #
# #
# #
# # # # core/individual.py
# # # import pygame
# # # import math
# # # from core.genome import Genome
# # # from models.reflex_module import ReflexModule
# # # from models.instinct_module import InstinctModule
# # # from core.base_strategy import Perception, ActionSuggestion
# # # import random
# # # from collections import deque
# # # from typing import List, Dict, Optional, Tuple, Any, Union
# # # import numpy as np
# # #
# # # class Individual:
# # #     def __init__(self, x=0, z=0, angle=0, move_delay=5, reflex_rules=None, instinct_patterns=None, genome=None,  max_buffer_size=10000):
# # #         self.x = x
# # #         self.z = z
# # #         self.angle = angle
# # #         self.body_w = 0.8
# # #         self.body_d = 0.5
# # #         self.body_h = 1.2
# # #         self.head_r = 0.3
# # #
# # #         # Если передан genome, используем его, иначе создаём из параметров
# # #         if genome is None:
# # #             genome = Genome({
# # #                 'move_delay': move_delay,
# # #                 'step_size': 2.0,
# # #                 'reflex_rules': reflex_rules if reflex_rules else [],
# # #                 'instinct_patterns': instinct_patterns if instinct_patterns else [],
# # #                 'max_steps': 500
# # #             })
# # #         self.genome = genome
# # #
# # #         self.fitness = 0.0
# # #         self.alive = True
# # #         # self.step_size = genome.get('step_size', 2.0)
# # #         self.step_size = 2.0  # всегда 2, игнорируем геном
# # #         self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
# # #         self.dir_index = 0
# # #
# # #         # Переменные для движения и сбора данных
# # #         self.visited_nodes = [(self.x, self.z)]
# # #         self.visited_edges = []
# # #         self.visited_edges_set = set()
# # #         self.frame_counter = 0
# # #         self.move_delay = genome.get('move_delay', 5)
# # #         self.max_steps = genome.get('max_steps', 1000)
# # #
# # #         # Модули рефлексов и инстинктов
# # #         self.reflex_module = ReflexModule(
# # #             reflex_rules if reflex_rules is not None else genome.get('reflex_rules', [])
# # #         )
# # #         self.instinct_module = InstinctModule(
# # #             instinct_patterns if instinct_patterns is not None else genome.get('instinct_patterns', [])
# # #         )
# # #
# # #         self.nearby_object = None
# # #         self.nearby_params = {}
# # #
# # #         # Состояние убегания (для взрыва и т.п.)
# # #         self.runaway_target = None
# # #         self.awaiting_steps = 0
# # #         self.moving = True
# # #
# # #         self.max_buffer_size = max_buffer_size
# # #         self.memory_buffer = deque(maxlen=self.max_buffer_size)  # список переходов (state, action, reward, next_state)
# # #
# # #         # self.max_buffer_size = 10000
# # #
# # #         self.runaway_attempts = 0
# # #
# # #         self.food_collected = 0  # счётчик собранной еды
# # #         self.total_reward = 0.0  # суммарная награда за эпизод
# # #
# # #         # НОВОЕ: счётчики для обратной связи
# # #         self.reflex_stats = {}  # {action_id: {'success': 0, 'total': 0}}
# # #         self.instinct_stats = {}  # {pattern_id: {'success': 0, 'total': 0}}
# # #
# # #         # НОВОЕ: флаги для приоритетов
# # #         self._has_reflex_action = False
# # #         self._has_instinct_action = False
# # #
# # #     # ---------- Взаимодействие с объектами ----------
# # #     def setInform(self, obj):
# # #         self.nearby_object = obj
# # #         self.nearby_params = obj.get(['type', 'temperature', 'smell', 'sound', 'name'])
# # #
# # #     def _add_edge(self, node1, node2):
# # #         if node1 > node2:
# # #             node1, node2 = node2, node1
# # #         self.visited_edges_set.add((node1, node2))
# # #
# # #     def _move_one_step(self, world):
# # #         dx, dz = self.directions[self.dir_index]
# # #         next_x = self.x + dx * self.step_size
# # #         next_z = self.z + dz * self.step_size
# # #         if world.is_within_world(next_x, next_z) and world.get_object_at(next_x, next_z) is None:
# # #             self.visited_edges.append(((self.x, self.z), (next_x, next_z)))
# # #             self._add_edge((self.x, self.z), (next_x, next_z))
# # #             self.x, self.z = next_x, next_z
# # #             self.visited_nodes.append((self.x, self.z))
# # #             return True
# # #         return False
# # #
# # #     def _turn_right(self):
# # #         self.dir_index = (self.dir_index + 1) % 4
# # #
# # #     # ---------- Действия от рефлексов ----------
# # #     def execute_action(self, action, world, state):
# # #         if action == 'move_on':
# # #             print("Рефлекс: move_on! Разворот и уход на 2 клетки.")
# # #             self.dir_index = (self.dir_index + 2) % 4
# # #             for _ in range(2):
# # #                 if not self._move_one_step(world):
# # #                     break
# # #         elif action == 'grab':
# # #             print("Рефлекс: grab! Захват еды.")
# # #             self._grab_object(world, state)  # передаём state
# # #         elif action == 'avoid':
# # #             print("Рефлекс: avoid! Отворачиваем.")
# # #             self._turn_right()
# # #         else:
# # #             print(f"Неизвестное действие: {action}")
# # #
# # #         print(f"[Action] executing {action}")
# # #
# # #
# # #     def _grab_object(self, world, state):
# # #         if self.nearby_object is None:
# # #             return False
# # #         target_x = self.nearby_object.x
# # #         target_z = self.nearby_object.z
# # #         dx = target_x - self.x
# # #         dz = target_z - self.z
# # #         if abs(dx) > self.step_size or abs(dz) > self.step_size:
# # #             print("Объект не в соседней клетке")
# # #             return False
# # #         # Удаляем объект до перемещения, чтобы сохранить правильное next_state
# # #         world.remove_object(self.nearby_object)
# # #         # Сохраняем старое состояние для next_state (после удаления объекта)
# # #         # Но next_state должно быть после перемещения, поэтому сначала перемещаемся
# # #         self.visited_edges.append(((self.x, self.z), (target_x, target_z)))
# # #         self._add_edge((self.x, self.z), (target_x, target_z))
# # #         self.x, self.z = target_x, target_z
# # #         self.visited_nodes.append((self.x, self.z))
# # #         self.food_collected += 1
# # #         # Награда за захват еды
# # #         reward = 10.0
# # #         # Получаем новое состояние
# # #         next_state = world.get_state(self)
# # #         # Добавляем опыт с action_id=4 (специальный код для "grab")
# # #         self.add_experience(state, 4, reward, next_state)
# # #         self.total_reward += reward
# # #         print(
# # #             f"Bot схватил {self.nearby_object.name if hasattr(self.nearby_object, 'name') else 'еду'} и переместился в ({self.x}, {self.z})")
# # #         self.nearby_object = None
# # #         self.nearby_params = {}
# # #         return True
# # #
# # #
# # #     # ---------- Инстинкты (взрыв) ----------
# # #     def notify(self, event_type, data):
# # #         print(f"notify called: event_type={event_type}, data={data}")
# # #         if event_type == 'explosion':
# # #             perception = Perception({
# # #                 'sound': data.get('sound'),
# # #                 'vision': data.get('vision'),
# # #                 'position': data.get('position')
# # #             })
# # #             print(f"Perception: {perception}")
# # #             suggestion = self.instinct_module.get_best_action(perception)
# # #             print(f"Suggestion: {suggestion}")
# # #             if suggestion:
# # #                 self.execute_instinct(suggestion.action_id, data.get('position'))
# # #
# # #
# # #     def execute_instinct(self, action_id, target_pos):
# # #         if action_id == 'run_away':
# # #             dx = self.x - target_pos[0]
# # #             dz = self.z - target_pos[1]
# # #             if abs(dx) >= abs(dz):
# # #                 dir_vec = (1, 0) if dx >= 0 else (-1, 0)
# # #             else:
# # #                 dir_vec = (0, 1) if dz >= 0 else (0, -1)
# # #             self.runaway_target = dir_vec
# # #             self.awaiting_steps = 0
# # #             self.moving = True
# # #             print(f"Убегаем в направлении {dir_vec}")
# # #
# # #     def _update_runaway(self, world):
# # #         dx, dz = self.runaway_target
# # #         next_x = self.x + dx * self.step_size
# # #         next_z = self.z + dz * self.step_size
# # #
# # #         # Если следующий шаг выходит за границы мира – сразу останавливаемся
# # #         if not world.is_within_world(next_x, next_z):
# # #             if self.awaiting_steps == 0:
# # #                 self.awaiting_steps = 200
# # #                 print(f"Достигнута граница мира, остановка убегания на {self.awaiting_steps} шагов")
# # #             else:
# # #                 self.awaiting_steps -= 1
# # #                 if self.awaiting_steps <= 0:
# # #                     self.runaway_target = None
# # #                     self.moving = True
# # #                     self.frame_counter = 0
# # #                     print("Возврат к обходу")
# # #             return
# # #
# # #         # Проверяем, можно ли двинуться прямо (в пределах мира и нет объекта)
# # #         if world.get_object_at(next_x, next_z) is None:
# # #             # Успешный шаг – сбрасываем счётчик неудач
# # #             self.runaway_attempts = 0
# # #             self.visited_edges.append(((self.x, self.z), (next_x, next_z)))
# # #             self._add_edge((self.x, self.z), (next_x, next_z))
# # #             self.x, self.z = next_x, next_z
# # #             self.visited_nodes.append((self.x, self.z))
# # #             self.awaiting_steps = 0
# # #             return
# # #
# # #         # Если прямо нельзя (в пределах мира, но занято объектом) – увеличиваем счётчик неудач
# # #         self.runaway_attempts += 1
# # #
# # #         # Если уже много неудачных попыток подряд – прекращаем попытки и ждём
# # #         if self.runaway_attempts > 5:
# # #             if self.awaiting_steps == 0:
# # #                 self.awaiting_steps = 200
# # #                 print(f"Достигнут предел попыток, остановка убегания на {self.awaiting_steps} шагов")
# # #             else:
# # #                 self.awaiting_steps -= 1
# # #                 if self.awaiting_steps <= 0:
# # #                     self.runaway_target = None
# # #                     self.moving = True
# # #                     self.frame_counter = 0
# # #                     print("Возврат к обходу")
# # #             return
# # #
# # #         # Пробуем повернуть влево или вправо
# # #         left = (dz, -dx)
# # #         right = (-dz, dx)
# # #         for (ndx, ndz) in [left, right]:
# # #             nnext_x = self.x + ndx * self.step_size
# # #             nnext_z = self.z + ndz * self.step_size
# # #             # Проверяем, что поворот ведёт в пределах мира и не занят объектом
# # #             if world.is_within_world(nnext_x, nnext_z) and world.get_object_at(nnext_x, nnext_z) is None:
# # #                 # Обновляем направление, сбрасываем счётчик неудач
# # #                 self.runaway_target = (ndx, ndz)
# # #                 self.runaway_attempts = 0
# # #                 self.visited_edges.append(((self.x, self.z), (nnext_x, nnext_z)))
# # #                 self._add_edge((self.x, self.z), (nnext_x, nnext_z))
# # #                 self.x, self.z = nnext_x, nnext_z
# # #                 self.visited_nodes.append((self.x, self.z))
# # #                 self.awaiting_steps = 0
# # #                 return
# # #
# # #         # Если ни один поворот не сработал (например, везде объекты) – ждём
# # #         if self.awaiting_steps == 0:
# # #             self.awaiting_steps = 200
# # #             print(f"Нет доступных направлений, остановка убегания на {self.awaiting_steps} шагов")
# # #         else:
# # #             self.awaiting_steps -= 1
# # #             if self.awaiting_steps <= 0:
# # #                 self.runaway_target = None
# # #                 self.moving = True
# # #                 self.frame_counter = 0
# # #                 print("Возврат к обходу")
# # #
# # #     # ---------- Основной цикл обновления ----------
# # #     def update(self, world):
# # #
# # #         # ============================================================
# # #         # УРОВЕНЬ 1: ИНСТИНКТЫ (ВЫСШИЙ ПРИОРИТЕТ - ВЫЖИВАНИЕ)
# # #         # ============================================================
# # #         if self.runaway_target:
# # #             # Проверяем, достигли ли безопасного расстояния
# # #             if self._is_safe(world):
# # #                 self.runaway_target = None
# # #                 print("✅ Безопасно, возврат к исследованию")
# # #             else:
# # #                 self._update_runaway(world)
# # #                 self._has_instinct_action = True
# # #                 return
# # #
# # #         # ============================================================
# # #         # УРОВЕНЬ 2: РЕФЛЕКСЫ (СРЕДНИЙ ПРИОРИТЕТ)
# # #         # ============================================================
# # #         state = world.get_state(self)
# # #         dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
# # #
# # #         # Проверяем все соседние клетки
# # #         for (dx, dz) in dirs:
# # #             check_x = self.x + dx * self.step_size
# # #             check_z = self.z + dz * self.step_size
# # #             obj = world.get_object_at(check_x, check_z)
# # #
# # #             if obj:
# # #                 self.setInform(obj)
# # #                 perception = Perception(self.nearby_params.copy())
# # #
# # #                 # Получаем пороги из генома
# # #                 thresholds = self.genome.get('reflex_thresholds', {})
# # #
# # #                 suggestion = self.reflex_module.get_best_action(
# # #                     perception,
# # #                     thresholds
# # #                 )
# # #
# # #                 if suggestion:
# # #                     # ВЫПОЛНЯЕМ РЕФЛЕКС
# # #                     self.execute_action(suggestion.action_id, world, state)
# # #
# # #                     # Сохраняем результат для обратной связи
# # #                     self._has_reflex_action = True
# # #                     self._record_reflex_outcome(suggestion.action_id, True)
# # #
# # #                     # Сохраняем опыт
# # #                     next_state = world.get_state(self)
# # #                     self.add_experience(state, suggestion.action_id, 1.0, next_state)
# # #                     return  # Рефлекс выполнен, выходим
# # #
# # #         # ============================================================
# # #         # УРОВЕНЬ 3: ИССЛЕДОВАНИЕ (НИЗШИЙ ПРИОРИТЕТ)
# # #         # ============================================================
# # #         self._explore(world)
# # #
# # #         # ============================================================
# # #         # ОБНОВЛЕНИЕ ЭМОЦИОНАЛЬНОЙ СИСТЕМЫ
# # #         # ============================================================
# # #         if hasattr(self, 'emotion_system') and self.emotion_system:
# # #             sensory_data = self._get_sensory_data(world)
# # #             self.emotion_system.update(sensory_data)
# # #
# # #     def _get_sensory_data(self, world) -> Dict:
# # #         """
# # #         Собирает сенсорные данные для эмоциональной системы.
# # #         """
# # #         sensory_data = {
# # #             'vision': self._get_vision(world),
# # #             'sound': self._get_sound(world),
# # #             'smell': self._get_smell(world),
# # #             'position': (self.x, self.z),
# # #             'context': {
# # #                 'nearby_object': self.nearby_params,
# # #                 'visited_count': len(self.visited_nodes),
# # #                 'food_collected': self.food_collected
# # #             }
# # #         }
# # #         return sensory_data
# # #
# # #     def _get_vision(self, world) -> np.ndarray:
# # #         """
# # #         Получает визуальные данные.
# # #         """
# # #         # Простая реализация: смотрим, что рядом
# # #         vision = np.zeros(8)
# # #         if self.nearby_object:
# # #             vision[0] = 1.0
# # #         return vision
# # #
# # #     def _get_sound(self, world) -> np.ndarray:
# # #         """
# # #         Получает звуковые данные.
# # #         """
# # #         sound = np.zeros(8)
# # #         # Простая реализация
# # #         return sound
# # #
# # #     def _get_smell(self, world) -> np.ndarray:
# # #         """
# # #         Получает данные о запахах.
# # #         """
# # #         smell = np.zeros(8)
# # #         if self.nearby_params:
# # #             if 'smell' in self.nearby_params:
# # #                 smell[0] = 1.0
# # #         return smell
# # #
# # #     def _is_safe(self, world) -> bool:
# # #         """
# # #         Проверяет, безопасно ли текущее местоположение.
# # #         """
# # #         # Проверяем, нет ли рядом опасных объектов
# # #         dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
# # #         for (dx, dz) in dirs:
# # #             check_x = self.x + dx * self.step_size * 3
# # #             check_z = self.z + dz * self.step_size * 3
# # #             obj = world.get_object_at(check_x, check_z)
# # #             if obj and hasattr(obj, 'danger_level') and obj.danger_level > 0.5:
# # #                 return False
# # #         return True
# # #
# # #     def _record_reflex_outcome(self, action_id: str, success: bool):
# # #         """
# # #         Записывает результат рефлекса для обратной связи.
# # #         """
# # #         if action_id not in self.reflex_stats:
# # #             self.reflex_stats[action_id] = {'success': 0, 'total': 0}
# # #         self.reflex_stats[action_id]['total'] += 1
# # #         if success:
# # #             self.reflex_stats[action_id]['success'] += 1
# # #
# # #     def _record_instinct_outcome(self, pattern_id: str, success: bool):
# # #         """
# # #         Записывает результат инстинкта для обратной связи.
# # #         """
# # #         if pattern_id not in self.instinct_stats:
# # #             self.instinct_stats[pattern_id] = {'success': 0, 'total': 0}
# # #         self.instinct_stats[pattern_id]['total'] += 1
# # #         if success:
# # #             self.instinct_stats[pattern_id]['success'] += 1
# # #
# # #     def get_reflex_success_rate(self, action_id: str) -> float:
# # #         """
# # #         Возвращает процент успешности рефлекса.
# # #         """
# # #         stats = self.reflex_stats.get(action_id, {'success': 0, 'total': 0})
# # #         if stats['total'] == 0:
# # #             return 0.0
# # #         return stats['success'] / stats['total']
# # #
# # #     def _explore(self, world):
# # #         """
# # #         УРОВЕНЬ 3: ИССЛЕДОВАНИЕ (низший приоритет).
# # #         Выполняется, если нет активных инстинктов и рефлексов.
# # #         """
# # #         # Получаем состояние до действия
# # #         state = world.get_state(self)
# # #
# # #         dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
# # #         candidates = []
# # #         fallback = []
# # #
# # #         for (dx, dz) in dirs:
# # #             next_x = self.x + dx * self.step_size
# # #             next_z = self.z + dz * self.step_size
# # #
# # #             # Проверяем границы
# # #             if not world.is_within_world(next_x, next_z):
# # #                 continue
# # #
# # #             # Проверяем, не занято ли объектом
# # #             if world.get_object_at(next_x, next_z) is not None:
# # #                 continue
# # #
# # #             node1 = (self.x, self.z)
# # #             node2 = (next_x, next_z)
# # #             if node1 > node2:
# # #                 node1, node2 = node2, node1
# # #
# # #             # Предпочитаем непосещённые узлы
# # #             if (node1, node2) not in self.visited_edges_set:
# # #                 candidates.append((dx, dz))
# # #             else:
# # #                 fallback.append((dx, dz))
# # #
# # #         # Выбираем направление
# # #         if candidates:
# # #             dx, dz = random.choice(candidates)
# # #             reward_step = 1.0  # Новый узел
# # #             action_idx = dirs.index((dx, dz))
# # #         elif fallback:
# # #             dx, dz = random.choice(fallback)
# # #             reward_step = -0.1  # Штраф за повтор
# # #             action_idx = dirs.index((dx, dz))
# # #         else:
# # #             # Нет доступных направлений - бот застрял
# # #             self.alive = False
# # #             reward_step = -1.0
# # #             next_state = world.get_state(self)
# # #             self.add_experience(state, 0, reward_step, next_state)
# # #             self.total_reward += reward_step
# # #             return
# # #
# # #         # Делаем шаг
# # #         next_x = self.x + dx * self.step_size
# # #         next_z = self.z + dz * self.step_size
# # #         self.visited_edges.append(((self.x, self.z), (next_x, next_z)))
# # #         self._add_edge((self.x, self.z), (next_x, next_z))
# # #         self.x, self.z = next_x, next_z
# # #         self.visited_nodes.append((self.x, self.z))
# # #
# # #         # Получаем новое состояние
# # #         next_state = world.get_state(self)
# # #
# # #         # Добавляем переход в буфер
# # #         self.add_experience(state, action_idx, reward_step, next_state)
# # #         self.total_reward += reward_step
# # #
# # #         # Проверяем объекты в соседних клетках (рефлексы)
# # #         for (dx_check, dz_check) in dirs:
# # #             check_x = self.x + dx_check * self.step_size
# # #             check_z = self.z + dz_check * self.step_size
# # #             obj = world.get_object_at(check_x, check_z)
# # #             if obj:
# # #                 self.setInform(obj)
# # #                 perception = Perception(self.nearby_params.copy())
# # #                 thresholds = self.genome.get('reflex_thresholds', {})
# # #                 suggestion = self.reflex_module.get_best_action(perception, thresholds)
# # #                 if suggestion:
# # #                     self.execute_action(suggestion.action_id, world, state)
# # #                     self._has_reflex_action = True
# # #                 break
# # #
# # #         # Проверяем лимит шагов
# # #         if len(self.visited_nodes) > self.max_steps:
# # #             self.alive = False
# # #             print("Достигнут лимит шагов")
# # #
# # #
# # #
# # #     # ---------- Фитнес ----------
# # #     def calculate_fitness(self):
# # #         """
# # #         Расчёт фитнеса с учётом разных факторов.
# # #         """
# # #         # 1. Количество посещённых узлов (максимум 500)
# # #         visited_score = len(self.visited_nodes) * 1.0
# # #
# # #         # 2. Бонус за собранную еду
# # #         food_score = self.food_collected * 10.0
# # #
# # #         # 3. Бонус за выживание
# # #         survival_score = 50.0 if self.alive else 0.0
# # #
# # #         # 4. Штраф за повторение узлов (чем меньше повторов, тем лучше)
# # #         unique_nodes = len(set(self.visited_nodes))
# # #         repeat_penalty = (len(self.visited_nodes) - unique_nodes) * 0.5
# # #
# # #         # 5. Бонус за разнообразие посещённых узлов
# # #         diversity_bonus = unique_nodes * 0.1
# # #
# # #         # Итоговый фитнес
# # #         fitness = visited_score + food_score + survival_score - repeat_penalty + diversity_bonus
# # #
# # #         return max(0, fitness)
# # #
# # #
# # #     # ---------- Отрисовка ----------
# # #     def draw_path(self, screen, world_to_screen_func):
# # #         if not self.visited_edges:
# # #             return
# # #         for (x1, z1), (x2, z2) in self.visited_edges:
# # #             p1 = world_to_screen_func(x1, z1, 0.1)
# # #             p2 = world_to_screen_func(x2, z2, 0.1)
# # #             pygame.draw.line(screen, (0, 255, 0), p1, p2, 3)
# # #
# # #     def draw(self, screen, world_to_screen_func, scale):
# # #         self.draw_path(screen, world_to_screen_func)
# # #
# # #         cx, cz = self.x, self.z
# # #         ang_rad = math.radians(self.angle)
# # #
# # #         def rotate_point(lx, lz):
# # #             rx = lx * math.cos(ang_rad) - lz * math.sin(ang_rad)
# # #             rz = lx * math.sin(ang_rad) + lz * math.cos(ang_rad)
# # #             return cx + rx, cz + rz
# # #
# # #         corners_local = [
# # #             (-self.body_w/2, -self.body_d/2),
# # #             (self.body_w/2, -self.body_d/2),
# # #             (self.body_w/2, self.body_d/2),
# # #             (-self.body_w/2, self.body_d/2)
# # #         ]
# # #         corners_world = [rotate_point(lx, lz) for (lx, lz) in corners_local]
# # #         base_points = [world_to_screen_func(wx, wz, 0) for (wx, wz) in corners_world]
# # #         top_points = [world_to_screen_func(wx, wz, self.body_h) for (wx, wz) in corners_world]
# # #
# # #         pygame.draw.polygon(screen, (0, 150, 200), top_points)
# # #         for i in range(4):
# # #             j = (i+1) % 4
# # #             pts = [base_points[i], base_points[j], top_points[j], top_points[i]]
# # #             pygame.draw.polygon(screen, (0, 100, 150), pts)
# # #         pygame.draw.polygon(screen, (0, 80, 120), base_points)
# # #
# # #         head_offset_z = 0.2
# # #         head_x, head_z = rotate_point(0, head_offset_z)
# # #         head_y = self.body_h + self.head_r * 0.8
# # #         head_screen = world_to_screen_func(head_x, head_z, head_y)
# # #         rad_px = int(self.head_r * scale)
# # #         if rad_px > 1:
# # #             pygame.draw.circle(screen, (255, 200, 150), head_screen, rad_px)
# # #
# # #     def reset(self, start_x=0, start_z=0):
# # #         """
# # #         Сбрасывает состояние бота для нового эпизода.
# # #         """
# # #         self.x = start_x
# # #         self.z = start_z
# # #         self.visited_nodes = [(self.x, self.z)]
# # #         self.visited_edges = []
# # #         self.visited_edges_set = set()
# # #         self.frame_counter = 0
# # #         self.runaway_target = None
# # #         self.awaiting_steps = 0
# # #         self.moving = True
# # #         self.alive = True
# # #         self.nearby_object = None
# # #         self.nearby_params = {}
# # #         self.fitness = 0.0
# # #         self.dir_index = 0
# # #
# # #     def evaluate(self, world, max_steps=500):
# # #         """
# # #         Запускает бота в мире на max_steps шагов и вычисляет фитнес.
# # #         """
# # #         self.reset()
# # #         for _ in range(max_steps):
# # #             if not self.alive:
# # #                 break
# # #             self.update(world)
# # #         self.fitness = self.calculate_fitness()
# # #         return self.fitness
# # #
# # #     def calculate_fitness(self):
# # #         # Базовая: количество посещённых узлов (чем больше, тем лучше)
# # #         return len(self.visited_nodes) * 1.0
# # #
# # #     def add_experience(self, state, action, reward, next_state):
# # #         """
# # #         Добавляет переход в буфер памяти. Автоматически удаляет старые при переполнении.
# # #         """
# # #         self.memory_buffer.append((state, action, reward, next_state))
# # #
# # #         # Если буфер слишком большой, удаляем старые записи
# # #         if len(self.memory_buffer) > self.max_buffer_size:
# # #             self.memory_buffer.pop(0)
# # #
# # #     def get_experiences(self):
# # #         """
# # #         Возвращает список всех переходов и очищает буфер."
# # #         """
# # #         data = list(self.memory_buffer)
# # #         self.memory_buffer.clear()
# # #         return data
# # #
