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

class Individual:
    def __init__(self, x=0, z=0, angle=0, move_delay=5, reflex_rules=None, instinct_patterns=None, genome=None,  max_buffer_size=10000):
        self.x = x
        self.z = z
        self.angle = angle
        self.body_w = 0.8
        self.body_d = 0.5
        self.body_h = 1.2
        self.head_r = 0.3

        # Если передан genome, используем его, иначе создаём из параметров
        if genome is None:
            genome = Genome({
                'move_delay': move_delay,
                'step_size': 2.0,
                'reflex_rules': reflex_rules if reflex_rules else [],
                'instinct_patterns': instinct_patterns if instinct_patterns else [],
                'max_steps': 500
            })
        self.genome = genome

        self.fitness = 0.0
        self.alive = True
        # self.step_size = genome.get('step_size', 2.0)
        self.step_size = 2.0  # всегда 2, игнорируем геном
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.dir_index = 0

        # Переменные для движения и сбора данных
        self.visited_nodes = [(self.x, self.z)]
        self.visited_edges = []
        self.visited_edges_set = set()
        self.frame_counter = 0
        self.move_delay = genome.get('move_delay', 5)
        self.max_steps = genome.get('max_steps', 1000)

        # Модули рефлексов и инстинктов
        self.reflex_module = ReflexModule(
            reflex_rules if reflex_rules is not None else genome.get('reflex_rules', [])
        )
        self.instinct_module = InstinctModule(
            instinct_patterns if instinct_patterns is not None else genome.get('instinct_patterns', [])
        )

        self.nearby_object = None
        self.nearby_params = {}

        # Состояние убегания (для взрыва и т.п.)
        self.runaway_target = None
        self.awaiting_steps = 0
        self.moving = True

        self.max_buffer_size = max_buffer_size
        self.memory_buffer = deque(maxlen=self.max_buffer_size)  # список переходов (state, action, reward, next_state)

        # self.max_buffer_size = 10000

        self.runaway_attempts = 0

        self.food_collected = 0  # счётчик собранной еды
        self.total_reward = 0.0  # суммарная награда за эпизод

        # НОВОЕ: счётчики для обратной связи
        self.reflex_stats = {}  # {action_id: {'success': 0, 'total': 0}}
        self.instinct_stats = {}  # {pattern_id: {'success': 0, 'total': 0}}

        # НОВОЕ: флаги для приоритетов
        self._has_reflex_action = False
        self._has_instinct_action = False

    # ---------- Взаимодействие с объектами ----------
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

    # ---------- Действия от рефлексов ----------
    def execute_action(self, action, world, state):
        if action == 'move_on':
            print("Рефлекс: move_on! Разворот и уход на 2 клетки.")
            self.dir_index = (self.dir_index + 2) % 4
            for _ in range(2):
                if not self._move_one_step(world):
                    break
        elif action == 'grab':
            print("Рефлекс: grab! Захват еды.")
            self._grab_object(world, state)  # передаём state
        elif action == 'avoid':
            print("Рефлекс: avoid! Отворачиваем.")
            self._turn_right()
        else:
            print(f"Неизвестное действие: {action}")

        print(f"[Action] executing {action}")


    def _grab_object(self, world, state):
        if self.nearby_object is None:
            return False
        target_x = self.nearby_object.x
        target_z = self.nearby_object.z
        dx = target_x - self.x
        dz = target_z - self.z
        if abs(dx) > self.step_size or abs(dz) > self.step_size:
            print("Объект не в соседней клетке")
            return False
        # Удаляем объект до перемещения, чтобы сохранить правильное next_state
        world.remove_object(self.nearby_object)
        # Сохраняем старое состояние для next_state (после удаления объекта)
        # Но next_state должно быть после перемещения, поэтому сначала перемещаемся
        self.visited_edges.append(((self.x, self.z), (target_x, target_z)))
        self._add_edge((self.x, self.z), (target_x, target_z))
        self.x, self.z = target_x, target_z
        self.visited_nodes.append((self.x, self.z))
        self.food_collected += 1
        # Награда за захват еды
        reward = 10.0
        # Получаем новое состояние
        next_state = world.get_state(self)
        # Добавляем опыт с action_id=4 (специальный код для "grab")
        self.add_experience(state, 4, reward, next_state)
        self.total_reward += reward
        print(
            f"Bot схватил {self.nearby_object.name if hasattr(self.nearby_object, 'name') else 'еду'} и переместился в ({self.x}, {self.z})")
        self.nearby_object = None
        self.nearby_params = {}
        return True


    # ---------- Инстинкты (взрыв) ----------
    def notify(self, event_type, data):
        print(f"notify called: event_type={event_type}, data={data}")
        if event_type == 'explosion':
            perception = Perception({
                'sound': data.get('sound'),
                'vision': data.get('vision'),
                'position': data.get('position')
            })
            print(f"Perception: {perception}")
            suggestion = self.instinct_module.get_best_action(perception)
            print(f"Suggestion: {suggestion}")
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
            print(f"Убегаем в направлении {dir_vec}")

    def _update_runaway(self, world):
        dx, dz = self.runaway_target
        next_x = self.x + dx * self.step_size
        next_z = self.z + dz * self.step_size

        # Если следующий шаг выходит за границы мира – сразу останавливаемся
        if not world.is_within_world(next_x, next_z):
            if self.awaiting_steps == 0:
                self.awaiting_steps = 200
                print(f"Достигнута граница мира, остановка убегания на {self.awaiting_steps} шагов")
            else:
                self.awaiting_steps -= 1
                if self.awaiting_steps <= 0:
                    self.runaway_target = None
                    self.moving = True
                    self.frame_counter = 0
                    print("Возврат к обходу")
            return

        # Проверяем, можно ли двинуться прямо (в пределах мира и нет объекта)
        if world.get_object_at(next_x, next_z) is None:
            # Успешный шаг – сбрасываем счётчик неудач
            self.runaway_attempts = 0
            self.visited_edges.append(((self.x, self.z), (next_x, next_z)))
            self._add_edge((self.x, self.z), (next_x, next_z))
            self.x, self.z = next_x, next_z
            self.visited_nodes.append((self.x, self.z))
            self.awaiting_steps = 0
            return

        # Если прямо нельзя (в пределах мира, но занято объектом) – увеличиваем счётчик неудач
        self.runaway_attempts += 1

        # Если уже много неудачных попыток подряд – прекращаем попытки и ждём
        if self.runaway_attempts > 5:
            if self.awaiting_steps == 0:
                self.awaiting_steps = 200
                print(f"Достигнут предел попыток, остановка убегания на {self.awaiting_steps} шагов")
            else:
                self.awaiting_steps -= 1
                if self.awaiting_steps <= 0:
                    self.runaway_target = None
                    self.moving = True
                    self.frame_counter = 0
                    print("Возврат к обходу")
            return

        # Пробуем повернуть влево или вправо
        left = (dz, -dx)
        right = (-dz, dx)
        for (ndx, ndz) in [left, right]:
            nnext_x = self.x + ndx * self.step_size
            nnext_z = self.z + ndz * self.step_size
            # Проверяем, что поворот ведёт в пределах мира и не занят объектом
            if world.is_within_world(nnext_x, nnext_z) and world.get_object_at(nnext_x, nnext_z) is None:
                # Обновляем направление, сбрасываем счётчик неудач
                self.runaway_target = (ndx, ndz)
                self.runaway_attempts = 0
                self.visited_edges.append(((self.x, self.z), (nnext_x, nnext_z)))
                self._add_edge((self.x, self.z), (nnext_x, nnext_z))
                self.x, self.z = nnext_x, nnext_z
                self.visited_nodes.append((self.x, self.z))
                self.awaiting_steps = 0
                return

        # Если ни один поворот не сработал (например, везде объекты) – ждём
        if self.awaiting_steps == 0:
            self.awaiting_steps = 200
            print(f"Нет доступных направлений, остановка убегания на {self.awaiting_steps} шагов")
        else:
            self.awaiting_steps -= 1
            if self.awaiting_steps <= 0:
                self.runaway_target = None
                self.moving = True
                self.frame_counter = 0
                print("Возврат к обходу")

    def update(self, world):
        """ОСНОВНОЙ ЦИКЛ с правильным порядком и приоритетами."""

        # ============================================================
        # УРОВЕНЬ 1: ИНСТИНКТЫ (ВЫСШИЙ ПРИОРИТЕТ - ВЫЖИВАНИЕ)
        # ============================================================
        if self.runaway_target:
            # Проверяем, достигли ли безопасного расстояния
            if self._is_safe(world):
                self.runaway_target = None
                print("✅ Безопасно, возврат к исследованию")
            else:
                self._update_runaway(world)
                self._has_instinct_action = True
                return

        # ============================================================
        # УРОВЕНЬ 2: РЕФЛЕКСЫ (СРЕДНИЙ ПРИОРИТЕТ)
        # ============================================================
        state = world.get_state(self)
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # Проверяем все соседние клетки
        for (dx, dz) in dirs:
            check_x = self.x + dx * self.step_size
            check_z = self.z + dz * self.step_size
            obj = world.get_object_at(check_x, check_z)

            if obj:
                self.setInform(obj)
                perception = Perception(self.nearby_params.copy())

                # Получаем пороги из генома
                thresholds = self.genome.get('reflex_thresholds', {})

                suggestion = self.reflex_module.get_best_action(
                    perception,
                    thresholds
                )

                if suggestion:
                    # ВЫПОЛНЯЕМ РЕФЛЕКС
                    self.execute_action(suggestion.action_id, world, state)

                    # Сохраняем результат для обратной связи
                    self._has_reflex_action = True
                    self._record_reflex_outcome(suggestion.action_id, True)

                    # Сохраняем опыт
                    next_state = world.get_state(self)
                    self.add_experience(state, suggestion.action_id, 1.0, next_state)
                    return  # Рефлекс выполнен, выходим

        # ============================================================
        # УРОВЕНЬ 3: ИССЛЕДОВАНИЕ (НИЗШИЙ ПРИОРИТЕТ)
        # ============================================================
        self._explore(world)

        # ============================================================
        # ОБНОВЛЕНИЕ ЭМОЦИОНАЛЬНОЙ СИСТЕМЫ
        # ============================================================
        if hasattr(self, 'emotion_system') and self.emotion_system:
            sensory_data = self._get_sensory_data(world)
            self.emotion_system.update(sensory_data)

    def _get_sensory_data(self, world) -> Dict:
        """Собирает сенсорные данные для эмоциональной системы."""
        sensory_data = {
            'vision': self._get_vision(world),
            'sound': self._get_sound(world),
            'smell': self._get_smell(world),
            'position': (self.x, self.z),
            'context': {
                'nearby_object': self.nearby_params,
                'visited_count': len(self.visited_nodes),
                'food_collected': self.food_collected
            }
        }
        return sensory_data

    def _get_vision(self, world) -> np.ndarray:
        """Получает визуальные данные."""
        # Простая реализация: смотрим, что рядом
        vision = np.zeros(8)
        if self.nearby_object:
            vision[0] = 1.0
        return vision

    def _get_sound(self, world) -> np.ndarray:
        """Получает звуковые данные."""
        sound = np.zeros(8)
        # Простая реализация
        return sound

    def _get_smell(self, world) -> np.ndarray:
        """Получает данные о запахах."""
        smell = np.zeros(8)
        if self.nearby_params:
            if 'smell' in self.nearby_params:
                smell[0] = 1.0
        return smell

    def _is_safe(self, world) -> bool:
        """Проверяет, безопасно ли当前位置."""
        # Проверяем, нет ли рядом опасных объектов
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for (dx, dz) in dirs:
            check_x = self.x + dx * self.step_size * 3
            check_z = self.z + dz * self.step_size * 3
            obj = world.get_object_at(check_x, check_z)
            if obj and hasattr(obj, 'danger_level') and obj.danger_level > 0.5:
                return False
        return True

    def _record_reflex_outcome(self, action_id: str, success: bool):
        """Записывает результат рефлекса для обратной связи."""
        if action_id not in self.reflex_stats:
            self.reflex_stats[action_id] = {'success': 0, 'total': 0}
        self.reflex_stats[action_id]['total'] += 1
        if success:
            self.reflex_stats[action_id]['success'] += 1

    def _record_instinct_outcome(self, pattern_id: str, success: bool):
        """Записывает результат инстинкта для обратной связи."""
        if pattern_id not in self.instinct_stats:
            self.instinct_stats[pattern_id] = {'success': 0, 'total': 0}
        self.instinct_stats[pattern_id]['total'] += 1
        if success:
            self.instinct_stats[pattern_id]['success'] += 1

    def get_reflex_success_rate(self, action_id: str) -> float:
        """Возвращает процент успешности рефлекса."""
        stats = self.reflex_stats.get(action_id, {'success': 0, 'total': 0})
        if stats['total'] == 0:
            return 0.0
        return stats['success'] / stats['total']

    def _explore(self, world):
        """
        УРОВЕНЬ 3: ИССЛЕДОВАНИЕ (низший приоритет).
        Выполняется, если нет активных инстинктов и рефлексов.
        """
        # Получаем состояние до действия
        state = world.get_state(self)

        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        candidates = []
        fallback = []

        for (dx, dz) in dirs:
            next_x = self.x + dx * self.step_size
            next_z = self.z + dz * self.step_size

            # Проверяем границы
            if not world.is_within_world(next_x, next_z):
                continue

            # Проверяем, не занято ли объектом
            if world.get_object_at(next_x, next_z) is not None:
                continue

            node1 = (self.x, self.z)
            node2 = (next_x, next_z)
            if node1 > node2:
                node1, node2 = node2, node1

            # Предпочитаем непосещённые узлы
            if (node1, node2) not in self.visited_edges_set:
                candidates.append((dx, dz))
            else:
                fallback.append((dx, dz))

        # Выбираем направление
        if candidates:
            dx, dz = random.choice(candidates)
            reward_step = 1.0  # Новый узел
            action_idx = dirs.index((dx, dz))
        elif fallback:
            dx, dz = random.choice(fallback)
            reward_step = -0.1  # Штраф за повтор
            action_idx = dirs.index((dx, dz))
        else:
            # Нет доступных направлений - бот застрял
            self.alive = False
            reward_step = -1.0
            next_state = world.get_state(self)
            self.add_experience(state, 0, reward_step, next_state)
            self.total_reward += reward_step
            return

        # Делаем шаг
        next_x = self.x + dx * self.step_size
        next_z = self.z + dz * self.step_size
        self.visited_edges.append(((self.x, self.z), (next_x, next_z)))
        self._add_edge((self.x, self.z), (next_x, next_z))
        self.x, self.z = next_x, next_z
        self.visited_nodes.append((self.x, self.z))

        # Получаем новое состояние
        next_state = world.get_state(self)

        # Добавляем переход в буфер
        self.add_experience(state, action_idx, reward_step, next_state)
        self.total_reward += reward_step

        # Проверяем объекты в соседних клетках (рефлексы)
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

        # Проверяем лимит шагов
        if len(self.visited_nodes) > self.max_steps:
            self.alive = False
            print("Достигнут лимит шагов")

    # ---------- Основной цикл обновления ----------
    # def update(self, world):
    #     if self.runaway_target:
    #         self._update_runaway(world)
    #         return
    #
    #     if not self.alive:
    #         return
    #
    #     self.frame_counter += 1
    #     if self.frame_counter < self.move_delay:
    #         return
    #     self.frame_counter = 0
    #
    #     # Получаем состояние до действия
    #     state = world.get_state(self)
    #
    #     dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    #     candidates = []
    #     fallback = []
    #     for (dx, dz) in dirs:
    #         next_x = self.x + dx * self.step_size
    #         next_z = self.z + dz * self.step_size
    #         if not world.is_within_world(next_x, next_z):
    #             continue
    #         if world.get_object_at(next_x, next_z) is not None:
    #             continue
    #         node1 = (self.x, self.z)
    #         node2 = (next_x, next_z)
    #         if node1 > node2:
    #             node1, node2 = node2, node1
    #         if (node1, node2) not in self.visited_edges_set:
    #             candidates.append((dx, dz))
    #         else:
    #             fallback.append((dx, dz))
    #
    #     if candidates:
    #         dx, dz = random.choice(candidates)
    #         reward_step = 1.0  # новый узел
    #         action_idx = dirs.index((dx, dz))  # 0-3
    #     elif fallback:
    #         dx, dz = random.choice(fallback)
    #         reward_step = -0.1  # штраф за повтор
    #         action_idx = dirs.index((dx, dz))
    #     else:
    #         self.alive = False
    #         reward_step = -1.0
    #         # Добавляем опыт с отрицательной наградой (действие 0 = стоять на месте)
    #         next_state = world.get_state(self)
    #         self.add_experience(state, 0, reward_step, next_state)
    #         self.total_reward += reward_step
    #         return
    #
    #     # Делаем шаг
    #     next_x = self.x + dx * self.step_size
    #     next_z = self.z + dz * self.step_size
    #     self.visited_edges.append(((self.x, self.z), (next_x, next_z)))
    #     self._add_edge((self.x, self.z), (next_x, next_z))
    #     self.x, self.z = next_x, next_z
    #     self.visited_nodes.append((self.x, self.z))
    #
    #     # Получаем новое состояние
    #     next_state = world.get_state(self)
    #
    #     # Добавляем переход в буфер
    #     self.add_experience(state, action_idx, reward_step, next_state)
    #     self.total_reward += reward_step
    #
    #     # Проверяем объекты в соседних клетках (рефлексы)
    #     for (dx_check, dz_check) in dirs:
    #         check_x = self.x + dx_check * self.step_size
    #         check_z = self.z + dz_check * self.step_size
    #         obj = world.get_object_at(check_x, check_z)
    #         if obj:
    #             self.setInform(obj)
    #             perception = Perception(self.nearby_params.copy())
    #             suggestion = self.reflex_module.get_best_action(perception)
    #             if suggestion:
    #                 # Передаём состояние до выполнения рефлекса (оно уже не актуально, но для grab используем текущее)
    #                 # Для простоты передадим текущее состояние (после шага) – но для grab важно, чтобы объект ещё был, но мы уже переместились.
    #                 # Лучше передать состояние до шага, но после шага состояние другое. Мы можем сохранить state до шага,
    #                 # но для рефлекса grab нам нужно состояние до захвата. Поэтому передадим state (которое было до шага) – но оно уже не соответствует текущему.
    #                 # Более правильно: вызов _grab_object использует текущее состояние, и мы передадим в него state (которое было до шага).
    #                 # Однако после шага мы уже переместились, и объект может быть не рядом. Поэтому логика должна быть пересмотрена.
    #                 # Упростим: будем вызывать execute_action с состоянием до шага, но если рефлекс grab, то он должен срабатывать до перемещения?
    #                 # Текущая логика: сначала шаг, потом проверка объектов. Это неправильно, потому что объект должен быть обнаружен до шага.
    #                 # Рекомендую перестроить логику: сначала проверять объекты, потом делать шаг.
    #                 # Но чтобы не переписывать всё сейчас, оставим как есть, и для grab будем использовать состояние до шага,
    #                 # но тогда нужно вызывать _grab_object до перемещения. Я изменю порядок: сначала проверка объектов, потом шаг.
    #                 # Однако это потребует значительных изменений. Предложу упрощённый вариант: пока собираем переходы только для шагов,
    #                 # а для grab добавим отдельный переход в _grab_object (мы уже это сделали).
    #                 # Поэтому в этом месте не будем добавлять опыт для рефлексов, только для шагов.
    #                 self.execute_action(suggestion.action_id, world, state)  # передаём state
    #             break
    #
    #             print(f"Object detected: {self.nearby_params}, perception: {perception}")
    #
    #     if len(self.visited_nodes) > self.max_steps:
    #         self.alive = False
    #         print("Достигнут лимит шагов")

    # ---------- Фитнес ----------
    def calculate_fitness(self):
        """
        Расчёт фитнеса с учётом разных факторов.
        """
        # 1. Количество посещённых узлов (максимум 500)
        visited_score = len(self.visited_nodes) * 1.0

        # 2. Бонус за собранную еду
        food_score = self.food_collected * 10.0

        # 3. Бонус за выживание
        survival_score = 50.0 if self.alive else 0.0

        # 4. Штраф за повторение узлов (чем меньше повторов, тем лучше)
        unique_nodes = len(set(self.visited_nodes))
        repeat_penalty = (len(self.visited_nodes) - unique_nodes) * 0.5

        # 5. Бонус за разнообразие посещённых узлов
        diversity_bonus = unique_nodes * 0.1

        # Итоговый фитнес
        fitness = visited_score + food_score + survival_score - repeat_penalty + diversity_bonus

        return max(0, fitness)


    # def calculate_fitness(self):
    #     # Компоненты:
    #     visited_score = len(self.visited_nodes) * 1.0  # покрытие территории
    #     food_score = self.food_collected * 10.0  # если добавить счётчик собранной еды
    #     survival_score = 1.0 if self.alive else 0.0  # бонус за выживание
    #     # Можно добавить штраф за столкновения с хищниками
    #     return visited_score + food_score + survival_score

    # ---------- Отрисовка ----------
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
            (-self.body_w/2, -self.body_d/2),
            (self.body_w/2, -self.body_d/2),
            (self.body_w/2, self.body_d/2),
            (-self.body_w/2, self.body_d/2)
        ]
        corners_world = [rotate_point(lx, lz) for (lx, lz) in corners_local]
        base_points = [world_to_screen_func(wx, wz, 0) for (wx, wz) in corners_world]
        top_points = [world_to_screen_func(wx, wz, self.body_h) for (wx, wz) in corners_world]

        pygame.draw.polygon(screen, (0, 150, 200), top_points)
        for i in range(4):
            j = (i+1) % 4
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

    def reset(self, start_x=0, start_z=0):
        """Сбрасывает состояние бота для нового эпизода."""
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
        """Запускает бота в мире на max_steps шагов и вычисляет фитнес."""
        self.reset()
        for _ in range(max_steps):
            if not self.alive:
                break
            self.update(world)
        self.fitness = self.calculate_fitness()
        return self.fitness

    def calculate_fitness(self):
        # Базовая: количество посещённых узлов (чем больше, тем лучше)
        return len(self.visited_nodes) * 1.0

    def add_experience(self, state, action, reward, next_state):
        """Добавляет переход в буфер памяти. Автоматически удаляет старые при переполнении."""
        self.memory_buffer.append((state, action, reward, next_state))

        # Если буфер слишком большой, удаляем старые записи
        if len(self.memory_buffer) > self.max_buffer_size:
            self.memory_buffer.pop(0)

    def get_experiences(self):
        """Возвращает список всех переходов и очищает буфер."""
        data = list(self.memory_buffer)
        self.memory_buffer.clear()
        return data

