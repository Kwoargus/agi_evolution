# models/reflex_module.py
"""
Модуль рефлексов - быстрые реакции на стимулы.
"""

from typing import Dict, Any, Optional, List
from core.base_strategy import Perception, ActionSuggestion


class ReflexModule:
    """
    Модуль рефлексов бота.

    Рефлексы - это быстрые, автоматические реакции на внешние стимулы.
    Они имеют приоритет ниже инстинктов, но выше исследования.
    """

    def __init__(self, rules: Optional[List[Dict]] = None):
        """
        Инициализирует модуль рефлексов.

        Args:
            rules: Список правил рефлексов. Каждое правило - словарь с полями:
                - sense_type: тип сенсора ('smell', 'sound', 'vision', 'touch')
                - signal_type: тип сигнала ('food_smell', 'predator_smell', и т.д.)
                - signal_threshold: порог срабатывания (0.0 - 1.0)
                - action: действие ('grab', 'move_on', 'avoid')
                - priority: приоритет (чем выше, тем важнее)
        """
        self.rules = rules or []
        self._boost_factors: Dict[str, float] = {}  # {action_id: boost_multiplier}
        self._suppress_factors: Dict[str, float] = {}  # {action_id: suppress_multiplier}

    def get_best_action(self, perception: Perception, thresholds: Optional[Dict[str, float]] = None) -> Optional[ActionSuggestion]:
        """
        Возвращает лучшее действие на основе восприятия.

        Args:
            perception: Словарь с сенсорными данными
            thresholds: Пороги из генома {'food_smell': 0.3, ...}

        Returns:
            ActionSuggestion или None, если нет подходящего рефлекса
        """
        thresholds = thresholds or {}

        # Получаем boost-факторы
        boosts = self._boost_factors

        # Сортируем правила по приоритету (чем выше, тем важнее)
        sorted_rules = sorted(
            self.rules,
            key=lambda r: r.get('priority', 0),
            reverse=True
        )

        for rule in sorted_rules:
            sense_type = rule.get('sense_type')
            signal_type = rule.get('signal_type')

            # Получаем значение восприятия
            value = perception.get(sense_type, 0)
            if signal_type:
                value = perception.get(signal_type, value)

            # Если значение - строка, сравниваем по точному совпадению
            if isinstance(value, str):
                if signal_type and value == signal_type:
                    # Строковое совпадение
                    pass
                else:
                    continue
            else:
                # Числовое значение - проверяем порог
                # Получаем порог (сначала из thresholds, потом из rule)
                threshold = thresholds.get(sense_type, rule.get('signal_threshold', 0.5))

                # Применяем boost
                action = rule.get('action')
                if action in boosts:
                    threshold *= (1.0 / boosts[action])

                # Применяем подавление
                if action in self._suppress_factors:
                    threshold *= (1.0 / self._suppress_factors[action])

                # Проверяем
                if value <= threshold:
                    continue

            # Нашли подходящее правило
            action = rule.get('action')
            if action:
                priority = rule.get('priority', 1.0)
                if action in boosts:
                    priority *= boosts[action]
                if action in self._suppress_factors:
                    priority *= self._suppress_factors[action]

                return ActionSuggestion(
                    action_id=action,
                    priority=priority,
                    params=rule.get('params', {})
                )

        return None

    def add_rule(self, rule: Dict) -> None:
        """Добавляет новое правило рефлекса."""
        self.rules.append(rule)

    def remove_rule(self, rule_id: str) -> bool:
        """Удаляет правило по ID."""
        for i, rule in enumerate(self.rules):
            if rule.get('id') == rule_id:
                self.rules.pop(i)
                return True
        return False

    def boost_reflex(self, action_id: str, factor: float = 1.5) -> None:
        """
        Усиливает рефлекс (например, под влиянием эмоций).

        Args:
            action_id: ID действия
            factor: Множитель усиления (>1.0 усиливает)
        """
        self._boost_factors[action_id] = factor
        print(f"🔊 Рефлекс {action_id} усилен в {factor}x")

    def suppress_reflex(self, action_id: str, factor: float = 0.5) -> None:
        """
        Подавляет рефлекс.

        Args:
            action_id: ID действия
            factor: Множитель подавления (<1.0 подавляет)
        """
        self._suppress_factors[action_id] = factor
        print(f"🔇 Рефлекс {action_id} подавлен в {factor}x")

    def reset_boosts(self) -> None:
        """Сбрасывает все усиления и подавления."""
        self._boost_factors.clear()
        self._suppress_factors.clear()

    def get_rules(self) -> List[Dict]:
        """Возвращает все правила рефлексов."""
        return self.rules.copy()

    def get_boost_factors(self) -> Dict[str, float]:
        """Возвращает текущие факторы усиления."""
        return self._boost_factors.copy()




# # models/reflex_module.py
# from __future__ import annotations
# from typing import List, Dict, Optional, Tuple, Any, Union
# from core.base_strategy import Perception, ActionSuggestion
#
#
# def update(self, world):
#     """ОСНОВНОЙ ЦИКЛ с правильным порядком и приоритетами."""
#
#     # ============================================================
#     # УРОВЕНЬ 1: ИНСТИНКТЫ (ВЫСШИЙ ПРИОРИТЕТ - ВЫЖИВАНИЕ)
#     # ============================================================
#     if self.runaway_target:
#         # Проверяем, достигли ли безопасного расстояния
#         if self._is_safe(world):
#             self.runaway_target = None
#             print("✅ Безопасно, возврат к исследованию")
#         else:
#             self._update_runaway(world)
#             self._has_instinct_action = True
#             return
#
#     # ============================================================
#     # УРОВЕНЬ 2: РЕФЛЕКСЫ (СРЕДНИЙ ПРИОРИТЕТ)
#     # ============================================================
#     state = world.get_state(self)
#     dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
#
#     # Проверяем все соседние клетки
#     for (dx, dz) in dirs:
#         check_x = self.x + dx * self.step_size
#         check_z = self.z + dz * self.step_size
#         obj = world.get_object_at(check_x, check_z)
#
#         if obj:
#             self.setInform(obj)
#             perception = Perception(self.nearby_params.copy())
#
#             # Получаем пороги из генома
#             thresholds = self.genome.get('reflex_thresholds', {})
#
#             suggestion = self.reflex_module.get_best_action(
#                 perception,
#                 thresholds
#             )
#
#             if suggestion:
#                 # ВЫПОЛНЯЕМ РЕФЛЕКС
#                 self.execute_action(suggestion.action_id, world, state)
#
#                 # Сохраняем результат для обратной связи
#                 self._has_reflex_action = True
#                 self._record_reflex_outcome(suggestion.action_id, True)
#
#                 # Сохраняем опыт
#                 next_state = world.get_state(self)
#                 self.add_experience(state, suggestion.action_id, 1.0, next_state)
#                 return  # Рефлекс выполнен, выходим
#
#     # ============================================================
#     # УРОВЕНЬ 3: ИССЛЕДОВАНИЕ (НИЗШИЙ ПРИОРИТЕТ)
#     # ============================================================
#     self._explore(world)
#
#
# def _is_safe(self, world) -> bool:
#     """Проверяет, безопасно ли当前位置."""
#     # Проверяем, нет ли рядом опасных объектов
#     dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
#     for (dx, dz) in dirs:
#         check_x = self.x + dx * self.step_size * 3
#         check_z = self.z + dz * self.step_size * 3
#         obj = world.get_object_at(check_x, check_z)
#         if obj and hasattr(obj, 'danger_level') and obj.danger_level > 0.5:
#             return False
#     return True
#
#
# def _record_reflex_outcome(self, action_id: str, success: bool):
#     """Записывает результат рефлекса для обратной связи."""
#     if action_id not in self.reflex_stats:
#         self.reflex_stats[action_id] = {'success': 0, 'total': 0}
#     self.reflex_stats[action_id]['total'] += 1
#     if success:
#         self.reflex_stats[action_id]['success'] += 1
#
#
# def _record_instinct_outcome(self, pattern_id: str, success: bool):
#     """Записывает результат инстинкта для обратной связи."""
#     if pattern_id not in self.instinct_stats:
#         self.instinct_stats[pattern_id] = {'success': 0, 'total': 0}
#     self.instinct_stats[pattern_id]['total'] += 1
#     if success:
#         self.instinct_stats[pattern_id]['success'] += 1
#
#
# def get_reflex_success_rate(self, action_id: str) -> float:
#     """Возвращает процент успешности рефлекса."""
#     stats = self.reflex_stats.get(action_id, {'success': 0, 'total': 0})
#     if stats['total'] == 0:
#         return 0.0
#     return stats['success'] / stats['total']
#
#
#
#
#
# # class ReflexModule:
# #     def __init__(self, rules: List[Dict]):
# #         self.rules = rules
# #
# #     def process(self, perception: Perception, context: Optional[Dict] = None) -> List[ActionSuggestion]:
# #         suggestions = []
# #         for rule in self.rules:
# #             sense_type = rule.get('sense_type')  # канал: smell, sound, vision, touch
# #             signal_type = rule.get('signal_type')  # конкретное значение: food_smell, predator_smell
# #             threshold = rule.get('signal_threshold')
# #             action = rule.get('action')
# #
# #             # Если указан канал (sense_type) и он есть в восприятии
# #             if sense_type and sense_type in perception:
# #                 value = perception[sense_type]
# #                 # Если это строковое значение, сравниваем с signal_type
# #                 if str(value) == str(signal_type):
# #                     suggestions.append(ActionSuggestion(action_id=action, priority=1.0))
# #             else:
# #                 # Если sense_type не указан, проверяем все значения в восприятии
# #                 for value in perception.values():
# #                     if str(value) == str(signal_type):
# #                         suggestions.append(ActionSuggestion(action_id=action, priority=1.0))
# #                         break
# #         print(f"[Reflex] perception: {perception}, rules: {self.rules}")
# #         return suggestions
# #
# #
# #     # def process(self, perception: Perception, context: Optional[Dict] = None) -> List[ActionSuggestion]:
# #     #     suggestions = []
# #     #     for rule in self.rules:
# #     #         sense_type = rule.get('sense_type')
# #     #         signal_type = rule.get('signal_type')
# #     #         threshold = rule.get('signal_threshold')
# #     #         action = rule.get('action')
# #     #
# #     #         if signal_type in perception:
# #     #             value = perception[signal_type]
# #     #             match = False
# #     #
# #     #             # Строковые сигналы: сравниваем с sense_type (если указан)
# #     #             if signal_type in ('type', 'smell', 'sound', 'vision', 'name'):
# #     #                 if sense_type is not None:
# #     #                     match = (str(value).strip().lower() == str(sense_type).strip().lower())
# #     #                 else:
# #     #                     match = True  # если sense_type не указан, считаем совпадением (не рекомендуется)
# #     #             else:
# #     #                 # Числовые сигналы
# #     #                 try:
# #     #                     num_value = float(value)
# #     #                     num_threshold = float(threshold) if threshold is not None else 0.0
# #     #                     match = num_value > num_threshold
# #     #                 except (ValueError, TypeError):
# #     #                     match = False
# #     #
# #     #             if match:
# #     #                 suggestions.append(ActionSuggestion(action_id=action, priority=1.0))
# #     #     return suggestions
# #
# #     def get_best_action(self, perception: Perception) -> Optional[ActionSuggestion]:
# #         suggestions = self.process(perception)
# #         if suggestions:
# #             return max(suggestions, key=lambda s: s.priority)
# #         return None
# #
# #     def update(self, reward: float, action_taken: ActionSuggestion) -> None:
# #         pass