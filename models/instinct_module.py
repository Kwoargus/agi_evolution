# models/instinct_module.py
from __future__ import annotations
from core.base_strategy import Perception, ActionSuggestion
from typing import List, Dict, Optional, Tuple, Any, Union


class InstinctModule:
    def __init__(self, patterns=None):
        self.patterns = patterns or []
        self._suppress_factors = {}  # {pattern_id: suppress_multiplier}
        self._success_history = {}  # {pattern_id: {'success': 0, 'total': 0}}

    def get_best_action(self, perception: Perception):
        """
        Возвращает лучшее действие на основе восприятия.
        """
        best_action = None
        best_score = -1.0

        for pattern in self.patterns:
            # Вычисляем схожесть
            similarity = self._compute_similarity(perception, pattern)

            # Применяем подавление
            pattern_id = pattern.get('id', id(pattern))
            suppress = self._suppress_factors.get(pattern_id, 1.0)

            # Учитываем успешность
            success_rate = self._get_success_rate(pattern_id)

            # Итоговый скор
            score = similarity * pattern.get('priority', 1.0) * suppress * (0.5 + 0.5 * success_rate)

            if score > best_score:
                best_score = score
                best_action = pattern.get('action')

        if best_action and best_score > 0.3:
            return ActionSuggestion(best_action, best_score)
        return None

    def _compute_similarity(self, perception: Perception, pattern: dict) -> float:
        """Вычисляет схожесть восприятия с паттерном."""
        # Простейшая реализация
        pattern_signals = pattern.get('signals', {})
        matches = 0
        total = len(pattern_signals)

        for key, expected_value in pattern_signals.items():
            actual_value = perception.get(key, 0)
            if abs(actual_value - expected_value) < 0.2:
                matches += 1

        return matches / total if total > 0 else 0.0

    def _get_success_rate(self, pattern_id: str) -> float:
        """Возвращает процент успешности паттерна."""
        stats = self._success_history.get(pattern_id, {'success': 0, 'total': 0})
        if stats['total'] == 0:
            return 0.5  # Нейтральное значение
        return stats['success'] / stats['total']

    def record_outcome(self, pattern_id: str, success: bool):
        """Записывает результат использования паттерна."""
        if pattern_id not in self._success_history:
            self._success_history[pattern_id] = {'success': 0, 'total': 0}
        self._success_history[pattern_id]['total'] += 1
        if success:
            self._success_history[pattern_id]['success'] += 1

    def suppress_instinct(self, pattern_id: str, factor: float = 0.5):
        """Подавляет инстинкт (например, под влиянием эмоций)."""
        self._suppress_factors[pattern_id] = factor
        print(f"🔇 Инстинкт {pattern_id} подавлен в {factor}x")

    def boost_instinct(self, pattern_id: str, factor: float = 1.5):
        """Усиливает инстинкт."""
        self._suppress_factors[pattern_id] = 1.0 / factor
        print(f"🔊 Инстинкт {pattern_id} усилен в {factor}x")


# class InstinctModule:
#     def __init__(self, patterns: List[Dict]):
#         self.patterns = patterns
#
#     def process(self, perception: Perception, context: Optional[Dict] = None) -> List[ActionSuggestion]:
#         print(f"process called with perception={perception}")
#         print(f"InstinctModule.process: perception={perception}, patterns={self.patterns}")
#         suggestions = []
#         for entry in self.patterns:
#             pattern = entry.get('pattern', {})
#             signals = pattern.get('signals', {})
#             match = True
#             for key, expected_value in signals.items():
#                 if key not in perception:
#                     match = False
#                     break
#                 actual = str(perception[key]).strip().lower()
#                 expected = str(expected_value).strip().lower()
#                 if actual != expected:
#                     match = False
#                     break
#             if match:
#                 action = entry.get('action', {})
#                 action_id = action.get('action', 'run_away')
#                 suggestions.append(ActionSuggestion(action_id=action_id, priority=1.0))
#         return suggestions
#
#     def get_best_action(self, perception: Perception) -> Optional[ActionSuggestion]:
#         suggestions = self.process(perception)
#         if suggestions:
#             return max(suggestions, key=lambda s: s.priority)
#         return None
#
#     def update(self, reward: float, action_taken: ActionSuggestion) -> None:
#         pass