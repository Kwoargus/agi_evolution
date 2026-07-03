# models/reflex_module.py
from __future__ import annotations
from typing import List, Dict, Optional
from core.base_strategy import Perception, ActionSuggestion

class ReflexModule:
    def __init__(self, rules: List[Dict]):
        self.rules = rules

    def process(self, perception: Perception, context: Optional[Dict] = None) -> List[ActionSuggestion]:
        suggestions = []
        for rule in self.rules:
            sense_type = rule.get('sense_type')  # канал: smell, sound, vision, touch
            signal_type = rule.get('signal_type')  # конкретное значение: food_smell, predator_smell
            threshold = rule.get('signal_threshold')
            action = rule.get('action')

            # Если указан канал (sense_type) и он есть в восприятии
            if sense_type and sense_type in perception:
                value = perception[sense_type]
                # Если это строковое значение, сравниваем с signal_type
                if str(value) == str(signal_type):
                    suggestions.append(ActionSuggestion(action_id=action, priority=1.0))
            else:
                # Если sense_type не указан, проверяем все значения в восприятии
                for value in perception.values():
                    if str(value) == str(signal_type):
                        suggestions.append(ActionSuggestion(action_id=action, priority=1.0))
                        break
        return suggestions


    # def process(self, perception: Perception, context: Optional[Dict] = None) -> List[ActionSuggestion]:
    #     suggestions = []
    #     for rule in self.rules:
    #         sense_type = rule.get('sense_type')
    #         signal_type = rule.get('signal_type')
    #         threshold = rule.get('signal_threshold')
    #         action = rule.get('action')
    #
    #         if signal_type in perception:
    #             value = perception[signal_type]
    #             match = False
    #
    #             # Строковые сигналы: сравниваем с sense_type (если указан)
    #             if signal_type in ('type', 'smell', 'sound', 'vision', 'name'):
    #                 if sense_type is not None:
    #                     match = (str(value).strip().lower() == str(sense_type).strip().lower())
    #                 else:
    #                     match = True  # если sense_type не указан, считаем совпадением (не рекомендуется)
    #             else:
    #                 # Числовые сигналы
    #                 try:
    #                     num_value = float(value)
    #                     num_threshold = float(threshold) if threshold is not None else 0.0
    #                     match = num_value > num_threshold
    #                 except (ValueError, TypeError):
    #                     match = False
    #
    #             if match:
    #                 suggestions.append(ActionSuggestion(action_id=action, priority=1.0))
    #     return suggestions

    def get_best_action(self, perception: Perception) -> Optional[ActionSuggestion]:
        suggestions = self.process(perception)
        if suggestions:
            return max(suggestions, key=lambda s: s.priority)
        return None

    def update(self, reward: float, action_taken: ActionSuggestion) -> None:
        pass