# models/instinct_module.py
from __future__ import annotations
from core.base_strategy import Perception, ActionSuggestion
from typing import List, Dict, Optional, Tuple, Any, Union

class InstinctModule:
    def __init__(self, patterns: List[Dict]):
        self.patterns = patterns

    def process(self, perception: Perception, context: Optional[Dict] = None) -> List[ActionSuggestion]:
        print(f"process called with perception={perception}")
        print(f"InstinctModule.process: perception={perception}, patterns={self.patterns}")
        suggestions = []
        for entry in self.patterns:
            pattern = entry.get('pattern', {})
            signals = pattern.get('signals', {})
            match = True
            for key, expected_value in signals.items():
                if key not in perception:
                    match = False
                    break
                actual = str(perception[key]).strip().lower()
                expected = str(expected_value).strip().lower()
                if actual != expected:
                    match = False
                    break
            if match:
                action = entry.get('action', {})
                action_id = action.get('action', 'run_away')
                suggestions.append(ActionSuggestion(action_id=action_id, priority=1.0))
        return suggestions

    def get_best_action(self, perception: Perception) -> Optional[ActionSuggestion]:
        suggestions = self.process(perception)
        if suggestions:
            return max(suggestions, key=lambda s: s.priority)
        return None

    def update(self, reward: float, action_taken: ActionSuggestion) -> None:
        pass