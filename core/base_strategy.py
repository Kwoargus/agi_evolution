# core/base_strategy.py
from typing import Dict, Any, Optional

class Perception(Dict[str, Any]):
    """
    [ru] Словарь с сенсорными данными.
    [en] Dictionary with sensory data.
    """
    pass

class ActionSuggestion:
    def __init__(self, action_id: str, priority: float = 1.0, params: Optional[Dict] = None):
        self.action_id = action_id
        self.priority = priority
        self.params = params or {}