# core/thinking/__init__.py
"""
[ru] Модуль мышления AGI.
[en] Thinking module AGI.
"""

from .understanding import UnderstandingEngine, UnderstandingResult, UnderstandingStatus
from .research import ResearchEngine, ResearchResult
from .models import MentalModel, ModelTemplate

# Экспортируем основные классы
__all__ = [
    'UnderstandingEngine',
    'UnderstandingResult',
    'UnderstandingStatus',
    'ResearchEngine',
    'ResearchResult',
    'MentalModel',
    'ModelTemplate'
]