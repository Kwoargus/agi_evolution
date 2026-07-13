# core/knowledge/test_result.py
"""
Модуль для работы с результатами проверки гипотез.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TestResult:
    """
    Результат проверки гипотезы в виртуальной среде.

    Attributes:
        hypothesis_id: ID проверяемой гипотезы
        success: Успешна ли проверка
        score: Оценка успешности (0-1)
        metrics: Словарь с метриками производительности
        details: Детали проверки
        timestamp: Время проверки
    """

    hypothesis_id: str
    success: bool = False
    score: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_simulation(cls, hypothesis_id: str,
                        simulation_result: Dict[str, Any]) -> 'TestResult':
        """
        Создает результат из симуляции.
        """
        return cls(
            hypothesis_id=hypothesis_id,
            success=simulation_result.get('success', False),
            score=simulation_result.get('score', 0.0),
            metrics=simulation_result.get('metrics', {}),
            details=simulation_result.get('details', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь."""
        return {
            'hypothesis_id': self.hypothesis_id,
            'success': self.success,
            'score': self.score,
            'metrics': self.metrics,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }