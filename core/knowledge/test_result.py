# core/knowledge/test_result.py
"""
[ru] Модуль для работы с результатами проверки гипотез.
[en] Module for working with hypothesis verification results.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TestResult:
    """
    [ru] Результат проверки гипотезы в виртуальной среде.
    [ru] Attributes:
        [ru] hypothesis_id: ID проверяемой гипотезы
        [ru] success: Успешна ли проверка
        [ru] score: Оценка успешности (0-1)
        [ru] metrics: Словарь с метриками производительности
        [ru] details: Детали проверки
        [ru] timestamp: Время проверки

    [en] Result of hypothesis verification in a virtual environment.
    [en] Attributes:
        [en] hypothesis_id: ID of the hypothesis being tested
        [en] success: Whether the verification was successful
        [en] score: Success score (0-1)
        [en] metrics: Dictionary with performance metrics
        [en] details: Verification details
        [en] timestamp: Verification time
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
        [ru] Создает результат из симуляции.
        [en] Creates a result from a simulation.
        """
        return cls(
            hypothesis_id=hypothesis_id,
            success=simulation_result.get('success', False),
            score=simulation_result.get('score', 0.0),
            metrics=simulation_result.get('metrics', {}),
            details=simulation_result.get('details', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        [ru] Преобразует в словарь.
        [en] Converts to a dictionary.
        """
        return {
            'hypothesis_id': self.hypothesis_id,
            'success': self.success,
            'score': self.score,
            'metrics': self.metrics,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }


# # core/knowledge/test_result.py
# """
# Модуль для работы с результатами проверки гипотез.
# """
#
# from typing import Dict, Any, Optional
# from dataclasses import dataclass, field
# from datetime import datetime
#
#
# @dataclass
# class TestResult:
#     """
#     Результат проверки гипотезы в виртуальной среде.
#
#     Attributes:
#         hypothesis_id: ID проверяемой гипотезы
#         success: Успешна ли проверка
#         score: Оценка успешности (0-1)
#         metrics: Словарь с метриками производительности
#         details: Детали проверки
#         timestamp: Время проверки
#     """
#
#     hypothesis_id: str
#     success: bool = False
#     score: float = 0.0
#     metrics: Dict[str, float] = field(default_factory=dict)
#     details: Dict[str, Any] = field(default_factory=dict)
#     timestamp: datetime = field(default_factory=datetime.now)
#
#     @classmethod
#     def from_simulation(cls, hypothesis_id: str,
#                         simulation_result: Dict[str, Any]) -> 'TestResult':
#         """
#         Создает результат из симуляции.
#         """
#         return cls(
#             hypothesis_id=hypothesis_id,
#             success=simulation_result.get('success', False),
#             score=simulation_result.get('score', 0.0),
#             metrics=simulation_result.get('metrics', {}),
#             details=simulation_result.get('details', {})
#         )
#
#     def to_dict(self) -> Dict[str, Any]:
#         """
#         Преобразует в словарь.
#         """
#         return {
#             'hypothesis_id': self.hypothesis_id,
#             'success': self.success,
#             'score': self.score,
#             'metrics': self.metrics,
#             'details': self.details,
#             'timestamp': self.timestamp.isoformat()
#         }