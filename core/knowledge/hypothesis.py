# core/knowledge/hypothesis.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import time
from enum import Enum
from .combination import Combination


class HypothesisStatus(Enum):
    PROPOSED = "proposed"
    TESTING = "testing"
    VALIDATED = "validated"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"


@dataclass
class Hypothesis:
    id: str
    task_description: str
    source_combination: Combination
    modifications: List[str] = field(default_factory=list)
    description: str = ""
    predicted_score: float = 0.0
    actual_score: float = 0.0
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)  # ← ДОБАВЛЕНО

    def __post_init__(self):
        if self.embedding is None:
            self._compute_embedding()

    def _compute_embedding(self):
        if self.source_combination.embedding is not None:
            dim = len(self.source_combination.embedding)
        else:
            dim = 64
        if self.source_combination.embedding is not None:
            self.embedding = self.source_combination.embedding.copy()
        else:
            self.embedding = np.zeros(dim)
        self.embedding += np.random.randn(dim) * 0.01
        norm = np.linalg.norm(self.embedding) + 1e-8
        self.embedding = self.embedding / norm

    def validate(self, score: float) -> bool:
        self.actual_score = score
        if score >= 0.7:
            self.status = HypothesisStatus.VALIDATED
            return True
        else:
            self.status = HypothesisStatus.REJECTED
            return False

    def add_test_result(self, result: Dict[str, Any]):
        self.test_results.append(result)
        if 'score' in result:
            self.actual_score = result['score']

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'task_description': self.task_description,
            'source_combination_id': self.source_combination.id,
            'modifications': self.modifications,
            'description': self.description,
            'predicted_score': self.predicted_score,
            'actual_score': self.actual_score,
            'status': self.status.value,
            'test_results': self.test_results,
            'metadata': self.metadata,
            'created_at': self.created_at
        }




# # core/knowledge/hypothesis.py
# """
# Модуль для работы с гипотезами.
# """
#
# from typing import List, Dict, Any, Optional
# from dataclasses import dataclass, field
# import numpy as np
# from enum import Enum
# from .combination import Combination
# import time
#
# class HypothesisStatus(Enum):
#     """Статус гипотезы."""
#     PROPOSED = "proposed"  # Предложена
#     TESTING = "testing"  # На стадии проверки
#     VALIDATED = "validated"  # Подтверждена
#     REJECTED = "rejected"  # Отвергнута
#     IN_PROGRESS = "in_progress"  # В разработке
#     created_at: float = field(default_factory=time.time)
#
# @dataclass
# class Hypothesis:
#     """
#     Гипотеза - новое знание, полученное из комбинации моделей.
#
#     Attributes:
#         id: Уникальный идентификатор гипотезы
#         task_description: Описание задачи, для которой создана гипотеза
#         source_combination: Исходная комбинация моделей
#         modifications: Список модификаций, примененных к комбинации
#         description: Описание гипотетического решения
#         predicted_score: Предсказанная оценка соответствия (0-1)
#         actual_score: Фактическая оценка после проверки
#         status: Текущий статус гипотезы
#         test_results: Результаты проверки
#         embedding: Векторное представление гипотезы
#         metadata: Дополнительные метаданные
#     """
#
#     id: str
#     task_description: str
#     source_combination: Combination
#     modifications: List[str] = field(default_factory=list)
#     description: str = ""
#     predicted_score: float = 0.0
#     actual_score: float = 0.0
#     status: HypothesisStatus = HypothesisStatus.PROPOSED
#     test_results: List[Dict[str, Any]] = field(default_factory=list)
#     embedding: Optional[np.ndarray] = None
#     metadata: Dict[str, Any] = field(default_factory=dict)
#     created_at: float = field(default_factory=time.time)
#
#     def __post_init__(self):
#         """Вычисляет эмбеддинг после инициализации."""
#         if self.embedding is None:
#             self._compute_embedding()
#
#     def _compute_embedding(self):
#         """Вычисляет эмбеддинг гипотезы."""
#         # Определяем размерность эмбеддинга
#         if self.source_combination.embedding is not None:
#             dim = len(self.source_combination.embedding)
#         else:
#             dim = 64  # размерность по умолчанию
#
#         # Копируем эмбеддинг комбинации
#         if self.source_combination.embedding is not None:
#             self.embedding = self.source_combination.embedding.copy()
#         else:
#             self.embedding = np.zeros(dim)
#
#         # Добавляем небольшой шум для разнообразия
#         self.embedding += np.random.randn(dim) * 0.01
#         norm = np.linalg.norm(self.embedding) + 1e-8
#         self.embedding = self.embedding / norm
#
#     # def _compute_embedding(self):
#     #     """Вычисляет эмбеддинг гипотезы."""
#     #     # Комбинируем эмбеддинг комбинации и описание задачи
#     #     if self.source_combination.embedding is not None:
#     #         self.embedding = self.source_combination.embedding.copy()
#     #     else:
#     #         self.embedding = np.zeros(128)
#     #
#     #     # Добавляем небольшое случайное смещение для разнообразия
#     #     self.embedding += np.random.randn(128) * 0.01
#     #     norm = np.linalg.norm(self.embedding) + 1e-8
#     #     self.embedding = self.embedding / norm
#
#     def validate(self, score: float) -> bool:
#         """
#         Подтверждает гипотезу.
#
#         Returns:
#             True если гипотеза подтверждена (score >= 0.7)
#         """
#         self.actual_score = score
#         if score >= 0.7:
#             self.status = HypothesisStatus.VALIDATED
#             return True
#         else:
#             self.status = HypothesisStatus.REJECTED
#             return False
#
#     def add_test_result(self, result: Dict[str, Any]):
#         """Добавляет результат проверки."""
#         self.test_results.append(result)
#         if 'score' in result:
#             self.actual_score = result['score']
#
#     def to_dict(self) -> Dict[str, Any]:
#         """Преобразует в словарь."""
#         return {
#             'id': self.id,
#             'task_description': self.task_description,
#             'source_combination_id': self.source_combination.id,
#             'modifications': self.modifications,
#             'description': self.description,
#             'predicted_score': self.predicted_score,
#             'actual_score': self.actual_score,
#             'status': self.status.value,
#             'test_results': self.test_results,
#             'metadata': self.metadata
#         }