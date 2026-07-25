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


