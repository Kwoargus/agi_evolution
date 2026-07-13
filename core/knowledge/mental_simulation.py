# core/knowledge/mental_simulation.py
from core.knowledge.hypothesis import Hypothesis
from .test_result import TestResult

class MentalSimulation:
    """
    Мысленный эксперимент - проверка гипотез в виртуальной среде.
    """

    def __init__(self, test_environment):
        self.environment = test_environment

    def test_hypothesis(self, hypothesis: Hypothesis) -> TestResult:
        """
        Проверяет гипотезу в виртуальной среде.
        """
        # 1. Создаем цифрового двойника
        model = self._build_digital_twin(hypothesis)

        # 2. Запускаем тест в среде
        result = self.environment.run_test(model)

        # 3. Возвращаем результат
        return TestResult(
            hypothesis=hypothesis,
            success=result.success,
            metrics=result.metrics,
            score=result.score
        )