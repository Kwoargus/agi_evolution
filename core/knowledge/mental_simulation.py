# core/knowledge/mental_simulation.py
from core.knowledge.hypothesis import Hypothesis
from .test_result import TestResult

class MentalSimulation:
    """
    [ru] Мысленный эксперимент - проверка гипотез в виртуальной среде.
    [en] Mental experiment - testing hypotheses in a virtual environment.
    """

    def __init__(self, test_environment):
        self.environment = test_environment

    def test_hypothesis(self, hypothesis: Hypothesis) -> TestResult:
        """
        [ru] Проверяет гипотезу в виртуальной среде.
        [en] Tests the hypothesis in a virtual environment.
        """
        # [ru] 1. Создаем цифрового двойника
        # [en] 1. Create a digital twin
        model = self._build_digital_twin(hypothesis)

        # [ru] 2. Запускаем тест в среде
        # [en] 2. Run the test in the environment
        result = self.environment.run_test(model)

        # [ru] 3. Возвращаем результат
        # [en] 3. Return the result
        return TestResult(
            hypothesis=hypothesis,
            success=result.success,
            metrics=result.metrics,
            score=result.score
        )


# # core/knowledge/mental_simulation.py
# from core.knowledge.hypothesis import Hypothesis
# from .test_result import TestResult
#
# class MentalSimulation:
#     """
#     Мысленный эксперимент - проверка гипотез в виртуальной среде.
#     """
#
#     def __init__(self, test_environment):
#         self.environment = test_environment
#
#     def test_hypothesis(self, hypothesis: Hypothesis) -> TestResult:
#         """
#         Проверяет гипотезу в виртуальной среде.
#         """
#         # 1. Создаем цифрового двойника
#         model = self._build_digital_twin(hypothesis)
#
#         # 2. Запускаем тест в среде
#         result = self.environment.run_test(model)
#
#         # 3. Возвращаем результат
#         return TestResult(
#             hypothesis=hypothesis,
#             success=result.success,
#             metrics=result.metrics,
#             score=result.score
#         )