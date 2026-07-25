# core/knowledge/hypothesis_validator.py
"""
[ru] Валидатор гипотез в тестовой среде. Проверяет, выполняет ли гипотеза функциональные требования.
[en] Hypothesis validator in a test environment. Checks whether a hypothesis satisfies functional requirements.
"""

from typing import List, Dict, Set, Optional, Tuple
import numpy as np

from core.knowledge.hypothesis import Hypothesis, HypothesisStatus
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from core.knowledge.knowledge_node import KnowledgeNode


class HypothesisValidator:
    """
    [ru] Проверяет гипотезы на выполнение функциональных требований. Использует правила, а не симуляцию (для простоты).
    [en] Validates hypotheses against functional requirements. Uses rules rather than simulation (for simplicity).
    """

    # [ru] Правила: какие узлы могут выполнять какие функции
    # [en] Rules: which nodes can perform which functions
    FUNCTION_RULES = {
        'создавать подъёмную силу': {
            'required_nodes': ['крыло', 'крылья', 'несущая поверхность'],
            'alternative_nodes': ['винт', 'ротор'],
            'min_count': 1
        },
        'создавать тягу': {
            'required_nodes': ['двигатель', 'мотор', 'турбина'],
            'alternative_nodes': ['винт', 'пропеллер'],
            'min_count': 1
        },
        'перевозить': {
            'required_nodes': ['фюзеляж', 'корпус', 'кузов', 'грузовой отсек'],
            'alternative_nodes': ['платформа', 'контейнер'],
            'min_count': 1
        },
        'поднимать груз': {
            'required_nodes': ['лебедка', 'кран', 'подъёмник', 'грузовой отсек'],
            'alternative_nodes': ['платформа', 'такелаж'],
            'min_count': 1
        },
        'управлять полётом': {
            'required_nodes': ['руль', 'автопилот', 'стабилизатор', 'руль высоты', 'руль направления'],
            'alternative_nodes': ['элерон', 'закрылок'],
            'min_count': 1
        },
        'преобразовывать энергию': {
            'required_nodes': ['двигатель', 'генератор', 'турбина'],
            'alternative_nodes': ['аккумулятор', 'топливный элемент'],
            'min_count': 1
        },
    }

    def __init__(self, global_graph: GlobalKnowledgeGraph):
        self.global_graph = global_graph

    def validate(self, hypothesis: Hypothesis, required_functions: List[str]) -> Tuple[bool, float, List[str]]:
        """
        [ru] Проверяет гипотезу.
        [en] Validates the hypothesis.

        Args:
            [ru] hypothesis: Гипотеза для проверки
            [en] hypothesis: Hypothesis to validate
            [ru] required_functions: Список функций, которые должна выполнять гипотеза
            [en] required_functions: List of functions the hypothesis must perform

        Returns:
            [ru] (is_valid, score, missing_functions)
            [en] (is_valid, score, missing_functions)
        """
        # [ru] Получаем имена узлов в гипотезе
        # [en] Get the names of nodes in the hypothesis
        node_names = [n.name.lower() for n in hypothesis.source_combination.nodes]

        passed = 0
        total = len(required_functions)
        missing_functions = []

        for func in required_functions:
            rule = self.FUNCTION_RULES.get(func)
            if not rule:
                # [ru] Если правила нет, считаем функцию выполненной (оптимистично)
                # [en] If there is no rule, consider the function fulfilled (optimistically)
                passed += 1
                continue

            found = False
            # [ru] Проверяем требуемые узлы
            # [en] Check the required nodes
            for req_node in rule['required_nodes']:
                if req_node in node_names:
                    found = True
                    break

            if not found:
                # [ru] Проверяем альтернативные узлы
                # [en] Check the alternative nodes
                for alt_node in rule['alternative_nodes']:
                    if alt_node in node_names:
                        found = True
                        break

            if found:
                passed += 1
            else:
                missing_functions.append(func)

        score = passed / total if total > 0 else 0.0
        # [ru] гипотеза валидна, если покрывает >= 70% функций
        # [en] the hypothesis is valid if it covers >= 70% of the functions
        is_valid = score >= 0.7

        # [ru] Обновляем гипотезу
        # [en] Update the hypothesis
        hypothesis.actual_score = score
        if is_valid:
            hypothesis.status = HypothesisStatus.VALIDATED
        else:
            hypothesis.status = HypothesisStatus.REJECTED

        return is_valid, score, missing_functions

    def get_failure_reasons(self, hypothesis: Hypothesis, required_functions: List[str]) -> List[str]:
        """
        [ru] Возвращает список причин, почему гипотеза не прошла.
        [en] Returns the list of reasons why the hypothesis failed.
        """
        _, _, missing = self.validate(hypothesis, required_functions)
        return [f"Недостающая функция: {f}" for f in missing]

    def suggest_improvements(self, hypothesis: Hypothesis, required_functions: List[str]) -> List[str]:
        """
        [ru] Предлагает улучшения для гипотезы.
        [en] Suggests improvements for the hypothesis.
        """
        _, _, missing = self.validate(hypothesis, required_functions)
        suggestions = []

        for func in missing:
            rule = self.FUNCTION_RULES.get(func)
            if rule:
                suggestions.append(f"Добавьте узел '{rule['required_nodes'][0]}' для выполнения функции '{func}'")

        return suggestions


# # core/knowledge/hypothesis_validator.py
# """
# Валидатор гипотез в тестовой среде. Проверяет, выполняет ли гипотеза функциональные требования.
# """
#
# from typing import List, Dict, Set, Optional, Tuple
# import numpy as np
#
# from core.knowledge.hypothesis import Hypothesis, HypothesisStatus
# from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
# from core.knowledge.knowledge_node import KnowledgeNode
#
#
# class HypothesisValidator:
#     """
#     Проверяет гипотезы на выполнение функциональных требований. Использует правила, а не симуляцию (для простоты).
#     """
#
#     # Правила: какие узлы могут выполнять какие функции
#     FUNCTION_RULES = {
#         'создавать подъёмную силу': {
#             'required_nodes': ['крыло', 'крылья', 'несущая поверхность'],
#             'alternative_nodes': ['винт', 'ротор'],
#             'min_count': 1
#         },
#         'создавать тягу': {
#             'required_nodes': ['двигатель', 'мотор', 'турбина'],
#             'alternative_nodes': ['винт', 'пропеллер'],
#             'min_count': 1
#         },
#         'перевозить': {
#             'required_nodes': ['фюзеляж', 'корпус', 'кузов', 'грузовой отсек'],
#             'alternative_nodes': ['платформа', 'контейнер'],
#             'min_count': 1
#         },
#         'поднимать груз': {
#             'required_nodes': ['лебедка', 'кран', 'подъёмник', 'грузовой отсек'],
#             'alternative_nodes': ['платформа', 'такелаж'],
#             'min_count': 1
#         },
#         'управлять полётом': {
#             'required_nodes': ['руль', 'автопилот', 'стабилизатор', 'руль высоты', 'руль направления'],
#             'alternative_nodes': ['элерон', 'закрылок'],
#             'min_count': 1
#         },
#         'преобразовывать энергию': {
#             'required_nodes': ['двигатель', 'генератор', 'турбина'],
#             'alternative_nodes': ['аккумулятор', 'топливный элемент'],
#             'min_count': 1
#         },
#     }
#
#     def __init__(self, global_graph: GlobalKnowledgeGraph):
#         self.global_graph = global_graph
#
#     def validate(self, hypothesis: Hypothesis, required_functions: List[str]) -> Tuple[bool, float, List[str]]:
#         """
#         Проверяет гипотезу.
#
#         Args:
#             hypothesis: Гипотеза для проверки
#             required_functions: Список функций, которые должна выполнять гипотеза
#
#         Returns:
#             (is_valid, score, missing_functions)
#         """
#         # Получаем имена узлов в гипотезе
#         node_names = [n.name.lower() for n in hypothesis.source_combination.nodes]
#
#         passed = 0
#         total = len(required_functions)
#         missing_functions = []
#
#         for func in required_functions:
#             rule = self.FUNCTION_RULES.get(func)
#             if not rule:
#                 # Если правила нет, считаем функцию выполненной (оптимистично)
#                 passed += 1
#                 continue
#
#             found = False
#             # Проверяем требуемые узлы
#             for req_node in rule['required_nodes']:
#                 if req_node in node_names:
#                     found = True
#                     break
#
#             if not found:
#                 # Проверяем альтернативные узлы
#                 for alt_node in rule['alternative_nodes']:
#                     if alt_node in node_names:
#                         found = True
#                         break
#
#             if found:
#                 passed += 1
#             else:
#                 missing_functions.append(func)
#
#         score = passed / total if total > 0 else 0.0
#         # гипотеза валидна, если покрывает >= 70% функций
#         is_valid = score >= 0.7
#
#         # Обновляем гипотезу
#         hypothesis.actual_score = score
#         if is_valid:
#             hypothesis.status = HypothesisStatus.VALIDATED
#         else:
#             hypothesis.status = HypothesisStatus.REJECTED
#
#         return is_valid, score, missing_functions
#
#     def get_failure_reasons(self, hypothesis: Hypothesis, required_functions: List[str]) -> List[str]:
#         """
#         Возвращает список причин, почему гипотеза не прошла.
#         """
#         _, _, missing = self.validate(hypothesis, required_functions)
#         return [f"Недостающая функция: {f}" for f in missing]
#
#     def suggest_improvements(self, hypothesis: Hypothesis, required_functions: List[str]) -> List[str]:
#         """
#         Предлагает улучшения для гипотезы.
#         """
#         _, _, missing = self.validate(hypothesis, required_functions)
#         suggestions = []
#
#         for func in missing:
#             rule = self.FUNCTION_RULES.get(func)
#             if rule:
#                 suggestions.append(f"Добавьте узел '{rule['required_nodes'][0]}' для выполнения функции '{func}'")
#
#         return suggestions
#
