# core/knowledge/hypothesis_validator.py
"""
Валидатор гипотез в тестовой среде.
Проверяет, выполняет ли гипотеза функциональные требования.
"""

from typing import List, Dict, Set, Optional, Tuple
import numpy as np

from core.knowledge.hypothesis import Hypothesis, HypothesisStatus
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from core.knowledge.knowledge_node import KnowledgeNode


class HypothesisValidator:
    """
    Проверяет гипотезы на выполнение функциональных требований.
    Использует правила, а не симуляцию (для простоты).
    """

    # Правила: какие узлы могут выполнять какие функции
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
        Проверяет гипотезу.

        Args:
            hypothesis: Гипотеза для проверки
            required_functions: Список функций, которые должна выполнять гипотеза

        Returns:
            (is_valid, score, missing_functions)
        """
        # Получаем имена узлов в гипотезе
        node_names = [n.name.lower() for n in hypothesis.source_combination.nodes]

        passed = 0
        total = len(required_functions)
        missing_functions = []

        for func in required_functions:
            rule = self.FUNCTION_RULES.get(func)
            if not rule:
                # Если правила нет, считаем функцию выполненной (оптимистично)
                passed += 1
                continue

            found = False
            # Проверяем требуемые узлы
            for req_node in rule['required_nodes']:
                if req_node in node_names:
                    found = True
                    break

            if not found:
                # Проверяем альтернативные узлы
                for alt_node in rule['alternative_nodes']:
                    if alt_node in node_names:
                        found = True
                        break

            if found:
                passed += 1
            else:
                missing_functions.append(func)

        score = passed / total if total > 0 else 0.0
        is_valid = score >= 0.7  # гипотеза валидна, если покрывает >= 70% функций

        # Обновляем гипотезу
        hypothesis.actual_score = score
        if is_valid:
            hypothesis.status = HypothesisStatus.VALIDATED
        else:
            hypothesis.status = HypothesisStatus.REJECTED

        return is_valid, score, missing_functions

    def get_failure_reasons(self, hypothesis: Hypothesis, required_functions: List[str]) -> List[str]:
        """Возвращает список причин, почему гипотеза не прошла."""
        _, _, missing = self.validate(hypothesis, required_functions)
        return [f"Недостающая функция: {f}" for f in missing]

    def suggest_improvements(self, hypothesis: Hypothesis, required_functions: List[str]) -> List[str]:
        """Предлагает улучшения для гипотезы."""
        _, _, missing = self.validate(hypothesis, required_functions)
        suggestions = []

        for func in missing:
            rule = self.FUNCTION_RULES.get(func)
            if rule:
                suggestions.append(f"Добавьте узел '{rule['required_nodes'][0]}' для выполнения функции '{func}'")

        return suggestions





# # core/knowledge/hypothesis_validator.py
# """
# Валидатор гипотез в тестовой среде.
# Проверяет, выполняет ли гипотеза функциональные требования.
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
#     Проверяет гипотезы на выполнение функциональных требований.
#     Использует правила, а не симуляцию (для простоты).
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
#         self.function_rules = self.FUNCTION_RULES
#
#     def validate(self, hypothesis: Hypothesis) -> Tuple[bool, float, List[str]]:
#         """
#         Проверяет гипотезу.
#
#         Args:
#             hypothesis: Гипотеза для проверки
#
#         Returns:
#             (is_valid, score, reasons)
#         """
#         # Получаем имена узлов в гипотезе
#         node_names = [n.name.lower() for n in hypothesis.source_combination.nodes]
#         node_props = set()
#         for node in hypothesis.source_combination.nodes:
#             node_props.update([p.lower() for p in node.properties])
#
#         # Проверяем каждую функцию
#         passed = 0
#         total = 0
#         reasons = []
#         missing_functions = []
#
#         for func in self.metadata.get('required_functions', []):
#             total += 1
#             rule = self.function_rules.get(func)
#             if not rule:
#                 # Если правила нет, считаем, что функция выполнена (оптимистично)
#                 passed += 1
#                 continue
#
#             # Проверяем, есть ли требуемый узел
#             found = False
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
#                 reasons.append(f"✅ {func} — выполнена")
#             else:
#                 missing_functions.append(func)
#                 reasons.append(f"❌ {func} — не выполнена (нужен узел: {rule['required_nodes'][0]})")
#
#         score = passed / total if total > 0 else 0.0
#         is_valid = score >= 0.7  # гипотеза валидна, если покрывает >= 70% функций
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
#     def get_failure_reasons(self, hypothesis: Hypothesis) -> List[str]:
#         """Возвращает список причин, почему гипотеза не прошла."""
#         _, _, missing = self.validate(hypothesis)
#         return [f"Недостающая функция: {f}" for f in missing]
#
#     def suggest_improvements(self, hypothesis: Hypothesis) -> List[str]:
#         """Предлагает улучшения для гипотезы."""
#         _, _, missing = self.validate(hypothesis)
#         suggestions = []
#
#         for func in missing:
#             rule = self.function_rules.get(func)
#             if rule:
#                 suggestions.append(f"Добавьте узел '{rule['required_nodes'][0]}' для выполнения функции '{func}'")
#
#         return suggestions




# # core/knowledge/hypothesis_validator.py
# """
# Проверка гипотез в тестовой среде (симуляция).
# """
#
# from typing import List, Dict, Set, Tuple, Optional
# import numpy as np
#
# from core.knowledge.hypothesis import Hypothesis
# from core.knowledge.knowledge_node import KnowledgeNode
# from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
#
#
# class HypothesisValidator:
#     """
#     Валидатор гипотез: проверяет, насколько комбинация узлов удовлетворяет функциям.
#     Использует симуляцию на основе правил и свойств узлов.
#     """
#
#     def __init__(self, global_graph: GlobalKnowledgeGraph):
#         self.global_graph = global_graph
#
#     def validate(self, hypothesis: Hypothesis, required_functions: List[str]) -> Dict:
#         """
#         Проверяет гипотезу и возвращает детальный отчёт.
#
#         Returns:
#             dict: {
#                 'score': float (0-1),
#                 'covered': List[str],
#                 'missing': List[str],
#                 'details': dict,
#                 'recommendations': List[str]
#             }
#         """
#         # Получаем узлы из гипотезы
#         nodes = hypothesis.source_combination.nodes
#
#         # 1. Проверяем покрытие функций
#         covered, missing = self._check_coverage(nodes, required_functions)
#
#         # 2. Проверяем связи между узлами (нужны ли они?)
#         connectivity_score = self._check_connectivity(nodes)
#
#         # 3. Проверяем "мощность" или "эффективность" (на основе свойств)
#         performance_score = self._check_performance(nodes)
#
#         # 4. Итоговая оценка
#         coverage_ratio = len(covered) / len(required_functions) if required_functions else 1.0
#         score = 0.5 * coverage_ratio + 0.3 * connectivity_score + 0.2 * performance_score
#
#         # 5. Рекомендации
#         recommendations = []
#         for func in missing:
#             # Найти подходящий узел для этой функции
#             candidates = self.global_graph.find_by_properties([func])
#             if candidates:
#                 recommendations.append(f"Добавьте узел '{candidates[0].name}' для выполнения функции '{func}'")
#             else:
#                 recommendations.append(f"Не найден узел для функции '{func}'")
#
#         return {
#             'score': score,
#             'covered': list(covered),
#             'missing': list(missing),
#             'details': {
#                 'coverage_ratio': coverage_ratio,
#                 'connectivity_score': connectivity_score,
#                 'performance_score': performance_score,
#                 'node_count': len(nodes)
#             },
#             'recommendations': recommendations
#         }
#
#     def _check_coverage(self, nodes: List[KnowledgeNode], required_functions: List[str]) -> Tuple[Set[str], Set[str]]:
#         """Проверяет, какие функции покрыты свойствами узлов."""
#         all_props = set()
#         for node in nodes:
#             all_props.update(p.lower() for p in node.properties)
#
#         req_set = set(f.lower() for f in required_functions)
#         covered = req_set & all_props
#         missing = req_set - covered
#         return covered, missing
#
#     def _check_connectivity(self, nodes: List[KnowledgeNode]) -> float:
#         """
#         Проверяет, насколько узлы связаны друг с другом в ГЗ.
#         Чем больше связей между узлами, тем выше оценка.
#         """
#         if len(nodes) < 2:
#             return 0.0
#
#         node_ids = set(n.id for n in nodes)
#         edge_count = 0
#         total_possible = len(nodes) * (len(nodes) - 1) / 2
#
#         for edge in self.global_graph.edges.values():
#             if edge.source_id in node_ids and edge.target_id in node_ids:
#                 edge_count += 1
#
#         if total_possible == 0:
#             return 0.0
#
#         # Нормализуем: если есть хотя бы одна связь, уже хорошо
#         return min(edge_count / total_possible, 1.0)
#
#     def _check_performance(self, nodes: List[KnowledgeNode]) -> float:
#         """
#         Оценивает "мощность" или "эффективность" узлов по наличию свойств,
#         указывающих на высокую производительность (например, "мощный", "эффективный").
#         """
#         performance_keywords = ['мощный', 'эффективный', 'высокопроизводительный', 'надёжный', 'быстрый', 'лёгкий']
#         score = 0.0
#         total = 0
#
#         for node in nodes:
#             total += 1
#             for prop in node.properties:
#                 if any(kw in prop.lower() for kw in performance_keywords):
#                     score += 1.0
#                     break
#
#         return score / total if total > 0 else 0.0