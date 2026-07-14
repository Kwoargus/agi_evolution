# core/knowledge/analogy_engine.py - ПОЛНОСТЬЮ ОБНОВЛЁННЫЙ

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import hashlib
import time

from core.knowledge.combination import Combination
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from core.knowledge.individual_knowledge_graph import IndividualKnowledgeGraph


class AnalogyEngine:
    """
    Движок аналогий - находит и модифицирует модели.
    Использует как ГГЗ, так и ИГЗ.
    """

    def __init__(self, global_graph: GlobalKnowledgeGraph,
                 individual_graph: IndividualKnowledgeGraph):
        self.global_graph = global_graph
        self.individual_graph = individual_graph
        self.analogy_cache = {}  # Кэш для ускорения поиска

    def find_analogies(self,
                       task_description: str,
                       required_properties: List[str],
                       max_results: int = 10) -> List[Combination]:
        """
        Находит аналогии для задачи.

        1. Поиск в ИГЗ (быстрый, образный)
        2. Поиск в ГГЗ (глубокий, точный)
        3. Комбинация и ранжирование результатов
        """
        cache_key = f"{task_description}_{'_'.join(sorted(required_properties))}"
        if cache_key in self.analogy_cache:
            return self.analogy_cache[cache_key][:max_results]

        all_analogies = []

        # 1. Ищем в ИГЗ (ментальные модели)
        mental_analogies = self._search_mental(required_properties)
        all_analogies.extend(mental_analogies)

        # 2. Ищем в ГГЗ (знания)
        knowledge_analogies = self._search_knowledge(required_properties)
        all_analogies.extend(knowledge_analogies)

        # 3. Объединяем и ранжируем
        ranked = self._merge_and_rank(all_analogies, required_properties)

        # Кэшируем
        self.analogy_cache[cache_key] = ranked

        return ranked[:max_results]

    def _search_mental(self, required_properties: List[str]) -> List[Combination]:
        """
        Ищет аналогии в индивидуальном графе знаний (образные модели).
        """
        analogies = []

        if not self.individual_graph or not hasattr(self.individual_graph, 'find_mental_models_by_properties'):
            return analogies

        # Ищем ментальные модели по свойствам
        found_models = self.individual_graph.find_mental_models_by_properties(required_properties)

        for model_info in found_models[:10]:
            model_data = model_info.get('model_data', {})

            # Создаем комбинацию
            combo = Combination(
                id=f"mental_{model_info['model_id']}",
                nodes=[],
                properties=model_data.get('properties', []),
                metadata={
                    'source': 'individual_graph',
                    'model_id': model_info['model_id'],
                    'model_data': model_data,
                    'match_count': model_info['match_count'],
                    'usage_count': model_info.get('usage_count', 0),
                    'score': model_info['match_count'] / max(len(required_properties), 1)
                }
            )
            analogies.append(combo)

        return analogies

    def _search_knowledge(self, required_properties: List[str]) -> List[Combination]:
        """
        Ищет аналогии в глобальном графе знаний.
        """
        analogies = []

        if not self.global_graph:
            return analogies

        # Ищем узлы по свойствам
        found_nodes = self.global_graph.find_by_properties(required_properties)

        # Группируем по типу
        nodes_by_type = {}
        for node in found_nodes:
            if node.node_type not in nodes_by_type:
                nodes_by_type[node.node_type] = []
            nodes_by_type[node.node_type].append(node)

        # Создаем комбинации из узлов
        for node_type, nodes in nodes_by_type.items():
            if len(nodes) >= 2:
                # Берем 2-3 узла одного типа
                import random
                selected = random.sample(nodes, min(3, len(nodes)))
                combo = Combination(
                    id=f"kg_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}",
                    nodes=selected,
                    properties=list(set([p for n in selected for p in n.properties])),
                    metadata={
                        'source': 'global_graph',
                        'node_type': node_type,
                        'node_count': len(selected)
                    }
                )
                analogies.append(combo)

        # Кросс-типовые комбинации
        types = list(nodes_by_type.keys())
        for _ in range(len(types) * 2):
            if len(types) < 2:
                break
            import random
            t1, t2 = random.sample(types, 2)
            n1 = random.choice(nodes_by_type[t1])
            n2 = random.choice(nodes_by_type[t2])

            combo = Combination(
                id=f"kg_cross_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}",
                nodes=[n1, n2],
                properties=list(set(n1.properties + n2.properties)),
                metadata={
                    'source': 'global_graph',
                    'node_type': f"{t1}+{t2}",
                    'node_count': 2
                }
            )
            analogies.append(combo)

        return analogies

    def _merge_and_rank(self, analogies: List[Combination],
                        required_properties: List[str]) -> List[Combination]:
        """
        Объединяет и ранжирует аналогии.
        """
        # Удаляем дубликаты по ID
        seen = set()
        unique = []
        for analogy in analogies:
            if analogy.id not in seen:
                seen.add(analogy.id)
                unique.append(analogy)

        # Вычисляем оценку для каждой аналогии
        for analogy in unique:
            # Базовый скор
            base_score = 0.0

            # 1. Покрытие свойств
            analogy_props = set(analogy.properties)
            covered = sum(1 for p in required_properties if p in analogy_props)
            coverage_score = covered / max(len(required_properties), 1)
            base_score += coverage_score * 0.5

            # 2. Источник (ИГЗ имеет приоритет)
            source = analogy.metadata.get('source', '')
            if source == 'individual_graph':
                base_score += 0.2  # Бонус за опыт

            # 3. Количество узлов
            node_count = len(analogy.nodes)
            if node_count > 0:
                base_score += min(node_count / 5, 0.3)  # Максимум 0.3

            # Сохраняем оценку
            analogy.metadata['score'] = base_score

        # Сортируем по оценке
        unique.sort(key=lambda x: x.metadata.get('score', 0), reverse=True)

        return unique

    def modify_analogy(self, analogy: Combination,
                       modifications: List[str]) -> Combination:
        """
        Модифицирует аналогию.
        """
        modified = analogy.copy()

        for mod in modifications:
            if mod == "replace_part":
                # Заменяет часть модели другой моделью
                if len(modified.nodes) > 1:
                    # Ищем замену в ГГЗ
                    replacement = self._find_replacement(modified.nodes[0])
                    if replacement:
                        modified.nodes[0] = replacement
                        modified.metadata['modified'] = True

            elif mod == "add_feature":
                # Добавляет свойство
                new_property = mod.replace("add_feature_", "")
                if new_property not in modified.properties:
                    modified.properties.append(new_property)

            elif mod == "combine":
                # Комбинирует с другой моделью
                if len(modified.nodes) > 1:
                    other = self._find_combination_partner(modified)
                    if other:
                        modified.nodes.append(other)
                        modified.id = f"combined_{modified.id}"

        return modified

    def _find_replacement(self, node) -> Optional[Any]:
        """Находит замену для узла в ГГЗ."""
        if not self.global_graph:
            return None

        # Ищем похожие узлы по типу
        if hasattr(node, 'node_type'):
            candidates = self.global_graph.find_by_type(node.node_type)
            if candidates:
                import random
                return random.choice([n for n in candidates if n.id != node.id])

        return None

    def _find_combination_partner(self, analogy: Combination) -> Optional[Any]:
        """Находит партнёра для комбинирования."""
        if not self.global_graph:
            return None

        # Ищем узел, который не входит в текущую комбинацию
        existing_ids = [n.id for n in analogy.nodes]
        for node in self.global_graph.nodes.values():
            if node.id not in existing_ids:
                return node

        return None

    def clear_cache(self):
        """Очищает кэш аналогий."""
        self.analogy_cache.clear()




# # core/knowledge/analogy_engine.py
# from typing import List, Dict, Any, Tuple, Optional
# import numpy as np
#
# from core.knowledge.combination import Combination
# from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
# from core.knowledge.individual_knowledge_graph import IndividualKnowledgeGraph
#
#
# class AnalogyEngine:
#     """
#     Движок аналогий - находит и модифицирует модели.
#     """
#
#     def __init__(self, global_graph: GlobalKnowledgeGraph,
#                  individual_graph: IndividualKnowledgeGraph):
#         self.global_graph = global_graph
#         self.individual_graph = individual_graph
#
#     def _search_mental(self, required_properties: List[str]) -> List[Combination]:
#         """
#         Ищет аналогии в индивидуальном графе знаний (образные модели).
#
#         Args:
#             required_properties: Список требуемых свойств
#
#         Returns:
#             Список комбинаций из ИГЗ
#         """
#         analogies = []
#
#         if not self.individual_graph:
#             return analogies
#
#         # Ищем ментальные модели по свойствам
#         found_models = self.individual_graph.find_mental_models_by_properties(required_properties)
#
#         for model_info in found_models[:10]:  # Ограничиваем
#             model_data = model_info.get('model_data', {})
#
#             # Создаем комбинацию
#             combo = Combination(
#                 id=f"mental_{model_info['model_id']}",
#                 nodes=[],  # В ИГЗ узлы могут быть неполными
#                 properties=model_data.get('properties', []),
#                 metadata={
#                     'source': 'individual_graph',
#                     'model_id': model_info['model_id'],
#                     'model_data': model_data,
#                     'match_count': model_info['match_count'],
#                     'usage_count': model_info['usage_count']
#                 }
#             )
#             analogies.append(combo)
#
#         # Сортируем по количеству совпадений
#         analogies.sort(key=lambda x: x.metadata.get('match_count', 0), reverse=True)
#
#         return analogies
#
#     def find_analogies(self,
#                        task_description: str,
#                        required_properties: List[str]) -> List[Combination]:
#         """
#         Находит аналогии для задачи.
#
#         1. Поиск в ИГЗ (быстрый, образный)
#         2. Поиск в ГГЗ (глубокий, точный)
#         3. Комбинация результатов
#         """
#         # Шаг 1: Ищем в ИГЗ по свойствам
#         mental_analogies = self._search_mental(required_properties)
#
#         # Шаг 2: Ищем в ГГЗ по свойствам
#         knowledge_analogies = self.global_graph.find_combinations(required_properties)
#
#         # Шаг 3: Объединяем и ранжируем
#         all_analogies = self._merge_and_rank(mental_analogies, knowledge_analogies)
#
#         return all_analogies
#
#     def modify_analogy(self, analogy: Combination,
#                        modifications: List[str]) -> Combination:
#         """
#         Модифицирует аналогию.
#         """
#         modified = analogy.copy()
#         for mod in modifications:
#             if mod == "replace_part":
#                 # Заменяет часть модели другой моделью
#                 pass
#             elif mod == "add_feature":
#                 # Добавляет свойство
#                 pass
#             elif mod == "combine":
#                 # Комбинирует с другой моделью
#                 pass
#         return modified