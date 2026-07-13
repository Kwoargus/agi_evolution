# core/thinking/research.py
"""
Модуль "Исследование" - генерация и проверка гипотез.
"""

import random
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from core.knowledge.knowledge_node import KnowledgeNode
from core.knowledge.knowledge_edge import KnowledgeEdge, EdgeType
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from core.knowledge.combination import Combination
from core.knowledge.hypothesis import Hypothesis, HypothesisStatus


class HypothesisStatus(Enum):
    PROPOSED = "proposed"
    TESTING = "testing"
    VALIDATED = "validated"
    REJECTED = "rejected"


@dataclass
class ResearchResult:
    """Результат этапа исследования."""
    problem_description: str
    required_properties: List[str]
    found_analogies: List[Combination]
    generated_hypotheses: List[Hypothesis]
    validated_hypotheses: List[Hypothesis]
    created_at: float = field(default_factory=time.time)


class ResearchEngine:
    """
    Движок "Исследования" - генерация и проверка гипотез.
    """

    def __init__(self, global_graph: GlobalKnowledgeGraph,
                 test_environment: Any = None):
        self.global_graph = global_graph
        self.test_environment = test_environment
        self.analogy_cache = {}

    def extract_requirements(self, problem_description: str) -> List[str]:
        """
        Извлекает требования из описания проблемы.
        """
        # Простейший подход: поиск ключевых слов
        requirements = []

        requirement_keywords = {
            "летать": ["flying", "airborne", "flight"],
            "двигаться": ["mobile", "moving", "transport"],
            "поднимать": ["lifting", "hoisting", "elevating"],
            "вращать": ["rotating", "spinning", "turning"],
            "управлять": ["controlled", "steered", "guided"],
            "передавать": ["transmitting", "transferring", "conveying"],
            "преобразовывать": ["converting", "transforming", "changing"],
            "накапливать": ["storing", "accumulating", "buffering"],
            "защищать": ["protecting", "shielding", "guarding"],
            "соединять": ["connecting", "joining", "linking"],
        }

        text_lower = problem_description.lower()
        for keyword, aliases in requirement_keywords.items():
            if keyword in text_lower or any(a in text_lower for a in aliases):
                requirements.append(keyword)

        return requirements

    def find_analogies(self, required_properties: List[str]) -> List[Combination]:
        """
        Находит аналогии в ГЗ по требуемым свойствам.
        """
        analogies = []

        # Ищем узлы с требуемыми свойствами
        found_nodes = []
        for prop in required_properties:
            # Ищем узлы, у которых есть это свойство
            for node in self.global_graph.nodes.values():
                if prop in [p.lower() for p in node.properties]:
                    found_nodes.append(node)

        # Группируем найденные узлы по типам
        nodes_by_type = {}
        for node in found_nodes:
            if node.node_type not in nodes_by_type:
                nodes_by_type[node.node_type] = []
            nodes_by_type[node.node_type].append(node)

        # Создаем комбинации из узлов разных типов
        for node_type, nodes in nodes_by_type.items():
            if len(nodes) >= 2:
                # Берем 2-3 узла одного типа
                selected = random.sample(nodes, min(3, len(nodes)))
                combo = Combination(
                    id=f"analogy_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}",
                    nodes=selected,
                    properties=list(set([p for n in selected for p in n.properties]))
                )
                analogies.append(combo)

        # Добавляем кросс-типовые комбинации
        types = list(nodes_by_type.keys())
        for _ in range(len(types) * 2):
            if len(types) < 2:
                break
            t1, t2 = random.sample(types, 2)
            n1 = random.choice(nodes_by_type[t1])
            n2 = random.choice(nodes_by_type[t2])

            combo = Combination(
                id=f"analogy_cross_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}",
                nodes=[n1, n2],
                properties=list(set(n1.properties + n2.properties))
            )
            analogies.append(combo)

        return analogies[:10]  # Ограничиваем количество

    def generate_hypothesis(self, analogy: Combination,
                            required_properties: List[str]) -> Hypothesis:
        """
        Генерирует гипотезу на основе аналогии.
        """
        # Создаем модификации аналогии
        modifications = []

        # 1. Добавляем недостающие свойства
        analogy_props = set(analogy.properties)
        missing = set(required_properties) - analogy_props
        if missing:
            modifications.append(f"add_{'_'.join(missing)}")

        # 2. Заменяем часть узлов
        if len(analogy.nodes) > 1:
            modifications.append("replace_part")

        # 3. Комбинируем с другим узлом
        random_node = random.choice(list(self.global_graph.nodes.values()))
        if random_node.id not in [n.id for n in analogy.nodes]:
            modifications.append(f"combine_with_{random_node.name}")

        # Создаем гипотезу
        hypothesis_id = f"hyp_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

        return Hypothesis(
            id=hypothesis_id,
            task_description="Generated from analogy",
            source_combination=analogy,
            modifications=modifications,
            predicted_score=random.uniform(0.3, 0.8),
            status=HypothesisStatus.PROPOSED,
            metadata={
                "generated_at": time.time(),
                "source": "research_engine",
                "required_properties": required_properties
            }
        )

    def test_hypothesis(self, hypothesis: Hypothesis) -> Tuple[bool, float]:
        """
        Проверяет гипотезу в тестовой среде.
        """
        # В реальной системе здесь была бы симуляция
        # Для демонстрации используем простую оценку

        # Оцениваем на основе покрытия требований
        required = hypothesis.metadata.get("required_properties", [])
        if not required:
            return True, 0.7

        # Проверяем, какие требования покрыты
        analogy_props = set(hypothesis.source_combination.properties)
        covered = sum(1 for r in required if r in analogy_props)
        score = covered / len(required) if required else 0.5

        # Добавляем случайность
        score += random.uniform(-0.1, 0.1)
        score = max(0, min(1, score))

        # Проверяем, достаточно ли хороша гипотеза
        validated = score >= 0.7

        # Обновляем гипотезу
        hypothesis.actual_score = score
        if validated:
            hypothesis.status = HypothesisStatus.VALIDATED
        else:
            hypothesis.status = HypothesisStatus.REJECTED

        return validated, score

    def research(self, problem_description: str) -> ResearchResult:
        """
        Основной метод "Исследования".

        Args:
            problem_description: Текстовое описание проблемы

        Returns:
            ResearchResult с результатами исследования
        """
        print("🔬 ЗАПУСК ИССЛЕДОВАНИЯ...")
        print(f"   Проблема: {problem_description[:100]}...")

        # 1. Извлекаем требования
        requirements = self.extract_requirements(problem_description)
        print(f"   📍 Требований: {len(requirements)}")

        # 2. Находим аналогии
        analogies = self.find_analogies(requirements)
        print(f"   📍 Найдено аналогий: {len(analogies)}")

        # 3. Генерируем гипотезы
        hypotheses = []
        for analogy in analogies[:5]:
            hyp = self.generate_hypothesis(analogy, requirements)
            hypotheses.append(hyp)

        print(f"   📍 Сгенерировано гипотез: {len(hypotheses)}")

        # 4. Проверяем гипотезы
        validated = []
        for hyp in hypotheses:
            is_valid, score = self.test_hypothesis(hyp)
            if is_valid:
                validated.append(hyp)
                print(f"   ✅ Гипотеза {hyp.id}: {score:.2f} - валидна")
            else:
                print(f"   ❌ Гипотеза {hyp.id}: {score:.2f} - отклонена")

        print(f"   📍 Валидных гипотез: {len(validated)}")

        return ResearchResult(
            problem_description=problem_description,
            required_properties=requirements,
            found_analogies=analogies,
            generated_hypotheses=hypotheses,
            validated_hypotheses=validated
        )