# core/knowledge/hypothesis_generator.py
from core.knowledge.hypothesis import Hypothesis
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

class HypothesisGenerator:
    """
    Генератор гипотез на основе GAN.
    """

    def __init__(self, global_graph: GlobalKnowledgeGraph):
        self.global_graph = global_graph
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

    def generate_hypotheses(self,
                            task_description: str,
                            required_properties: List[str],
                            n_hypotheses: int = 5) -> List[Hypothesis]:
        """
        Генерирует гипотезы для задачи.
        """
        # 1. Находим аналогии
        analogies = self.analogy_engine.find_analogies(task_description, required_properties)

        # 2. Для каждой аналогии генерируем модификации
        hypotheses = []
        for analogy in analogies[:n_hypotheses * 2]:
            # Генератор предлагает модификации
            modifications = self.generator.generate(analogy, task_description)
            for mod in modifications:
                hypothesis = self._create_hypothesis(analogy, mod)
                hypotheses.append(hypothesis)

        # 3. Оцениваем дискриминатором
        scored = [(h, self.discriminator.evaluate(h, task_description)) for h in hypotheses]
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:n_hypotheses]