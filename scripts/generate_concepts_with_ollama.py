# scripts/generate_concepts_with_ollama.py (исправленный)
"""
Генерация концептов через Ollama с правильным парсингом.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import requests
import re
import time
import random  # <-- ДОБАВЛЯЕМ ИМПОРТ
import hashlib
from typing import List, Dict, Any

from core.knowledge.knowledge_node import KnowledgeNode
from core.knowledge.knowledge_edge import KnowledgeEdge, EdgeType
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from db.knowledge_db import KnowledgeDB


class OllamaConceptGenerator:
    """
    Генератор концептов через Ollama.
    """

    def __init__(self, model: str = "llama3.1:8b", ollama_url: str = "http://localhost:11434"):
        self.model = model
        self.ollama_url = ollama_url
        self.graph = GlobalKnowledgeGraph()
        self.db = KnowledgeDB()
        self.generated_count = 0

    def _generate_short_id(self, name: str) -> str:
        """Генерирует короткий ID."""
        hash_obj = hashlib.md5(name.encode())
        return f"n_{hash_obj.hexdigest()[:8]}"

    def _generate_edge_id(self, source: str, target: str, edge_type: str) -> str:
        """Генерирует короткий ID для ребра."""
        combined = f"{source}_{target}_{edge_type}"
        hash_obj = hashlib.md5(combined.encode())
        return f"e_{hash_obj.hexdigest()[:8]}"

    def generate_concepts(self, category: str, count: int = 5) -> List[Dict]:
        """
        Генерирует концепты для категории.
        """
        prompt = f"""
        Generate exactly {count} engineering concepts in the category "{category}".

        Return ONLY valid JSON array. Each concept must have:
        - name: string (in Russian)
        - properties: array of 3-5 strings
        - functions: array of 2-3 strings
        - category: string (same as provided)

        Example:
        [
            {{"name": "Крыло", "properties": ["аэродинамический", "легкий"], "functions": ["создавать подъемную силу"], "category": "aerospace"}}
        ]

        Only return JSON, no other text.
        """

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,
                    "max_tokens": 2000
                },
                timeout=120
            )

            if response.status_code != 200:
                print(f"   ⚠️ HTTP {response.status_code}")
                return []

            result = response.json()
            text = result.get("response", "")

            # Извлекаем JSON
            concepts = self._extract_json(text)

            # Добавляем категорию, если отсутствует
            for concept in concepts:
                if "category" not in concept:
                    concept["category"] = category

            if concepts:
                self.generated_count += len(concepts)
                return concepts
            else:
                print(f"   ⚠️ Не удалось извлечь JSON")
                print(f"   Ответ: {text[:200]}...")
                return []

        except requests.exceptions.Timeout:
            print(f"   ⚠️ Таймаут")
            return []
        except Exception as e:
            print(f"   ⚠️ Ошибка: {e}")
            return []

    def _extract_json(self, text: str) -> List[Dict]:
        """
        Извлекает JSON из текста.
        """
        # Пробуем найти массив
        patterns = [
            r'\[\s*\{.*\}\s*\]',
            r'```json\s*(\[\s*\{.*\}\s*\])\s*```',
            r'```\s*(\[\s*\{.*\}\s*\])\s*```'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match)
                    if isinstance(data, list) and len(data) > 0:
                        return data
                except:
                    continue

        # Пробуем найти объект
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
        except:
            pass

        return []

    def generate_all_categories(self, categories: List[str], per_category: int = 5) -> List[Dict]:
        """
        Генерирует концепты для всех категорий.
        """
        all_concepts = []

        for category in categories:
            print(f"\n🔄 Категория: {category}")
            concepts = self.generate_concepts(category, per_category)
            all_concepts.extend(concepts)
            print(f"   ✅ {len(concepts)} концептов")
            time.sleep(2)  # Пауза между запросами

        return all_concepts

    def build_graph_from_concepts(self, concepts: List[Dict]) -> GlobalKnowledgeGraph:
        """
        Строит граф из сгенерированных концептов.
        """
        print(f"\n📦 Создание графа из {len(concepts)} концептов...")

        # Создаем узлы
        for concept in concepts:
            name = concept.get("name", "Unknown")
            node_id = self._generate_short_id(name)

            node = KnowledgeNode(
                id=node_id,
                name=name,
                node_type=concept.get("category", "concept"),
                properties=concept.get("properties", []),
                description=f"Generated concept: {name}"
            )
            self.graph.add_node(node)

            # Добавляем функции
            from core.knowledge.function import Function
            for func in concept.get("functions", []):
                func_obj = Function(
                    id=f"func_{node_id}_{func.replace(' ', '_')[:20]}",
                    name=func,
                    description=f"{node.name} can {func}"
                )
                node.add_function(func_obj)

        # Создаем связи между узлами
        print("🔗 Создание связей...")
        edge_counter = 0
        nodes = list(self.graph.nodes.values())

        # Связи внутри категорий
        for i, node in enumerate(nodes):
            for j, other in enumerate(nodes):
                if i == j:
                    continue

                # Связи внутри одной категории
                if node.node_type == other.node_type and random.random() < 0.3:
                    edge_id = self._generate_edge_id(node.id, other.id, "related")
                    edge = KnowledgeEdge(
                        id=edge_id,
                        source_id=node.id,
                        target_id=other.id,
                        edge_type=EdgeType.RELATED_TO,
                        weight=0.6,
                        description=f"{node.name} related to {other.name}"
                    )
                    self.graph.add_edge(edge)
                    edge_counter += 1

                # Связи между разными категориями (реже)
                elif node.node_type != other.node_type and random.random() < 0.1:
                    edge_id = self._generate_edge_id(node.id, other.id, "related")
                    edge = KnowledgeEdge(
                        id=edge_id,
                        source_id=node.id,
                        target_id=other.id,
                        edge_type=EdgeType.RELATED_TO,
                        weight=0.4,
                        description=f"{node.name} related to {other.name}"
                    )
                    self.graph.add_edge(edge)
                    edge_counter += 1

        print(f"✅ Создано {len(self.graph.nodes)} узлов и {edge_counter} связей")
        return self.graph

    def save_to_db(self):
        """Сохраняет граф в БД."""
        print("\n💾 Сохранение в БД...")
        saved_nodes = 0
        saved_edges = 0

        for node in self.graph.nodes.values():
            if self.db.save_node(node):
                saved_nodes += 1

        for edge in self.graph.edges.values():
            if self.db.save_edge(edge):
                saved_edges += 1

        print(f"✅ Сохранено {saved_nodes} узлов и {saved_edges} связей")


def main():
    """
    Основная функция.
    """
    print("=" * 60)
    print("🧠 ГЕНЕРАЦИЯ КОНЦЕПТОВ ЧЕРЕЗ OLLAMA")
    print("=" * 60)

    # Проверяем доступность Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            print("❌ Ollama недоступен")
            return
        models = response.json().get("models", [])
        print(f"✅ Ollama доступен")
        print(f"   Модели: {[m['name'] for m in models]}")
    except:
        print("❌ Ollama недоступен")
        return

    # Категории для генерации
    categories = [
        "mechanical", "electrical", "aerospace",
        "materials", "processes", "systems",
        "energy", "structures"
    ]

    # Генерируем концепты
    generator = OllamaConceptGenerator(model="llama3.1:8b")
    concepts = generator.generate_all_categories(categories, per_category=5)

    print(f"\n📊 Всего сгенерировано: {len(concepts)} концептов")

    if not concepts:
        print("\n⚠️ Не удалось сгенерировать концепты. Проверьте Ollama.")
        return

    # Строим граф
    graph = generator.build_graph_from_concepts(concepts)

    # Сохраняем в БД
    generator.save_to_db()

    # Статистика
    stats = graph.get_statistics()
    print("\n" + "=" * 60)
    print("📊 СТАТИСТИКА ГРАФА")
    print("=" * 60)
    print(f"   Узлов: {stats['total_nodes']}")
    print(f"   Связей: {stats['total_edges']}")
    print(f"   Типы узлов: {stats['node_types']}")
    print("=" * 60)

    # Примеры
    print("\n🔍 ПРИМЕРЫ СГЕНЕРИРОВАННЫХ КОНЦЕПТОВ:")
    for node in list(graph.nodes.values())[:5]:
        print(f"\n   {node.name}:")
        print(f"      ID: {node.id}")
        print(f"      Тип: {node.node_type}")
        print(f"      Свойства: {node.properties}")
        neighbors = graph.get_neighbors(node.id)
        print(f"      Соседей: {len(neighbors)}")


if __name__ == "__main__":
    main()