# scripts/generate_with_mistral.py
"""
Генерация концептов через Mistral 7B.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import hashlib
import random
import requests
from typing import List, Dict

from core.knowledge.knowledge_node import KnowledgeNode
from core.knowledge.knowledge_edge import KnowledgeEdge, EdgeType
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from db.knowledge_db import KnowledgeDB


class MistralGenerator:
    """
    Генератор концептов через Mistral 7B.
    """

    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.model = "mistral:7b"
        self.db = KnowledgeDB()
        self.graph = GlobalKnowledgeGraph()
        self.total_generated = 0

        # Загружаем существующие узлы
        print("📥 Загрузка существующих узлов...")
        existing_nodes = self.db.load_all_nodes()
        for node in existing_nodes:
            self.graph.add_node(node)

        print(f"📊 Существующих узлов: {len(self.graph.nodes)}")

    def check_model(self) -> bool:
        """Проверяет, доступна ли модель."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "test",
                    "stream": False
                },
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

    def generate_concepts(self, category: str, count: int = 10) -> List[Dict]:
        """
        Генерирует концепты для категории через Mistral.
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
                    "temperature": 0.7,
                    "max_tokens": 2000
                },
                timeout=120
            )

            if response.status_code != 200:
                print(f"   ⚠️ HTTP {response.status_code}")
                return []

            result = response.json()
            text = result.get("response", "")

            concepts = self._extract_json(text)

            for concept in concepts:
                if "category" not in concept:
                    concept["category"] = category
                concept["source"] = "mistral"

            return concepts

        except Exception as e:
            print(f"   ⚠️ Ошибка: {e}")
            return []

    def _extract_json(self, text: str) -> List[Dict]:
        """Извлекает JSON из текста."""
        import re

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

        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
        except:
            pass

        return []

    def generate_all_categories(self, categories: List[str], per_category: int = 5):
        """
        Генерирует концепты для всех категорий.
        """
        print("=" * 60)
        print(f"🚀 ГЕНЕРАЦИЯ ЧЕРЕЗ {self.model.upper()}")
        print("=" * 60)

        all_concepts = []

        for category in categories:
            print(f"\n🔄 Категория: {category}")

            # Пробуем сгенерировать
            concepts = self.generate_concepts(category, per_category)

            # Фильтруем дубликаты
            existing_names = [n.name for n in self.graph.nodes.values()]
            new_concepts = []
            for c in concepts:
                name = c.get("name", "")
                if name and name not in existing_names:
                    new_concepts.append(c)
                    existing_names.append(name)
                elif name:
                    print(f"      ⏭️ Пропущен дубликат: {name}")

            all_concepts.extend(new_concepts)
            print(f"   ✅ {len(new_concepts)} новых концептов (из {len(concepts)} сгенерированных)")

            # Пауза между запросами
            time.sleep(2)

        print(f"\n📊 Всего новых концептов: {len(all_concepts)}")
        return all_concepts

    def build_graph(self, concepts: List[Dict]):
        """Строит граф из концептов."""
        print("\n📦 Создание узлов...")

        added = 0
        for concept in concepts:
            name = concept.get("name", "Unknown")

            if self.graph.get_node_by_name(name):
                continue

            node_id = f"mist_{hashlib.md5(name.encode()).hexdigest()[:8]}"

            node = KnowledgeNode(
                id=node_id,
                name=name,
                node_type=concept.get("category", "concept"),
                properties=concept.get("properties", []),
                description=f"Generated by Mistral: {name}"
            )

            node.metadata = {
                "source": "mistral",
                "generation_time": time.time()
            }

            self.graph.add_node(node)
            added += 1

        print(f"✅ Добавлено {added} новых узлов")

        # Создаем связи
        print("🔗 Создание связей...")
        edges_added = self._create_edges()
        print(f"✅ Добавлено {edges_added} связей")

    def _create_edges(self) -> int:
        """Создает связи между узлами."""
        edges_added = 0
        nodes = list(self.graph.nodes.values())

        # Группируем по типу
        by_type = {}
        for node in nodes:
            if node.node_type not in by_type:
                by_type[node.node_type] = []
            by_type[node.node_type].append(node)

        # Связи внутри типов
        for node_type, type_nodes in by_type.items():
            if len(type_nodes) < 2:
                continue

            for i, node in enumerate(type_nodes):
                others = [n for n in type_nodes if n.id != node.id]
                if not others:
                    continue

                num_edges = min(2, len(others))
                selected = random.sample(others, num_edges)

                for target in selected:
                    # Проверяем, есть ли уже связь
                    existing = False
                    for edge in self.graph.edges.values():
                        if edge.source_id == node.id and edge.target_id == target.id:
                            existing = True
                            break

                    if existing:
                        continue

                    edge = KnowledgeEdge(
                        id=f"mist_edge_{hashlib.md5(f'{node.id}_{target.id}'.encode()).hexdigest()[:8]}",
                        source_id=node.id,
                        target_id=target.id,
                        edge_type=EdgeType.RELATED_TO,
                        weight=0.6,
                        description=f"{node.name} related to {target.name}"
                    )
                    self.graph.add_edge(edge)
                    edges_added += 1

        return edges_added

    def save_to_db(self):
        """Сохраняет граф в БД."""
        print("\n💾 Сохранение в БД...")

        saved_nodes = 0
        for node in self.graph.nodes.values():
            if self.db.save_node(node):
                saved_nodes += 1

        saved_edges = 0
        for edge in self.graph.edges.values():
            if self.db.save_edge(edge):
                saved_edges += 1

        print(f"✅ Сохранено {saved_nodes} узлов и {saved_edges} связей")
        print(f"📊 Всего в БД: {saved_nodes} узлов, {saved_edges} связей")


def main():
    generator = MistralGenerator()

    # Проверяем, запущена ли Mistral
    if not generator.check_model():
        print("❌ Mistral 7B не запущена!")
        print("   Запустите в другом терминале: ollama run mistral:7b")
        return

    print(f"✅ {generator.model} запущена и доступна")

    # Категории для генерации
    categories = [
        "mechanical", "electrical", "aerospace",
        "materials", "processes", "systems",
        "energy", "structures", "electronics",
        "robotics", "automation", "instrumentation"
    ]

    # Генерируем
    concepts = generator.generate_all_categories(categories, per_category=5)

    if not concepts:
        print("⚠️ Не удалось сгенерировать концепты")
        return

    # Строим граф
    generator.build_graph(concepts)

    # Сохраняем
    generator.save_to_db()

    # Статистика
    stats = generator.graph.get_statistics()
    print("\n" + "=" * 60)
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)
    print(f"   Всего узлов: {stats['total_nodes']}")
    print(f"   Всего связей: {stats['total_edges']}")
    print(f"   Типы узлов: {stats['node_types']}")
    print("=" * 60)

    # Показываем примеры новых узлов
    print("\n🔍 ПРИМЕРЫ НОВЫХ УЗЛОВ:")
    new_nodes = [n for n in generator.graph.nodes.values() if n.metadata.get("source") == "mistral"]
    for node in new_nodes[:5]:
        print(f"\n   {node.name}:")
        print(f"      Тип: {node.node_type}")
        print(f"      Свойства: {node.properties[:3]}")
        neighbors = generator.graph.get_neighbors(node.id)
        print(f"      Соседей: {len(neighbors)}")


if __name__ == "__main__":
    main()