# scripts/generate_knowledge_graph.py (исправленный - больше связей)
"""
Генерация расширенного графа знаний.
"""

import sys
import os
import hashlib
import random
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.knowledge.knowledge_node import KnowledgeNode
from core.knowledge.knowledge_edge import KnowledgeEdge, EdgeType
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from db.knowledge_db import KnowledgeDB


class KnowledgeGraphGenerator:
    """
    Генератор расширенного графа знаний.
    """

    CATEGORIES = {
        "mechanical": ["механизмы", "детали", "приводы", "передачи"],
        "electrical": ["электроника", "схемы", "компоненты", "системы"],
        "aerospace": ["летательные аппараты", "аэродинамика", "двигатели"],
        "materials": ["металлы", "полимеры", "композиты", "сплавы"],
        "processes": ["технологии", "производство", "обработка", "сборка"],
        "systems": ["системы управления", "гидравлика", "пневматика"],
        "energy": ["источники энергии", "преобразование", "хранение"],
        "structures": ["конструкции", "фермы", "рамы", "крепления"],
    }

    PROPERTY_TEMPLATES = {
        "mechanical": ["подвижный", "вращающийся", "скользящий", "упругий", "жесткий"],
        "electrical": ["проводящий", "изолирующий", "полупроводниковый", "емкостный"],
        "aerospace": ["аэродинамический", "легкий", "прочный", "обтекаемый"],
        "materials": ["твердый", "эластичный", "хрупкий", "вязкий", "жаростойкий"],
        "processes": ["последовательный", "циклический", "непрерывный", "дискретный"],
        "systems": ["управляемый", "автономный", "адаптивный", "стабильный"],
        "energy": ["эффективный", "интенсивный", "аккумулирующий", "преобразующий"],
        "structures": ["несущий", "опорный", "соединительный", "усиливающий"],
    }

    def __init__(self, target_nodes: int = 1000):
        self.target_nodes = target_nodes
        self.graph = GlobalKnowledgeGraph()
        self.db = KnowledgeDB()
        self.concepts = []

    def _generate_short_id(self, *args) -> str:
        """Генерирует короткий ID на основе хеша."""
        combined = "_".join(str(a) for a in args)
        hash_obj = hashlib.md5(combined.encode())
        return f"n_{hash_obj.hexdigest()[:8]}"

    def _generate_edge_id(self, *args) -> str:
        """Генерирует короткий ID для ребра."""
        combined = "_".join(str(a) for a in args)
        hash_obj = hashlib.md5(combined.encode())
        return f"e_{hash_obj.hexdigest()[:8]}"

    def generate_concepts(self) -> List[Dict]:
        """Генерирует концепты."""
        print("🧠 Генерация концептов...")

        concepts = []
        for category, subcategories in self.CATEGORIES.items():
            for sub in subcategories:
                for i in range(5):
                    concept = self._generate_concept(category, sub, i)
                    concepts.append(concept)

        self.concepts = concepts
        print(f"✅ Сгенерировано {len(concepts)} концептов")
        return concepts

    def _generate_concept(self, category: str, subcategory: str, index: int) -> Dict:
        """Генерирует один концепт."""
        base_names = {
            "mechanical": ["шестерня", "вал", "подшипник", "муфта", "пружина",
                           "рычаг", "винт", "шпонка", "звездочка", "ремень"],
            "electrical": ["резистор", "конденсатор", "транзистор", "диод", "микросхема",
                           "трансформатор", "индуктор", "тиристор", "оптрон", "кварц"],
            "aerospace": ["крыло", "фюзеляж", "стабилизатор", "элерон", "закрылок",
                          "руль", "лонжерон", "нервюра", "стрингер", "обшивка"],
            "materials": ["сталь", "алюминий", "титан", "медь", "пластик",
                          "композит", "стекловолокно", "керамика", "графит", "сплав"],
            "processes": ["сварка", "пайка", "клепка", "болтовое соединение", "прессовка",
                          "литье", "ковка", "штамповка", "фрезерование", "точка"],
            "systems": ["регулятор", "усилитель", "преобразователь", "датчик", "привод",
                        "клапан", "насос", "компрессор", "распределитель", "фильтр"],
            "energy": ["аккумулятор", "генератор", "турбина", "солнечная панель", "топливный элемент",
                       "ветрогенератор", "гидротурбина", "теплообменник", "двигатель", "электромотор"],
            "structures": ["ферма", "рама", "балка", "колонна", "раскос",
                           "стяжка", "профиль", "панель", "плита", "ребро"],
        }

        names = base_names.get(category, base_names["mechanical"])
        name = names[index % len(names)]
        full_name = f"{name}_{subcategory.replace(' ', '_')}_{index}"

        properties = self.PROPERTY_TEMPLATES.get(category, ["физический"])
        selected_props = random.sample(properties, min(3, len(properties)))

        concept_id = self._generate_short_id(category, subcategory, full_name)

        return {
            "id": concept_id,
            "name": full_name.replace('_', ' ').title(),
            "category": category,
            "subcategory": subcategory,
            "properties": selected_props,
            "description": f"{full_name} - компонент категории {category}",
            "functions": self._generate_functions(category, name)
        }

    def _generate_functions(self, category: str, name: str) -> List[str]:
        """Генерирует функции для концепта."""
        functions_map = {
            "mechanical": ["передавать движение", "преобразовывать вращение", "обеспечивать опору"],
            "electrical": ["преобразовывать сигнал", "усиливать мощность", "фильтровать помехи"],
            "aerospace": ["создавать подъемную силу", "обеспечивать устойчивость", "управлять полетом"],
            "materials": ["выдерживать нагрузку", "обеспечивать прочность", "защищать от коррозии"],
            "processes": ["соединять детали", "обрабатывать поверхность", "формировать конструкцию"],
            "systems": ["регулировать параметры", "контролировать процессы", "преобразовывать энергию"],
            "energy": ["генерировать энергию", "накапливать энергию", "преобразовывать энергию"],
            "structures": ["поддерживать конструкцию", "перераспределять нагрузку", "обеспечивать жесткость"],
        }

        funcs = functions_map.get(category, ["выполнять функцию"])
        return random.sample(funcs, min(2, len(funcs)))

    def _add_relations(self, concepts: List[Dict]):
        """
        Добавляет связи между концептами.
        Создает ~2-3 связи на каждый узел.
        """
        print("🔗 Генерация связей...")

        relation_types = [
            ("mechanical", "aerospace", EdgeType.HAS_PART, 0.8),
            ("aerospace", "mechanical", EdgeType.DEPENDS_ON, 0.9),
            ("electrical", "systems", EdgeType.HAS_PART, 0.8),
            ("systems", "electrical", EdgeType.DEPENDS_ON, 0.9),
            ("materials", "mechanical", EdgeType.RELATED_TO, 0.7),
            ("processes", "structures", EdgeType.CAUSES, 0.7),
            ("energy", "systems", EdgeType.CAUSES, 0.8),
            ("structures", "aerospace", EdgeType.HAS_PART, 0.8),
        ]

        edge_counter = 0
        processed_pairs = set()

        # Для каждого концепта создаем связи
        for i, concept in enumerate(concepts):
            # Находим подходящие пары
            candidates = []

            for j, other in enumerate(concepts):
                if i == j:
                    continue

                pair_key = tuple(sorted([concept["id"], other["id"]]))
                if pair_key in processed_pairs:
                    continue

                # Связи внутри категории (похожие концепты)
                if other["category"] == concept["category"]:
                    candidates.append((other, EdgeType.RELATED_TO, 0.6))

                # Связи между категориями
                for cat1, cat2, edge_type, weight in relation_types:
                    if concept["category"] == cat1 and other["category"] == cat2:
                        candidates.append((other, edge_type, weight * 0.9))

            # Берем до 3 случайных кандидатов
            if candidates:
                selected = random.sample(candidates, min(3, len(candidates)))
                for other, edge_type, weight in selected:
                    pair_key = tuple(sorted([concept["id"], other["id"]]))
                    if pair_key in processed_pairs:
                        continue

                    edge_id = self._generate_edge_id(concept["id"], other["id"], edge_type.value)
                    edge = KnowledgeEdge(
                        id=edge_id,
                        source_id=concept["id"],
                        target_id=other["id"],
                        edge_type=edge_type,
                        weight=weight,
                        description=f"{concept['name']} {edge_type.value} {other['name']}"
                    )
                    self.graph.add_edge(edge)
                    processed_pairs.add(pair_key)
                    edge_counter += 1

        print(f"✅ Создано {edge_counter} связей")

    def build_graph(self):
        """Строит полный граф."""
        concepts = self.generate_concepts()

        # Создаем узлы
        print("\n📦 Создание узлов...")
        for concept in concepts:
            node = KnowledgeNode(
                id=concept["id"],
                name=concept["name"],
                node_type=concept["category"],
                properties=concept["properties"],
                description=concept["description"]
            )
            self.graph.add_node(node)

            from core.knowledge.function import Function
            for func in concept.get("functions", []):
                func_obj = Function(
                    id=f"func_{node.id}_{func.replace(' ', '_')[:20]}",
                    name=func,
                    description=f"{node.name} can {func}"
                )
                node.add_function(func_obj)

        print(f"✅ Создано {len(concepts)} узлов")

        # Создаем связи
        self._add_relations(concepts)

        # Сохраняем в БД
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

        return self.graph


def main():
    print("=" * 60)
    print("🏗️ ГЕНЕРАЦИЯ РАСШИРЕННОГО ГРАФА ЗНАНИЙ")
    print("=" * 60)

    generator = KnowledgeGraphGenerator(target_nodes=1000)
    graph = generator.build_graph()

    stats = graph.get_statistics()
    print("\n" + "=" * 60)
    print("📊 СТАТИСТИКА ГРАФА")
    print("=" * 60)
    print(f"   Узлов: {stats['total_nodes']}")
    print(f"   Связей: {stats['total_edges']}")
    print(f"   Типы узлов: {stats['node_types']}")
    print("=" * 60)

    # Проверяем связи
    print("\n🔗 ПРОВЕРКА СВЯЗЕЙ:")
    # Находим узел с наибольшим количеством связей
    max_neighbors = 0
    max_node = None
    for node in graph.nodes.values():
        neighbors = graph.get_neighbors(node.id)
        if len(neighbors) > max_neighbors:
            max_neighbors = len(neighbors)
            max_node = node

    if max_node:
        print(f"\n   Узел с наибольшим количеством связей: {max_node.name}")
        print(f"   Количество связей: {max_neighbors}")
        neighbors = graph.get_neighbors(max_node.id)
        print(f"   Примеры соседей:")
        for n in neighbors[:5]:
            print(f"      → {n.name} ({n.node_type})")


if __name__ == "__main__":
    main()



