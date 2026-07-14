# scripts/generate_1000_nodes.py
"""
Генерация 1000+ инженерных концептов с использованием систематического подхода.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hashlib
import random
import time
from typing import List, Dict, Tuple, Set
from itertools import product

from core.knowledge.knowledge_node import KnowledgeNode
from core.knowledge.knowledge_edge import KnowledgeEdge, EdgeType
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from db.knowledge_db import KnowledgeDB


class MassiveKnowledgeGenerator:
    """
    Генератор 1000+ инженерных концептов.
    Использует комбинаторный подход для создания разнообразных концептов.
    """

    # Базовые корни для комбинаций
    ROOTS = {
        "mechanical": [
            "вал", "шестерня", "подшипник", "муфта", "пружина", "рычаг", "ремень", "цепь",
            "шкив", "звездочка", "винт", "гайка", "болт", "шпонка", "штифт", "палец",
            "ось", "ступица", "маховик", "шкив", "кулачок", "ползун", "кривошип", "шатун",
            "поршень", "цилиндр", "клапан", "золотник", "дроссель", "редуктор", "мультипликатор",
            "преобразователь", "усилитель", "демпфер", "амортизатор"
        ],
        "electrical": [
            "резистор", "конденсатор", "транзистор", "диод", "микросхема", "трансформатор",
            "индуктор", "тиристор", "оптрон", "кварцевый резонатор", "фоторезистор", "термистор",
            "варистор", "стабилитрон", "светодиод", "фотодиод", "лазер", "пьезоэлемент",
            "датчик", "акселерометр", "гироскоп", "магнитометр", "барометр", "гигрометр"
        ],
        "aerospace": [
            "крыло", "фюзеляж", "стабилизатор", "элерон", "закрылок", "предкрылок",
            "руль", "лонжерон", "нервюра", "стрингер", "обшивка", "пилон", "кессон",
            "спарка", "элерон", "интерцептор", "спойлер", "воздушный тормоз", "киль",
            "шпангоут", "стрингер", "обтекатель", "зализ", "носовой обтекатель"
        ],
        "materials": [
            "сталь", "алюминий", "титан", "медь", "пластик", "композит", "керамика",
            "стекло", "резина", "полимер", "сплав", "бронза", "латунь", "свинец",
            "цинк", "никель", "хром", "марганец", "кремний", "графит", "алмаз", "карбид"
        ],
        "systems": [
            "гидравлическая", "пневматическая", "электрическая", "механическая", "оптическая",
            "термальная", "акустическая", "электронная", "цифровая", "аналоговая", "смешанная",
            "инерциальная", "спутниковая", "радиолокационная", "инфракрасная", "ультразвуковая"
        ],
        "energy": [
            "двигатель", "генератор", "турбина", "аккумулятор", "топливный элемент",
            "солнечная панель", "ветрогенератор", "гидротурбина", "теплообменник",
            "компрессор", "насос", "вентилятор", "нагреватель", "охладитель", "конденсатор",
            "испаритель", "рекуператор", "адсорбер", "абсорбер"
        ],
        "structures": [
            "ферма", "балка", "колонна", "рама", "фундамент", "плита", "стенка",
            "перекрытие", "покрытие", "связь", "раскос", "узел", "опора", "консоль",
            "арка", "свод", "купол", "башня", "мачта", "пилон"
        ],
        "processes": [
            "сварка", "пайка", "литье", "ковка", "фрезерование", "точение", "шлифовка",
            "полировка", "наплавка", "напыление", "термообработка", "закалка", "отпуск",
            "отжиг", "нормализация", "цементация", "азотирование", "хромирование", "оцинковка"
        ],
        "electronics": [
            "микроконтроллер", "процессор", "память", "интерфейс", "шина", "порт",
            "разъем", "переключатель", "предохранитель", "реле", "контактор", "пускатель",
            "контроллер", "регулятор", "стабилизатор", "преобразователь", "инвертор",
            "выпрямитель", "фильтр", "усилитель", "аттенюатор"
        ],
        "robotics": [
            "манипулятор", "захват", "шарнир", "привод", "сервопривод", "шаговый двигатель",
            "датчик", "камера", "лидар", "сонар", "радар", "гироскоп", "акселерометр",
            "инклинометр", "одометр", "энкодер", "резольвер", "таходатчик"
        ]
    }

    # Префиксы для создания вариаций
    PREFIXES = [
        "двойной", "тройной", "много", "комбинированный", "универсальный", "специальный",
        "быстрый", "медленный", "легкий", "тяжелый", "большой", "малый", "компактный",
        "мощный", "слабый", "точный", "грубый", "гибкий", "жесткий", "надежный",
        "простой", "сложный", "автоматический", "ручной", "цифровой", "аналоговый"
    ]

    # Суффиксы для создания вариаций
    SUFFIXES = [
        "механизм", "устройство", "система", "агрегат", "блок", "модуль", "компонент",
        "узел", "деталь", "элемент", "прибор", "аппарат", "установка", "машина"
    ]

    # Ключевые слова для расширения
    EXTRA_KEYWORDS = [
        "регулируемый", "управляемый", "настраиваемый", "программируемый", "интеллектуальный",
        "автономный", "адаптивный", "самонастраивающийся", "самодиагностируемый", "самовосстанавливающийся"
    ]

    def __init__(self):
        self.db = KnowledgeDB()
        self.graph = GlobalKnowledgeGraph()
        self.existing_names: Set[str] = set()
        self.generated_count = 0

        # Загружаем существующие узлы
        print("📥 Загрузка существующих узлов...")
        existing_nodes = self.db.load_all_nodes()
        for node in existing_nodes:
            self.graph.add_node(node)
            self.existing_names.add(node.name)

        print(f"📊 Существующих узлов: {len(self.graph.nodes)}")
        print(f"📊 Существующих имен: {len(self.existing_names)}")

    def _generate_id(self, name: str) -> str:
        """Генерирует ID."""
        hash_obj = hashlib.md5(name.encode())
        return f"mass_{hash_obj.hexdigest()[:8]}"

    def _generate_edge_id(self, source: str, target: str) -> str:
        """Генерирует ID для ребра."""
        combined = f"{source}_{target}"
        return f"me_{hashlib.md5(combined.encode()).hexdigest()[:8]}"

    def _normalize_name(self, name: str) -> str:
        """Нормализует имя для сравнения."""
        name = name.lower().strip()
        name = name.replace("  ", " ")
        return name

    def _is_duplicate(self, name: str) -> bool:
        """Проверяет, существует ли уже такое имя."""
        normalized = self._normalize_name(name)
        for existing in self.existing_names:
            if self._normalize_name(existing) == normalized:
                return True
        return False

    def _add_name(self, name: str):
        """Добавляет имя в множество существующих."""
        self.existing_names.add(name)

    def generate_root_concepts(self):
        """Генерирует концепты из корней."""
        print("\n🌱 Генерация базовых концептов...")
        concepts = []

        for category, roots in self.ROOTS.items():
            for root in roots:
                name = root.capitalize()
                if not self._is_duplicate(name):
                    concepts.append({
                        "name": name,
                        "category": category,
                        "properties": [f"{category}_property_{random.randint(1, 5)}"],
                        "functions": [f"{category}_function_{random.randint(1, 3)}"]
                    })
                    self._add_name(name)
                    self.generated_count += 1

        print(f"   ✅ Создано {len(concepts)} базовых концептов")
        return concepts

    def generate_prefixed_variations(self):
        """Генерирует вариации с префиксами."""
        print("\n🔤 Генерация вариаций с префиксами...")
        concepts = []

        # Берем существующие концепты как основу
        base_names = list(self.ROOTS.keys())
        for category in base_names:
            for root in self.ROOTS.get(category, []):
                for prefix in random.sample(self.PREFIXES, min(5, len(self.PREFIXES))):
                    name = f"{prefix} {root}".capitalize()
                    if not self._is_duplicate(name):
                        concepts.append({
                            "name": name,
                            "category": category,
                            "properties": [f"{prefix}_modified", f"{category}_enhanced"],
                            "functions": [f"improved_{category}_function", f"enhanced_performance"]
                        })
                        self._add_name(name)
                        self.generated_count += 1

        print(f"   ✅ Создано {len(concepts)} вариаций с префиксами")
        return concepts

    def generate_compound_concepts(self):
        """Генерирует составные концепты."""
        print("\n🧩 Генерация составных концептов...")
        concepts = []

        categories = list(self.ROOTS.keys())
        for _ in range(200):  # Генерируем 200 составных концептов
            # Берем два случайных корня из разных категорий
            cat1 = random.choice(categories)
            cat2 = random.choice(categories)
            root1 = random.choice(self.ROOTS.get(cat1, ["устройство"]))
            root2 = random.choice(self.ROOTS.get(cat2, ["система"]))

            # Соединяем их
            connectors = [" с ", " и ", " на основе ", " с использованием "]
            connector = random.choice(connectors)
            name = f"{root1.capitalize()}{connector}{root2}".capitalize()

            if not self._is_duplicate(name) and len(name) < 60:
                concepts.append({
                    "name": name,
                    "category": "combined",
                    "properties": [f"{cat1}_property", f"{cat2}_property", "integrated"],
                    "functions": [f"combine_{cat1}_and_{cat2}", "integrated_function"]
                })
                self._add_name(name)
                self.generated_count += 1

        print(f"   ✅ Создано {len(concepts)} составных концептов")
        return concepts

    def generate_suffix_concepts(self):
        """Генерирует концепты с суффиксами."""
        print("\n📦 Генерация концептов с суффиксами...")
        concepts = []

        # Берем базовые корни и добавляем суффиксы
        for category, roots in self.ROOTS.items():
            for root in random.sample(roots, min(10, len(roots))):
                for suffix in random.sample(self.SUFFIXES, min(3, len(self.SUFFIXES))):
                    name = f"{root} {suffix}".capitalize()
                    if not self._is_duplicate(name):
                        concepts.append({
                            "name": name,
                            "category": category,
                            "properties": [f"{category}_with_suffix", "complete_system"],
                            "functions": [f"function_with_{suffix}", "integrated_operation"]
                        })
                        self._add_name(name)
                        self.generated_count += 1

        print(f"   ✅ Создано {len(concepts)} концептов с суффиксами")
        return concepts

    def generate_specialized_concepts(self):
        """Генерирует специализированные концепты с ключевыми словами."""
        print("\n🎯 Генерация специализированных концептов...")
        concepts = []

        for category, roots in self.ROOTS.items():
            for root in random.sample(roots, min(8, len(roots))):
                for keyword in random.sample(self.EXTRA_KEYWORDS, min(4, len(self.EXTRA_KEYWORDS))):
                    name = f"{keyword} {root}".capitalize()
                    if not self._is_duplicate(name):
                        concepts.append({
                            "name": name,
                            "category": category,
                            "properties": [keyword, "advanced", "smart"],
                            "functions": [f"{keyword}_function", f"adaptive_operation"]
                        })
                        self._add_name(name)
                        self.generated_count += 1

        print(f"   ✅ Создано {len(concepts)} специализированных концептов")
        return concepts

    def generate_random_combinations(self):
        """Генерирует случайные комбинации."""
        print("\n🎲 Генерация случайных комбинаций...")
        concepts = []

        # Создаем список всех корней
        all_roots = []
        for roots in self.ROOTS.values():
            all_roots.extend(roots)

        # Генерируем случайные комбинации
        for _ in range(200):
            root1 = random.choice(all_roots)
            root2 = random.choice(all_roots)

            if random.random() > 0.5:
                name = f"{root1}-{root2}".capitalize()
            else:
                name = f"{root1} с {root2}".capitalize()

            if not self._is_duplicate(name) and len(name) < 50:
                concepts.append({
                    "name": name,
                    "category": "mixed",
                    "properties": ["комбинированный", "гибридный", "универсальный"],
                    "functions": ["комбинировать функции", "универсальное применение"]
                })
                self._add_name(name)
                self.generated_count += 1

        print(f"   ✅ Создано {len(concepts)} случайных комбинаций")
        return concepts

    def generate_all_concepts(self) -> List[Dict]:
        """Генерирует все концепты."""
        all_concepts = []

        all_concepts.extend(self.generate_root_concepts())
        all_concepts.extend(self.generate_prefixed_variations())
        all_concepts.extend(self.generate_compound_concepts())
        all_concepts.extend(self.generate_suffix_concepts())
        all_concepts.extend(self.generate_specialized_concepts())
        all_concepts.extend(self.generate_random_combinations())

        print(f"\n📊 Всего сгенерировано: {len(all_concepts)} уникальных концептов")
        return all_concepts

    def build_graph(self, concepts: List[Dict]) -> GlobalKnowledgeGraph:
        """Строит граф из концептов."""
        print("\n📦 Создание узлов...")

        added = 0
        for concept in concepts:
            name = concept["name"]
            node_id = self._generate_id(name)

            node = KnowledgeNode(
                id=node_id,
                name=name,
                node_type=concept.get("category", "concept"),
                properties=concept.get("properties", []),
                description=f"Массово сгенерированный концепт: {name}"
            )

            node.metadata = {
                "source": "mass_generation",
                "generation_time": time.time(),
                "concept_type": "generated"
            }

            self.graph.add_node(node)
            added += 1

        print(f"✅ Добавлено {added} новых узлов")

        # Создаем связи
        print("🔗 Создание связей...")
        edges_added = self._create_edges()
        print(f"✅ Добавлено {edges_added} связей")

        return self.graph

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

            # Каждый узел связываем с 2-3 случайными
            for node in type_nodes:
                others = [n for n in type_nodes if n.id != node.id]
                if not others:
                    continue

                num_edges = min(3, len(others))
                selected = random.sample(others, num_edges)

                for target in selected:
                    # Проверяем существование связи
                    existing = False
                    for edge in self.graph.edges.values():
                        if edge.source_id == node.id and edge.target_id == target.id:
                            existing = True
                            break

                    if existing:
                        continue

                    edge = KnowledgeEdge(
                        id=self._generate_edge_id(node.id, target.id),
                        source_id=node.id,
                        target_id=target.id,
                        edge_type=EdgeType.RELATED_TO,
                        weight=0.5,
                        description=f"{node.name} связан с {target.name}"
                    )
                    self.graph.add_edge(edge)
                    edges_added += 1

        # Межкатегорийные связи
        print("   🔗 Межкатегорийные связи...")
        categories = list(by_type.keys())
        for _ in range(len(nodes) * 2):
            if len(categories) < 2:
                break

            cat1 = random.choice(categories)
            cat2 = random.choice(categories)

            if cat1 == cat2 or cat1 not in by_type or cat2 not in by_type:
                continue

            node1 = random.choice(by_type[cat1])
            node2 = random.choice(by_type[cat2])

            # Проверяем существование связи
            existing = False
            for edge in self.graph.edges.values():
                if edge.source_id == node1.id and edge.target_id == node2.id:
                    existing = True
                    break

            if existing:
                continue

            edge = KnowledgeEdge(
                id=self._generate_edge_id(node1.id, node2.id),
                source_id=node1.id,
                target_id=node2.id,
                edge_type=EdgeType.RELATED_TO,
                weight=0.4,
                description=f"Междисциплинарная: {node1.name} → {node2.name}"
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
        return saved_nodes, saved_edges


def main():
    print("=" * 60)
    print("🚀 МАССОВАЯ ГЕНЕРАЦИЯ 1000+ КОНЦЕПТОВ")
    print("=" * 60)

    generator = MassiveKnowledgeGenerator()

    # Генерируем все концепты
    concepts = generator.generate_all_concepts()

    if not concepts:
        print("⚠️ Не удалось сгенерировать концепты")
        return

    # Строим граф
    graph = generator.build_graph(concepts)

    # Сохраняем
    saved_nodes, saved_edges = generator.save_to_db()

    # Статистика
    stats = graph.get_statistics()
    print("\n" + "=" * 60)
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)
    print(f"   Всего узлов в БД: {stats['total_nodes']}")
    print(f"   Всего связей в БД: {stats['total_edges']}")
    print(f"   Типы узлов:")
    for node_type, count in stats['node_types'].items():
        print(f"      - {node_type}: {count}")
    print("=" * 60)

    # Показываем примеры новых узлов
    print("\n🔍 ПРИМЕРЫ НОВЫХ УЗЛОВ (последние 10):")
    all_nodes = list(graph.nodes.values())
    for node in all_nodes[-10:]:
        if node.metadata.get("source") == "mass_generation":
            print(f"\n   {node.name}:")
            print(f"      Тип: {node.node_type}")
            print(f"      Свойства: {node.properties[:3]}")
            neighbors = graph.get_neighbors(node.id)
            print(f"      Соседей: {len(neighbors)}")


if __name__ == "__main__":
    main()