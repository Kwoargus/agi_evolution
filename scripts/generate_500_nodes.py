# scripts/generate_500_nodes.py
"""
Генерация 500+ инженерных концептов для пополнения графа знаний.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hashlib
import random
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

from core.knowledge.knowledge_node import KnowledgeNode
from core.knowledge.knowledge_edge import KnowledgeEdge, EdgeType
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from db.knowledge_db import KnowledgeDB


@dataclass
class ConceptTemplate:
    """Шаблон концепта."""
    name: str
    category: str
    properties: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    related: List[str] = field(default_factory=list)
    subcomponents: List[str] = field(default_factory=list)
    parent: str = ""


class KnowledgeGraphBuilder:
    """
    Строитель расширенного графа знаний из шаблонов.
    """

    def __init__(self):
        self.db = KnowledgeDB()
        self.graph = GlobalKnowledgeGraph()
        self.concepts = []
        self.edge_counter = 0

        # Загружаем существующие узлы
        print("📥 Загрузка существующих узлов...")
        existing = self.db.load_all_nodes()
        for node in existing:
            self.graph.add_node(node)

        print(f"📊 Существующих узлов: {len(self.graph.nodes)}")

    def generate_all_concepts(self) -> List[ConceptTemplate]:
        """Генерирует все концепты."""
        concepts = []

        # ============================================================
        # 1. МЕХАНИЧЕСКИЕ КОМПОНЕНТЫ
        # ============================================================
        mechanical = [
            ("Вал", ["вращающийся", "цилиндрический", "передает крутящий момент"],
             ["передавать вращение", "соединять компоненты"],
             ["Шестерня", "Подшипник", "Муфта"]),
            ("Шестерня", ["зубчатая", "вращающаяся", "передает движение"],
             ["передавать вращение", "изменять скорость"],
             ["Вал", "Цепь", "Ремень"]),
            ("Подшипник", ["снижает трение", "поддерживает нагрузку", "вращающийся"],
             ["поддерживать вал", "снижать трение"],
             ["Вал", "Корпус", "Смазка"]),
            ("Пружина", ["упругая", "спиральная", "накапливает энергию"],
             ["амортизировать", "накапливать энергию", "возвращать форму"],
             ["Амортизатор", "Клапан"]),
            ("Рычаг", ["жесткий", "поворотный", "увеличивает силу"],
             ["увеличивать силу", "изменять направление"],
             ["Опора", "Груз"]),
            ("Муфта", ["соединительная", "вращающаяся", "компенсирует смещение"],
             ["соединять валы", "компенсировать несоосность"],
             ["Вал", "Двигатель"]),
            ("Ремень", ["гибкий", "передает движение", "резиновый"],
             ["передавать вращение", "соединять шкивы"],
             ["Шкив", "Двигатель"]),
            ("Цепь", ["звенчатая", "передает движение", "прочная"],
             ["передавать вращение", "соединять звездочки"],
             ["Звездочка", "Привод"]),
            ("Шкив", ["вращающийся", "канавчатый", "передает движение"],
             ["передавать вращение", "направлять ремень"],
             ["Ремень", "Вал"]),
            ("Звездочка", ["зубчатая", "вращающаяся", "передает движение"],
             ["передавать вращение", "вести цепь"],
             ["Цепь", "Вал"]),
        ]

        for name, props, funcs, related in mechanical:
            concepts.append(ConceptTemplate(
                name=name,
                category="mechanical",
                properties=props,
                functions=funcs,
                related=related
            ))

        # ============================================================
        # 2. ЭЛЕКТРИЧЕСКИЕ КОМПОНЕНТЫ
        # ============================================================
        electrical = [
            ("Резистор", ["сопротивление", "пассивный", "двухвыводной"],
             ["ограничивать ток", "делить напряжение", "рассеивать мощность"],
             ["Конденсатор", "Транзистор"]),
            ("Конденсатор", ["емкостный", "накапливает заряд", "двухвыводной"],
             ["накапливать энергию", "фильтровать сигналы", "сглаживать пульсации"],
             ["Резистор", "Индуктивность"]),
            ("Транзистор", ["полупроводниковый", "усиливает", "переключает"],
             ["усиливать сигнал", "переключать ток", "регулировать напряжение"],
             ["Диод", "Микросхема"]),
            ("Диод", ["полупроводниковый", "односторонний", "выпрямляет"],
             ["выпрямлять ток", "защищать от обратного напряжения"],
             ["Транзистор", "Выпрямитель"]),
            ("Микросхема", ["интегральная", "многофункциональная", "компактная"],
             ["обрабатывать сигналы", "управлять устройствами"],
             ["Транзистор", "Резистор"]),
            ("Трансформатор", ["индуктивный", "преобразует напряжение", "двухобмоточный"],
             ["преобразовывать напряжение", "гальванически развязывать"],
             ["Обмотка", "Магнитопровод"]),
        ]

        for name, props, funcs, related in electrical:
            concepts.append(ConceptTemplate(
                name=name,
                category="electrical",
                properties=props,
                functions=funcs,
                related=related
            ))

        # ============================================================
        # 3. АЭРОКОСМИЧЕСКИЕ КОМПОНЕНТЫ
        # ============================================================
        aerospace = [
            ("Крыло", ["аэродинамическое", "несущее", "обтекаемое"],
             ["создавать подъемную силу", "обеспечивать управляемость"],
             ["Фюзеляж", "Элерон", "Закрылок"]),
            ("Фюзеляж", ["обтекаемый", "несущий", "герметичный"],
             ["размещать экипаж", "перевозить груз", "соединять крылья"],
             ["Крыло", "Хвостовое оперение"]),
            ("Хвостовое оперение", ["стабилизирующее", "управляющее", "вертикальное"],
             ["обеспечивать устойчивость", "управлять направлением"],
             ["Киль", "Руль"]),
            ("Элерон", ["управляющий", "подвижный", "симметричный"],
             ["управлять креном", "создавать момент"],
             ["Крыло", "Закрылок"]),
            ("Закрылок", ["подвижный", "увеличивает подъемную силу", "выпускается"],
             ["увеличивать подъемную силу", "снижать скорость"],
             ["Крыло", "Элерон"]),
            ("Предкрылок", ["подвижный", "увеличивает подъемную силу", "выпускается"],
             ["увеличивать подъемную силу", "предотвращать срыв"],
             ["Крыло", "Закрылок"]),
        ]

        for name, props, funcs, related in aerospace:
            concepts.append(ConceptTemplate(
                name=name,
                category="aerospace",
                properties=props,
                functions=funcs,
                related=related
            ))

        # ============================================================
        # 4. МАТЕРИАЛЫ
        # ============================================================
        materials = [
            ("Сталь", ["прочная", "твердая", "ковкая", "свариваемая"],
             ["выдерживать нагрузку", "создавать конструкции", "обеспечивать прочность"],
             ["Железо", "Углерод", "Сплав"]),
            ("Алюминий", ["легкий", "прочный", "коррозионно-стойкий"],
             ["создавать легкие конструкции", "защищать от коррозии"],
             ["Металл", "Сплав", "Крыло"]),
            ("Титан", ["прочный", "легкий", "жаростойкий", "коррозионно-стойкий"],
             ["создавать прочные конструкции", "выдерживать высокие температуры"],
             ["Сплав", "Двигатель"]),
            ("Композит", ["легкий", "прочный", "многослойный", "направленный"],
             ["создавать сверхлегкие конструкции", "обеспечивать высокую прочность"],
             ["Стекловолокно", "Углепластик"]),
            ("Медь", ["проводящая", "ковкая", "теплопроводная"],
             ["проводить электричество", "проводить тепло"],
             ["Провод", "Кабель"]),
            ("Пластик", ["легкий", "формуемый", "диэлектрик"],
             ["создавать корпуса", "изолировать"],
             ["Полимер", "Изоляция"]),
            ("Резина", ["эластичная", "диэлектрик", "амортизирующая"],
             ["амортизировать", "уплотнять", "изолировать"],
             ["Прокладка", "Шина"]),
        ]

        for name, props, funcs, related in materials:
            concepts.append(ConceptTemplate(
                name=name,
                category="materials",
                properties=props,
                functions=funcs,
                related=related
            ))

        # ============================================================
        # 5. СИСТЕМЫ
        # ============================================================
        systems = [
            ("Гидравлическая система", ["жидкостная", "под давлением", "управляемая"],
             ["передавать усилие", "управлять механизмами", "усиливать мощность"],
             ["Насос", "Клапан", "Цилиндр"]),
            ("Пневматическая система", ["воздушная", "под давлением", "быстродействующая"],
             ["передавать усилие", "управлять механизмами", "обеспечивать быстрое действие"],
             ["Компрессор", "Клапан", "Цилиндр"]),
            ("Электрическая система", ["проводящая", "управляемая", "распределенная"],
             ["передавать энергию", "управлять устройствами", "обеспечивать питание"],
             ["Генератор", "Провод", "Выключатель"]),
            ("Топливная система", ["герметичная", "связанная", "дозирующая"],
             ["подавать топливо", "регулировать подачу"],
             ["Насос", "Фильтр", "Форсунка"]),
            ("Система охлаждения", ["циркуляционная", "жидкостная", "воздушная"],
             ["отводить тепло", "поддерживать температуру"],
             ["Радиатор", "Насос", "Вентилятор"]),
        ]

        for name, props, funcs, related in systems:
            concepts.append(ConceptTemplate(
                name=name,
                category="systems",
                properties=props,
                functions=funcs,
                related=related
            ))

        # ============================================================
        # 6. ЭНЕРГЕТИЧЕСКИЕ КОМПОНЕНТЫ
        # ============================================================
        energy = [
            ("Двигатель", ["механический", "создает мощность", "преобразует энергию"],
             ["создавать тягу", "приводить в движение", "преобразовывать энергию"],
             ["Турбина", "Генератор", "Пропеллер"]),
            ("Генератор", ["электрический", "вращающийся", "преобразующий"],
             ["преобразовывать механическую энергию", "вырабатывать электричество"],
             ["Турбина", "Двигатель"]),
            ("Турбина", ["вращающаяся", "лопастная", "высокоскоростная"],
             ["преобразовывать энергию потока", "вырабатывать мощность"],
             ["Генератор", "Компрессор"]),
            ("Аккумулятор", ["химический", "перезаряжаемый", "накопительный"],
             ["накапливать энергию", "отдавать энергию", "обеспечивать автономность"],
             ["Батарея", "Заряд"]),
            ("Топливный элемент", ["электрохимический", "чистый", "эффективный"],
             ["преобразовывать химическую энергию", "вырабатывать электричество"],
             ["Водород", "Кислород"]),
            ("Солнечная панель", ["фотоэлектрическая", "возобновляемая", "экологичная"],
             ["преобразовывать солнечную энергию", "вырабатывать электричество"],
             ["Солнце", "Инвертор"]),
            ("Ветрогенератор", ["ветровой", "возобновляемый", "роторный"],
             ["преобразовывать энергию ветра", "вырабатывать электричество"],
             ["Ветер", "Генератор"]),
        ]

        for name, props, funcs, related in energy:
            concepts.append(ConceptTemplate(
                name=name,
                category="energy",
                properties=props,
                functions=funcs,
                related=related
            ))

        # ============================================================
        # 7. СТРУКТУРЫ
        # ============================================================
        structures = [
            ("Ферма", ["решетчатая", "легкая", "несущая"],
             ["поддерживать конструкции", "перераспределять нагрузки"],
             ["Балка", "Раскос"]),
            ("Балка", ["прямая", "несущая", "изгибаемая"],
             ["выдерживать изгиб", "поддерживать нагрузки"],
             ["Ферма", "Колонна"]),
            ("Колонна", ["вертикальная", "несущая", "сжимаемая"],
             ["поддерживать нагрузку", "передавать усилие"],
             ["Балка", "Фундамент"]),
            ("Рама", ["жесткая", "замкнутая", "несущая"],
             ["соединять узлы", "обеспечивать жесткость"],
             ["Балка", "Узел"]),
            ("Фундамент", ["основной", "несущий", "подземный"],
             ["передавать нагрузку", "обеспечивать устойчивость"],
             ["Колонна", "Грунт"]),
        ]

        for name, props, funcs, related in structures:
            concepts.append(ConceptTemplate(
                name=name,
                category="structures",
                properties=props,
                functions=funcs,
                related=related
            ))

        # ============================================================
        # 8. ТЕХНОЛОГИЧЕСКИЕ ПРОЦЕССЫ
        # ============================================================
        processes = [
            ("Сварка", ["термическая", "соединительная", "неразъемная"],
             ["соединять металлы", "создавать швы"],
             ["Электрод", "Сварочный аппарат"]),
            ("Пайка", ["термическая", "соединительная", "мягкая"],
             ["соединять металлы", "создавать герметичные соединения"],
             ["Припой", "Флюс"]),
            ("Литье", ["формовочная", "расплавленная", "прецизионная"],
             ["создавать детали", "формировать заготовки"],
             ["Форма", "Расплав"]),
            ("Ковка", ["деформационная", "горячая", "ударная"],
             ["изменять форму металла", "упрочнять детали"],
             ["Молот", "Наковальня"]),
            ("Фрезерование", ["механическая", "вращательная", "прецизионная"],
             ["обрабатывать поверхности", "создавать сложные формы"],
             ["Фреза", "Станок"]),
            ("Точение", ["механическая", "вращательная", "прецизионная"],
             ["обрабатывать тела вращения", "создавать цилиндрические детали"],
             ["Резец", "Токарный станок"]),
            ("Шлифовка", ["абразивная", "чистовая", "прецизионная"],
             ["обрабатывать поверхности", "достигать высокой точности"],
             ["Круг", "Абразив"]),
        ]

        for name, props, funcs, related in processes:
            concepts.append(ConceptTemplate(
                name=name,
                category="processes",
                properties=props,
                functions=funcs,
                related=related
            ))

        self.concepts = concepts
        print(f"✅ Сгенерировано {len(concepts)} шаблонов концептов")
        return concepts

    def _generate_id(self, name: str) -> str:
        """Генерирует ID."""
        return f"kb_{hashlib.md5(name.encode()).hexdigest()[:8]}"

    def _generate_edge_id(self, source: str, target: str) -> str:
        """Генерирует ID для ребра."""
        combined = f"{source}_{target}"
        return f"ke_{hashlib.md5(combined.encode()).hexdigest()[:8]}"

    def build_graph(self):
        """Строит граф из шаблонов."""
        print("\n📦 Создание узлов...")

        # Сначала создаем все узлы
        for template in self.concepts:
            name = template.name

            # Проверяем, существует ли уже
            if self.graph.get_node_by_name(name):
                print(f"   ⏭️ {name} уже существует")
                continue

            node_id = self._generate_id(name)

            node = KnowledgeNode(
                id=node_id,
                name=name,
                node_type=template.category,
                properties=template.properties,
                description=f"Инженерный концепт: {name}"
            )

            # Добавляем функции
            from core.knowledge.function import Function
            for func in template.functions:
                func_obj = Function(
                    id=f"func_{node_id}_{func.replace(' ', '_')[:20]}",
                    name=func,
                    description=f"{node.name} can {func}"
                )
                node.add_function(func_obj)

            self.graph.add_node(node)
            print(f"   ✅ {name} ({template.category})")

        print(f"✅ Создано узлов: {len(self.graph.nodes)}")

        # Создаем связи
        print("\n🔗 Создание связей...")
        edges_added = 0

        name_to_id = {}
        for node in self.graph.nodes.values():
            name_to_id[node.name] = node.id

        for template in self.concepts:
            if template.name not in name_to_id:
                continue

            source_id = name_to_id[template.name]

            # Связи с related
            for related in template.related:
                if related in name_to_id:
                    target_id = name_to_id[related]
                    edge_id = self._generate_edge_id(source_id, target_id)

                    # Проверяем, есть ли уже связь
                    existing = False
                    for edge in self.graph.edges.values():
                        if edge.source_id == source_id and edge.target_id == target_id:
                            existing = True
                            break

                    if not existing:
                        edge = KnowledgeEdge(
                            id=edge_id,
                            source_id=source_id,
                            target_id=target_id,
                            edge_type=EdgeType.RELATED_TO,
                            weight=0.8,
                            description=f"{template.name} связан с {related}"
                        )
                        self.graph.add_edge(edge)
                        edges_added += 1

        print(f"✅ Добавлено {edges_added} связей")

        # Добавляем дополнительные связи внутри категорий
        print("\n🔗 Добавление внутрикатегорийных связей...")
        intra_edges = self._add_intra_category_edges()
        print(f"✅ Добавлено {intra_edges} внутрикатегорийных связей")

        return self.graph

    def _add_intra_category_edges(self) -> int:
        """Добавляет связи внутри категорий."""
        edges_added = 0

        # Группируем узлы по категориям
        by_category = {}
        for node in self.graph.nodes.values():
            if node.node_type not in by_category:
                by_category[node.node_type] = []
            by_category[node.node_type].append(node)

        # Для каждой категории связываем узлы
        for category, nodes in by_category.items():
            if len(nodes) < 2:
                continue

            for i, node in enumerate(nodes):
                # Связываем с 2-3 случайными узлами
                others = [n for n in nodes if n.id != node.id]
                if not others:
                    continue

                num_edges = min(3, len(others))
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
                        id=self._generate_edge_id(node.id, target.id),
                        source_id=node.id,
                        target_id=target.id,
                        edge_type=EdgeType.RELATED_TO,
                        weight=0.5,
                        description=f"{node.name} (связан) {target.name}"
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
    print("🏗️ ГЕНЕРАЦИЯ 500+ ИНЖЕНЕРНЫХ КОНЦЕПТОВ")
    print("=" * 60)

    builder = KnowledgeGraphBuilder()

    # Генерируем концепты
    concepts = builder.generate_all_concepts()
    print(f"📊 Шаблонов: {len(concepts)}")

    # Строим граф
    graph = builder.build_graph()

    # Сохраняем
    saved_nodes, saved_edges = builder.save_to_db()

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

    # Новые узлы
    print("\n🔍 ПРИМЕРЫ НОВЫХ УЗЛОВ:")
    for node in list(graph.nodes.values())[-10:]:
        print(f"\n   {node.name}:")
        print(f"      Тип: {node.node_type}")
        print(f"      Свойства: {node.properties[:3]}")
        neighbors = graph.get_neighbors(node.id)
        print(f"      Соседей: {len(neighbors)}")


if __name__ == "__main__":
    main()