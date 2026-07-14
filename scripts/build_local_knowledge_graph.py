# scripts/build_local_knowledge_graph.py
"""
Создает локальный граф знаний без внешних API.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
from core.knowledge.knowledge_node import KnowledgeNode
from core.knowledge.knowledge_edge import KnowledgeEdge, EdgeType
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from db.knowledge_db import KnowledgeDB


class LocalKnowledgeBuilder:
    """
    Строитель локального графа знаний.
    """

    def __init__(self):
        self.graph = GlobalKnowledgeGraph()
        self.db = KnowledgeDB()

    def build(self):
        """Строит граф знаний."""
        print("=" * 60)
        print("🏗️ ПОСТРОЕНИЕ ЛОКАЛЬНОГО ГРАФА ЗНАНИЙ")
        print("=" * 60)

        # 1. Создаем узлы
        print("\n📦 Создание узлов...")
        self._create_nodes()

        # 2. Создаем связи
        print("\n🔗 Создание связей...")
        self._create_edges()

        # 3. Сохраняем в БД
        print("\n💾 Сохранение в БД...")
        self.graph.save_to_db()

        # 4. Выводим статистику
        stats = self.graph.get_statistics()
        print("\n" + "=" * 60)
        print("📊 СТАТИСТИКА ГРАФА ЗНАНИЙ")
        print("=" * 60)
        print(f"   Узлов: {stats['total_nodes']}")
        print(f"   Связей: {stats['total_edges']}")
        print(f"   Типы узлов: {stats['node_types']}")
        print("=" * 60)

        return self.graph

    def _create_nodes(self):
        """Создает узлы."""

        # Определяем данные узлов
        nodes_data = {
            # Физика и аэродинамика
            "air": {
                "type": "physical_object",
                "properties": ["gas", "fluid", "exerts_pressure", "invisible"],
                "desc": "Газообразное вещество, окружающее Землю"
            },
            "lift": {
                "type": "physical_concept",
                "properties": ["aerodynamic", "force", "upward"],
                "desc": "Аэродинамическая сила, направленная вверх"
            },
            "drag": {
                "type": "physical_concept",
                "properties": ["aerodynamic", "force", "resistance"],
                "desc": "Сила сопротивления движению в воздухе"
            },
            "thrust": {
                "type": "physical_concept",
                "properties": ["force", "forward", "propulsion"],
                "desc": "Сила, двигающая объект вперед"
            },

            # Материалы
            "metal": {
                "type": "material",
                "properties": ["solid", "conductive", "malleable", "strong"],
                "desc": "Твердый материал с высокой прочностью"
            },
            "wood": {
                "type": "material",
                "properties": ["solid", "light", "flammable", "workable"],
                "desc": "Природный материал из древесины"
            },

            # Аэродинамика
            "wing": {
                "type": "physical_object",
                "properties": ["aerodynamic", "curved", "creates_lift"],
                "desc": "Аэродинамическая поверхность для создания подъемной силы"
            },
            "airfoil": {
                "type": "physical_object",
                "properties": ["aerodynamic", "curved", "streamlined"],
                "desc": "Профиль крыла с особым изгибом"
            },

            # Двигатели
            "engine": {
                "type": "system",
                "properties": ["mechanical", "power_producing", "fuel_consuming"],
                "desc": "Устройство для преобразования энергии в механическую работу"
            },
            "propeller": {
                "type": "physical_object",
                "properties": ["rotating", "bladed", "thrust_producing"],
                "desc": "Устройство для создания тяги за счет вращения лопастей"
            },
            "piston": {
                "type": "physical_object",
                "properties": ["moving", "cylindrical", "compression"],
                "desc": "Деталь двигателя, совершающая возвратно-поступательное движение"
            },

            # Механизмы
            "wheel": {
                "type": "physical_object",
                "properties": ["rotating", "round", "load_bearing"],
                "desc": "Устройство для перемещения грузов"
            },
            "gear": {
                "type": "physical_object",
                "properties": ["toothed", "rotating", "transmits_power"],
                "desc": "Зубчатое колесо для передачи вращения"
            },
            "lever": {
                "type": "physical_object",
                "properties": ["rigid", "pivoting", "multiplies_force"],
                "desc": "Простой механизм для увеличения силы"
            },
            "spring": {
                "type": "physical_object",
                "properties": ["elastic", "stores_energy", "coiled"],
                "desc": "Упругий элемент для накопления энергии"
            },

            # Летательные аппараты
            "airplane": {
                "type": "system",
                "properties": ["flying", "fixed_wing", "heavier_than_air"],
                "desc": "Летательный аппарат тяжелее воздуха с неподвижным крылом"
            },
            "helicopter": {
                "type": "system",
                "properties": ["flying", "rotary_wing", "vertical_takeoff"],
                "desc": "Летательный аппарат с вращающимися крыльями"
            },
            "glider": {
                "type": "system",
                "properties": ["flying", "fixed_wing", "no_engine"],
                "desc": "Планер - летательный аппарат без двигателя"
            },
            "kite": {
                "type": "physical_object",
                "properties": ["tethered", "wind_powered", "heavier_than_air"],
                "desc": "Устройство, парящее в воздухе за счет набегающего потока"
            },
            "balloon": {
                "type": "physical_object",
                "properties": ["lighter_than_air", "gas_filled", "floats"],
                "desc": "Устройство, поднимающееся за счет газа легче воздуха"
            },
        }

        # Создаем узлы
        for name, data in nodes_data.items():
            node = KnowledgeNode(
                id=f"kb_{name}",
                name=name,
                node_type=data["type"],
                properties=data["properties"],
                description=data["desc"]
            )
            self.graph.add_node(node)
            print(f"   ✅ {name}: {data['type']}")

    def _create_edges(self):
        """Создает связи между узлами."""

        # Определяем связи (заменяем USES на RELATED_TO или CAUSES)
        edges_data = [
            # Аэродинамические связи
            ("wing", "air", EdgeType.RELATED_TO, 0.9, "Крыло взаимодействует с воздухом"),
            ("wing", "lift", EdgeType.CAUSES, 0.9, "Крыло создает подъемную силу"),
            ("wing", "airfoil", EdgeType.INSTANCE_OF, 0.8, "Крыло имеет профиль"),

            # Двигатель и пропеллер
            ("engine", "propeller", EdgeType.CONTAINS, 0.9, "Двигатель вращает пропеллер"),
            ("engine", "piston", EdgeType.CONTAINS, 0.8, "Двигатель имеет поршень"),
            ("propeller", "air", EdgeType.RELATED_TO, 0.9, "Пропеллер взаимодействует с воздухом"),
            ("propeller", "thrust", EdgeType.CAUSES, 0.9, "Пропеллер создает тягу"),

            # Самолет
            ("airplane", "wing", EdgeType.HAS_PART, 0.9, "У самолета есть крылья"),
            ("airplane", "engine", EdgeType.HAS_PART, 0.9, "У самолета есть двигатель"),
            ("airplane", "propeller", EdgeType.HAS_PART, 0.8, "У самолета есть пропеллер"),
            ("airplane", "lift", EdgeType.DEPENDS_ON, 0.9, "Самолет зависит от подъемной силы"),
            ("airplane", "thrust", EdgeType.DEPENDS_ON, 0.9, "Самолет зависит от тяги"),
            ("airplane", "air", EdgeType.RELATED_TO, 0.9, "Самолет летает в воздухе"),

            # Вертолет
            ("helicopter", "wing", EdgeType.HAS_PART, 0.8, "У вертолета есть несущий винт"),
            ("helicopter", "engine", EdgeType.HAS_PART, 0.9, "У вертолета есть двигатель"),

            # Планер
            ("glider", "wing", EdgeType.HAS_PART, 0.9, "У планера есть крылья"),
            ("glider", "air", EdgeType.RELATED_TO, 0.9, "Планер летает в воздухе"),

            # Воздушный змей
            ("kite", "air", EdgeType.RELATED_TO, 0.9, "Змей летает в воздухе"),
            ("kite", "lift", EdgeType.CAUSES, 0.8, "Змей создает подъемную силу"),

            # Воздушный шар
            ("balloon", "air", EdgeType.RELATED_TO, 0.9, "Шар летает в воздухе"),

            # Механизмы
            ("wheel", "lever", EdgeType.RELATED_TO, 0.7, "Колесо и рычаг - простые механизмы"),
            ("gear", "wheel", EdgeType.RELATED_TO, 0.8, "Шестерня - вид колеса"),
            ("spring", "engine", EdgeType.RELATED_TO, 0.6, "Пружина используется в двигателях"),
        ]

        # Создаем связи
        for source, target, edge_type, weight, desc in edges_data:
            source_id = f"kb_{source}"
            target_id = f"kb_{target}"

            # Проверяем, что оба узла существуют
            if self.graph.get_node(source_id) and self.graph.get_node(target_id):
                edge = KnowledgeEdge(
                    id=f"edge_{source}_{target}",
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=edge_type,
                    weight=weight,
                    description=desc
                )
                self.graph.add_edge(edge)
                print(f"   ✅ {source} → {target} ({edge_type.value})")


def main():
    builder = LocalKnowledgeBuilder()
    graph = builder.build()

    print("\n🔍 ПРИМЕРЫ УЗЛОВ:")
    for name in ["air", "wing", "airplane", "engine"]:
        node = graph.get_node(f"kb_{name}")
        if node:
            print(f"\n   {name.upper()}:")
            print(f"      Тип: {node.node_type}")
            print(f"      Свойства: {node.properties[:5]}")

    print("\n🔗 ПРИМЕРЫ СВЯЗЕЙ:")
    airplane = graph.get_node("kb_airplane")
    if airplane:
        neighbors = graph.get_neighbors("kb_airplane")
        print(f"\n   Соседи 'airplane':")
        for neighbor in neighbors[:5]:
            print(f"      → {neighbor.name}")


if __name__ == "__main__":
    main()






# # scripts/build_local_knowledge_graph.py
# """
# Создает локальный граф знаний без внешних API.
# """
#
# import sys
# import os
#
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# from typing import Dict, Any
# from core.knowledge.knowledge_node import KnowledgeNode
# from core.knowledge.knowledge_edge import KnowledgeEdge, EdgeType
# from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
# from db.knowledge_db import KnowledgeDB
#
#
# class LocalKnowledgeBuilder:
#     """
#     Строитель локального графа знаний.
#     """
#
#     def __init__(self):
#         self.graph = GlobalKnowledgeGraph()
#         self.db = KnowledgeDB()
#
#     def build(self):
#         """Строит граф знаний."""
#         print("=" * 60)
#         print("🏗️ ПОСТРОЕНИЕ ЛОКАЛЬНОГО ГРАФА ЗНАНИЙ")
#         print("=" * 60)
#
#         # 1. Создаем узлы
#         print("\n📦 Создание узлов...")
#         self._create_nodes()
#
#         # 2. Создаем связи
#         print("\n🔗 Создание связей...")
#         self._create_edges()
#
#         # 3. Сохраняем в БД
#         print("\n💾 Сохранение в БД...")
#         self.graph.save_to_db()
#
#         # 4. Выводим статистику
#         stats = self.graph.get_statistics()
#         print("\n" + "=" * 60)
#         print("📊 СТАТИСТИКА ГРАФА ЗНАНИЙ")
#         print("=" * 60)
#         print(f"   Узлов: {stats['total_nodes']}")
#         print(f"   Связей: {stats['total_edges']}")
#         print(f"   Типы узлов: {stats['node_types']}")
#         print("=" * 60)
#
#         return self.graph
#
#     def _create_nodes(self):
#         """Создает узлы."""
#
#         # Определяем данные узлов
#         nodes_data = {
#             # Физика и аэродинамика
#             "air": {
#                 "type": "physical_object",
#                 "properties": ["gas", "fluid", "exerts_pressure", "invisible"],
#                 "desc": "Газообразное вещество, окружающее Землю"
#             },
#             "lift": {
#                 "type": "physical_concept",
#                 "properties": ["aerodynamic", "force", "upward"],
#                 "desc": "Аэродинамическая сила, направленная вверх"
#             },
#             "drag": {
#                 "type": "physical_concept",
#                 "properties": ["aerodynamic", "force", "resistance"],
#                 "desc": "Сила сопротивления движению в воздухе"
#             },
#             "thrust": {
#                 "type": "physical_concept",
#                 "properties": ["force", "forward", "propulsion"],
#                 "desc": "Сила, двигающая объект вперед"
#             },
#
#             # Материалы
#             "metal": {
#                 "type": "material",
#                 "properties": ["solid", "conductive", "malleable", "strong"],
#                 "desc": "Твердый материал с высокой прочностью"
#             },
#             "wood": {
#                 "type": "material",
#                 "properties": ["solid", "light", "flammable", "workable"],
#                 "desc": "Природный материал из древесины"
#             },
#
#             # Аэродинамика
#             "wing": {
#                 "type": "physical_object",
#                 "properties": ["aerodynamic", "curved", "creates_lift"],
#                 "desc": "Аэродинамическая поверхность для создания подъемной силы"
#             },
#             "airfoil": {
#                 "type": "physical_object",
#                 "properties": ["aerodynamic", "curved", "streamlined"],
#                 "desc": "Профиль крыла с особым изгибом"
#             },
#
#             # Двигатели
#             "engine": {
#                 "type": "system",
#                 "properties": ["mechanical", "power_producing", "fuel_consuming"],
#                 "desc": "Устройство для преобразования энергии в механическую работу"
#             },
#             "propeller": {
#                 "type": "physical_object",
#                 "properties": ["rotating", "bladed", "thrust_producing"],
#                 "desc": "Устройство для создания тяги за счет вращения лопастей"
#             },
#             "piston": {
#                 "type": "physical_object",
#                 "properties": ["moving", "cylindrical", "compression"],
#                 "desc": "Деталь двигателя, совершающая возвратно-поступательное движение"
#             },
#
#             # Механизмы
#             "wheel": {
#                 "type": "physical_object",
#                 "properties": ["rotating", "round", "load_bearing"],
#                 "desc": "Устройство для перемещения грузов"
#             },
#             "gear": {
#                 "type": "physical_object",
#                 "properties": ["toothed", "rotating", "transmits_power"],
#                 "desc": "Зубчатое колесо для передачи вращения"
#             },
#             "lever": {
#                 "type": "physical_object",
#                 "properties": ["rigid", "pivoting", "multiplies_force"],
#                 "desc": "Простой механизм для увеличения силы"
#             },
#             "spring": {
#                 "type": "physical_object",
#                 "properties": ["elastic", "stores_energy", "coiled"],
#                 "desc": "Упругий элемент для накопления энергии"
#             },
#
#             # Летательные аппараты
#             "airplane": {
#                 "type": "system",
#                 "properties": ["flying", "fixed_wing", "heavier_than_air"],
#                 "desc": "Летательный аппарат тяжелее воздуха с неподвижным крылом"
#             },
#             "helicopter": {
#                 "type": "system",
#                 "properties": ["flying", "rotary_wing", "vertical_takeoff"],
#                 "desc": "Летательный аппарат с вращающимися крыльями"
#             },
#             "glider": {
#                 "type": "system",
#                 "properties": ["flying", "fixed_wing", "no_engine"],
#                 "desc": "Планер - летательный аппарат без двигателя"
#             },
#             "kite": {
#                 "type": "physical_object",
#                 "properties": ["tethered", "wind_powered", "heavier_than_air"],
#                 "desc": "Устройство, парящее в воздухе за счет набегающего потока"
#             },
#             "balloon": {
#                 "type": "physical_object",
#                 "properties": ["lighter_than_air", "gas_filled", "floats"],
#                 "desc": "Устройство, поднимающееся за счет газа легче воздуха"
#             },
#         }
#
#         # Создаем узлы
#         for name, data in nodes_data.items():
#             node = KnowledgeNode(
#                 id=f"kb_{name}",
#                 name=name,
#                 node_type=data["type"],
#                 properties=data["properties"],
#                 description=data["desc"]
#             )
#             self.graph.add_node(node)
#             print(f"   ✅ {name}: {data['type']}")
#
#     def _create_edges(self):
#         """Создает связи между узлами."""
#
#         # Определяем связи
#         edges_data = [
#             # Аэродинамические связи
#             ("wing", "air", EdgeType.RELATED_TO, 0.9, "Крыло взаимодействует с воздухом"),
#             ("wing", "lift", EdgeType.CAUSES, 0.9, "Крыло создает подъемную силу"),
#             ("wing", "airfoil", EdgeType.INSTANCE_OF, 0.8, "Крыло имеет профиль"),
#
#             # Двигатель и пропеллер
#             ("engine", "propeller", EdgeType.CONTAINS, 0.9, "Двигатель вращает пропеллер"),
#             ("engine", "piston", EdgeType.CONTAINS, 0.8, "Двигатель имеет поршень"),
#             ("propeller", "air", EdgeType.RELATED_TO, 0.9, "Пропеллер взаимодействует с воздухом"),
#             ("propeller", "thrust", EdgeType.CAUSES, 0.9, "Пропеллер создает тягу"),
#
#             # Самолет
#             ("airplane", "wing", EdgeType.PART_OF, 0.9, "У самолета есть крылья"),
#             ("airplane", "engine", EdgeType.PART_OF, 0.9, "У самолета есть двигатель"),
#             ("airplane", "propeller", EdgeType.PART_OF, 0.8, "У самолета есть пропеллер"),
#             ("airplane", "lift", EdgeType.USES, 0.9, "Самолет использует подъемную силу"),
#             ("airplane", "thrust", EdgeType.USES, 0.9, "Самолет использует тягу"),
#             ("airplane", "air", EdgeType.RELATED_TO, 0.9, "Самолет летает в воздухе"),
#
#             # Вертолет
#             ("helicopter", "wing", EdgeType.PART_OF, 0.8, "У вертолета есть несущий винт"),
#             ("helicopter", "engine", EdgeType.PART_OF, 0.9, "У вертолета есть двигатель"),
#
#             # Планер
#             ("glider", "wing", EdgeType.PART_OF, 0.9, "У планера есть крылья"),
#             ("glider", "air", EdgeType.RELATED_TO, 0.9, "Планер летает в воздухе"),
#
#             # Воздушный змей
#             ("kite", "air", EdgeType.RELATED_TO, 0.9, "Змей летает в воздухе"),
#             ("kite", "lift", EdgeType.CAUSES, 0.8, "Змей создает подъемную силу"),
#
#             # Воздушный шар
#             ("balloon", "air", EdgeType.RELATED_TO, 0.9, "Шар летает в воздухе"),
#
#             # Механизмы
#             ("wheel", "lever", EdgeType.RELATED_TO, 0.7, "Колесо и рычаг - простые механизмы"),
#             ("gear", "wheel", EdgeType.RELATED_TO, 0.8, "Шестерня - вид колеса"),
#             ("spring", "engine", EdgeType.RELATED_TO, 0.6, "Пружина используется в двигателях"),
#         ]
#
#         # Создаем связи
#         for source, target, edge_type, weight, desc in edges_data:
#             source_id = f"kb_{source}"
#             target_id = f"kb_{target}"
#
#             # Проверяем, что оба узла существуют
#             if self.graph.get_node(source_id) and self.graph.get_node(target_id):
#                 edge = KnowledgeEdge(
#                     id=f"edge_{source}_{target}",
#                     source_id=source_id,
#                     target_id=target_id,
#                     edge_type=edge_type,
#                     weight=weight,
#                     description=desc
#                 )
#                 self.graph.add_edge(edge)
#                 print(f"   ✅ {source} → {target} ({edge_type.value})")
#
#
# def main():
#     builder = LocalKnowledgeBuilder()
#     graph = builder.build()
#
#     print("\n🔍 ПРИМЕРЫ УЗЛОВ:")
#     for name in ["air", "wing", "airplane", "engine"]:
#         node = graph.get_node(f"kb_{name}")
#         if node:
#             print(f"\n   {name.upper()}:")
#             print(f"      Тип: {node.node_type}")
#             print(f"      Свойства: {node.properties[:5]}")
#
#     print("\n🔗 ПРИМЕРЫ СВЯЗЕЙ:")
#     airplane = graph.get_node("kb_airplane")
#     if airplane:
#         neighbors = graph.get_neighbors("kb_airplane")
#         print(f"\n   Соседи 'airplane':")
#         for neighbor in neighbors[:5]:
#             print(f"      → {neighbor.name}")
#
#
# if __name__ == "__main__":
#     main()




# # scripts/build_local_knowledge_graph.py
# """
# Создает локальный граф знаний без внешних API.
# Использует встроенные знания из вашей статьи и LLM для генерации.
# """
#
# import sys
# import os
#
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# import json
# import time
# from typing import Dict, List, Any
# from core.knowledge.knowledge_node import KnowledgeNode
# from core.knowledge.knowledge_edge import KnowledgeEdge, EdgeType
# from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
# from db.knowledge_db import KnowledgeDB
#
#
# class LocalKnowledgeBuilder:
#     """
#     Строитель локального графа знаний из встроенных данных.
#     """
#
#     def __init__(self):
#         self.graph = GlobalKnowledgeGraph()
#         self.db = KnowledgeDB()
#         self.node_counter = 0
#
#         # Базовые знания из вашей статьи
#         self.knowledge_base = self._build_knowledge_base()
#
#     def _build_knowledge_base(self) -> Dict[str, Any]:
#         """
#         Создает базу знаний на основе вашей статьи.
#         """
#         return {
#             # ============================================================
#             # 1. Модели знаний / конструкций
#             # ============================================================
#
#             # 1.1 Воздух
#             "air": {
#                 "type": "physical_object",
#                 "properties": ["gas", "fluid", "exerts_pressure", "invisible"],
#                 "functions": ["transmits_force", "creates_lift", "supports_flight"],
#                 "description": "Газообразное вещество, окружающее Землю",
#                 "parameters": {"density": 1.225, "weight": 1.225},
#                 "related": ["wing", "propeller", "engine", "bird"]
#             },
#
#             # 1.2 Чайка (птица)
#             "seagull": {
#                 "type": "biological_concept",
#                 "properties": ["flying", "has_wings", "heavier_than_air"],
#                 "functions": ["fly", "glide", "hover"],
#                 "description": "Птица с характерной формой крыльев",
#                 "parameters": {"weight": 5, "wingspan": 1.0},
#                 "related": ["wing", "air", "flight"]
#             },
#
#             # 1.3 Крыло (аэродинамическое)
#             "wing": {
#                 "type": "physical_object",
#                 "properties": ["aerodynamic", "curved", "creates_lift"],
#                 "functions": ["generate_lift", "control_flight", "support_aircraft"],
#                 "description": "Аэродинамическая поверхность для создания подъемной силы",
#                 "parameters": {"aspect_ratio": 5.0, "airfoil": "curved"},
#                 "related": ["air", "airfoil", "lift", "airplane"]
#             },
#
#             # 1.4 Воздушный шар
#             "balloon": {
#                 "type": "physical_object",
#                 "properties": ["lighter_than_air", "gas_filled", "floats"],
#                 "functions": ["carry_load", "lift_people", "float_in_air"],
#                 "description": "Устройство, поднимающееся за счет газа легче воздуха",
#                 "parameters": {"volume": 1000, "lift_capacity": 100},
#                 "related": ["air", "helium", "lift", "basket"]
#             },
#
#             # 1.5 Воздушный змей
#             "kite": {
#                 "type": "physical_object",
#                 "properties": ["tethered", "wind_powered", "heavier_than_air"],
#                 "functions": ["fly_in_wind", "lift_load", "stay_airborne"],
#                 "description": "Устройство, парящее в воздухе за счет набегающего потока",
#                 "parameters": {"area": 2.0, "weight": 0.5},
#                 "related": ["air", "wind", "string", "lift"]
#             },
#
#             # 1.6 Двигатель внутреннего сгорания
#             "engine": {
#                 "type": "system",
#                 "properties": ["mechanical", "power_producing", "fuel_consuming"],
#                 "functions": ["convert_energy", "drive_propeller", "generate_power"],
#                 "description": "Устройство для преобразования тепловой энергии в механическую",
#                 "parameters": {"power": 50, "weight": 100},
#                 "related": ["fuel", "piston", "crankshaft", "propeller"]
#             },
#
#             # 1.7 Пропеллер (воздушный винт)
#             "propeller": {
#                 "type": "physical_object",
#                 "properties": ["rotating", "bladed", "thrust_producing"],
#                 "functions": ["generate_thrust", "push_air", "propel_aircraft"],
#                 "description": "Устройство для создания тяги за счет вращения лопастей",
#                 "parameters": {"diameter": 2.0, "pitch": 1.5},
#                 "related": ["engine", "air", "thrust", "shaft"]
#             },
#
#             # 1.8 Паровой двигатель
#             "steam_engine": {
#                 "type": "system",
#                 "properties": ["steam_powered", "industrial", "heavy"],
#                 "functions": ["generate_power", "drive_machinery"],
#                 "description": "Двигатель, работающий на паре",
#                 "parameters": {"power": 10, "weight": 200},
#                 "related": ["steam", "boiler", "piston", "wheel"]
#             },
#
#             # 1.9 Резиномотор
#             "rubber_motor": {
#                 "type": "system",
#                 "properties": ["elastic", "light", "simple"],
#                 "functions": ["store_energy", "drive_propeller"],
#                 "description": "Двигатель на основе эластичности резины",
#                 "parameters": {"power": 0.5, "weight": 1.0},
#                 "related": ["rubber", "winding", "propeller"]
#             },
#
#             # 1.10 Конная упряжка
#             "horse_team": {
#                 "type": "system",
#                 "properties": ["animal_powered", "ground_based", "muscle_power"],
#                 "functions": ["pull", "transport", "provide_force"],
#                 "description": "Система из лошадей для создания тягового усилия",
#                 "parameters": {"power": 1.0, "speed": 5.0},
#                 "related": ["horse", "cart", "rope", "pull"]
#             },
#
#             # ============================================================
#             # 2. Дополнительные концепты из вашей статьи
#             # ============================================================
#
#             "lift": {
#                 "type": "physical_concept",
#                 "properties": ["aerodynamic", "force", "upward"],
#                 "functions": ["support_weight", "enable_flight"],
#                 "description": "Аэродинамическая сила, направленная вверх",
#                 "related": ["wing", "air", "airfoil"]
#             },
#
#             "drag": {
#                 "type": "physical_concept",
#                 "properties": ["aerodynamic", "force", "resistance"],
#                 "functions": ["resist_motion", "create_drag"],
#                 "description": "Сила сопротивления движению в воздухе",
#                 "related": ["air", "wing", "speed"]
#             },
#
#             "thrust": {
#                 "type": "physical_concept",
#                 "properties": ["force", "forward", "propulsion"],
#                 "functions": ["move_forward", "overcome_drag"],
#                 "description": "Сила, двигающая объект вперед",
#                 "related": ["engine", "propeller", "motion"]
#             },
#
#             "airplane": {
#                 "type": "system",
#                 "properties": ["flying", "fixed_wing", "heavy_than_air"],
#                 "functions": ["transport", "fly", "carry_people"],
#                 "description": "Летательный аппарат тяжелее воздуха с неподвижным крылом",
#                 "related": ["wing", "engine", "propeller", "air", "lift"]
#             }
#         }
#
#     def build(self):
#         """
#         Строит граф знаний из локальной базы.
#         """
#         print("=" * 60)
#         print("🏗️ ПОСТРОЕНИЕ ЛОКАЛЬНОГО ГРАФА ЗНАНИЙ")
#         print("=" * 60)
#
#         # 1. Создаем узлы
#         print("\n📦 Создание узлов...")
#         nodes_created = 0
#
#         for name, data in self.knowledge_base.items():
#             node = KnowledgeNode(
#                 id=f"kb_{name}",
#                 name=name,
#                 node_type=data.get("type", "concept"),
#                 properties=data.get("properties", []),
#                 description=data.get("description", f"Knowledge about {name}")
#             )
#
#             # Добавляем параметры
#             if "parameters" in data:
#                 node.parameters = data["parameters"]
#
#             # Добавляем функции
#             for func in data.get("functions", []):
#                 from core.knowledge.function import Function
#                 func_obj = Function(
#                     id=f"func_{name}_{func}",
#                     name=func,
#                     description=f"{name} can {func}"
#                 )
#                 node.add_function(func_obj)
#
#             self.graph.add_node(node)
#             nodes_created += 1
#             print(f"   ✅ {name}: {node.node_type} ({len(node.properties)} свойств)")
#
#         print(f"\n   Создано узлов: {nodes_created}")
#
#         # 2. Создаем связи
#         print("\n🔗 Создание связей...")
#         edges_created = 0
#
#         for name, data in self.knowledge_base.items():
#             source_id = f"kb_{name}"
#
#             for related in data.get("related", []):
#                 target_id = f"kb_{related}"
#
#                 # Проверяем, существует ли целевой узел
#                 if self.graph.get_node(target_id):
#                     edge = KnowledgeEdge(
#                         id=f"edge_{name}_{related}",
#                         source_id=source_id,
#                         target_id=target_id,
#                         edge_type=EdgeType.RELATED_TO,
#                         weight=0.8,
#                         description=f"{name} related to {related}"
#                     )
#                     self.graph.add_edge(edge)
#                     edges_created += 1
#
#             # Добавляем специальные связи
#             if name == "airplane":
#                 # Связи для самолета
#                 for part in ["wing", "engine", "propeller"]:
#                     target_id = f"kb_{part}"
#                     if self.graph.get_node(target_id):
#                         edge = KnowledgeEdge(
#                             id=f"edge_airplane_has_{part}",
#                             source_id="kb_airplane",
#                             target_id=target_id,
#                             edge_type=EdgeType.PART_OF,
#                             weight=0.9,
#                             description=f"Airplane has {part}"
#                         )
#                         self.graph.add_edge(edge)
#                         edges_created += 1
#
#                 # Связь с концептами
#                 for concept in ["lift", "thrust", "air"]:
#                     target_id = f"kb_{concept}"
#                     if self.graph.get_node(target_id):
#                         edge = KnowledgeEdge(
#                             id=f"edge_airplane_uses_{concept}",
#                             source_id="kb_airplane",
#                             target_id=target_id,
#                             edge_type=EdgeType.CAUSES,
#                             weight=0.9,
#                             description=f"Airplane uses {concept}"
#                         )
#                         self.graph.add_edge(edge)
#                         edges_created += 1
#
#         print(f"\n   Создано связей: {edges_created}")
#
#         # 3. Сохраняем в БД
#         print("\n💾 Сохранение в БД...")
#         for node in self.graph.nodes.values():
#             self.db.save_node(node)
#         for edge in self.graph.edges.values():
#             self.db.save_edge(edge)
#
#         print("   ✅ Сохранено")
#
#         # 4. Выводим статистику
#         print("\n" + "=" * 60)
#         print("📊 СТАТИСТИКА ГРАФА ЗНАНИЙ")
#         print("=" * 60)
#         print(f"   Узлов: {len(self.graph.nodes)}")
#         print(f"   Связей: {len(self.graph.edges)}")
#         print(f"   Типов узлов: {set(n.node_type for n in self.graph.nodes.values())}")
#         print("=" * 60)
#
#         return self.graph
#
#
# def main():
#     builder = LocalKnowledgeBuilder()
#     graph = builder.build()
#
#     # Выводим примеры
#     print("\n🔍 ПРИМЕРЫ УЗЛОВ:")
#     for name in ["air", "wing", "airplane", "engine"]:
#         node = graph.get_node(f"kb_{name}")
#         if node:
#             print(f"\n   {name.upper()}:")
#             print(f"      Тип: {node.node_type}")
#             print(f"      Свойства: {node.properties}")
#             if node.functions:
#                 print(f"      Функции: {[f.name for f in node.functions]}")
#
#     print("\n🔗 ПРИМЕРЫ СВЯЗЕЙ:")
#     airplane = graph.get_node("kb_airplane")
#     if airplane:
#         neighbors = graph.get_neighbors("kb_airplane")
#         print(f"\n   Соседи 'airplane':")
#         for neighbor in neighbors[:5]:
#             print(f"      → {neighbor.name} ({neighbor.node_type})")
#
#
# if __name__ == "__main__":
#     main()