# core/thinking/understanding.py (исправленный импорт)
"""
Модуль "Понимание" - создание новых моделей из описания задачи.
"""

import re
import json
import time
import hashlib
import random
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from core.knowledge.knowledge_node import KnowledgeNode
from core.knowledge.knowledge_edge import KnowledgeEdge, EdgeType
from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
from core.knowledge.individual_knowledge_graph import IndividualKnowledgeGraph
from core.knowledge.combination import Combination
from core.knowledge.function import Function
from .models import MentalModel  # <-- Исправленный импорт




# # core/thinking/understanding.py
# """
# Модуль "Понимание" - создание новых моделей из описания задачи.
# """
#
# import re
# import json
# import time
# import hashlib
# import random
# from typing import List, Dict, Any, Optional, Set, Tuple
# from dataclasses import dataclass, field
# from enum import Enum
#
# from core.knowledge.knowledge_node import KnowledgeNode
# from core.knowledge.knowledge_edge import KnowledgeEdge, EdgeType
# from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
# from core.knowledge.individual_knowledge_graph import IndividualKnowledgeGraph
# from core.knowledge.combination import Combination
# from core.knowledge.function import Function
# from core.knowledge.mental_model import MentalModel


class UnderstandingStatus(Enum):
    """Статус понимания."""
    EXTRACTING = "extracting"  # Извлечение концептов
    SEARCHING = "searching"  # Поиск в ГЗ
    BUILDING = "building"  # Построение модели
    SAVING = "saving"  # Сохранение в ИГЗ
    COMPLETED = "completed"  # Завершено


@dataclass
class UnderstandingResult:
    """Результат этапа понимания."""
    task_description: str
    status: UnderstandingStatus
    extracted_concepts: List[str] = field(default_factory=list)
    found_nodes: List[KnowledgeNode] = field(default_factory=list)
    new_model: Optional[Combination] = None
    mental_model: Optional[MentalModel] = None
    created_at: float = field(default_factory=time.time)
    experience: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0


class UnderstandingEngine:
    """
    Движок "Понимания" - создает новые модели из описания задачи.

    Этапы:
    1. Извлечение ключевых концептов из текста
    2. Поиск соответствующих узлов в ГЗ
    3. Построение новой модели из найденных узлов
    4. Сохранение модели в ИГЗ бота
    5. Формирование опыта использования
    """

    # Стоп-слова для фильтрации
    STOP_WORDS = {
        'это', 'что', 'как', 'для', 'с', 'на', 'по', 'из', 'у', 'в',
        'к', 'от', 'до', 'за', 'через', 'без', 'вместе', 'около',
        'очень', 'слишком', 'почти', 'примерно', 'около'
    }

    # Инженерные ключевые слова
    ENGINEERING_KEYWORDS = [
        # Аэрокосмические
        "летательный аппарат", "крыло", "фюзеляж", "двигатель", "турбина", "пропеллер", "самолет",
        "вертолет", "ракета", "спутник", "беспилотник", "дрон",

        # Механические
        "вал", "шестерня", "подшипник", "пружина", "рычаг", "ремень", "цепь",
        "муфта", "шкив", "звездочка", "винт", "гайка", "болт", "шпонка",
        "поршень", "цилиндр", "клапан", "редуктор", "амортизатор",

        # Электрические
        "двигатель", "генератор", "аккумулятор", "резистор", "конденсатор",
        "транзистор", "диод", "микросхема", "трансформатор", "индуктор",
        "датчик", "контроллер", "привод", "сервопривод", "шаговый двигатель",

        # Системы
        "система", "управление", "регулирование", "автоматизация", "контроль",
        "навигация", "связь", "безопасность", "диагностика", "мониторинг",

        # Физика
        "энергия", "сила", "давление", "температура", "скорость", "ускорение",
        "момент", "мощность", "частота", "вибрация", "нагрев", "охлаждение",

        # Материалы
        "сталь", "алюминий", "титан", "медь", "композит", "пластик", "стекло",
        "керамика", "резина", "полимер", "сплав", "бронза", "латунь",

        # Процессы
        "сварка", "пайка", "литье", "ковка", "фрезерование", "точение",
        "шлифовка", "полировка", "термообработка", "закалка", "отжиг",
        "сборка", "монтаж", "настройка", "калибровка", "испытание",

        # Общие инженерные термины
        "механизм", "устройство", "агрегат", "конструкция", "деталь", "узел",
        "прототип", "модель", "система", "комплекс", "установка", "аппарат",
        "машина", "станок", "прибор", "инструмент", "оборудование"
    ]

    # Синонимы для поиска
    SYNONYM_MAP = {
        "самолет": ["аэроплан", "лайнер", "биплан", "моноплан"],
        "двигатель": ["мотор", "силовой агрегат", "установка"],
        "крыло": ["крылья", "несущая поверхность", "аэродинамическая поверхность"],
        "система": ["комплекс", "установка", "агрегат", "компоновка"],
        "механизм": ["привод", "исполнительный механизм", "трансмиссия"],
        "подшипник": ["опора", "подвес"],
        "пружина": ["упругий элемент", "рессора"],
        "энергия": ["мощность", "работа", "ёмкость"],
        "управление": ["регулировка", "контроль", "руление"]
    }

    def __init__(self, global_graph: GlobalKnowledgeGraph = None,
                 individual_graph: IndividualKnowledgeGraph = None,
                 load_from_db: bool = True):  # ← НОВЫЙ ПАРАМЕТР
        """
        Инициализирует движок понимания.

        Args:
            global_graph: Глобальный граф знаний (если None — загружается из БД)
            individual_graph: Индивидуальный граф знаний
            load_from_db: Загружать ли ГЗ из БД
        """
        self.global_graph = global_graph
        self.individual_graph = individual_graph or IndividualKnowledgeGraph()

        # Если ГЗ не передан, загружаем из БД
        if self.global_graph is None and load_from_db:
            print("📥 Загрузка ГЗ из БД...")
            self.global_graph = GlobalKnowledgeGraph()
            self.global_graph.load_from_db()
            print(f"   ✅ Загружено: {len(self.global_graph.nodes)} узлов, {len(self.global_graph.edges)} связей")
        elif self.global_graph is None:
            self.global_graph = GlobalKnowledgeGraph()

        # Индексы для быстрого поиска
        self._name_index: Dict[str, str] = {}
        self._property_index: Dict[str, List[str]] = {}
        self._build_indexes()

        # История понимания
        self.understanding_history: List[UnderstandingResult] = []

        print("🧠 Движок понимания инициализирован")
        print(f"   Узлов в ГЗ: {len(self.global_graph.nodes)}")
        print(f"   Связей в ГЗ: {len(self.global_graph.edges)}")

    # def __init__(self, global_graph: GlobalKnowledgeGraph = None,
    #              individual_graph: IndividualKnowledgeGraph = None):
    #     self.global_graph = global_graph or GlobalKnowledgeGraph()
    #     self.individual_graph = individual_graph or IndividualKnowledgeGraph()
    #
    #     # Индексы для быстрого поиска
    #     self._name_index: Dict[str, str] = {}
    #     self._property_index: Dict[str, List[str]] = {}
    #     self._build_indexes()
    #
    #     # История понимания
    #     self.understanding_history: List[UnderstandingResult] = []
    #
    #     print("🧠 Движок понимания инициализирован")
    #     print(f"   Узлов в ГЗ: {len(self.global_graph.nodes)}")
    #     print(f"   Связей в ГЗ: {len(self.global_graph.edges)}")

    def _build_indexes(self):
        """Строит индексы для быстрого поиска."""
        self._name_index = {}
        self._property_index = {}

        for node in self.global_graph.nodes.values():
            # Индекс по имени
            self._name_index[node.name.lower()] = node.id

            # Индекс по свойствам
            for prop in node.properties:
                prop_lower = prop.lower()
                if prop_lower not in self._property_index:
                    self._property_index[prop_lower] = []
                self._property_index[prop_lower].append(node.id)

    def extract_concepts(self, text: str) -> List[str]:
        """
        Извлекает ключевые концепты из текста.
        """
        text_lower = text.lower()
        concepts = []
        seen = set()

        # 1. Поиск по инженерным ключевым словам (включая фразы)
        for keyword in self.ENGINEERING_KEYWORDS:
            if keyword in text_lower and keyword not in seen:
                concepts.append(keyword)
                seen.add(keyword)

        # 2. Поиск по синонимам
        for key, synonyms in self.SYNONYM_MAP.items():
            if key in text_lower and key not in seen:
                concepts.append(key)
                seen.add(key)
            for syn in synonyms:
                if syn in text_lower and syn not in seen:
                    concepts.append(syn)
                    seen.add(syn)

        # 3. ИЗВЛЕЧЕНИЕ БИГРАММ (ДВУХСЛОВНЫХ ФРАЗ) — НОВОЕ!
        words = re.findall(r'[а-яА-Яa-zA-Z]+', text_lower)
        for i in range(len(words) - 1):
            # Пропускаем короткие слова
            if len(words[i]) < 3 or len(words[i + 1]) < 3:
                continue

            # Создаём биграмму
            bigram = f"{words[i]} {words[i + 1]}"

            # Проверяем, есть ли такой узел в ГЗ или это ключевое слово
            if bigram not in seen:
                # Проверяем в индексе имён
                if bigram in self._name_index:
                    concepts.append(bigram)
                    seen.add(bigram)
                # Проверяем в ключевых словах
                elif bigram in self.ENGINEERING_KEYWORDS:
                    concepts.append(bigram)
                    seen.add(bigram)

        # 4. Извлечение фраз "существительное + существительное" (уже было)
        for i in range(len(words) - 1):
            if len(words[i]) > 3 and len(words[i + 1]) > 3:
                phrase = f"{words[i]} {words[i + 1]}"
                if phrase not in seen and phrase in text_lower:
                    if phrase in self._name_index or any(phrase in n.name.lower() for n in self.global_graph.nodes.values()):
                        concepts.append(phrase)
                        seen.add(phrase)

        # 5. Удаляем стоп-слова
        concepts = [c for c in concepts if c not in self.STOP_WORDS]

        # 6. Удаляем дубликаты и сортируем
        concepts = list(set(concepts))
        concepts.sort(key=len, reverse=True)

        return concepts

    # def extract_concepts(self, text: str) -> List[str]:
    #     """
    #     Извлекает ключевые концепты из текста.
    #
    #     Args:
    #         text: Текстовое описание задачи
    #
    #     Returns:
    #         Список извлеченных концептов
    #     """
    #     text_lower = text.lower()
    #     concepts = []
    #     seen = set()
    #
    #     # 1. Поиск по инженерным ключевым словам
    #     for keyword in self.ENGINEERING_KEYWORDS:
    #         if keyword in text_lower and keyword not in seen:
    #             concepts.append(keyword)
    #             seen.add(keyword)
    #
    #     # 2. Поиск по синонимам
    #     for key, synonyms in self.SYNONYM_MAP.items():
    #         if key in text_lower and key not in seen:
    #             concepts.append(key)
    #             seen.add(key)
    #         for syn in synonyms:
    #             if syn in text_lower and syn not in seen:
    #                 concepts.append(syn)
    #                 seen.add(syn)
    #
    #     # 3. Извлечение фраз "существительное + существительное"
    #     words = re.findall(r'[а-яА-Яa-zA-Z]+', text_lower)
    #     for i in range(len(words) - 1):
    #         if len(words[i]) > 3 and len(words[i + 1]) > 3:
    #             phrase = f"{words[i]} {words[i + 1]}"
    #             if phrase not in seen and phrase in text_lower:
    #                 # Проверяем, есть ли такой узел
    #                 if phrase in self._name_index or any(phrase in n.name.lower() for n in self.global_graph.nodes.values()):
    #                     concepts.append(phrase)
    #                     seen.add(phrase)
    #
    #     # 4. Удаляем стоп-слова
    #     concepts = [c for c in concepts if c not in self.STOP_WORDS]
    #
    #     # 5. Удаляем дубликаты и сортируем
    #     concepts = list(set(concepts))
    #     concepts.sort(key=len, reverse=True)
    #
    #     return concepts

    def find_nodes_by_concepts(self, concepts: List[str],
                               max_nodes: int = 10) -> List[KnowledgeNode]:
        """
        Находит узлы в ГЗ по списку концептов.

        Args:
            concepts: Список концептов
            max_nodes: Максимальное количество узлов

        Returns:
            Список найденных узлов
        """
        found_nodes = []
        seen_ids = set()

        for concept in concepts:
            concept_lower = concept.lower()

            # 1. Точное совпадение по имени
            if concept_lower in self._name_index:
                node_id = self._name_index[concept_lower]
                if node_id not in seen_ids:
                    node = self.global_graph.get_node(node_id)
                    if node:
                        found_nodes.append(node)
                        seen_ids.add(node_id)
                continue

            # 2. Частичное совпадение по имени
            for node in self.global_graph.nodes.values():
                if node.id in seen_ids:
                    continue
                if concept_lower in node.name.lower() or node.name.lower() in concept_lower:
                    found_nodes.append(node)
                    seen_ids.add(node.id)
                    break

            # 3. Поиск по свойствам
            if len(found_nodes) < 5:  # Если еще мало узлов
                for prop, node_ids in self._property_index.items():
                    if concept_lower in prop or prop in concept_lower:
                        for node_id in node_ids[:3]:  # Берем первые 3
                            if node_id not in seen_ids:
                                node = self.global_graph.get_node(node_id)
                                if node:
                                    found_nodes.append(node)
                                    seen_ids.add(node_id)

        # Ограничиваем количество
        return found_nodes[:max_nodes]

    def build_mental_model(self, nodes: List[KnowledgeNode],
                           task_description: str) -> MentalModel:
        """
        Строит ментальную модель из найденных узлов.
        """
        import uuid

        # Создаем UUID
        model_id = str(uuid.uuid4())

        # Собираем все свойства
        all_properties = []
        all_functions = []
        for node in nodes:
            all_properties.extend(node.properties)
            for func in node.functions:
                all_functions.append(func.name)

        # Создаем последовательность (для ментальной модели)
        # В упрощенном виде - список описаний узлов
        sequence = [f"{node.name}: {node.description[:50]}" for node in nodes[:5]]

        # Создаем эмбеддинг
        embedding = self._create_embedding(nodes)

        # Создаем модель
        model = MentalModel(
            id=model_id,
            name=f"Модель: {task_description[:30]}...",
            sequence=sequence,
            embedding=embedding
        )

        # Добавляем свойства из узлов
        model.properties = list(set(all_properties))

        # Добавляем метаданные
        model.metadata = {
            "task": task_description,
            "node_ids": [n.id for n in nodes],
            "node_names": [n.name for n in nodes],
            "created_at": time.time(),
            "source": "understanding_engine"
        }

        return model

    # def build_mental_model(self, nodes: List[KnowledgeNode],
    #                        task_description: str) -> MentalModel:
    #     """
    #     Строит ментальную модель из найденных узлов.
    #
    #     Args:
    #         nodes: Список узлов
    #         task_description: Описание задачи
    #
    #     Returns:
    #         Ментальная модель
    #     """
    #     # Создаем ID модели
    #     model_id = f"mm_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
    #
    #     # Собираем все свойства
    #     all_properties = []
    #     all_functions = []
    #     for node in nodes:
    #         all_properties.extend(node.properties)
    #         for func in node.functions:
    #             all_functions.append(func.name)
    #
    #     # Создаем последовательность (для ментальной модели)
    #     # В упрощенном виде - список описаний узлов
    #     sequence = [f"{node.name}: {node.description[:50]}" for node in nodes[:5]]
    #
    #     model = MentalModel(
    #         id=model_id,
    #         name=f"Модель: {task_description[:30]}...",
    #         sequence=sequence,
    #         embedding=self._create_embedding(nodes)
    #     )
    #
    #     # Добавляем свойства из узлов
    #     model.properties = list(set(all_properties))
    #
    #     # Добавляем метаданные
    #     model.metadata = {
    #         "task": task_description,
    #         "node_ids": [n.id for n in nodes],
    #         "node_names": [n.name for n in nodes],
    #         "created_at": time.time(),
    #         "source": "understanding_engine"
    #     }
    #
    #     return model

    def _create_embedding(self, nodes: List[KnowledgeNode]) -> List[float]:
        """
        Создает эмбеддинг для модели из узлов.
        """
        import random
        # Простой эмбеддинг - усреднение свойств
        embedding = [0.0] * 64
        for node in nodes[:3]:
            for i, prop in enumerate(node.properties[:5]):
                if i < len(embedding):
                    embedding[i] += hash(prop) % 100 / 100.0

        # Нормализуем
        max_val = max(embedding) if embedding else 1
        if max_val > 0:
            embedding = [v / max_val for v in embedding]

        return embedding

    def understand(self, task_description: str) -> UnderstandingResult:
        """
        Основной метод "Понимания".
        """
        start_time = time.time()
        result = UnderstandingResult(
            task_description=task_description,
            status=UnderstandingStatus.EXTRACTING
        )

        try:
            # 1. ИЗВЛЕЧЕНИЕ КОНЦЕПТОВ
            print("\n🧠 ПОНИМАНИЕ ЗАДАЧИ")
            print("=" * 60)
            print(f"📝 {task_description[:100]}...")

            result.status = UnderstandingStatus.EXTRACTING
            concepts = self.extract_concepts(task_description)
            result.extracted_concepts = concepts
            print(f"📌 Извлечено концептов: {len(concepts)}")
            if concepts:
                print(f"   {', '.join(concepts[:5])}")

            if not concepts:
                result.status = UnderstandingStatus.COMPLETED
                result.errors.append("Не удалось извлечь концепты из текста")
                print("⚠️ Не удалось извлечь концепты")
                result.execution_time = time.time() - start_time
                self.understanding_history.append(result)
                return result

            # 2. ПОИСК УЗЛОВ
            result.status = UnderstandingStatus.SEARCHING
            nodes = self.find_nodes_by_concepts(concepts)
            result.found_nodes = nodes
            print(f"📌 Найдено узлов: {len(nodes)}")
            if nodes:
                print(f"   {', '.join([n.name for n in nodes[:5]])}")

            if not nodes:
                result.status = UnderstandingStatus.COMPLETED
                result.errors.append("Не найдено подходящих узлов в ГЗ")
                print("⚠️ Не найдено подходящих узлов")
                result.execution_time = time.time() - start_time
                self.understanding_history.append(result)
                return result

            # 3. ПОСТРОЕНИЕ МОДЕЛИ
            result.status = UnderstandingStatus.BUILDING
            model = self.build_mental_model(nodes, task_description)
            result.mental_model = model
            print(f"📌 Создана ментальная модель: {model.id}")
            print(f"   Свойств: {len(model.properties)}")
            print(f"   Узлов: {len(nodes)}")

            # ============================================================
            # 4. СОХРАНЕНИЕ В БД
            # ============================================================
            try:
                from db.knowledge_db import KnowledgeDB
                db = KnowledgeDB()

                # Сохраняем ментальную модель в БД
                if db.save_mental_model(model):
                    print(f"   💾 Ментальная модель сохранена в БД: {model.id}")
                else:
                    print(f"   ⚠️ Не удалось сохранить модель в БД")

            except Exception as e:
                print(f"   ❌ Ошибка сохранения в БД: {e}")
                result.errors.append(f"Ошибка сохранения: {e}")

            # 5. СОХРАНЕНИЕ В ИГЗ
            result.status = UnderstandingStatus.SAVING
            if self.individual_graph:
                self._save_to_individual_graph(model, task_description, nodes)
                print("📌 Сохранено в ИГЗ")

            # 6. ФОРМИРОВАНИЕ ОПЫТА
            result.experience = {
                "task": task_description,
                "concepts": concepts,
                "node_ids": [n.id for n in nodes],
                "node_names": [n.name for n in nodes],
                "model_id": model.id,
                "model_properties": model.properties,
                "created_at": time.time()
            }

            # Создаем комбинацию для совместимости
            combo = Combination(
                id=f"combo_{model.id}",
                nodes=nodes,
                properties=model.properties,
                metadata=model.metadata
            )
            result.new_model = combo

            result.status = UnderstandingStatus.COMPLETED
            print("✅ Понимание завершено")

        except Exception as e:
            result.status = UnderstandingStatus.COMPLETED
            result.errors.append(str(e))
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()

        result.execution_time = time.time() - start_time
        self.understanding_history.append(result)
        return result

    # def understand(self, task_description: str) -> UnderstandingResult:
    #     """
    #     Основной метод "Понимания".
    #     """
    #     start_time = time.time()
    #     result = UnderstandingResult(
    #         task_description=task_description,
    #         status=UnderstandingStatus.EXTRACTING
    #     )
    #
    #     try:
    #         # ... существующий код (извлечение концептов, поиск узлов) ...
    #
    #         # 3. ПОСТРОЕНИЕ МОДЕЛИ
    #         result.status = UnderstandingStatus.BUILDING
    #         model = self.build_mental_model(nodes, task_description)
    #         result.mental_model = model
    #         print(f"📌 Создана ментальная модель: {model.id}")
    #         print(f"   Свойств: {len(model.properties)}")
    #         print(f"   Узлов: {len(nodes)}")
    #
    #         # ============================================================
    #         # 4. СОХРАНЕНИЕ В БД (НОВОЕ!)
    #         # ============================================================
    #         try:
    #             from db.knowledge_db import KnowledgeDB
    #             db = KnowledgeDB()
    #
    #             # Сохраняем ментальную модель в БД
    #             if db.save_mental_model(model):
    #                 print(f"   💾 Ментальная модель сохранена в БД: {model.id}")
    #             else:
    #                 print(f"   ⚠️ Не удалось сохранить модель в БД")
    #
    #         except Exception as e:
    #             print(f"   ❌ Ошибка сохранения в БД: {e}")
    #             result.errors.append(f"Ошибка сохранения: {e}")
    #
    #         # 5. СОХРАНЕНИЕ В ИГЗ
    #         result.status = UnderstandingStatus.SAVING
    #         if self.individual_graph:
    #             self._save_to_individual_graph(model, task_description, nodes)
    #             print("📌 Сохранено в ИГЗ")
    #
    #         # ... остальной код ...
    #
    #     except Exception as e:
    #         result.status = UnderstandingStatus.COMPLETED
    #         result.errors.append(str(e))
    #         print(f"❌ Ошибка: {e}")
    #
    #     result.execution_time = time.time() - start_time
    #     self.understanding_history.append(result)
    #     return result

    # def understand(self, task_description: str) -> UnderstandingResult:
    #     """
    #     Основной метод "Понимания".
    #
    #     Args:
    #         task_description: Текстовое описание задачи
    #
    #     Returns:
    #         UnderstandingResult с результатами понимания
    #     """
    #     start_time = time.time()
    #     result = UnderstandingResult(
    #         task_description=task_description,
    #         status=UnderstandingStatus.EXTRACTING
    #     )
    #
    #     try:
    #         # 1. ИЗВЛЕЧЕНИЕ КОНЦЕПТОВ
    #         print("\n🧠 ПОНИМАНИЕ ЗАДАЧИ")
    #         print("=" * 60)
    #         print(f"📝 {task_description[:100]}...")
    #
    #         result.status = UnderstandingStatus.EXTRACTING
    #         concepts = self.extract_concepts(task_description)
    #         result.extracted_concepts = concepts
    #         print(f"📌 Извлечено концептов: {len(concepts)}")
    #         if concepts:
    #             print(f"   {', '.join(concepts[:5])}")
    #
    #         if not concepts:
    #             result.status = UnderstandingStatus.COMPLETED
    #             result.errors.append("Не удалось извлечь концепты из текста")
    #             print("⚠️ Не удалось извлечь концепты")
    #             result.execution_time = time.time() - start_time
    #             self.understanding_history.append(result)
    #             return result
    #
    #         # 2. ПОИСК УЗЛОВ
    #         result.status = UnderstandingStatus.SEARCHING
    #         nodes = self.find_nodes_by_concepts(concepts)
    #         result.found_nodes = nodes
    #         print(f"📌 Найдено узлов: {len(nodes)}")
    #         if nodes:
    #             print(f"   {', '.join([n.name for n in nodes[:5]])}")
    #
    #         if not nodes:
    #             result.status = UnderstandingStatus.COMPLETED
    #             result.errors.append("Не найдено подходящих узлов в ГЗ")
    #             print("⚠️ Не найдено подходящих узлов")
    #             result.execution_time = time.time() - start_time
    #             self.understanding_history.append(result)
    #             return result
    #
    #         # 3. ПОСТРОЕНИЕ МОДЕЛИ
    #         result.status = UnderstandingStatus.BUILDING
    #         model = self.build_mental_model(nodes, task_description)
    #         result.mental_model = model
    #         print(f"📌 Создана ментальная модель: {model.id}")
    #         print(f"   Свойств: {len(model.properties)}")
    #         print(f"   Узлов: {len(nodes)}")
    #
    #         # Создаем комбинацию для совместимости
    #         combo = Combination(
    #             id=f"combo_{model.id}",
    #             nodes=nodes,
    #             properties=model.properties,
    #             metadata=model.metadata
    #         )
    #         result.new_model = combo
    #
    #         # 4. СОХРАНЕНИЕ В ИГЗ
    #         result.status = UnderstandingStatus.SAVING
    #         if self.individual_graph:
    #             self._save_to_individual_graph(model, task_description, nodes)
    #             print("📌 Сохранено в ИГЗ")
    #
    #         # 5. ФОРМИРОВАНИЕ ОПЫТА
    #         result.experience = {
    #             "task": task_description,
    #             "concepts": concepts,
    #             "node_ids": [n.id for n in nodes],
    #             "node_names": [n.name for n in nodes],
    #             "model_id": model.id,
    #             "model_properties": model.properties,
    #             "created_at": time.time()
    #         }
    #
    #         result.status = UnderstandingStatus.COMPLETED
    #         print("✅ Понимание завершено")
    #
    #     except Exception as e:
    #         result.status = UnderstandingStatus.COMPLETED
    #         result.errors.append(str(e))
    #         print(f"❌ Ошибка: {e}")
    #
    #     result.execution_time = time.time() - start_time
    #     self.understanding_history.append(result)
    #     return result

    def _save_to_individual_graph(self, model: MentalModel,
                                  task_description: str,
                                  nodes: List[KnowledgeNode]):
        """Сохраняет модель в индивидуальный граф знаний и синхронизирует с ГГЗ."""
        if not self.individual_graph:
            return

        # 1. Сохраняем как обычное знание
        record = {
            "id": model.id,
            "type": "understanding",
            "name": model.name,
            "nodes": [n.id for n in nodes],
            "properties": model.properties,
            "task": task_description,
            "created_at": time.time(),
            "source": "understanding_engine"
        }
        self.individual_graph.add_knowledge(record)

        # 2. Сохраняем как ментальную модель
        self.individual_graph.add_mental_model(model.id, {
            'name': model.name,
            'properties': model.properties,
            'nodes': [n.id for n in nodes],
            'task': task_description,
            'model_type': getattr(model, 'model_type', 'mental_model')
        })

        # 3. Синхронизируем узлы с ГГЗ
        node_ids = [n.id for n in nodes]
        results = self.individual_graph.sync_with_global(self.global_graph, node_ids)

        success_count = sum(1 for v in results.values() if v)
        print(f"📌 Синхронизировано с ГГЗ: {success_count}/{len(node_ids)} узлов")

        # 4. Синхронизируем ментальную модель с ГГЗ
        if self.individual_graph.sync_mental_model_to_global(self.global_graph, model.id):
            print(f"📌 Ментальная модель {model.id} синхронизирована с ГГЗ")


    # def _save_to_individual_graph(self, model: MentalModel,
    #                               task_description: str,
    #                               nodes: List[KnowledgeNode]):
    #     """Сохраняет модель в индивидуальный граф знаний."""
    #     if not self.individual_graph:
    #         return
    #
    #     # Создаем запись
    #     record = {
    #         "id": model.id,
    #         "type": "understanding",
    #         "name": model.name,
    #         "nodes": [n.id for n in nodes],
    #         "properties": model.properties,
    #         "task": task_description,
    #         "created_at": time.time(),
    #         "source": "understanding_engine"
    #     }
    #
    #     # Добавляем в ИГЗ
    #     self.individual_graph.add_knowledge(record)

    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику понимания."""
        return {
            "total_understandings": len(self.understanding_history),
            "successful": sum(1 for r in self.understanding_history
                              if not r.errors and r.status == UnderstandingStatus.COMPLETED),
            "failed": sum(1 for r in self.understanding_history if r.errors),
            "avg_time": sum(r.execution_time for r in self.understanding_history) / len(self.understanding_history) if self.understanding_history else 0,
            "avg_concepts": sum(len(r.extracted_concepts) for r in self.understanding_history) / len(self.understanding_history) if self.understanding_history else 0,
            "avg_nodes": sum(len(r.found_nodes) for r in self.understanding_history) / len(self.understanding_history) if self.understanding_history else 0
        }


def understand_task(task_description: str, load_from_db: bool = True) -> UnderstandingResult:
    """
    Быстрый запуск понимания задачи.

    Args:
        task_description: Описание задачи
        load_from_db: Загружать ли ГЗ из БД

    Returns:
        Результат понимания
    """
    from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
    from db.knowledge_db import KnowledgeDB

    graph = GlobalKnowledgeGraph()

    if load_from_db:
        print("📥 Загрузка ГЗ из БД...")
        db = KnowledgeDB()
        nodes = db.load_all_nodes()
        for node in nodes:
            graph.add_node(node)
        edges = db.load_all_edges()
        for edge in edges:
            graph.add_edge(edge)
        print(f"   ✅ Загружено {len(nodes)} узлов и {len(edges)} связей")
    else:
        print("⚠️ Используется пустой ГЗ")

    engine = UnderstandingEngine(graph, load_from_db=False)
    return engine.understand(task_description)

# # Функция-обертка для быстрого использования
# def understand_task(task_description: str) -> UnderstandingResult:
#     """
#     Быстрый запуск понимания задачи.
#
#     Args:
#         task_description: Описание задачи
#
#     Returns:
#         Результат понимания
#     """
#     from core.knowledge.global_knowledge_graph import GlobalKnowledgeGraph
#     from db.knowledge_db import KnowledgeDB
#
#     # Загружаем ГЗ из БД
#     graph = GlobalKnowledgeGraph()
#     db = KnowledgeDB()
#     nodes = db.load_all_nodes()
#     for node in nodes:
#         graph.add_node(node)
#
#     engine = UnderstandingEngine(graph)
#     return engine.understand(task_description)