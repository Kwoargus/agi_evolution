# core/emotions/emotion_graph.py (добавляем методы add_event и add_emotion)

import networkx as nx
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Set
from .emotion_base import EmotionalEvent, EmotionalResponse, MentalModel, EmotionType
from .links import (
    CausalLink, EmotionChainLink, EventEmotionLink, EmotionEventLink,
    BaseLink, LinkFactory
)
from core.emotions.emotion_base import EmotionalEvent, EmotionalResponse, EmotionType

class EmotionGraph:
    """
    Биграф событий и эмоциональных реакций с объектами-связями.
    """

    def __init__(self):
        # Узлы
        self.events: Dict[str, EmotionalEvent] = {}
        self.emotions: Dict[str, EmotionalResponse] = {}
        self.models: Dict[str, MentalModel] = {}

        # Связи как объекты
        self.causal_links: Dict[str, CausalLink] = {}
        self.emotion_chain_links: Dict[str, EmotionChainLink] = {}
        self.event_emotion_links: Dict[str, EventEmotionLink] = {}
        self.emotion_event_links: Dict[str, EmotionEventLink] = {}

        # Индексы для быстрого поиска
        self._event_outgoing: Dict[str, Set[str]] = {}
        self._event_incoming: Dict[str, Set[str]] = {}
        self._emotion_outgoing: Dict[str, Set[str]] = {}
        self._emotion_incoming: Dict[str, Set[str]] = {}

        # Графы для визуализации
        self.event_graph = nx.DiGraph()
        self.emotion_graph = nx.DiGraph()
        self.event_to_emotion = nx.DiGraph()
        self.emotion_to_event = nx.DiGraph()

        # Эмбеддинги для сравнения
        self.event_embeddings: List[np.ndarray] = []
        self.emotion_embeddings: List[np.ndarray] = []

        self._initialize_indexes()

        print("✅ Биграф событий/эмоций инициализирован (с объектами-связями)")

    def _initialize_indexes(self):
        """Инициализирует индексы для быстрого поиска."""
        self._event_outgoing = {}
        self._event_incoming = {}
        self._emotion_outgoing = {}
        self._emotion_incoming = {}

    def _add_to_index(self, index: Dict[str, Set[str]], key: str, link_id: str):
        """Добавляет связь в индекс."""
        if key not in index:
            index[key] = set()
        index[key].add(link_id)

    # ============================================================
    # МЕТОДЫ ДЛЯ ДОБАВЛЕНИЯ УЗЛОВ
    # ============================================================

    def add_event(self, event: EmotionalEvent):
        """
        Добавляет событие в граф.
        """
        self.events[event.id] = event
        self.event_graph.add_node(event.id, embedding=event.embedding)
        self.event_embeddings.append(event.embedding)

    def add_emotion(self, emotion: EmotionalResponse):
        """
        Добавляет эмоциональную реакцию в граф.
        """
        # Получаем строковое представление типа эмоции
        emotion_type_str = emotion.emotion_type.value if hasattr(emotion.emotion_type, 'value') else str(emotion.emotion_type)

        self.emotions[emotion_type_str] = emotion
        self.emotion_graph.add_node(emotion_type_str,
                                    embedding=emotion.embedding)
        self.emotion_embeddings.append(emotion.embedding)

    # ============================================================
    # МЕТОДЫ ДЛЯ СВЯЗЕЙ
    # ============================================================

    def add_causal_link(self, event1_id: str, event2_id: str,
                        weight: float = 1.0, delay: float = 0.0,
                        probability: float = 0.5) -> CausalLink:
        """
        Добавляет причинно-следственную связь между событиями.
        """
        link = LinkFactory.create_causal_link(
            event1_id, event2_id, weight, delay, probability
        )

        self.causal_links[link.id] = link
        self._add_to_index(self._event_outgoing, event1_id, link.id)
        self._add_to_index(self._event_incoming, event2_id, link.id)

        # Обновляем граф для визуализации
        self.event_graph.add_edge(event1_id, event2_id,
                                  weight=weight, delay=delay, link_id=link.id)

        return link

    def add_emotion_chain(self, emotion1_type: EmotionType,
                          emotion2_type: EmotionType,
                          weight: float = 1.0) -> EmotionChainLink:
        """
        Добавляет связь: эмоция1 → эмоция2.
        """
        link = LinkFactory.create_emotion_chain_link(
            emotion1_type.value, emotion2_type.value,
            weight=weight
        )

        self.emotion_chain_links[link.id] = link
        self._add_to_index(self._emotion_outgoing, emotion1_type.value, link.id)
        self._add_to_index(self._emotion_incoming, emotion2_type.value, link.id)

        self.emotion_graph.add_edge(emotion1_type.value, emotion2_type.value,
                                    weight=weight, link_id=link.id)

        return link

    def add_event_emotion_link(self, event_id: str,
                               emotion_type: EmotionType,
                               probability: float = 0.5,
                               intensity_factor: float = 1.0) -> EventEmotionLink:
        """
        Добавляет связь: событие → эмоция.
        """
        link = LinkFactory.create_event_emotion_link(
            event_id, emotion_type.value,
            probability=probability,
            intensity_factor=intensity_factor
        )

        self.event_emotion_links[link.id] = link
        self._add_to_index(self._event_outgoing, event_id, link.id)
        self._add_to_index(self._emotion_incoming, emotion_type.value, link.id)

        self.event_to_emotion.add_edge(event_id, emotion_type.value,
                                       probability=probability, link_id=link.id)

        return link

    def add_emotion_event_link(self, emotion_type: EmotionType,
                               event_id: str,
                               probability: float = 0.5,
                               action_urgency: float = 0.5) -> EmotionEventLink:
        """
        Добавляет связь: эмоция → событие (действие).
        """
        link = LinkFactory.create_emotion_event_link(
            emotion_type.value, event_id,
            probability=probability,
            action_urgency=action_urgency
        )

        self.emotion_event_links[link.id] = link
        self._add_to_index(self._emotion_outgoing, emotion_type.value, link.id)
        self._add_to_index(self._event_incoming, event_id, link.id)

        self.emotion_to_event.add_edge(emotion_type.value, event_id,
                                       probability=probability, link_id=link.id)

        return link

    # ============================================================
    # МЕТОДЫ ДЛЯ ПОЛУЧЕНИЯ СВЯЗЕЙ
    # ============================================================

    def get_causal_links_from_event(self, event_id: str) -> List[CausalLink]:
        """Возвращает все причинные связи, исходящие из события."""
        link_ids = self._event_outgoing.get(event_id, set())
        return [self.causal_links[lid] for lid in link_ids
                if lid in self.causal_links]

    def get_causal_links_to_event(self, event_id: str) -> List[CausalLink]:
        """Возвращает все причинные связи, входящие в событие."""
        link_ids = self._event_incoming.get(event_id, set())
        return [self.causal_links[lid] for lid in link_ids
                if lid in self.causal_links]

    def get_event_emotion_links(self, event_id: str) -> List[EventEmotionLink]:
        """Возвращает все связи событие→эмоция."""
        link_ids = self._event_outgoing.get(event_id, set())
        return [self.event_emotion_links[lid] for lid in link_ids
                if lid in self.event_emotion_links]

    def get_emotion_event_links(self, emotion_type: str) -> List[EmotionEventLink]:
        """Возвращает все связи эмоция→событие."""
        link_ids = self._emotion_outgoing.get(emotion_type, set())
        return [self.emotion_event_links[lid] for lid in link_ids
                if lid in self.emotion_event_links]

    def successors(self, node_id: str) -> List[str]:
        """Возвращает наследников узла."""
        successors = []

        for link in self.causal_links.values():
            if link.source_id == node_id:
                successors.append(link.target_id)

        for link in self.emotion_chain_links.values():
            if link.source_id == node_id:
                successors.append(link.target_id)

        for link in self.event_emotion_links.values():
            if link.source_id == node_id:
                successors.append(link.target_id)

        for link in self.emotion_event_links.values():
            if link.source_id == node_id:
                successors.append(link.target_id)

        return successors

    def predecessors(self, node_id: str) -> List[str]:
        """Возвращает предшественников узла."""
        predecessors = []

        for link in self.causal_links.values():
            if link.target_id == node_id:
                predecessors.append(link.source_id)

        for link in self.emotion_chain_links.values():
            if link.target_id == node_id:
                predecessors.append(link.source_id)

        for link in self.event_emotion_links.values():
            if link.target_id == node_id:
                predecessors.append(link.source_id)

        for link in self.emotion_event_links.values():
            if link.target_id == node_id:
                predecessors.append(link.source_id)

        return predecessors

    # ============================================================
    # МЕТОДЫ ДЛЯ ПОИСКА
    # ============================================================

    def get_similar_events(self, embedding: np.ndarray,
                           top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Находит похожие события по эмбеддингу.
        """
        similarities = []
        for event_id, event in self.events.items():
            sim = self._cosine_similarity(embedding, event.embedding)
            similarities.append((event_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def trace_event_chain(self, event_id: str) -> List[List[str]]:
        """
        Трассирует цепочку событий: причины данного события.
        """
        chains = []

        def trace_back(current: str, path: List[str]):
            path.append(current)
            chains.append(path.copy())

            for predecessor in self.event_graph.predecessors(current):
                trace_back(predecessor, path.copy())

        # Проверяем, существует ли событие
        if event_id not in self.events:
            return [[event_id]]

        trace_back(event_id, [])
        return chains

    def get_emotion_chain(self, start_emotion: EmotionType,
                          max_depth: int = 10) -> List[List[str]]:
        """
        Находит все цепочки эмоций, начинающиеся с данной.
        """
        chains = []

        def dfs(current: str, path: List[str], depth: int):
            if depth > max_depth:
                return

            path.append(current)
            chains.append(path.copy())

            for successor in self.emotion_graph.successors(current):
                dfs(successor, path.copy(), depth + 1)

        dfs(start_emotion.value, [], 0)
        return chains

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Косинусное сходство."""
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # ============================================================
    # МЕТОДЫ ДЛЯ ВИЗУАЛИЗАЦИИ
    # ============================================================

    def to_dict(self) -> Dict:
        """Сериализация графа."""
        return {
            'event_graph': nx.to_dict_of_dicts(self.event_graph),
            'emotion_graph': nx.to_dict_of_dicts(self.emotion_graph),
            'event_to_emotion': nx.to_dict_of_dicts(self.event_to_emotion),
            'emotion_to_event': nx.to_dict_of_dicts(self.emotion_to_event),
            'events': {k: v.__dict__ for k, v in self.events.items()},
            'emotions': {k: v.__dict__ for k, v in self.emotions.items()},
            'models': {k: v.__dict__ for k, v in self.models.items()}
        }

    def visualize(self, path: str = 'emotion_graph.png'):
        """Визуализация биграфа."""
        import matplotlib.pyplot as plt

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Граф событий
        nx.draw(self.event_graph, ax=ax1, with_labels=True,
                node_color='lightblue', node_size=500, font_size=8)
        ax1.set_title('Граф событий (причинно-следственные связи)')

        # 2. Граф эмоций
        nx.draw(self.emotion_graph, ax=ax2, with_labels=True,
                node_color='pink', node_size=500, font_size=8)
        ax2.set_title('Граф эмоций (порождение эмоций)')

        # 3. Связи событие → эмоция
        nx.draw(self.event_to_emotion, ax=ax3, with_labels=True,
                node_color='lightgreen', node_size=500, font_size=8)
        ax3.set_title('Связи: Событие → Эмоция')

        # 4. Связи эмоция → событие
        nx.draw(self.emotion_to_event, ax=ax4, with_labels=True,
                node_color='orange', node_size=500, font_size=8)
        ax4.set_title('Связи: Эмоция → Событие')

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        print(f"✅ Биграф сохранен в {path}")

    def get_links_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по всем связям."""
        total_links = (len(self.causal_links) +
                       len(self.emotion_chain_links) +
                       len(self.event_emotion_links) +
                       len(self.emotion_event_links))

        # Считаем средние веса
        def avg_weight(links_dict):
            if not links_dict:
                return 0.0
            return sum(l.weight for l in links_dict.values()) / len(links_dict)

        return {
            'total_links': total_links,
            'causal_links': len(self.causal_links),
            'emotion_chain_links': len(self.emotion_chain_links),
            'event_emotion_links': len(self.event_emotion_links),
            'emotion_event_links': len(self.emotion_event_links),
            'avg_causal_weight': avg_weight(self.causal_links),
            'avg_emotion_chain_weight': avg_weight(self.emotion_chain_links),
            'avg_event_emotion_weight': avg_weight(self.event_emotion_links),
            'avg_emotion_event_weight': avg_weight(self.emotion_event_links),
        }

    def has_node(self, node_id: str, graph_type: str = 'event') -> bool:
        """
        Проверяет существование узла в указанном графе.

        Args:
            node_id: ID узла
            graph_type: 'event', 'emotion', 'event_to_emotion', 'emotion_to_event'
        """
        if graph_type == 'event':
            return node_id in self.events or node_id in self.event_graph
        elif graph_type == 'emotion':
            return node_id in self.emotions or node_id in self.emotion_graph
        elif graph_type == 'event_to_emotion':
            return node_id in self.event_to_emotion
        elif graph_type == 'emotion_to_event':
            return node_id in self.emotion_to_event
        return False




# # core/emotions/emotion_graph.py (обновленный)
#
# import networkx as nx
# import numpy as np
# from typing import List, Dict, Optional, Tuple, Any, Set
# from .emotion_base import EmotionalEvent, EmotionalResponse, MentalModel, EmotionType
# from .links import (
#     CausalLink, EmotionChainLink, EventEmotionLink, EmotionEventLink,
#     BaseLink, LinkFactory
# )
#
#
# class EmotionGraph:
#     """
#     Биграф событий и эмоциональных реакций с объектами-связями.
#     """
#
#     def __init__(self):
#         # Узлы
#         self.events: Dict[str, EmotionalEvent] = {}
#         self.emotions: Dict[str, EmotionalResponse] = {}
#         self.models: Dict[str, MentalModel] = {}
#
#         # Связи как объекты
#         self.causal_links: Dict[str, CausalLink] = {}  # событие → событие
#         self.emotion_chain_links: Dict[str, EmotionChainLink] = {}  # эмоция → эмоция
#         self.event_emotion_links: Dict[str, EventEmotionLink] = {}  # событие → эмоция
#         self.emotion_event_links: Dict[str, EmotionEventLink] = {}  # эмоция → событие
#
#         # Индексы для быстрого поиска
#         self._event_outgoing: Dict[str, Set[str]] = {}  # source → [link_ids]
#         self._event_incoming: Dict[str, Set[str]] = {}  # target → [link_ids]
#         self._emotion_outgoing: Dict[str, Set[str]] = {}
#         self._emotion_incoming: Dict[str, Set[str]] = {}
#
#         # Графы для визуализации
#         self.event_graph = nx.DiGraph()
#         self.emotion_graph = nx.DiGraph()
#         self.event_to_emotion = nx.DiGraph()
#         self.emotion_to_event = nx.DiGraph()
#
#         self._initialize_indexes()
#
#         print("✅ Биграф событий/эмоций инициализирован (с объектами-связями)")
#
#     def _initialize_indexes(self):
#         """Инициализирует индексы для быстрого поиска."""
#         self._event_outgoing = {}
#         self._event_incoming = {}
#         self._emotion_outgoing = {}
#         self._emotion_incoming = {}
#
#     def _add_to_index(self, index: Dict[str, Set[str]], key: str, link_id: str):
#         """Добавляет связь в индекс."""
#         if key not in index:
#             index[key] = set()
#         index[key].add(link_id)
#
#     # ============================================================
#     # МЕТОДЫ ДЛЯ СВЯЗЕЙ (ОСНОВНЫЕ)
#     # ============================================================
#
#     def add_causal_link(self, event1_id: str, event2_id: str,
#                         weight: float = 1.0, delay: float = 0.0,
#                         probability: float = 0.5) -> CausalLink:
#         """
#         Добавляет причинно-следственную связь между событиями.
#         """
#         link = LinkFactory.create_causal_link(
#             event1_id, event2_id, weight, delay, probability
#         )
#
#         self.causal_links[link.id] = link
#         self._add_to_index(self._event_outgoing, event1_id, link.id)
#         self._add_to_index(self._event_incoming, event2_id, link.id)
#
#         # Обновляем граф для визуализации
#         self.event_graph.add_edge(event1_id, event2_id,
#                                   weight=weight, delay=delay, link_id=link.id)
#
#         return link
#
#     def add_emotion_chain_link(self, emotion1_type: EmotionType,
#                                emotion2_type: EmotionType,
#                                weight: float = 1.0,
#                                intensity_amplification: float = 1.0,
#                                threshold: float = 0.3) -> EmotionChainLink:
#         """
#         Добавляет связь между эмоциями.
#         """
#         link = LinkFactory.create_emotion_chain_link(
#             emotion1_type.value, emotion2_type.value,
#             weight, intensity_amplification, threshold
#         )
#
#         self.emotion_chain_links[link.id] = link
#         self._add_to_index(self._emotion_outgoing, emotion1_type.value, link.id)
#         self._add_to_index(self._emotion_incoming, emotion2_type.value, link.id)
#
#         self.emotion_graph.add_edge(emotion1_type.value, emotion2_type.value,
#                                     weight=weight, link_id=link.id)
#
#         return link
#
#     def add_event_emotion_link(self, event_id: str,
#                                emotion_type: EmotionType,
#                                probability: float = 0.5,
#                                intensity_factor: float = 1.0) -> EventEmotionLink:
#         """
#         Добавляет связь: событие → эмоция.
#         """
#         link = LinkFactory.create_event_emotion_link(
#             event_id, emotion_type.value,
#             probability=probability,
#             intensity_factor=intensity_factor
#         )
#
#         self.event_emotion_links[link.id] = link
#         self._add_to_index(self._event_outgoing, event_id, link.id)
#         self._add_to_index(self._emotion_incoming, emotion_type.value, link.id)
#
#         self.event_to_emotion.add_edge(event_id, emotion_type.value,
#                                        probability=probability, link_id=link.id)
#
#         return link
#
#     def add_emotion_event_link(self, emotion_type: EmotionType,
#                                event_id: str,
#                                probability: float = 0.5,
#                                action_urgency: float = 0.5) -> EmotionEventLink:
#         """
#         Добавляет связь: эмоция → событие (действие).
#         """
#         link = LinkFactory.create_emotion_event_link(
#             emotion_type.value, event_id,
#             probability=probability,
#             action_urgency=action_urgency
#         )
#
#         self.emotion_event_links[link.id] = link
#         self._add_to_index(self._emotion_outgoing, emotion_type.value, link.id)
#         self._add_to_index(self._event_incoming, event_id, link.id)
#
#         self.emotion_to_event.add_edge(emotion_type.value, event_id,
#                                        probability=probability, link_id=link.id)
#
#         return link
#
#     # ============================================================
#     # МЕТОДЫ ДЛЯ ПОЛУЧЕНИЯ СВЯЗЕЙ
#     # ============================================================
#
#     def get_causal_links_from_event(self, event_id: str) -> List[CausalLink]:
#         """Возвращает все причинные связи, исходящие из события."""
#         link_ids = self._event_outgoing.get(event_id, set())
#         return [self.causal_links[lid] for lid in link_ids
#                 if lid in self.causal_links]
#
#     def get_causal_links_to_event(self, event_id: str) -> List[CausalLink]:
#         """Возвращает все причинные связи, входящие в событие."""
#         link_ids = self._event_incoming.get(event_id, set())
#         return [self.causal_links[lid] for lid in link_ids
#                 if lid in self.causal_links]
#
#     def get_event_emotion_links(self, event_id: str) -> List[EventEmotionLink]:
#         """Возвращает все связи событие→эмоция."""
#         link_ids = self._event_outgoing.get(event_id, set())
#         return [self.event_emotion_links[lid] for lid in link_ids
#                 if lid in self.event_emotion_links]
#
#     def get_emotion_event_links(self, emotion_type: str) -> List[EmotionEventLink]:
#         """Возвращает все связи эмоция→событие."""
#         link_ids = self._emotion_outgoing.get(emotion_type, set())
#         return [self.emotion_event_links[lid] for lid in link_ids
#                 if lid in self.emotion_event_links]
#
#     # ============================================================
#     # МЕТОДЫ ДЛЯ ОБНОВЛЕНИЯ И АНАЛИЗА
#     # ============================================================
#
#     def update_link_from_experience(self, link_id: str,
#                                     success: bool,
#                                     context: Dict = None):
#         """
#         Обновляет связь на основе опыта.
#         """
#         link = self.get_link_by_id(link_id)
#         if link:
#             if isinstance(link, CausalLink):
#                 link.update_from_experience(success, context)
#             elif isinstance(link, BaseLink):
#                 link.record_usage(success)
#
#     def get_link_by_id(self, link_id: str) -> Optional[BaseLink]:
#         """Находит связь по ID."""
#         for links in [self.causal_links, self.emotion_chain_links,
#                       self.event_emotion_links, self.emotion_event_links]:
#             if link_id in links:
#                 return links[link_id]
#         return None
#
#     def get_links_statistics(self) -> Dict[str, Any]:
#         """Возвращает статистику по всем связям."""
#         total_links = (len(self.causal_links) +
#                        len(self.emotion_chain_links) +
#                        len(self.event_emotion_links) +
#                        len(self.emotion_event_links))
#
#        # Считаем средние веса
#        def avg_weight(links_dict):
#            if not links_dict:
#                return 0.0
#            return sum(l.weight for l in links_dict.values()) / len(links_dict)
#
#         return {
#             'total_links': total_links,
#             'causal_links': len(self.causal_links),
#             'emotion_chain_links': len(self.emotion_chain_links),
#             'event_emotion_links': len(self.event_emotion_links),
#             'emotion_event_links': len(self.emotion_event_links),
#             'avg_causal_weight': avg_weight(self.causal_links),
#             'avg_emotion_chain_weight': avg_weight(self.emotion_chain_links),
#             'avg_event_emotion_weight': avg_weight(self.event_emotion_links),
#             'avg_emotion_event_weight': avg_weight(self.emotion_event_links),
#         }
#
#     def find_weak_links(self, threshold: float = 0.3) -> List[BaseLink]:
#         """Находит слабые связи (с низким весом)."""
#         weak = []
#         for links in [self.causal_links, self.emotion_chain_links,
#                       self.event_emotion_links, self.emotion_event_links]:
#             for link in links.values():
#                 if link.weight < threshold:
#                     weak.append(link)
#         return weak
#
#     def find_strong_links(self, threshold: float = 0.8) -> List[BaseLink]:
#         """Находит сильные связи (с высоким весом)."""
#         strong = []
#         for links in [self.causal_links, self.emotion_chain_links,
#                       self.event_emotion_links, self.emotion_event_links]:
#             for link in links.values():
#                 if link.weight >= threshold:
#                     strong.append(link)
#         return strong
#
#     # ============================================================
#     # МЕТОДЫ СОВМЕСТИМОСТИ (ДЛЯ СТАРОГО КОДА)
#     # ============================================================
#
#     def successors(self, node_id: str) -> List[str]:
#         """Возвращает наследников узла (совместимость со старым кодом)."""
#         successors = []
#
#         # Проверяем все типы связей
#         for link in self.causal_links.values():
#             if link.source_id == node_id:
#                 successors.append(link.target_id)
#
#         for link in self.emotion_chain_links.values():
#             if link.source_id == node_id:
#                 successors.append(link.target_id)
#
#         for link in self.event_emotion_links.values():
#             if link.source_id == node_id:
#                 successors.append(link.target_id)
#
#         for link in self.emotion_event_links.values():
#             if link.source_id == node_id:
#                 successors.append(link.target_id)
#
#         return successors
#
#     def predecessors(self, node_id: str) -> List[str]:
#         """Возвращает предшественников узла."""
#         predecessors = []
#
#         for link in self.causal_links.values():
#             if link.target_id == node_id:
#                 predecessors.append(link.source_id)
#
#         for link in self.emotion_chain_links.values():
#             if link.target_id == node_id:
#                 predecessors.append(link.source_id)
#
#         for link in self.event_emotion_links.values():
#             if link.target_id == node_id:
#                 predecessors.append(link.source_id)
#
#         for link in self.emotion_event_links.values():
#             if link.target_id == node_id:
#                 predecessors.append(link.source_id)
#
#         return predecessors
#
#
#
# # # core/emotions/emotion_graph.py
# # import networkx as nx
# # import numpy as np
# # from typing import List, Dict, Optional, Tuple, Any, Union
# # from .emotion_base import EmotionalEvent, EmotionalResponse, MentalModel, EmotionType
# #
# #
# # class EmotionGraph:
# #     """
# #     Биграф событий и эмоциональных реакций.
# #
# #     Особенности:
# #     - Причинно-следственные связи между событиями
# #     - Порождение одних эмоций другими
# #     - Двунаправленные связи: события → эмоции → события
# #     - Поддержка ветвящихся цепочек
# #     """
# #
# #     def __init__(self):
# #         # Граф событий (причинно-следственные связи)
# #         self.event_graph = nx.DiGraph()
# #
# #         # Граф эмоций (порождение эмоций)
# #         self.emotion_graph = nx.DiGraph()
# #
# #         # Связи: событие → эмоция
# #         self.event_to_emotion = nx.DiGraph()
# #
# #         # Связи: эмоция → событие (действие)
# #         self.emotion_to_event = nx.DiGraph()
# #
# #         # Хранилище данных
# #         self.events: Dict[str, EmotionalEvent] = {}
# #         self.emotions: Dict[str, EmotionalResponse] = {}
# #         self.models: Dict[str, MentalModel] = {}
# #
# #         # Эмбеддинги для сравнения
# #         self.event_embeddings: List[np.ndarray] = []
# #         self.emotion_embeddings: List[np.ndarray] = []
# #
# #         print("✅ Биграф событий/эмоций инициализирован")
# #
# #     def add_event(self, event: EmotionalEvent):
# #         """Добавляет событие в граф."""
# #         self.events[event.id] = event
# #         self.event_graph.add_node(event.id, embedding=event.embedding)
# #         self.event_embeddings.append(event.embedding)
# #
# #     def add_emotion(self, emotion: EmotionalResponse):
# #         """Добавляет эмоциональную реакцию в граф."""
# #         self.emotions[emotion.emotion_type.value] = emotion
# #         self.emotion_graph.add_node(emotion.emotion_type.value,
# #                                     embedding=emotion.embedding)
# #         self.emotion_embeddings.append(emotion.embedding)
# #
# #     def add_causal_link(self, event1_id: str, event2_id: str,
# #                         weight: float = 1.0, delay: float = 0.0):
# #         """
# #         Добавляет причинно-следственную связь между событиями.
# #
# #         event1 → event2 (event1 вызывает event2)
# #         """
# #         self.event_graph.add_edge(event1_id, event2_id,
# #                                   weight=weight, delay=delay)
# #
# #     def add_emotion_chain(self, emotion1_type: EmotionType,
# #                           emotion2_type: EmotionType,
# #                           weight: float = 1.0):
# #         """
# #         Добавляет связь: эмоция1 → эмоция2.
# #
# #         Например: Обида → Ненависть
# #         """
# #         self.emotion_graph.add_edge(emotion1_type.value,
# #                                     emotion2_type.value,
# #                                     weight=weight)
# #
# #     def add_event_emotion_link(self, event_id: str,
# #                                emotion_type: EmotionType,
# #                                probability: float = 0.5):
# #         """
# #         Добавляет связь: событие → эмоциональная реакция.
# #         """
# #         self.event_to_emotion.add_edge(event_id, emotion_type.value,
# #                                        probability=probability)
# #
# #     def add_emotion_event_link(self, emotion_type: EmotionType,
# #                                event_id: str,
# #                                probability: float = 0.5):
# #         """
# #         Добавляет связь: эмоция → действие (событие).
# #         """
# #         self.emotion_to_event.add_edge(emotion_type.value, event_id,
# #                                        probability=probability)
# #
# #     def get_emotion_chain(self, start_emotion: EmotionType,
# #                           max_depth: int = 10) -> List[List[str]]:
# #         """
# #         Находит все цепочки эмоций, начинающиеся с данной.
# #         """
# #         chains = []
# #
# #         def dfs(current: str, path: List[str], depth: int):
# #             if depth > max_depth:
# #                 return
# #
# #             path.append(current)
# #             chains.append(path.copy())
# #
# #             for successor in self.emotion_graph.successors(current):
# #                 dfs(successor, path.copy(), depth + 1)
# #
# #         dfs(start_emotion.value, [], 0)
# #         return chains
# #
# #     def trace_emotion_origin(self, emotion_type: EmotionType) -> List[List[str]]:
# #         """
# #         Трассирует происхождение эмоции: откуда она возникла.
# #         Идет назад по графу эмоций и событий.
# #         """
# #         # Находим все пути к этой эмоции через события
# #         origins = []
# #
# #         # 1. Находим события, которые вызывают эту эмоцию
# #         for event_id in self.event_to_emotion.predecessors(emotion_type.value):
# #             path = [event_id, emotion_type.value]
# #             origins.append(path)
# #
# #             # 2. Трассируем причинно-следственные связи событий
# #             event_chain = self.trace_event_chain(event_id)
# #             for chain in event_chain:
# #                 origins.append(chain + [emotion_type.value])
# #
# #         return origins
# #
# #     def trace_event_chain(self, event_id: str) -> List[List[str]]:
# #         """
# #         Трассирует цепочку событий: причины данного события.
# #         """
# #         chains = []
# #
# #         def trace_back(current: str, path: List[str]):
# #             path.append(current)
# #             chains.append(path.copy())
# #
# #             for predecessor in self.event_graph.predecessors(current):
# #                 trace_back(predecessor, path.copy())
# #
# #         # Проверяем, существует ли событие
# #         if event_id not in self.events:
# #             return [[event_id]]
# #
# #         trace_back(event_id, [])
# #         return chains
# #
# #     def get_similar_events(self, embedding: np.ndarray,
# #                            top_k: int = 5) -> List[Tuple[str, float]]:
# #         """
# #         Находит похожие события по эмбеддингу.
# #         """
# #         similarities = []
# #         for event_id, event in self.events.items():
# #             sim = self._cosine_similarity(embedding, event.embedding)
# #             similarities.append((event_id, sim))
# #
# #         similarities.sort(key=lambda x: x[1], reverse=True)
# #         return similarities[:top_k]
# #
# #     def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
# #         """Косинусное сходство."""
# #         if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
# #             return 0.0
# #         return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
# #
# #     def to_dict(self) -> Dict:
# #         """Сериализация графа."""
# #         return {
# #             'event_graph': nx.to_dict_of_dicts(self.event_graph),
# #             'emotion_graph': nx.to_dict_of_dicts(self.emotion_graph),
# #             'event_to_emotion': nx.to_dict_of_dicts(self.event_to_emotion),
# #             'emotion_to_event': nx.to_dict_of_dicts(self.emotion_to_event),
# #             'events': {k: v.__dict__ for k, v in self.events.items()},
# #             'emotions': {k: v.__dict__ for k, v in self.emotions.items()},
# #             'models': {k: v.__dict__ for k, v in self.models.items()}
# #         }
# #
# #     def visualize(self, path: str = 'emotion_graph.png'):
# #         """Визуализация биграфа."""
# #         import matplotlib.pyplot as plt
# #
# #         fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
# #
# #         # 1. Граф событий
# #         nx.draw(self.event_graph, ax=ax1, with_labels=True,
# #                 node_color='lightblue', node_size=500, font_size=8)
# #         ax1.set_title('Граф событий (причинно-следственные связи)')
# #
# #         # 2. Граф эмоций
# #         nx.draw(self.emotion_graph, ax=ax2, with_labels=True,
# #                 node_color='pink', node_size=500, font_size=8)
# #         ax2.set_title('Граф эмоций (порождение эмоций)')
# #
# #         # 3. Связи событие → эмоция
# #         nx.draw(self.event_to_emotion, ax=ax3, with_labels=True,
# #                 node_color='lightgreen', node_size=500, font_size=8)
# #         ax3.set_title('Связи: Событие → Эмоция')
# #
# #         # 4. Связи эмоция → событие
# #         nx.draw(self.emotion_to_event, ax=ax4, with_labels=True,
# #                 node_color='orange', node_size=500, font_size=8)
# #         ax4.set_title('Связи: Эмоция → Событие')
# #
# #         plt.tight_layout()
# #         plt.savefig(path, dpi=150)
# #         print(f"✅ Биграф сохранен в {path}")