# core/emotions/emotion_graph.py
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

    [ru] Биграф событий и эмоциональных реакций с объектами-связями.
    [en] Bigraph of events and emotional reactions with objects-links.
    """

    def __init__(self):
        # [ru] Узлы
        # [en] Nodes
        self.events: Dict[str, EmotionalEvent] = {}
        self.emotions: Dict[str, EmotionalResponse] = {}
        self.models: Dict[str, MentalModel] = {}

        # [ru] Связи как объекты
        # [en] Relationships as objects
        self.causal_links: Dict[str, CausalLink] = {}
        self.emotion_chain_links: Dict[str, EmotionChainLink] = {}
        self.event_emotion_links: Dict[str, EventEmotionLink] = {}
        self.emotion_event_links: Dict[str, EmotionEventLink] = {}

        # [ru] Индексы для быстрого поиска
        # [en] Indexes for quick searching
        self._event_outgoing: Dict[str, Set[str]] = {}
        self._event_incoming: Dict[str, Set[str]] = {}
        self._emotion_outgoing: Dict[str, Set[str]] = {}
        self._emotion_incoming: Dict[str, Set[str]] = {}

        # [ru] Графы для визуализации
        # [en] Graphs for visualization
        self.event_graph = nx.DiGraph()
        self.emotion_graph = nx.DiGraph()
        self.event_to_emotion = nx.DiGraph()
        self.emotion_to_event = nx.DiGraph()

        # [ru] Эмбеддинги для сравнения
        # [en] Embeddings for comparison
        self.event_embeddings: List[np.ndarray] = []
        self.emotion_embeddings: List[np.ndarray] = []

        self._initialize_indexes()

        print("[ru] Биграф событий/эмоций инициализирован (с объектами-связями)")
        print("[en] Event/emotion bigraph initialized (with relationship objects)")

    def _initialize_indexes(self):
        """
        [ru] Инициализирует индексы для быстрого поиска.
        [en] Initializes indexes for fast searching.
        """
        self._event_outgoing = {}
        self._event_incoming = {}
        self._emotion_outgoing = {}
        self._emotion_incoming = {}

    def _add_to_index(self, index: Dict[str, Set[str]], key: str, link_id: str):
        """
        [ru] Добавляет связь в индекс.
        [en] Adds a relationship to the index.
        """
        if key not in index:
            index[key] = set()
        index[key].add(link_id)

    # ============================================================
    # [ru] МЕТОДЫ ДЛЯ ДОБАВЛЕНИЯ УЗЛОВ
    # [en] METHODS FOR ADDING NODES
    # ============================================================

    def add_event(self, event: EmotionalEvent):
        """
        [ru] Добавляет событие в граф.
        [en] Adds an event to the graph.
        """
        self.events[event.id] = event
        self.event_graph.add_node(event.id, embedding=event.embedding)
        self.event_embeddings.append(event.embedding)

    def add_emotion(self, emotion: EmotionalResponse):
        """
        [ru] Добавляет эмоциональную реакцию в граф.
        [en] Adds an emotional reaction to the graph.
        """
        # [ru] Получаем строковое представление типа эмоции
        # [en] We get a string representation of the emotion type
        emotion_type_str = emotion.emotion_type.value if hasattr(emotion.emotion_type, 'value') else str(emotion.emotion_type)

        self.emotions[emotion_type_str] = emotion
        self.emotion_graph.add_node(emotion_type_str, embedding=emotion.embedding)
        self.emotion_embeddings.append(emotion.embedding)

    # ============================================================
    # [ru] МЕТОДЫ ДЛЯ СВЯЗЕЙ
    # [en] METHODS FOR CONNECTIONS
    # ============================================================

    def add_causal_link(self, event1_id: str, event2_id: str,
                        weight: float = 1.0, delay: float = 0.0,
                        probability: float = 0.5) -> CausalLink:
        """
        [ru] Добавляет причинно-следственную связь между событиями.
        [en] Adds a cause and effect relationship between events.
        """
        link = LinkFactory.create_causal_link(
            event1_id, event2_id, weight, delay, probability
        )

        self.causal_links[link.id] = link
        self._add_to_index(self._event_outgoing, event1_id, link.id)
        self._add_to_index(self._event_incoming, event2_id, link.id)

        # [ru] Обновляем граф для визуализации
        # [en] Updating the graph for visualization
        self.event_graph.add_edge(event1_id, event2_id,
                                  weight=weight, delay=delay, link_id=link.id)

        return link

    def add_emotion_chain(self, emotion1_type: EmotionType,
                          emotion2_type: EmotionType,
                          weight: float = 1.0) -> EmotionChainLink:
        """
        [ru] Добавляет связь: эмоция1 → эмоция2.
        [en] Adds a connection: emotion1 → emotion2.
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

        [ru] Добавляет связь: событие → эмоция.
        [en] Adds a connection: event → emotion.
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

        [ru] Добавляет связь: эмоция → событие (действие).
        [en] Adds a connection: emotion → event (action).
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
    # [ru] МЕТОДЫ ДЛЯ ПОЛУЧЕНИЯ СВЯЗЕЙ
    # [en] METHODS FOR GETTING CONNECTIONS
    # ============================================================

    def get_causal_links_from_event(self, event_id: str) -> List[CausalLink]:
        """
        [ru] Возвращает все причинные связи, исходящие из события.
        [en] Returns all causal relationships emanating from an event.
        """
        link_ids = self._event_outgoing.get(event_id, set())
        return [self.causal_links[lid] for lid in link_ids
                if lid in self.causal_links]

    def get_causal_links_to_event(self, event_id: str) -> List[CausalLink]:
        """
        [ru] Возвращает все причинные связи, входящие в событие.
        [en] Returns all causal relationships included in the event.
        """
        link_ids = self._event_incoming.get(event_id, set())
        return [self.causal_links[lid] for lid in link_ids
                if lid in self.causal_links]

    def get_event_emotion_links(self, event_id: str) -> List[EventEmotionLink]:
        """
        [ru] Возвращает все связи событие→эмоция.
        [en] Returns all event→emotion relationships.
        """
        link_ids = self._event_outgoing.get(event_id, set())
        return [self.event_emotion_links[lid] for lid in link_ids
                if lid in self.event_emotion_links]

    def get_emotion_event_links(self, emotion_type: str) -> List[EmotionEventLink]:
        """
        [ru] Возвращает все связи эмоция→событие.
        [en] Returns all emotion→event connections.
        """
        link_ids = self._emotion_outgoing.get(emotion_type, set())
        return [self.emotion_event_links[lid] for lid in link_ids
                if lid in self.emotion_event_links]

    def successors(self, node_id: str) -> List[str]:
        """
        [ru] Возвращает наследников узла.
        [en] Returns the descendants of a node.
        """
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
        """
        [ru] Возвращает предшественников узла.
        [en] Returns the predecessors of a node.
        """
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
    # [ru] МЕТОДЫ ДЛЯ ПОИСКА
    # [en]SEARCH METHODS
    # ============================================================

    def get_similar_events(self, embedding: np.ndarray,
                           top_k: int = 5) -> List[Tuple[str, float]]:
        """
        [ru] Находит похожие события по эмбеддингу.
        [en] Finds similar events by embedding.
        """
        similarities = []
        for event_id, event in self.events.items():
            sim = self._cosine_similarity(embedding, event.embedding)
            similarities.append((event_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def trace_event_chain(self, event_id: str) -> List[List[str]]:
        """
        [ru] Трассирует цепочку событий: причины данного события.
        [en] Traces the chain of events: the causes of a given event.
        """
        chains = []

        def trace_back(current: str, path: List[str]):
            path.append(current)
            chains.append(path.copy())

            for predecessor in self.event_graph.predecessors(current):
                trace_back(predecessor, path.copy())

        #
        # [ru] Проверяем, существует ли событие.
        # [en] Checking if an event exists.
        if event_id not in self.events:
            return [[event_id]]

        trace_back(event_id, [])
        return chains

    def get_emotion_chain(self, start_emotion: EmotionType,
                          max_depth: int = 10) -> List[List[str]]:
        """
        [ru] Находит все цепочки эмоций, начинающиеся с данной.
        [en] Finds all emotion chains that start with the given one.
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
        """
        [ru] Косинусное сходство.
        [en] Cosine similarity.
        """
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # ============================================================
    # [ru] МЕТОДЫ ДЛЯ ВИЗУАЛИЗАЦИИ
    # [en] METHODS FOR VISUALIZATION
    # ============================================================

    def to_dict(self) -> Dict:
        """
        [ru] Сериализация графа..
        [en] Graph serialization..
        """
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
        """
        [ru] Визуализация биграфа.
        [en] Bigraph visualization.
        """
        import matplotlib.pyplot as plt

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # [ru] 1. Граф событий
        # [en] 1. Event graph
        nx.draw(self.event_graph, ax=ax1, with_labels=True,
                node_color='lightblue', node_size=500, font_size=8)
        ax1.set_title('Граф событий (причинно-следственные связи)')

        # [ru] 2. Граф эмоций
        # [en] 2. Emotion graph
        nx.draw(self.emotion_graph, ax=ax2, with_labels=True,
                node_color='pink', node_size=500, font_size=8)
        ax2.set_title('Граф эмоций (порождение эмоций)')

        # [ru] 3. Связи событие → эмоция
        # [en] 3. Event → emotion links
        nx.draw(self.event_to_emotion, ax=ax3, with_labels=True,
                node_color='lightgreen', node_size=500, font_size=8)
        ax3.set_title('Связи: Событие → Эмоция')

        # [ru] 4. Связи эмоция → событие
        # [en] 4. Emotion → event connections
        nx.draw(self.emotion_to_event, ax=ax4, with_labels=True,
                node_color='orange', node_size=500, font_size=8)
        ax4.set_title('[ru] Связи: Эмоция → Событие [en] Connections: Emotion → Event')

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        print(f"✅ [ru] Биграф сохранен в {path}")
        print(f"✅ [en] The bigraph is saved in {path}")

    def get_links_statistics(self) -> Dict[str, Any]:
        """
        [ru] Возвращает статистику по всем связям.
        [en] Returns statistics for all connections.
        """
        total_links = (len(self.causal_links) +
                       len(self.emotion_chain_links) +
                       len(self.event_emotion_links) +
                       len(self.emotion_event_links))

        # [ru] Считаем средние веса
        # [en] We calculate average weights
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
        [ru]

        Args:
            node_id: ID of node
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

