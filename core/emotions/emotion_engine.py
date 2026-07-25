# core/emotions/emotion_engine.py
import numpy as np
from typing import List, Dict, Optional, Tuple
from core.emotions.emotion_base import EmotionalEvent, EmotionalResponse, EmotionType
from core.emotions.emotion_graph import EmotionGraph


class EmotionEngine:
    """
    [ru] Движок эмоциональных реакций.
    Основные функции:
    1. Распознавание эмоций по событиям
    2. Генерация цепочек эмоций
    3. Трассировка причин ЭР
    4. Прогнозирование развития эмоциональных состояний

    [en] Emotional Response Engine.
    Main functions:
    1. Recognizing emotions from events
    2. Generating emotion chains
    3. Tracing the causes of emotional reactions
    4. Predicting the development of emotional states
    """

    def __init__(self):
        self.graph = EmotionGraph()
        self._initialize_base_emotions()
        self._initialize_base_links()

        # [ru] Обучаем связи событие→эмоция
        # [en] Teaching the event→emotion connection
        self.train_event_emotion_links()  # <-- ДОБАВЬТЕ ЭТУ СТРОКУ

        # [ru] Кэш для быстрого поиска
        # [en] Cache for fast searching
        self._embedding_cache = {}

        print("✅ Движок эмоциональных реакций инициализирован")

    def _initialize_base_emotions(self):
        """
        [ru] Инициализирует базовые эмоции.
        [en] Initializes basic emotions.
        """
        # [ru] Определяем словарь базовых эмоций
        # [en] Defining the vocabulary of basic emotions
        base_emotions = {
            EmotionType.JOY: (1.0, 0.8, 0.6),  # (valence, arousal, intensity)
            EmotionType.SADNESS: (-0.8, -0.5, 0.6),
            EmotionType.ANGER: (-0.7, 0.9, 0.8),
            EmotionType.FEAR: (-0.9, 0.7, 0.9),
            EmotionType.SURPRISE: (0.3, 0.8, 0.5),
            EmotionType.DISGUST: (-0.6, 0.1, 0.4),
            EmotionType.TRUST: (0.6, -0.2, 0.3),
            EmotionType.ANTICIPATION: (0.4, 0.5, 0.4),
        }

        for emotion_type, (valence, arousal, intensity) in base_emotions.items():
            emotion = EmotionalResponse(
                emotion_type=emotion_type,
                intensity=intensity,
                valence=valence,
                arousal=arousal,
                embedding=self._emotion_to_embedding(emotion_type, valence, arousal)
            )
            self.graph.add_emotion(emotion)

    def _initialize_base_links(self):
        """
        [ru] Инициализирует базовые связи между эмоциями.
        [en] Initializes basic connections between emotions.
        """
        # [ru] Типичные цепочки эмоций
        # [en] Typical chains of emotions
        emotion_chains = [
            (EmotionType.ANTICIPATION, EmotionType.JOY),  # Ожидание → Радость
            (EmotionType.JOY, EmotionType.TRUST),  # Радость → Доверие
            (EmotionType.ANGER, EmotionType.DISGUST),  # Гнев → Отвращение
            (EmotionType.FEAR, EmotionType.SURPRISE),  # Страх → Удивление
            (EmotionType.SADNESS, EmotionType.TRUST),  # Печаль → Доверие
            (EmotionType.SURPRISE, EmotionType.JOY),  # Удивление → Радость
        ]

        for em1, em2 in emotion_chains:
            self.graph.add_emotion_chain(em1, em2, weight=0.7)

    def process_event(self, event: EmotionalEvent) -> List[EmotionalResponse]:
        """

        [ru] Обрабатывает событие и возвращает эмоциональные реакции
        [en] Processes an event and returns emotional responses
        """
        responses = []

        # ============================================================
        # [ru] 1. СНАЧАЛА ИЩЕМ ПРЯМЫЕ СВЯЗИ ПО ID СОБЫТИЯ
        # [en] 1. FIRST, WE LOOK FOR DIRECT CONNECTIONS BY EVENT ID
        # ============================================================
        # [ru] Проверяем, есть ли прямые связи для этого события
        # [en] We check if there are direct links for this event
        event_links = self.graph.get_event_emotion_links(event.id)

        if event_links:
            # [ru] Используем прямые связи
            # [en] We use direct connections
            for link in event_links:
                emotion_type = EmotionType(link.target_id)
                emotion_data = self.graph.emotions.get(link.target_id)

                if emotion_data:
                    # [ru] Интенсивность зависит от вероятности связи
                    # [en] The intensity depends on the probability of the connection
                    intensity = link.probability * emotion_data.intensity * link.intensity_factor

                    response = EmotionalResponse(
                        emotion_type=emotion_type,
                        intensity=min(1.0, intensity),
                        valence=emotion_data.valence,
                        arousal=emotion_data.arousal,
                        trigger_event_id=event.id,
                        embedding=emotion_data.embedding * intensity
                    )
                    responses.append(response)

        # ============================================================
        # [ru] 2. ЕСЛИ НЕТ ПРЯМЫХ СВЯЗЕЙ - ИЩЕМ ПО ПОХОЖИМ СОБЫТИЯМ
        # [en] 2. IF THERE ARE NO DIRECT CONNECTIONS, WE SEARCH FOR SIMILAR EVENTS
        # ============================================================
        if not responses:
            similar_events = self.graph.get_similar_events(event.embedding, top_k=5)

            for event_id, similarity in similar_events:
                if similarity < 0.5:
                    continue

                # [ru] Ищем связи для похожего события
                # [en] Looking for connections for a similar event
                similar_links = self.graph.get_event_emotion_links(event_id)
                for link in similar_links:
                    emotion_type = EmotionType(link.target_id)
                    emotion_data = self.graph.emotions.get(link.target_id)

                    if emotion_data:
                        intensity = similarity * link.probability * emotion_data.intensity * link.intensity_factor

                        response = EmotionalResponse(
                            emotion_type=emotion_type,
                            intensity=min(1.0, intensity),
                            valence=emotion_data.valence,
                            arousal=emotion_data.arousal,
                            trigger_event_id=event_id,
                            embedding=emotion_data.embedding * intensity
                        )
                        responses.append(response)

        # ============================================================
        # [ru] 3. ЕСЛИ НЕТ РЕАКЦИЙ - ВОЗВРАЩАЕМ НЕЙТРАЛЬНУЮ
        # [en] 3. IF THERE IS NO REACTION - RETURN TO NEUTRAL
        # ============================================================
        if not responses:
            neutral = EmotionalResponse(
                emotion_type=EmotionType.TRUST,
                intensity=0.1,
                valence=0.0,
                arousal=0.0,
                trigger_event_id=event.id,
                embedding=np.zeros(64)
            )
            responses.append(neutral)

        return responses


    def trace_response_chain(self, response: EmotionalResponse) -> List[Dict]:
        """
        [ru] Трассирует цепочку эмоций, приведших к данной реакции.
        [en] Traces the chain of emotions that led to a given reaction.
        Returns:
        [ru]    Список шагов: событие → эмоция → событие → ...
        [en]    List of steps: event → emotion → event → ...
        """
        chain = []

        # [ru] Идем назад по графу
        # [en] Let's go back along the graph
        current_event = response.trigger_event_id
        current_emotion = response.emotion_type

        chain.append({
            'type': 'emotion',
            'data': current_emotion.value,
            'intensity': response.intensity
        })

        # [ru] Трассируем причину эмоции (событие)
        # [en] Tracing the cause of the emotion (event)
        if current_event:
            # [ru] Получаем событие из графа
            # [en] We receive an event from the graph
            event_obj = self.graph.events.get(current_event)
            if event_obj:
                if hasattr(event_obj, 'description'):
                    event_description = event_obj.description
                else:
                    event_description = str(event_obj)
            else:
                event_description = f"Событие {current_event}"

            chain.append({
                'type': 'event',
                'data': event_description,
                'id': current_event
            })

            # [ru] Трассируем причины события
            # [en] Tracing the causes of an event
            event_chain = self.graph.trace_event_chain(current_event)
            for ec in event_chain:
                for event_id in ec:
                    event_obj = self.graph.events.get(event_id)
                    if event_obj:
                        if hasattr(event_obj, 'description'):
                            event_description = event_obj.description
                        else:
                            event_description = str(event_obj)
                    else:
                        event_description = f"Событие {event_id}"

                    chain.append({
                        'type': 'event_cause',
                        'data': event_description,
                        'id': event_id
                    })

        return chain

    def predict_emotion_chain(self, emotion_type: EmotionType,
                              max_depth: int = 5) -> List[List[str]]:
        """
        [ru] Предсказывает развитие эмоциональной цепочки.
        [en] Predicts the development of an emotional chain.
        Args:
            emotion_type: [ru] Начальная эмоция [en] Initial emotion
            max_depth: [ru] Максимальная глубина прогноза [en] Maximum forecast depth
        Returns:
            [ru] Список возможных цепочек эмоций
            [en] List of possible chains of emotions
        """
        return self.graph.get_emotion_chain(emotion_type, max_depth)

    def _emotion_to_embedding(self, emotion_type: EmotionType,
                              valence: float, arousal: float) -> np.ndarray:
        """
        [ru]Преобразует эмоцию в векторное представление.
        [en] Converts emotion into vector representation.
        """
        # [ru] Базовый вектор из 64 компонент
        # [en] Basic vector of 64 components
        embedding = np.zeros(64)

        # [ru] Кодируем тип эмоции (one-hot)
        # [en] Encoding the type of emotion (one-hot)
        emotion_idx = list(EmotionType).index(emotion_type)
        embedding[emotion_idx % 8] = 1.0

        # [ru] Добавляем валентность и возбуждение
        # [en] Adding valence and arousal
        embedding[8:12] = [valence, arousal,
                           (valence + arousal) / 2,
                           abs(valence - arousal)]

        # [ru] Добавляем случайный шум для разнообразия
        # [en] Adding random noise for variety
        embedding[12:] = np.random.randn(52) * 0.1

        # [ru] Нормализуем
        # [en] Let's normalize
        norm = np.linalg.norm(embedding) + 1e-8
        return embedding / norm

    def get_current_state(self) -> Dict:
        """
        [ru] Возвращает текущее эмоциональное состояние.
        [en] Returns the current emotional state.
        """
        return {
            'active_emotions': [e.emotion_type.value for e in self.graph.emotions.values()],
            'graph_stats': {
                'events': len(self.graph.events),
                'emotions': len(self.graph.emotions),
                'event_edges': len(self.graph.event_graph.edges),
                'emotion_edges': len(self.graph.emotion_graph.edges)
            }
        }

    def train_event_emotion_links(self):
        """
        [ru] Обучает систему связывать события с эмоциями. Создает реалистичные связи между типами событий и эмоциональными реакциями.
        [en] Trains the system to associate events with emotions. Creates realistic connections between event types and emotional responses.
        """
        print("[ru] Обучение связей событие→эмоция...")
        print("[en] Learning event→emotion connections...")

        # [ru]  Связи: событие → эмоция
        # [en]  Relationships: event → emotion
        event_emotion_pairs = [
            # [ru] Опасность → Страх
            # [en] Danger → Fear
            ('danger', EmotionType.FEAR, 0.9, 1.2),
            ('threat', EmotionType.FEAR, 0.8, 1.1),
            ('predator', EmotionType.FEAR, 0.9, 1.3),

            # [ru] Успех → Радость
            # [en] Success → Joy
            ('success', EmotionType.JOY, 0.8, 1.1),
            ('achievement', EmotionType.JOY, 0.7, 1.0),
            ('victory', EmotionType.JOY, 0.9, 1.2),

            # [ru] Неудача → Печаль
            # [en] Failure → Sadness
            ('failure', EmotionType.SADNESS, 0.7, 1.0),
            ('loss', EmotionType.SADNESS, 0.8, 1.1),
            ('defeat', EmotionType.SADNESS, 0.7, 1.0),

            # [ru] Несправедливость → Гнев
            # [en] Injustice → Anger
            ('injustice', EmotionType.ANGER, 0.8, 1.2),
            ('betrayal', EmotionType.ANGER, 0.9, 1.3),
            ('offense', EmotionType.ANGER, 0.7, 1.1),

            # [ru] Неожиданность → Удивление
            # [en] Unexpectedness → Surprise
            ('surprise', EmotionType.SURPRISE, 0.7, 1.0),
            ('unexpected', EmotionType.SURPRISE, 0.6, 0.9),
            ('shock', EmotionType.SURPRISE, 0.8, 1.1),

            # [ru] Любовь → Любовь
            # [en] Love → Love
            ('love', EmotionType.LOVE, 0.8, 1.1),
            ('care', EmotionType.LOVE, 0.7, 1.0),
            ('affection', EmotionType.LOVE, 0.8, 1.2),

            # [ru] Предательство → Обида
            # [en] Betrayal → Resentment
            ('betrayal', EmotionType.RESENTMENT, 0.8, 1.2),
            ('deception', EmotionType.RESENTMENT, 0.7, 1.1),
            ('lie', EmotionType.RESENTMENT, 0.7, 1.0),

            # Угроза → Страх
            # [ru]
            # [en]
            ('danger', EmotionType.FEAR, 0.9, 1.2),
            ('risk', EmotionType.FEAR, 0.7, 1.0),

            # [ru] Достижение → Гордость
            # [en] Achievement → Pride
            ('achievement', EmotionType.JOY, 0.7, 1.0),
            ('milestone', EmotionType.JOY, 0.8, 1.1),

            # [ru] Потеря → Печаль
            # [en] Loss → Sadness
            ('loss', EmotionType.SADNESS, 0.8, 1.1),
            ('grief', EmotionType.SADNESS, 0.9, 1.2),
        ]

        # [ru] Добавляем связи в граф
        # [en] Adding connections to the graph
        for event_id, emotion_type, probability, intensity in event_emotion_pairs:
            # [ru] Проверяем, существует ли событие
            # [en] Checking if an event exists
            if event_id not in self.graph.events:
                # [ru] Создаем событие, если его нет
                # [en] Create an event if it doesn't exist
                from .emotion_base import EmotionalEvent
                import numpy as np
                event = EmotionalEvent(
                    id=event_id,
                    description=f"Событие: {event_id}",
                    timestamp=0,
                    context={'type': event_id},
                    participants=['system'],
                    embedding=np.random.randn(128)
                )
                self.graph.add_event(event)

            # [ru] Добавляем связь
            # [en] Adding a connection
            self.graph.add_event_emotion_link(
                event_id,
                emotion_type,
                probability=probability,
                intensity_factor=intensity
            )

        print(f"✅ Обучено {len(event_emotion_pairs)} связей событие→эмоция")
        print(f"✅ Trained {len(event_emotion_pairs)} event→emotion connections")