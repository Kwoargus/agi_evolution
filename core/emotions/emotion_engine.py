# core/emotions/emotion_engine.py (исправленный)
import numpy as np
from typing import List, Dict, Optional, Tuple
from core.emotions.emotion_base import EmotionalEvent, EmotionalResponse, EmotionType
from core.emotions.emotion_graph import EmotionGraph


class EmotionEngine:
    """
    Движок эмоциональных реакций.

    Основные функции:
    1. Распознавание эмоций по событиям
    2. Генерация цепочек эмоций
    3. Трассировка причин ЭР
    4. Прогнозирование развития эмоциональных состояний
    """

    def __init__(self):
        self.graph = EmotionGraph()
        self._initialize_base_emotions()
        self._initialize_base_links()

                # Обучаем связи событие→эмоция
        self.train_event_emotion_links()  # <-- ДОБАВЬТЕ ЭТУ СТРОКУ

        # Кэш для быстрого поиска
        self._embedding_cache = {}

        print("✅ Движок эмоциональных реакций инициализирован")

    def _initialize_base_emotions(self):
        """Инициализирует базовые эмоции."""
        # Определяем словарь базовых эмоций
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
        """Инициализирует базовые связи между эмоциями."""
        # Типичные цепочки эмоций
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
        Обрабатывает событие и возвращает эмоциональные реакции.
        """
        responses = []

        # ============================================================
        # 1. СНАЧАЛА ИЩЕМ ПРЯМЫЕ СВЯЗИ ПО ID СОБЫТИЯ
        # ============================================================
        # Проверяем, есть ли прямые связи для этого события
        event_links = self.graph.get_event_emotion_links(event.id)

        if event_links:
            # Используем прямые связи
            for link in event_links:
                emotion_type = EmotionType(link.target_id)
                emotion_data = self.graph.emotions.get(link.target_id)

                if emotion_data:
                    # Интенсивность зависит от вероятности связи
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
        # 2. ЕСЛИ НЕТ ПРЯМЫХ СВЯЗЕЙ - ИЩЕМ ПО ПОХОЖИМ СОБЫТИЯМ
        # ============================================================
        if not responses:
            similar_events = self.graph.get_similar_events(event.embedding, top_k=5)

            for event_id, similarity in similar_events:
                if similarity < 0.5:
                    continue

                # Ищем связи для похожего события
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
        # 3. ЕСЛИ НЕТ РЕАКЦИЙ - ВОЗВРАЩАЕМ НЕЙТРАЛЬНУЮ
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


    # def process_event(self, event: EmotionalEvent) -> List[EmotionalResponse]:
    #     """
    #     Обрабатывает событие и возвращает эмоциональные реакции.
    #
    #     Args:
    #         event: Событие, вызвавшее реакцию
    #
    #     Returns:
    #         Список эмоциональных реакций
    #     """
    #     responses = []
    #
    #     # 1. Находим похожие события в памяти
    #     similar_events = self.graph.get_similar_events(event.embedding, top_k=5)
    #
    #     # 2. Для каждого похожего события проверяем связи с эмоциями
    #     for event_id, similarity in similar_events:
    #         if similarity < 0.5:
    #             continue
    #
    #         # Находим эмоции, связанные с этим событием
    #         for emotion_type in self.graph.event_to_emotion.successors(event_id):
    #             emotion_data = self.graph.emotions.get(emotion_type)
    #             if emotion_data:
    #                 # Интенсивность зависит от схожести события
    #                 intensity = similarity * emotion_data.intensity
    #                 response = EmotionalResponse(
    #                     emotion_type=EmotionType(emotion_type),
    #                     intensity=intensity,
    #                     valence=emotion_data.valence,
    #                     arousal=emotion_data.arousal,
    #                     trigger_event_id=event_id,
    #                     embedding=emotion_data.embedding * intensity
    #                 )
    #                 responses.append(response)
    #
    #     # 3. Если нет сильных реакций, используем нейтральную
    #     if not responses:
    #         neutral = EmotionalResponse(
    #             emotion_type=EmotionType.TRUST,
    #             intensity=0.1,
    #             valence=0.0,
    #             arousal=0.0,
    #             trigger_event_id=event.id,
    #             embedding=np.zeros(64)
    #         )
    #         responses.append(neutral)
    #
    #     return responses

    def trace_response_chain(self, response: EmotionalResponse) -> List[Dict]:
        """
        Трассирует цепочку эмоций, приведших к данной реакции.

        Returns:
            Список шагов: событие → эмоция → событие → ...
        """
        chain = []

        # Идем назад по графу
        current_event = response.trigger_event_id
        current_emotion = response.emotion_type

        chain.append({
            'type': 'emotion',
            'data': current_emotion.value,
            'intensity': response.intensity
        })

        # Трассируем причину эмоции (событие)
        if current_event:
            # Получаем событие из графа
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

            # Трассируем причины события
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
        Предсказывает развитие эмоциональной цепочки.

        Args:
            emotion_type: Начальная эмоция
            max_depth: Максимальная глубина прогноза

        Returns:
            Список возможных цепочек эмоций
        """
        return self.graph.get_emotion_chain(emotion_type, max_depth)

    def _emotion_to_embedding(self, emotion_type: EmotionType,
                              valence: float, arousal: float) -> np.ndarray:
        """
        Преобразует эмоцию в векторное представление.
        """
        # Базовый вектор из 64 компонент
        embedding = np.zeros(64)

        # Кодируем тип эмоции (one-hot)
        emotion_idx = list(EmotionType).index(emotion_type)
        embedding[emotion_idx % 8] = 1.0

        # Добавляем валентность и возбуждение
        embedding[8:12] = [valence, arousal,
                           (valence + arousal) / 2,
                           abs(valence - arousal)]

        # Добавляем случайный шум для разнообразия
        embedding[12:] = np.random.randn(52) * 0.1

        # Нормализуем
        norm = np.linalg.norm(embedding) + 1e-8
        return embedding / norm

    def get_current_state(self) -> Dict:
        """Возвращает текущее эмоциональное состояние."""
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
        Обучает систему связывать события с эмоциями.
        Создает реалистичные связи между типами событий и эмоциональными реакциями.
        """
        print("🧠 Обучение связей событие→эмоция...")

        # Связи: событие → эмоция
        event_emotion_pairs = [
            # Опасность → Страх
            ('danger', EmotionType.FEAR, 0.9, 1.2),
            ('threat', EmotionType.FEAR, 0.8, 1.1),
            ('predator', EmotionType.FEAR, 0.9, 1.3),

            # Успех → Радость
            ('success', EmotionType.JOY, 0.8, 1.1),
            ('achievement', EmotionType.JOY, 0.7, 1.0),
            ('victory', EmotionType.JOY, 0.9, 1.2),

            # Неудача → Печаль
            ('failure', EmotionType.SADNESS, 0.7, 1.0),
            ('loss', EmotionType.SADNESS, 0.8, 1.1),
            ('defeat', EmotionType.SADNESS, 0.7, 1.0),

            # Несправедливость → Гнев
            ('injustice', EmotionType.ANGER, 0.8, 1.2),
            ('betrayal', EmotionType.ANGER, 0.9, 1.3),
            ('offense', EmotionType.ANGER, 0.7, 1.1),

            # Неожиданность → Удивление
            ('surprise', EmotionType.SURPRISE, 0.7, 1.0),
            ('unexpected', EmotionType.SURPRISE, 0.6, 0.9),
            ('shock', EmotionType.SURPRISE, 0.8, 1.1),

            # Любовь → Любовь
            ('love', EmotionType.LOVE, 0.8, 1.1),
            ('care', EmotionType.LOVE, 0.7, 1.0),
            ('affection', EmotionType.LOVE, 0.8, 1.2),

            # Предательство → Обида
            ('betrayal', EmotionType.RESENTMENT, 0.8, 1.2),
            ('deception', EmotionType.RESENTMENT, 0.7, 1.1),
            ('lie', EmotionType.RESENTMENT, 0.7, 1.0),

            # Угроза → Страх
            ('danger', EmotionType.FEAR, 0.9, 1.2),
            ('risk', EmotionType.FEAR, 0.7, 1.0),

            # Достижение → Гордость
            ('achievement', EmotionType.JOY, 0.7, 1.0),
            ('milestone', EmotionType.JOY, 0.8, 1.1),

            # Потеря → Печаль
            ('loss', EmotionType.SADNESS, 0.8, 1.1),
            ('grief', EmotionType.SADNESS, 0.9, 1.2),
        ]

        # Добавляем связи в граф
        for event_id, emotion_type, probability, intensity in event_emotion_pairs:
            # Проверяем, существует ли событие
            if event_id not in self.graph.events:
                # Создаем событие, если его нет
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

            # Добавляем связь
            self.graph.add_event_emotion_link(
                event_id,
                emotion_type,
                probability=probability,
                intensity_factor=intensity
            )

        print(f"✅ Обучено {len(event_emotion_pairs)} связей событие→эмоция")