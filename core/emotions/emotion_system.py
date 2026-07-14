# core/emotions/emotion_system.py
"""
Главный модуль эмоциональной системы.

Интегрирует все компоненты:
- Биграф событий/эмоций
- Движок эмоциональных реакций
- Ментальные модели
- Интуицию
- Эволюцию эмоций
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from .emotion_engine import EmotionEngine
from .mental_model import MentalModelManager
from core.emotions.emotion_base import MentalModel
from .intuition_engine import IntuitionEngine
from core.emotions.emotion_base import EmotionalEvent, EmotionalResponse, EmotionType
from core.emotions.emotion_graph import EmotionGraph


class EmotionSystem:
    """
    Полная эмоциональная подсистема AGI.

    Интеграция всех компонентов эмоционального восприятия и реагирования.
    """

    def __init__(self):
        # Основные компоненты
        self.engine = EmotionEngine()
        self.models = MentalModelManager()
        self.intuition = IntuitionEngine(self.engine.graph)

        # Состояние системы
        self.current_emotions: List[EmotionalResponse] = []
        self.emotion_history: List[Dict] = []
        self.insight_history: List[Dict] = []

        # Связь с другими подсистемами
        self.reflex_system = None  # Будет подключен позже
        self.instinct_system = None  # Будет подключен позже

        print("✅ Полная эмоциональная подсистема инициализирована")

    def synchronize_with_knowledge_graph(self, model_id: str) -> bool:
        """
        Синхронизирует ментальную модель с глобальным графом знаний.
        """
        model = self.models.get_model(model_id)
        if not model:
            return False

        try:
            from core.knowledge.knowledge_node import KnowledgeNode
            from db.knowledge_db import KnowledgeDB

            db = KnowledgeDB()

            # Создаём узел в ГЗ
            node = KnowledgeNode(
                id=model.id,
                name=model.name,
                node_type="mental_model",
                properties=list(model.attributes.keys()),
                description=f"Ментальная модель: {model.name} (тип: {model.type})"
            )

            # Добавляем эмбеддинг
            if hasattr(model, 'embedding') and model.embedding is not None:
                node.embedding = model.embedding

            # Сохраняем
            db.save_node(node)
            print(f"✅ Модель {model.name} синхронизирована с ГЗ")
            return True

        except Exception as e:
            print(f"⚠️ Ошибка синхронизации модели {model.name}: {e}")
            return False

    def process_sensory_input(self, sensory_data: Dict) -> List[EmotionalResponse]:
        """
        Обрабатывает сенсорные данные и генерирует эмоциональные реакции.
        """
        # 1. Создаем событие из сенсорных данных
        event = self._sensory_to_event(sensory_data)

        # 2. Генерируем эмоциональные реакции
        responses = self.engine.process_event(event)

        # 3. Обновляем текущее состояние
        self.current_emotions = responses

        # 4. Сохраняем в историю
        self.emotion_history.append({
            'timestamp': event.timestamp,
            'event': event,
            'responses': responses
        })

        # 5. Проверяем интуитивные инсайты (с обработкой ошибок)
        try:
            insight = self.intuition.get_insight(event)
            if insight and insight.get('confidence', 0) > 0.7:
                self.insight_history.append(insight)
                print(f"💡 Интуитивный инсайт: {insight.get('explanation', '')}")
        except Exception as e:
            # Игнорируем ошибки интуиции, чтобы не прерывать основной процесс
            pass

        return responses


    # def process_sensory_input(self, sensory_data: Dict) -> List[EmotionalResponse]:
    #     """
    #     Обрабатывает сенсорные данные и генерирует эмоциональные реакции.
    #
    #     Args:
    #         sensory_data: {
    #             'vision': np.ndarray,
    #             'sound': np.ndarray,
    #             'smell': np.ndarray,
    #             'context': dict
    #         }
    #     """
    #     # 1. Создаем событие из сенсорных данных
    #     event = self._sensory_to_event(sensory_data)
    #
    #     # 2. Генерируем эмоциональные реакции
    #     responses = self.engine.process_event(event)
    #
    #     # 3. Обновляем текущее состояние
    #     self.current_emotions = responses
    #
    #     # 4. Сохраняем в историю
    #     self.emotion_history.append({
    #         'timestamp': event.timestamp,
    #         'event': event,
    #         'responses': responses
    #     })
    #
    #     # 5. Проверяем интуитивные инсайты
    #     insight = self.intuition.get_insight(event)
    #     if insight and insight['confidence'] > 0.7:
    #         self.insight_history.append(insight)
    #         print(f"💡 Интуитивный инсайт: {insight['explanation']}")
    #
    #     return responses

    def get_emotional_state(self) -> Dict:
        """
        Возвращает текущее эмоциональное состояние.
        """
        if not self.current_emotions:
            return {'state': 'neutral', 'intensity': 0.0}

        # Определяем доминирующую эмоцию
        dominant = max(self.current_emotions, key=lambda x: x.intensity)

        return {
            'dominant_emotion': dominant.emotion_type.value,
            'intensity': dominant.intensity,
            'valence': dominant.valence,
            'arousal': dominant.arousal,
            'all_emotions': [{
                'type': e.emotion_type.value,
                'intensity': e.intensity,
                'valence': e.valence
            } for e in self.current_emotions]
        }

    def trace_emotional_chain(self, depth: int = 10) -> List[Dict]:
        """
        Трассирует цепочку эмоциональных реакций.
        """
        if not self.emotion_history:
            return []

        # Начинаем с последней реакции
        last_response = self.emotion_history[-1]['responses'][0]
        chain = self.engine.trace_response_chain(last_response)

        return chain[:depth]

    def predict_emotional_development(self, emotion_type: EmotionType,
                                      max_depth: int = 5) -> List[List[str]]:
        """
        Прогнозирует развитие эмоциональной цепочки.
        """
        return self.engine.predict_emotion_chain(emotion_type, max_depth)

    def compare_mental_models(self, model1_name: str, model2_name: str) -> Dict:
        """
        Сравнивает две ментальные модели.
        """
        # Находим модели по имени
        model1 = None
        model2 = None

        for m in self.models.models.values():
            if m.name == model1_name:
                model1 = m
            if m.name == model2_name:
                model2 = m

        if not model1 or not model2:
            return {'error': 'Model not found'}

        return self.models.compare_models(model1.id, model2.id)

    def synchronize_mental_models(self, model1_name: str, model2_name: str):
        """
        Синхронизирует две ментальные модели.
        """
        # Находим модели по имени
        model1 = None
        model2 = None

        for m in self.models.models.values():
            if m.name == model1_name:
                model1 = m
            if m.name == model2_name:
                model2 = m

        if not model1 or not model2:
            print(f"❌ Модели не найдены: {model1_name}, {model2_name}")
            return None

        return self.models.synchronize_models(model1.id, model2.id)

    def _sensory_to_event(self, sensory_data: Dict) -> EmotionalEvent:
        """
        Преобразует сенсорные данные в событие.
        """
        import time

        # Создаем эмбеддинг из сенсорных данных
        embedding = self._sensory_to_embedding(sensory_data)

        event = EmotionalEvent(
            id=f"event_{len(self.emotion_history)}",
            description=self._describe_sensory(sensory_data),
            timestamp=time.time(),
            context=sensory_data.get('context', {}),
            participants=sensory_data.get('participants', []),
            embedding=embedding
        )

        return event

    def _sensory_to_embedding(self, sensory_data: Dict) -> np.ndarray:
        """Преобразует сенсорные данные в эмбеддинг."""
        embedding = np.zeros(128)

        # Vision
        if 'vision' in sensory_data:
            vision = sensory_data['vision']
            if len(vision) >= 64:
                embedding[:64] = vision[:64]
            else:
                embedding[:len(vision)] = vision

        # Sound
        if 'sound' in sensory_data:
            sound = sensory_data['sound']
            if len(sound) >= 32:
                embedding[64:96] = sound[:32]
            else:
                embedding[64:64 + len(sound)] = sound

        # Smell
        if 'smell' in sensory_data:
            smell = sensory_data['smell']
            if len(smell) >= 32:
                embedding[96:128] = smell[:32]
            else:
                embedding[96:96 + len(smell)] = smell

        # Нормализуем
        norm = np.linalg.norm(embedding) + 1e-8
        return embedding / norm

    def _describe_sensory(self, sensory_data: Dict) -> str:
        """Генерирует текстовое описание сенсорных данных."""
        parts = []
        if 'vision' in sensory_data:
            parts.append("визуальный стимул")
        if 'sound' in sensory_data:
            parts.append("звуковой стимул")
        if 'smell' in sensory_data:
            parts.append("запаховой стимул")

        return f"Событие: {', '.join(parts)}"

    def connect_systems(self, reflex_system, instinct_system):
        """
        Подключает эмоциональную систему к рефлексам и инстинктам.
        """
        self.reflex_system = reflex_system
        self.instinct_system = instinct_system
        print("✅ Эмоциональная система подключена к рефлексам и инстинктам")

    def influence_reflexes(self):
        """
        Эмоции влияют на рефлексы.
        Например: страх усиливает рефлекс убегания.
        """
        if not self.reflex_system:
            return

        state = self.get_emotional_state()

        # Пример: страх усиливает рефлексы
        if state['dominant_emotion'] == EmotionType.FEAR.value:
            self.reflex_system.boost_reflex('run_away', state['intensity'])

        # Гнев усиливает агрессивные рефлексы
        if state['dominant_emotion'] == EmotionType.ANGER.value:
            self.reflex_system.boost_reflex('attack', state['intensity'])

    def influence_instincts(self):
        """
        Эмоции влияют на инстинкты.
        Например: страх может подавлять инстинкт исследования.
        """
        if not self.instinct_system:
            return

        state = self.get_emotional_state()

        # Пример: страх подавляет инстинкт исследования
        if state['dominant_emotion'] == EmotionType.FEAR.value:
            self.instinct_system.suppress_instinct('explore', state['intensity'])

        # Радость усиливает инстинкт исследования
        if state['dominant_emotion'] == EmotionType.JOY.value:
            self.instinct_system.boost_instinct('explore', state['intensity'])