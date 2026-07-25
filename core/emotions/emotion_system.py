# core/emotions/emotion_system.py
"""
[ru] Главный модуль эмоциональной системы.

Интегрирует все компоненты:
- Биграф событий/эмоций
- Движок эмоциональных реакций
- Ментальные модели
- Интуицию
- Эволюцию эмоций

[en] The main module of the emotional system.

Integrates all components:
- Event/emotion bigraph
- Emotional response engine
- Mental models
- Intuition
- Emotional evolution
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
    [ru] Полная эмоциональная подсистема AGI.
    Интеграция всех компонентов эмоционального восприятия и реагирования.

    [en]AGI's complete emotional subsystem.
    Integration of all components of emotional perception and response.
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

        print("✅ [ru] Полная эмоциональная подсистема инициализирована")
        print("✅ [en] The complete emotional subsystem is initialized")

    def synchronize_with_knowledge_graph(self, model_id: str) -> bool:
        """
        [ru] Синхронизирует ментальную модель с глобальным графом знаний.
        [en] Synchronizes the mental model with the global knowledge graph.
        """
        model = self.models.get_model(model_id)
        if not model:
            return False

        try:
            from core.knowledge.knowledge_node import KnowledgeNode
            from db.knowledge_db import KnowledgeDB

            db = KnowledgeDB()

            # Создаём узел в ГЗ
            # [ru]
            # [en]
            node = KnowledgeNode(
                id=model.id,
                name=model.name,
                node_type="mental_model",
                properties=list(model.attributes.keys()),
                description=f"Ментальная модель: {model.name} (тип: {model.type})"
            )

            # [ru] Добавляем эмбеддинг
            # [en] Adding embedding
            if hasattr(model, 'embedding') and model.embedding is not None:
                node.embedding = model.embedding

            # [ru] Сохраняем
            # [en] Save
            db.save_node(node)
            print(f"✅ Модель {model.name} синхронизирована с ГЗ")
            print(f"✅ Model {model.name} synchronized with the KG")
            return True

        except Exception as e:
            print(f"Ошибка синхронизации модели {model.name}: {e}")
            print(f"Model synchronization error {model.name}: {e}")
            return False

    def process_sensory_input(self, sensory_data: Dict) -> List[EmotionalResponse]:
        """
        [ru] Обрабатывает сенсорные данные и генерирует эмоциональные реакции.
        [en] Processes sensory data and generates emotional responses.
        """
        # [ru] 1. Создаем событие из сенсорных данных
        # [en] 1. Create an event from sensor data
        event = self._sensory_to_event(sensory_data)

        # [ru] 2. Генерируем эмоциональные реакции
        # [en] 2. Generate emotional reactions
        responses = self.engine.process_event(event)

        # [ru] 3. Обновляем текущее состояние
        # [en] 3. Update the current state
        self.current_emotions = responses

        # [ru] 4. Сохраняем в историю
        # [en] 4. Save to history
        self.emotion_history.append({
            'timestamp': event.timestamp,
            'event': event,
            'responses': responses
        })

        # [ru] 5. Проверяем интуитивные инсайты (с обработкой ошибок)
        # [en] 5. Testing intuitive insights (with error handling)
        try:
            insight = self.intuition.get_insight(event)
            if insight and insight.get('confidence', 0) > 0.7:
                self.insight_history.append(insight)
                print(f"[ru] Интуитивный инсайт: {insight.get('explanation', '')}")
                print(f"[en] Intuitive insight: {insight.get('explanation', '')}")
        except Exception as e:
            # [ru] Игнорируем ошибки интуиции, чтобы не прерывать основной процесс
            # [en] Ignore intuition errors to avoid interrupting the main process
            pass

        return responses


    def get_emotional_state(self) -> Dict:
        """
        [ru] Возвращает текущее эмоциональное состояние.
        [en] Returns the current emotional state.
        """
        if not self.current_emotions:
            return {'state': 'neutral', 'intensity': 0.0}

        # [ru] Определяем доминирующую эмоцию
        # [en] Identifying the dominant emotion
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
        [ru] Трассирует цепочку эмоциональных реакций.
        [en] Traces the chain of emotional reactions.
        """
        if not self.emotion_history:
            return []

        # Начинаем с последней реакции
        # [ru]
        # [en]
        last_response = self.emotion_history[-1]['responses'][0]
        chain = self.engine.trace_response_chain(last_response)

        return chain[:depth]

    def predict_emotional_development(self, emotion_type: EmotionType,
                                      max_depth: int = 5) -> List[List[str]]:
        """
        [ru] Прогнозирует развитие эмоциональной цепочки.
        [en] Predicts the development of the emotional chain.
        """
        return self.engine.predict_emotion_chain(emotion_type, max_depth)

    def compare_mental_models(self, model1_name: str, model2_name: str) -> Dict:
        """
        [ru] Сравнивает две ментальные модели.
        [en] Compares two mental models.

        """
        # [ru] Находим модели по имени
        # [en] Find models by name
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
        [ru] Синхронизирует две ментальные модели.
        [en] Synchronizes two mental models.
        """

        # [ru] Находим модели по имени
        # [en] Find models by name
        model1 = None
        model2 = None

        for m in self.models.models.values():
            if m.name == model1_name:
                model1 = m
            if m.name == model2_name:
                model2 = m

        if not model1 or not model2:
            print(f"[ru] Модели не найдены: {model1_name}, {model2_name}")
            print(f"[en] No models found: {model1_name}, {model2_name}")
            return None

        return self.models.synchronize_models(model1.id, model2.id)

    def _sensory_to_event(self, sensory_data: Dict) -> EmotionalEvent:
        """
        [ru] Преобразует сенсорные данные в событие.
        [en] Converts sensor data into an event.

        """
        import time

        # [ru] Создаем эмбеддинг из сенсорных данных
        # [en] Creating an embedding from sensory data
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
        """
        [ru] Преобразует сенсорные данные в эмбеддинг.
        [en] Converts sensory data into embedding.
        """
        embedding = np.zeros(128)

        # [ru] Зрение
        # [en] Vision
        if 'vision' in sensory_data:
            vision = sensory_data['vision']
            if len(vision) >= 64:
                embedding[:64] = vision[:64]
            else:
                embedding[:len(vision)] = vision

        # [ru] Слух
        # [en] Sound
        if 'sound' in sensory_data:
            sound = sensory_data['sound']
            if len(sound) >= 32:
                embedding[64:96] = sound[:32]
            else:
                embedding[64:64 + len(sound)] = sound

        # [ru] Запах
        # [en] Smell
        if 'smell' in sensory_data:
            smell = sensory_data['smell']
            if len(smell) >= 32:
                embedding[96:128] = smell[:32]
            else:
                embedding[96:96 + len(smell)] = smell

        # [ru] Нормализуем
        # [en] Let's normalize
        norm = np.linalg.norm(embedding) + 1e-8
        return embedding / norm

    def _describe_sensory(self, sensory_data: Dict) -> str:
        """
        [ru] Генерирует текстовое описание сенсорных данных.
        [en] Generates a text description of sensory data.
        """
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
        [ru] Подключает эмоциональную систему к рефлексам и инстинктам.
        [en] Connects the emotional system to reflexes and instincts.
        """
        self.reflex_system = reflex_system
        self.instinct_system = instinct_system
        print("[ru] Эмоциональная система подключена к рефлексам и инстинктам")
        print("[en] The emotional system is connected to reflexes and instincts.")

    def influence_reflexes(self):
        """
        [ru] Эмоции влияют на рефлексы. Например: страх усиливает рефлекс убегания.
        [en] Emotions influence reflexes. For example, fear increases the flight reflex.
        """
        if not self.reflex_system:
            return

        state = self.get_emotional_state()

        # [ru] Пример: страх усиливает рефлексы
        # [en] Example: fear enhances reflexes
        if state['dominant_emotion'] == EmotionType.FEAR.value:
            self.reflex_system.boost_reflex('run_away', state['intensity'])

        # [ru] Гнев усиливает агрессивные рефлексы
        # [en] Anger increases aggressive reflexes
        if state['dominant_emotion'] == EmotionType.ANGER.value:
            self.reflex_system.boost_reflex('attack', state['intensity'])

    def influence_instincts(self):
        """
        [ru] Эмоции влияют на инстинкты. Например: страх может подавлять инстинкт исследования.
        [en] Emotions influence instincts. For example, fear can suppress the instinct to explore.
        """
        if not self.instinct_system:
            return

        state = self.get_emotional_state()

        # [ru] Пример: страх подавляет инстинкт исследования
        # [en] Example: fear suppresses the instinct of exploration
        if state['dominant_emotion'] == EmotionType.FEAR.value:
            self.instinct_system.suppress_instinct('explore', state['intensity'])

         # [ru] Радость усиливает инстинкт исследования
        # [en] Joy strengthens the instinct of exploration
        if state['dominant_emotion'] == EmotionType.JOY.value:
            self.instinct_system.boost_instinct('explore', state['intensity'])