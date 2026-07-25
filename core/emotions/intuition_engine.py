# core/emotions/intuition_engine.py (расширенный)

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
from .emotion_graph import EmotionGraph
from .emotion_base import EmotionalEvent, EmotionalResponse, EmotionType
import time

class IntuitionEngine:
    """

    [ru] Движок интуиции. Предсказывает последствия действий на основе прошлого опыта.
    [en] Intuition Engine. Predicts the consequences of actions based on past experience.
    """

    def __init__(self, emotion_graph: EmotionGraph):
        self.graph = emotion_graph
        self.insight_cache = {}
        self.chain_cache = {}

        # Память интуиции: (ситуация) → (последствие, эмоция, уверенность)
        self.intuition_memory: Dict[str, List[Dict]] = defaultdict(list)

        # История опыта для обучения интуиции
        self.experience_history: List[Dict] = []

        print("✅ Движок интуиции инициализирован")

    def learn_from_experience(self, situation: str, action: str,
                              consequence: str, emotion: str,
                              success: bool, intensity: float = 1.0):
        """
        Обучает интуицию на основе опыта.
        """
        memory_key = f"{situation}_{action}"

        memory = {
            'situation': situation,
            'action': action,
            'consequence': consequence,
            'emotion': emotion,
            'success': success,
            'intensity': intensity,
            'weight': 1.0 if success else -1.0,
            'timestamp': time.time()
        }

        self.intuition_memory[memory_key].append(memory)

        # [ru] Ограничиваем размер памяти
        # [en] Limiting memory size
        if len(self.intuition_memory[memory_key]) > 100:
            self.intuition_memory[memory_key] = self.intuition_memory[memory_key][-100:]

        # Сохраняем в историю опыта
        # [ru]
        # [en]
        self.experience_history.append(memory)

        # [ru] СОХРАНЯЕМ В БД
        # [en] SAVE IN THE DB
        try:
            from db.emotion_db import EmotionDB
            db = EmotionDB()
            db.save_intuition_memory(
                situation=situation,
                action=action,
                consequence=consequence,
                emotion=emotion,
                success=success,
                intensity=intensity
            )
        except Exception as e:
            print(f"[ru] Ошибка сохранения интуитивной памяти: {e}")
            print(f"[en] Intuitive Memory Preservation Error: {e}")

        print(f"[ru] Интуиция запомнила: {situation} → {action} → {consequence} ({emotion})")
        print(f"[en] Intuition remembered: {situation} → {action} → {consequence} ({emotion})")


    def predict_consequence(self, situation: str, action: str) -> Optional[Dict]:
        """
        [ru] Предсказывает последствие действия на основе интуиции.
        [en] Predicts the consequences of an action based on intuition.
        Returns:
            dict: {
                'consequence': str,
                'emotion': str,
                'probability': float,
                'intensity': float
            }
        """
        memory_key = f"{situation}_{action}"

        if memory_key not in self.intuition_memory:
            return None

        memories = self.intuition_memory[memory_key]

        # [ru] Анализируем память
        # [en] Analyzing memory
        consequences = {}
        emotions = {}
        total_weight = 0

        for mem in memories:
            key = (mem['consequence'], mem['emotion'])
            if key not in consequences:
                consequences[key] = 0
                emotions[key] = mem['emotion']
            consequences[key] += mem['weight']
            total_weight += abs(mem['weight'])

        if total_weight == 0:
            return None

        # [ru] Находим наиболее вероятное последствие
        # [en] We find the most probable consequence
        best_key = max(consequences.items(), key=lambda x: x[1])
        (consequence, emotion), weight = best_key


        # [ru] Вычисляем вероятность и интенсивность
        # [en] We calculate the probability and intensity
        probability = abs(weight) / total_weight
        intensity = min(1.0, abs(weight) / 10.0)

        return {
            'consequence': consequence,
            'emotion': emotion,
            'probability': probability,
            'intensity': intensity,
            'confidence': min(1.0, len(memories) / 20.0)
        }

    def get_intuition_decision(self, situation: str, actions: List[str]) -> Dict:
        """
        [ru] Принимает интуитивное решение на основе опыта.
        [en] Makes an intuitive decision based on experience.
        Args:
            situation: Текущая ситуация
            actions: Список возможных действий

        Returns:
            dict: {
                'best_action': str,
                'predictions': List[Dict],
                'confidence': float
            }
        """
        predictions = []

        for action in actions:
            pred = self.predict_consequence(situation, action)
            if pred:
                predictions.append({
                    'action': action,
                    'consequence': pred['consequence'],
                    'emotion': pred['emotion'],
                    'probability': pred['probability'],
                    'intensity': pred['intensity'],
                    'confidence': pred['confidence']
                })

        if not predictions:
            return {
                'best_action': None,
                'predictions': [],
                'confidence': 0.0,
                'message': 'Нет опыта для принятия решения'
            }


        # [ru]  Выбираем действие с наилучшим прогнозом (предпочитаем действия с положительными эмоциями)
        # [en] We choose the action with the best prognosis (we prefer actions with positive emotions)
        best = max(predictions, key=lambda x:
        x['probability'] * (1 if x['emotion'] in ['joy', 'love', 'trust'] else -0.5))

        return {
            'best_action': best['action'],
            'predictions': predictions,
            'confidence': best['confidence'],
            'message': f"Интуиция подсказывает: {best['action']} → {best['consequence']}"
        }

    def get_insight(self, event: EmotionalEvent) -> Optional[Dict]:
        """
        [ru] Генерирует "озарение" - неожиданное решение или понимание.
        [en] Generates "insight" - an unexpected solution or understanding.
        """
        insights = []

        similar_events = self.graph.get_similar_events(event.embedding, top_k=10)

        for event_id, similarity in similar_events:
            if similarity < 0.6:
                continue

            if event_id not in self.graph.event_to_emotion:
                continue

            for emotion_type in self.graph.event_to_emotion.successors(event_id):
                if emotion_type not in self.graph.emotion_to_event:
                    continue

                for consequence in self.graph.emotion_to_event.successors(emotion_type):
                    if consequence != event_id:
                        insights.append({
                            'from_event': event_id,
                            'through_emotion': emotion_type,
                            'to_event': consequence,
                            'confidence': similarity * 0.7
                        })

        if insights:
            best_insight = max(insights, key=lambda x: x['confidence'])
            return {
                'insight': best_insight,
                'explanation': f"Через эмоцию {best_insight['through_emotion']}",
                'confidence': best_insight['confidence'],
                'chain': self.find_path(event.id, best_insight['to_event'], 5)
            }

        return None

    def find_path(self, start_event: str, target_event: str,
                  max_depth: int = 10) -> List[List[str]]:
        """
        [ru] Находит пути от начального события к целевому.
        [en] Finds paths from a start event to a target event.
        """
        paths = []

        event_paths = self._find_event_path(start_event, target_event, max_depth)
        for path in event_paths:
            paths.append([f"event:{p}" for p in path])

        emotion_paths = self._find_emotion_mediated_path(start_event, target_event, max_depth)
        for path in emotion_paths:
            paths.append(path)

        paths.sort(key=len)

        cache_key = f"{start_event}->{target_event}"
        self.chain_cache[cache_key] = paths

        return paths

    def _find_event_path(self, start: str, target: str,
                         max_depth: int) -> List[List[str]]:
        """
        [ru] Находит пути в графе событий.
        [en] Finds paths in an event graph.
        """
        paths = []

        def dfs(current: str, path: List[str], depth: int):
            if depth > max_depth:
                return

            if current == target:
                paths.append(path.copy())
                return

            if current not in self.graph.event_graph:
                return

            for successor in self.graph.event_graph.successors(current):
                dfs(successor, path + [successor], depth + 1)

        if start in self.graph.event_graph:
            dfs(start, [start], 0)

        return paths

    def _find_emotion_mediated_path(self, start: str, target: str,
                                    max_depth: int) -> List[List[str]]:
        """
        [ru] Находит пути через эмоции.
        [en] Finds ways through emotions.
        """
        paths = []

        if start not in self.graph.event_to_emotion:
            return paths

        start_emotions = list(self.graph.event_to_emotion.successors(start))
        target_emotions = list(self.graph.event_to_emotion.predecessors(target))

        for em1 in start_emotions:
            for em2 in target_emotions:
                if em1 == em2:
                    paths.append([f"event:{start}", f"emotion:{em1}", f"event:{target}"])
                else:
                    emotion_path = self._find_emotion_path(em1, em2, max_depth)
                    if emotion_path:
                        full_path = [f"event:{start}"]
                        for e in emotion_path:
                            full_path.append(f"emotion:{e}")
                        full_path.append(f"event:{target}")
                        paths.append(full_path)

        return paths

    def _find_emotion_path(self, start_emotion: str, target_emotion: str,
                           max_depth: int) -> List[str]:
        """
        [ru] Находит путь в графе эмоций.
        [en] Finds a way in the emotion graph.
        """

        def dfs(current: str, path: List[str], depth: int):
            if depth > max_depth:
                return None

            if current == target_emotion:
                return path.copy()

            if current not in self.graph.emotion_graph:
                return None

            for successor in self.graph.emotion_graph.successors(current):
                result = dfs(successor, path + [successor], depth + 1)
                if result:
                    return result

            return None

        if start_emotion not in self.graph.emotion_graph:
            return []

        return dfs(start_emotion, [start_emotion], 0) or []

    def _evaluate_path_confidence(self, path: List[str]) -> float:
        """
        [ru] Оценивает вероятность пути.
        [en] Estimates the probability of a path.
        """
        if not path:
            return 0.0

        confidence = 1.0
        for i in range(len(path) - 1):
            if path[i].startswith('event:'):
                event_id = path[i].replace('event:', '')
                if path[i + 1].startswith('emotion:'):
                    emotion_type = path[i + 1].replace('emotion:', '')
                    if emotion_type in self.graph.event_to_emotion and event_id in self.graph.event_to_emotion:
                        if self.graph.event_to_emotion.has_edge(event_id, emotion_type):
                            prob = self.graph.event_to_emotion[event_id][emotion_type].get('probability', 0.5)
                            confidence *= prob
                    else:
                        confidence *= 0.5
                else:
                    confidence *= 0.5
            elif path[i].startswith('emotion:'):
                emotion_type = path[i].replace('emotion:', '')
                if path[i + 1].startswith('emotion:'):
                    next_emotion = path[i + 1].replace('emotion:', '')
                    if next_emotion in self.graph.emotion_graph and emotion_type in self.graph.emotion_graph:
                        if self.graph.emotion_graph.has_edge(emotion_type, next_emotion):
                            weight = self.graph.emotion_graph[emotion_type][next_emotion].get('weight', 0.5)
                            confidence *= weight
                    else:
                        confidence *= 0.5
                else:
                    confidence *= 0.5

        return float(np.clip(confidence, 0.0, 1.0))

    def _predict_likely_emotion(self, path: List[str]) -> Optional[str]:
        """
        [ru] Предсказывает наиболее вероятную эмоцию в конце пути.
        [en] Predicts the most likely emotion at the end of the journey.
        """
        for item in reversed(path):
            if item.startswith('emotion:'):
                return item.replace('emotion:', '')
        return None

