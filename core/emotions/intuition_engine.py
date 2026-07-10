# core/emotions/intuition_engine.py (расширенный)

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
from .emotion_graph import EmotionGraph
from .emotion_base import EmotionalEvent, EmotionalResponse, EmotionType


class IntuitionEngine:
    """
    Движок интуиции.
    Предсказывает последствия действий на основе прошлого опыта.
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

        Args:
            situation: Описание ситуации (например, "вижу яблоко рядом с волком")
            action: Действие (например, "пойти за яблоком")
            consequence: Последствие (например, "волк напал")
            emotion: Эмоция, связанная с последствием
            success: Успешно ли было действие
            intensity: Интенсивность эмоции
        """
        memory_key = f"{situation}_{action}"

        self.intuition_memory[memory_key].append({
            'situation': situation,
            'action': action,
            'consequence': consequence,
            'emotion': emotion,
            'success': success,
            'intensity': intensity,
            'weight': 1.0 if success else -1.0
        })

        # Ограничиваем размер памяти
        if len(self.intuition_memory[memory_key]) > 100:
            self.intuition_memory[memory_key] = self.intuition_memory[memory_key][-100:]

        # Сохраняем в историю опыта
        self.experience_history.append({
            'situation': situation,
            'action': action,
            'consequence': consequence,
            'emotion': emotion,
            'success': success,
            'intensity': intensity
        })

        print(f"🧠 Интуиция запомнила: {situation} → {action} → {consequence} ({emotion})")

    def predict_consequence(self, situation: str, action: str) -> Optional[Dict]:
        """
        Предсказывает последствие действия на основе интуиции.

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

        # Анализируем память
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

        # Находим наиболее вероятное последствие
        best_key = max(consequences.items(), key=lambda x: x[1])
        (consequence, emotion), weight = best_key

        # Вычисляем вероятность и интенсивность
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
        Принимает интуитивное решение на основе опыта.

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

        # Выбираем действие с наилучшим прогнозом
        # (предпочитаем действия с положительными эмоциями)
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
        Генерирует "озарение" - неожиданное решение или понимание.
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
        """Находит пути от начального события к целевому."""
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
        """Находит пути в графе событий."""
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
        """Находит пути через эмоции."""
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
        """Находит путь в графе эмоций."""

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
        """Оценивает вероятность пути."""
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
        """Предсказывает наиболее вероятную эмоцию в конце пути."""
        for item in reversed(path):
            if item.startswith('emotion:'):
                return item.replace('emotion:', '')
        return None



# # core/emotions/intuition_engine.py (исправленный - абсолютные импорты)
#
# import numpy as np
# from typing import List, Dict, Optional, Tuple
# from core.emotions.emotion_graph import EmotionGraph
# from core.emotions.emotion_base import EmotionalEvent, EmotionalResponse, EmotionType
#
#
# class IntuitionEngine:
#     """
#     Движок интуиции.
#
#     Интуиция - это бессознательный перебор цепочек в биграфе событий/эмоций,
#     приводящий к "озарению" - нахождению решения без явного логического анализа.
#     """
#
#     def __init__(self, emotion_graph: EmotionGraph):
#         self.graph = emotion_graph
#         self.insight_cache = {}
#         self.chain_cache = {}
#
#         print("✅ Движок интуиции инициализирован")
#
#     def find_path(self, start_event: str, target_event: str,
#                   max_depth: int = 10) -> List[List[str]]:
#         """
#         Находит пути от начального события к целевому.
#         """
#         paths = []
#
#         # 1. Прямой путь через события
#         event_paths = self._find_event_path(start_event, target_event, max_depth)
#         for path in event_paths:
#             paths.append([f"event:{p}" for p in path])
#
#         # 2. Путь через эмоции
#         emotion_paths = self._find_emotion_mediated_path(start_event, target_event, max_depth)
#         for path in emotion_paths:
#             paths.append(path)
#
#         paths.sort(key=len)
#
#         cache_key = f"{start_event}->{target_event}"
#         self.chain_cache[cache_key] = paths
#
#         return paths
#
#     def _find_event_path(self, start: str, target: str,
#                          max_depth: int) -> List[List[str]]:
#         """Находит пути в графе событий."""
#         paths = []
#
#         def dfs(current: str, path: List[str], depth: int):
#             if depth > max_depth:
#                 return
#
#             if current == target:
#                 paths.append(path.copy())
#                 return
#
#             # Проверяем, существует ли узел
#             if current not in self.graph.event_graph:
#                 return
#
#             for successor in self.graph.event_graph.successors(current):
#                 dfs(successor, path + [successor], depth + 1)
#
#         if start in self.graph.event_graph:
#             dfs(start, [start], 0)
#
#         return paths
#
#     def _find_emotion_mediated_path(self, start: str, target: str,
#                                     max_depth: int) -> List[List[str]]:
#         """
#         Находит пути через эмоции: событие → эмоция → событие.
#         """
#         paths = []
#
#         # Проверяем существование узлов
#         if start not in self.graph.event_to_emotion:
#             return paths
#
#         # Находим эмоции, связанные с начальным событием
#         start_emotions = list(self.graph.event_to_emotion.successors(start))
#
#         # Находим события, связанные с целевой эмоцией
#         target_emotions = list(self.graph.event_to_emotion.predecessors(target))
#
#         for em1 in start_emotions:
#             for em2 in target_emotions:
#                 if em1 == em2:
#                     # Прямая связь: событие → эмоция → событие
#                     paths.append([f"event:{start}", f"emotion:{em1}", f"event:{target}"])
#                 else:
#                     # Находим путь в графе эмоций
#                     emotion_path = self._find_emotion_path(em1, em2, max_depth)
#                     if emotion_path:
#                         full_path = [f"event:{start}"]
#                         for e in emotion_path:
#                             full_path.append(f"emotion:{e}")
#                         full_path.append(f"event:{target}")
#                         paths.append(full_path)
#
#         return paths
#
#     def _find_emotion_path(self, start_emotion: str, target_emotion: str,
#                            max_depth: int) -> List[str]:
#         """
#         Находит путь в графе эмоций.
#         """
#
#         def dfs(current: str, path: List[str], depth: int):
#             if depth > max_depth:
#                 return None
#
#             if current == target_emotion:
#                 return path.copy()
#
#             # Проверяем, существует ли узел
#             if current not in self.graph.emotion_graph:
#                 return None
#
#             for successor in self.graph.emotion_graph.successors(current):
#                 result = dfs(successor, path + [successor], depth + 1)
#                 if result:
#                     return result
#
#             return None
#
#         if start_emotion not in self.graph.emotion_graph:
#             return []
#
#         return dfs(start_emotion, [start_emotion], 0) or []
#
#     def get_insight(self, event: EmotionalEvent) -> Optional[Dict]:
#         """
#         Генерирует "озарение" - неожиданное решение или понимание.
#         """
#         insights = []
#
#         # Проверяем все похожие события
#         similar_events = self.graph.get_similar_events(event.embedding, top_k=10)
#
#         for event_id, similarity in similar_events:
#             if similarity < 0.6:
#                 continue
#
#             # Проверяем существование узла
#             if event_id not in self.graph.event_to_emotion:
#                 continue
#
#             # Находим неочевидные связи через эмоции
#             for emotion_type in self.graph.event_to_emotion.successors(event_id):
#                 # Проверяем существование узла в emotion_to_event
#                 if emotion_type not in self.graph.emotion_to_event:
#                     continue
#
#                 for consequence in self.graph.emotion_to_event.successors(emotion_type):
#                     if consequence != event_id:
#                         insights.append({
#                             'from_event': event_id,
#                             'through_emotion': emotion_type,
#                             'to_event': consequence,
#                             'confidence': similarity * 0.7
#                         })
#
#         if insights:
#             best_insight = max(insights, key=lambda x: x['confidence'])
#             return {
#                 'insight': best_insight,
#                 'explanation': f"Через эмоцию {best_insight['through_emotion']}",
#                 'confidence': best_insight['confidence'],
#                 'chain': self.find_path(event.id, best_insight['to_event'], 5)
#             }
#
#         return None
#
#     def predict_intuitive(self, event: EmotionalEvent) -> Dict:
#         """
#         Интуитивное предсказание развития событий.
#         """
#         paths = self.find_path(event.id, '', 5)
#
#         predictions = []
#         for path in paths:
#             confidence = self._evaluate_path_confidence(path)
#             predictions.append({
#                 'path': path,
#                 'confidence': confidence,
#                 'likely_emotion': self._predict_likely_emotion(path)
#             })
#
#         predictions.sort(key=lambda x: x['confidence'], reverse=True)
#
#         return {
#             'predictions': predictions[:3],
#             'intuition_score': predictions[0]['confidence'] if predictions else 0.0
#         }
#
#     def _evaluate_path_confidence(self, path: List[str]) -> float:
#         """Оценивает вероятность пути."""
#         if not path:
#             return 0.0
#
#         confidence = 1.0
#         for i in range(len(path) - 1):
#             if path[i].startswith('event:'):
#                 event_id = path[i].replace('event:', '')
#                 if path[i + 1].startswith('emotion:'):
#                     emotion_type = path[i + 1].replace('emotion:', '')
#                     if emotion_type in self.graph.event_to_emotion and event_id in self.graph.event_to_emotion:
#                         if self.graph.event_to_emotion.has_edge(event_id, emotion_type):
#                             prob = self.graph.event_to_emotion[event_id][emotion_type].get('probability', 0.5)
#                             confidence *= prob
#                     else:
#                         confidence *= 0.5
#                 else:
#                     confidence *= 0.5
#             elif path[i].startswith('emotion:'):
#                 emotion_type = path[i].replace('emotion:', '')
#                 if path[i + 1].startswith('emotion:'):
#                     next_emotion = path[i + 1].replace('emotion:', '')
#                     if next_emotion in self.graph.emotion_graph and emotion_type in self.graph.emotion_graph:
#                         if self.graph.emotion_graph.has_edge(emotion_type, next_emotion):
#                             weight = self.graph.emotion_graph[emotion_type][next_emotion].get('weight', 0.5)
#                             confidence *= weight
#                     else:
#                         confidence *= 0.5
#                 else:
#                     confidence *= 0.5
#
#         return float(np.clip(confidence, 0.0, 1.0))
#
#     def _predict_likely_emotion(self, path: List[str]) -> Optional[str]:
#         """Предсказывает наиболее вероятную эмоцию в конце пути."""
#         for item in reversed(path):
#             if item.startswith('emotion:'):
#                 return item.replace('emotion:', '')
#         return None
#
#
#
# # # core/emotions/intuition_engine.py
# # import numpy as np
# # from typing import List, Dict, Optional, Tuple, Any, Union
# # from .emotion_graph import EmotionGraph
# # from .emotion_base import EmotionalEvent, EmotionalResponse, EmotionType
# #
# #
# #
# # class IntuitionEngine:
# #     """
# #     Движок интуиции.
# #
# #     Интуиция - это бессознательный перебор цепочек в биграфе событий/эмоций,
# #     приводящий к "озарению" - нахождению решения без явного логического анализа.
# #     """
# #
# #     def __init__(self, emotion_graph: EmotionGraph):
# #         self.graph = emotion_graph
# #         self.insight_cache = {}
# #         self.chain_cache = {}
# #
# #         print("✅ Движок интуиции инициализирован")
# #
# #     def find_path(self, start_event: str, target_event: str,
# #                   max_depth: int = 10) -> List[List[str]]:
# #         """
# #         Находит пути от начального события к целевому.
# #         Использует бессознательный перебор цепочек.
# #
# #         Returns:
# #             Список возможных путей (через события и эмоции)
# #         """
# #         paths = []
# #
# #         # 1. Прямой путь через события
# #         event_paths = self._find_event_path(start_event, target_event, max_depth)
# #         for path in event_paths:
# #             paths.append([f"event:{p}" for p in path])
# #
# #         # 2. Путь через эмоции
# #         emotion_paths = self._find_emotion_mediated_path(start_event, target_event, max_depth)
# #         for path in emotion_paths:
# #             paths.append(path)
# #
# #         # Сортируем по длине
# #         paths.sort(key=len)
# #
# #         # Кэшируем для быстрого доступа
# #         cache_key = f"{start_event}->{target_event}"
# #         self.chain_cache[cache_key] = paths
# #
# #         return paths
# #
# #     def _find_event_path(self, start: str, target: str,
# #                          max_depth: int) -> List[List[str]]:
# #         """
# #         Находит пути в графе событий.
# #         """
# #         paths = []
# #
# #         def dfs(current: str, path: List[str], depth: int):
# #             if depth > max_depth:
# #                 return
# #
# #             if current == target:
# #                 paths.append(path.copy())
# #                 return
# #
# #             for successor in self.graph.event_graph.successors(current):
# #                 dfs(successor, path + [successor], depth + 1)
# #
# #         dfs(start, [start], 0)
# #         return paths
# #
# #     def _find_emotion_mediated_path(self, start: str, target: str,
# #                                     max_depth: int) -> List[List[str]]:
# #         """
# #         Находит пути через эмоции: событие → эмоция → событие.
# #         """
# #         paths = []
# #
# #         # Находим эмоции, связанные с начальным событием
# #         start_emotions = list(self.graph.event_to_emotion.successors(start))
# #
# #         # Находим события, связанные с целевой эмоцией
# #         target_emotions = list(self.graph.event_to_emotion.predecessors(target))
# #
# #         for em1 in start_emotions:
# #             for em2 in target_emotions:
# #                 if em1 == em2:
# #                     # Прямая связь: событие → эмоция → событие
# #                     paths.append([f"event:{start}", f"emotion:{em1}", f"event:{target}"])
# #                 else:
# #                     # Находим путь в графе эмоций
# #                     emotion_path = self._find_emotion_path(em1, em2, max_depth)
# #                     if emotion_path:
# #                         full_path = [f"event:{start}"]
# #                         for e in emotion_path:
# #                             full_path.append(f"emotion:{e}")
# #                         full_path.append(f"event:{target}")
# #                         paths.append(full_path)
# #
# #         return paths
# #
# #     def _find_emotion_path(self, start_emotion: str, target_emotion: str,
# #                            max_depth: int) -> List[str]:
# #         """
# #         Находит путь в графе эмоций.
# #         """
# #
# #         def dfs(current: str, path: List[str], depth: int):
# #             if depth > max_depth:
# #                 return None
# #
# #             if current == target_emotion:
# #                 return path.copy()
# #
# #             for successor in self.graph.emotion_graph.successors(current):
# #                 result = dfs(successor, path + [successor], depth + 1)
# #                 if result:
# #                     return result
# #
# #             return None
# #
# #         return dfs(start_emotion, [start_emotion], 0)
# #
# #     def get_insight(self, event: EmotionalEvent) -> Optional[Dict]:
# #         """
# #         Генерирует "озарение" - неожиданное решение или понимание.
# #
# #         Args:
# #             event: Текущее событие
# #
# #         Returns:
# #             dict: Инсайт с explanation и confidence
# #         """
# #         # 1. Ищем неочевидные связи
# #         insights = []
# #
# #         # Проверяем все похожие события
# #         similar_events = self.graph.get_similar_events(event.embedding, top_k=10)
# #
# #         for event_id, similarity in similar_events:
# #             if similarity < 0.6:
# #                 continue
# #
# #             # Находим неочевидные связи через эмоции
# #             for emotion_type in self.graph.event_to_emotion.successors(event_id):
# #                 # Проверяем, есть ли у этой эмоции неочевидные следствия
# #                 for consequence in self.graph.emotion_to_event.successors(emotion_type):
# #                     if consequence != event_id:
# #                         insights.append({
# #                             'from_event': event_id,
# #                             'through_emotion': emotion_type,
# #                             'to_event': consequence,
# #                             'confidence': similarity * 0.7
# #                         })
# #
# #         # 2. Выбираем лучший инсайт
# #         if insights:
# #             best_insight = max(insights, key=lambda x: x['confidence'])
# #             return {
# #                 'insight': best_insight,
# #                 'explanation': f"Через эмоцию {best_insight['through_emotion']}",
# #                 'confidence': best_insight['confidence'],
# #                 'chain': self.find_path(event.id, best_insight['to_event'], 5)
# #             }
# #
# #         return None
# #
# #     def predict_intuitive(self, event: EmotionalEvent) -> Dict:
# #         """
# #         Интуитивное предсказание развития событий.
# #         """
# #         # 1. Находим возможные пути развития
# #         paths = self.find_path(event.id, '', 5)  # Пути без конкретной цели
# #
# #         # 2. Оцениваем вероятности каждого пути
# #         predictions = []
# #         for path in paths:
# #             confidence = self._evaluate_path_confidence(path)
# #             predictions.append({
# #                 'path': path,
# #                 'confidence': confidence,
# #                 'likely_emotion': self._predict_likely_emotion(path)
# #             })
# #
# #         # 3. Выбираем наиболее вероятные
# #         predictions.sort(key=lambda x: x['confidence'], reverse=True)
# #
# #         return {
# #             'predictions': predictions[:3],
# #             'intuition_score': predictions[0]['confidence'] if predictions else 0.0
# #         }
# #
# #     def _evaluate_path_confidence(self, path: List[str]) -> float:
# #         """Оценивает вероятность пути."""
# #         if not path:
# #             return 0.0
# #
# #         confidence = 1.0
# #         for i in range(len(path) - 1):
# #             # Проверяем, есть ли связь
# #             if path[i].startswith('event:'):
# #                 event_id = path[i].replace('event:', '')
# #                 if path[i + 1].startswith('emotion:'):
# #                     emotion_type = path[i + 1].replace('emotion:', '')
# #                     # Проверяем связь событие → эмоция
# #                     if self.graph.event_to_emotion.has_edge(event_id, emotion_type):
# #                         prob = self.graph.event_to_emotion[event_id][emotion_type].get('probability', 0.5)
# #                         confidence *= prob
# #                 else:
# #                     confidence *= 0.5
# #             elif path[i].startswith('emotion:'):
# #                 emotion_type = path[i].replace('emotion:', '')
# #                 if path[i + 1].startswith('emotion:'):
# #                     next_emotion = path[i + 1].replace('emotion:', '')
# #                     if self.graph.emotion_graph.has_edge(emotion_type, next_emotion):
# #                         weight = self.graph.emotion_graph[emotion_type][next_emotion].get('weight', 0.5)
# #                         confidence *= weight
# #                 else:
# #                     confidence *= 0.5
# #
# #         return float(np.clip(confidence, 0.0, 1.0))
# #
# #     def _predict_likely_emotion(self, path: List[str]) -> Optional[str]:
# #         """Предсказывает наиболее вероятную эмоцию в конце пути."""
# #         for item in reversed(path):
# #             if item.startswith('emotion:'):
# #                 return item.replace('emotion:', '')
# #         return None