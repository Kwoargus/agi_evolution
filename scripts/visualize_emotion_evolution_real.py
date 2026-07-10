# scripts/visualize_emotion_evolution_real.py (исправленный с отладкой)
"""
Визуализация эволюции эмоциональной подсистемы с реалистичными сценариями.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import warnings

warnings.filterwarnings('ignore')

from core.emotions.emotion_system import EmotionSystem
from core.emotions.emotion_base import EmotionalEvent, EmotionalResponse, EmotionType


class RealEmotionVisualizer:
    """
    Визуализатор эмоций с реалистичными сценариями.
    """

    def __init__(self):
        print("🔧 Инициализация визуализатора...")
        self.emotion_system = EmotionSystem()
        self.emotion_history = []
        self.emotion_counts = {}
        self.emotion_chains = []

        # Принудительно создаем события и связи
        self._create_events_and_links()
        print("✅ Визуализатор инициализирован")

    def _create_events_and_links(self):
        """
        Принудительно создает события и связи между ними и эмоциями.
        """
        print("🔗 Создание событий и связей с эмоциями...")

        # Создаем события с осмысленными эмбеддингами
        events_data = {
            'danger': {'desc': 'Опасность!', 'pattern': [0.9, 0.8, 0.1, 0.0, 0.0]},
            'success': {'desc': 'Успех!', 'pattern': [0.1, 0.0, 0.9, 0.8, 0.0]},
            'failure': {'desc': 'Неудача...', 'pattern': [0.0, 0.9, 0.8, 0.1, 0.0]},
            'injustice': {'desc': 'Несправедливость!', 'pattern': [0.8, 0.1, 0.0, 0.0, 0.9]},
            'surprise': {'desc': 'Сюрприз!', 'pattern': [0.1, 0.9, 0.1, 0.0, 0.0]},
            'love': {'desc': 'Любовь ❤️', 'pattern': [0.1, 0.0, 0.1, 0.9, 0.8]},
            'betrayal': {'desc': 'Предательство!', 'pattern': [0.9, 0.1, 0.0, 0.8, 0.1]},
            'achievement': {'desc': 'Достижение!', 'pattern': [0.1, 0.0, 0.8, 0.9, 0.1]},
            'loss': {'desc': 'Потеря...', 'pattern': [0.0, 0.9, 0.1, 0.0, 0.8]},
            'gift': {'desc': 'Подарок 🎁', 'pattern': [0.2, 0.0, 0.7, 0.8, 0.3]},
        }

        # Карта: событие → эмоция
        event_emotion_map = {
            'danger': EmotionType.FEAR,
            'success': EmotionType.JOY,
            'failure': EmotionType.SADNESS,
            'injustice': EmotionType.ANGER,
            'surprise': EmotionType.SURPRISE,
            'love': EmotionType.LOVE,
            'betrayal': EmotionType.RESENTMENT,
            'achievement': EmotionType.JOY,
            'loss': EmotionType.SADNESS,
            'gift': EmotionType.JOY,
        }

        # Дополнительные эмоции для сложных реакций
        additional_emotions = {
            'danger': [EmotionType.SURPRISE],
            'injustice': [EmotionType.RESENTMENT],
            'betrayal': [EmotionType.ANGER, EmotionType.SADNESS],
            'achievement': [EmotionType.ANTICIPATION],
            'loss': [EmotionType.DISGUST],
            'success': [EmotionType.ANTICIPATION],
            'love': [EmotionType.TRUST],
            'gift': [EmotionType.SURPRISE],
        }

        # Создаем события и связи
        for event_id, data in events_data.items():
            # Создаем эмбеддинг
            embedding = np.zeros(128)
            embedding[:len(data['pattern'])] = data['pattern']
            embedding += np.random.randn(128) * 0.05
            norm = np.linalg.norm(embedding) + 1e-8
            embedding = embedding / norm

            # Создаем событие
            event = EmotionalEvent(
                id=event_id,
                description=data['desc'],
                timestamp=0,
                context={'type': event_id},
                participants=['system'],
                embedding=embedding
            )

            # Добавляем событие в граф
            self.emotion_system.engine.graph.add_event(event)

            # Добавляем основную связь событие→эмоция
            main_emotion = event_emotion_map[event_id]
            self.emotion_system.engine.graph.add_event_emotion_link(
                event_id,
                main_emotion,
                probability=0.9,
                intensity_factor=1.2
            )

            # Добавляем дополнительные связи
            for extra_emotion in additional_emotions.get(event_id, []):
                self.emotion_system.engine.graph.add_event_emotion_link(
                    event_id,
                    extra_emotion,
                    probability=0.5,
                    intensity_factor=0.8
                )

        print(f"✅ Создано {len(events_data)} событий и связей с эмоциями")
        print(f"   Всего связей событие→эмоция: {len(self.emotion_system.engine.graph.event_emotion_links)}")

    def run_emotion_simulation(self, n_events: int = 80):
        """
        Запускает симуляцию эмоциональных реакций.
        """
        print("🧠 Запуск симуляции эмоций...")

        # Список доступных событий
        event_types = ['danger', 'success', 'failure', 'injustice',
                       'surprise', 'love', 'betrayal', 'achievement',
                       'loss', 'gift']

        # Веса для выбора событий
        weights = [0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05, 0.15, 0.05, 0.1]

        # Счетчики эмоций
        self.emotion_counts = {e.value: 0 for e in EmotionType}

        for i in range(n_events):
            # Выбираем событие
            event_type = np.random.choice(event_types, p=weights)

            # Получаем событие из графа
            event = self.emotion_system.engine.graph.events.get(event_type)
            if not event:
                continue

            # Обрабатываем событие через сенсорный ввод
            responses = self.emotion_system.process_sensory_input({
                'vision': event.embedding[:64],
                'sound': event.embedding[64:96],
                'smell': event.embedding[96:128],
                'context': {'type': event_type},
                'participants': ['bot']
            })

            # Сохраняем историю
            state = self.emotion_system.get_emotional_state()
            self.emotion_history.append({
                'step': i,
                'event_type': event_type,
                'description': event.description,
                'responses': responses,
                'dominant': state['dominant_emotion'],
                'intensity': state['intensity'],
                'valence': state['valence'],
                'arousal': state['arousal']
            })

            # Обновляем счетчики
            for response in responses:
                self.emotion_counts[response.emotion_type.value] += 1

            # Создаем цепочки эмоций
            if len(responses) > 1:
                for j in range(len(responses) - 1):
                    self.emotion_chains.append((
                        responses[j].emotion_type.value,
                        responses[j + 1].emotion_type.value
                    ))

            # Иногда добавляем новые связи (эволюция)
            if i > 0 and i % 20 == 0:
                # Добавляем случайную новую связь
                random_event = np.random.choice(event_types)
                random_emotion = np.random.choice(list(EmotionType))
                if random_event in self.emotion_system.engine.graph.events:
                    self.emotion_system.engine.graph.add_event_emotion_link(
                        random_event,
                        random_emotion,
                        probability=np.random.uniform(0.3, 0.7),
                        intensity_factor=np.random.uniform(0.5, 1.0)
                    )
                    print(f"  🔄 Добавлена новая связь: {random_event} → {random_emotion.value}")

        print(f"✅ Симуляция завершена. Обработано {n_events} событий.")
        self._print_statistics()

    def _print_statistics(self):
        """Выводит статистику эмоций."""
        print("\n📊 СТАТИСТИКА ЭМОЦИЙ:")
        print("-" * 50)
        sorted_emotions = sorted(
            [(k, v) for k, v in self.emotion_counts.items() if v > 0],
            key=lambda x: x[1],
            reverse=True
        )
        total = sum(self.emotion_counts.values())
        if total > 0:
            for emotion, count in sorted_emotions[:15]:
                print(f"  {emotion:15} : {count:4} раз ({count / total * 100:5.1f}%)")
        else:
            print("  ⚠️ Нет зарегистрированных эмоций!")
        print("-" * 50)

        # Цепочки эмоций
        if self.emotion_chains:
            print("\n🔗 ПОПУЛЯРНЫЕ ЦЕПОЧКИ ЭМОЦИЙ:")
            from collections import Counter
            chain_counts = Counter(self.emotion_chains)
            for (e1, e2), count in chain_counts.most_common(5):
                print(f"  {e1:12} → {e2:12} : {count:3} раз")

    def visualize_emotions(self):
        """
        Визуализирует эмоциональную эволюцию.
        """
        print("\n🎨 Создание визуализации...")

        if not self.emotion_history:
            print("⚠️ Нет данных. Сначала запустите run_emotion_simulation()")
            return

        print(f"📊 Данных для визуализации: {len(self.emotion_history)} записей")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Эмоциональная эволюция бота', fontsize=16)

        # 1. Временная линия эмоций
        ax1 = axes[0, 0]
        self._plot_timeline(ax1)

        # 2. Интенсивность и валентность
        ax2 = axes[0, 1]
        self._plot_intensity_valence(ax2)

        # 3. Распределение эмоций
        ax3 = axes[0, 2]
        self._plot_distribution(ax3)

        # 4. Матрица переходов
        ax4 = axes[1, 0]
        self._plot_transition_matrix(ax4)

        # 5. Эволюция сложности
        ax5 = axes[1, 1]
        self._plot_complexity_evolution(ax5)

        # 6. Статистика
        ax6 = axes[1, 2]
        self._plot_stats(ax6)

        plt.tight_layout()
        print("🖥️ Отображение диаграмм...")
        plt.show()
        print("✅ Визуализация завершена")

    def _plot_timeline(self, ax):
        """Рисует временную линию эмоций."""
        steps = [h['step'] for h in self.emotion_history]
        emotions = [h['dominant'] for h in self.emotion_history]
        intensities = [h['intensity'] for h in self.emotion_history]

        # Цветовая карта для эмоций
        emotion_colors = {
            'joy': '#FFD700',
            'sadness': '#4169E1',
            'anger': '#FF0000',
            'fear': '#8B008B',
            'surprise': '#FFA500',
            'disgust': '#228B22',
            'trust': '#00CED1',
            'anticipation': '#FF69B4',
            'love': '#FF1493',
            'resentment': '#8B0000',
            'hatred': '#000000',
        }

        # Рисуем точки
        for step, emotion, intensity in zip(steps, emotions, intensities):
            color = emotion_colors.get(emotion, '#808080')
            ax.scatter(step, 0, c=[color], s=intensity * 300 + 50, alpha=0.7)

        # Добавляем названия событий
        for i, h in enumerate(self.emotion_history[::10]):
            ax.text(h['step'], -0.5, h['event_type'][:8],
                    rotation=45, ha='right', fontsize=7)

        ax.set_title('Временная линия эмоций')
        ax.set_xlabel('Шаг')
        ax.set_yticks([])
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3)

        # Легенда
        patches = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color, markersize=8, label=emotion)
                   for emotion, color in list(emotion_colors.items())
                   if emotion in set(emotions)]
        if patches:
            ax.legend(handles=patches[:8], loc='upper right', fontsize=7, ncol=2)

    def _plot_intensity_valence(self, ax):
        """Рисует интенсивность и валентность."""
        steps = [h['step'] for h in self.emotion_history]
        intensities = [h['intensity'] for h in self.emotion_history]
        valences = [h['valence'] for h in self.emotion_history]

        ax.plot(steps, intensities, 'b-', label='Интенсивность', alpha=0.7)
        ax.plot(steps, valences, 'r-', label='Валентность', alpha=0.7)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.set_title('Интенсивность и валентность эмоций')
        ax.set_xlabel('Шаг')
        ax.set_ylabel('Значение')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)

    def _plot_distribution(self, ax):
        """Рисует распределение эмоций."""
        emotions = [h['dominant'] for h in self.emotion_history]

        unique, counts = np.unique(emotions, return_counts=True)

        if len(unique) == 0:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
            return

        colors = plt.cm.tab20(np.linspace(0, 1, len(unique)))

        bars = ax.bar(range(len(unique)), counts, color=colors, alpha=0.7)
        ax.set_xticks(range(len(unique)))
        ax.set_xticklabels(unique, rotation=45, ha='right')
        ax.set_title('Распределение доминирующих эмоций')
        ax.set_xlabel('Эмоция')
        ax.set_ylabel('Частота')

        # Добавляем значения
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=9)

    def _plot_transition_matrix(self, ax):
        """Рисует матрицу переходов между эмоциями."""
        transitions = {}
        for i in range(len(self.emotion_history) - 1):
            e1 = self.emotion_history[i]['dominant']
            e2 = self.emotion_history[i + 1]['dominant']
            if e1 != e2:
                key = (e1, e2)
                transitions[key] = transitions.get(key, 0) + 1

        if not transitions:
            ax.text(0.5, 0.5, 'Нет переходов', ha='center', va='center')
            return

        all_emotions = list(set([e for pair in transitions.keys() for e in pair]))
        n = len(all_emotions)
        matrix = np.zeros((n, n))
        emotion_to_idx = {e: i for i, e in enumerate(all_emotions)}

        for (e1, e2), count in transitions.items():
            if e1 in emotion_to_idx and e2 in emotion_to_idx:
                matrix[emotion_to_idx[e1], emotion_to_idx[e2]] = count

        # Нормализуем
        row_sums = matrix.sum(axis=1, keepdims=True)
        if row_sums.sum() > 0:
            matrix = np.divide(matrix, row_sums, where=row_sums != 0)

        im = ax.imshow(matrix, cmap='Blues', aspect='auto')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(all_emotions, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(all_emotions, fontsize=7)
        ax.set_title('Матрица переходов между эмоциями')
        plt.colorbar(im, ax=ax)

    def _plot_complexity_evolution(self, ax):
        """Рисует эволюцию сложности эмоциональных реакций."""
        steps = [h['step'] for h in self.emotion_history]
        complexity = [len(h['responses']) for h in self.emotion_history]

        if not complexity:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
            return

        window = min(10, len(complexity))
        if window > 1:
            moving_avg = np.convolve(complexity, np.ones(window) / window, mode='valid')
            ax.plot(steps[window - 1:], moving_avg, 'r-', linewidth=2, label='Среднее')

        ax.plot(steps, complexity, 'b.', alpha=0.3, label='Реакции')
        ax.set_title('Эволюция сложности эмоциональных реакций')
        ax.set_xlabel('Шаг')
        ax.set_ylabel('Количество эмоций в реакции')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_stats(self, ax):
        """Рисует статистику."""
        ax.axis('off')

        all_emotions = [h['dominant'] for h in self.emotion_history]
        unique_emotions = len(set(all_emotions))
        event_types = [h['event_type'] for h in self.emotion_history]
        unique_events = len(set(event_types))

        stats_text = f"""
        ╔══════════════════════════════════════════╗
        ║      СТАТИСТИКА ЭМОЦИОНАЛЬНОЙ СИСТЕМЫ    ║
        ╠══════════════════════════════════════════╣
        ║  Всего событий:        {len(self.emotion_history):<6}              ║
        ║  Уникальных эмоций:    {unique_emotions:<6}              ║
        ║  Уникальных событий:   {unique_events:<6}              ║
        ║                                        ║
        ║  Средняя интенсивность: {np.mean([h['intensity'] for h in self.emotion_history]):.3f}   ║
        ║  Средняя валентность:   {np.mean([h['valence'] for h in self.emotion_history]):.3f}   ║
        ║  Средняя сложность:     {np.mean([len(h['responses']) for h in self.emotion_history]):.3f}   ║
        ║                                        ║
        ║  Цепочек эмоций:       {len(self.emotion_chains):<6}              ║
        ║  Уникальных цепочек:   {len(set(self.emotion_chains)):<6}              ║
        ╚══════════════════════════════════════════╝
        """

        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')


def main():
    """
    Главная функция для визуализации реальной эмоциональной эволюции.
    """
    print("=" * 60)
    print("РЕАЛЬНАЯ ЭМОЦИОНАЛЬНАЯ ЭВОЛЮЦИЯ БОТА")
    print("=" * 60)

    # Создаем визуализатор
    visualizer = RealEmotionVisualizer()

    # Запускаем симуляцию
    visualizer.run_emotion_simulation(n_events=80)

    # Визуализируем результаты
    visualizer.visualize_emotions()

    print("\n✅ Визуализация завершена!")


if __name__ == "__main__":
    main()



# # core/emotions/intuition_engine.py (исправленный)
#
# import numpy as np
# from typing import List, Dict, Optional, Tuple
# from core.emotions.emotion_graph import EmotionGraph
# from core.emotions.emotion_engine import EmotionEngine
# from core.emotions.mental_model import MentalModelManager
# from core.emotions.intuition_engine import IntuitionEngine
# from core.emotions.emotion_base import EmotionalEvent, EmotionalResponse, EmotionType
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
#
# # # scripts/visualize_emotion_evolution_real.py (исправленный)
# # """
# # Визуализация эволюции эмоциональной подсистемы с реалистичными сценариями.
# # """
# #
# # import sys
# # import os
# #
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# #
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from typing import List, Dict, Any
# # import warnings
# #
# # warnings.filterwarnings('ignore')
# #
# # from core.emotions.emotion_system import EmotionSystem
# # from core.emotions.emotion_base import EmotionalEvent, EmotionalResponse, EmotionType
# #
# #
# # class RealEmotionVisualizer:
# #     """
# #     Визуализатор эмоций с реалистичными сценариями.
# #     """
# #
# #     def __init__(self):
# #         self.emotion_system = EmotionSystem()
# #
# #         # Обучаем систему связывать события с эмоциями
# #         self.emotion_system.engine.train_event_emotion_links()
# #
# #         self.emotion_history = []
# #         self.emotion_counts = {}
# #         self.emotion_chains = []
# #
# #     def generate_test_events(self, n_events: int = 50):
# #         """
# #         Генерирует тестовые события с разными типами.
# #         """
# #         # События с их эмбеддингами
# #         event_types = {
# #             'danger': {'embedding': self._create_embedding([0.9, 0.8, 0.1, 0.0, 0.0]), 'desc': 'Опасность!'},
# #             'success': {'embedding': self._create_embedding([0.1, 0.0, 0.9, 0.8, 0.0]), 'desc': 'Успех!'},
# #             'failure': {'embedding': self._create_embedding([0.0, 0.9, 0.8, 0.1, 0.0]), 'desc': 'Неудача...'},
# #             'injustice': {'embedding': self._create_embedding([0.8, 0.1, 0.0, 0.0, 0.9]), 'desc': 'Несправедливость!'},
# #             'surprise': {'embedding': self._create_embedding([0.1, 0.9, 0.1, 0.0, 0.0]), 'desc': 'Сюрприз!'},
# #             'love': {'embedding': self._create_embedding([0.1, 0.0, 0.1, 0.9, 0.8]), 'desc': 'Любовь ❤️'},
# #             'betrayal': {'embedding': self._create_embedding([0.9, 0.1, 0.0, 0.8, 0.1]), 'desc': 'Предательство!'},
# #             'achievement': {'embedding': self._create_embedding([0.1, 0.0, 0.8, 0.9, 0.1]), 'desc': 'Достижение!'},
# #             'loss': {'embedding': self._create_embedding([0.0, 0.9, 0.1, 0.0, 0.8]), 'desc': 'Потеря...'},
# #             'gift': {'embedding': self._create_embedding([0.2, 0.0, 0.7, 0.8, 0.3]), 'desc': 'Подарок 🎁'},
# #         }
# #
# #         # Создаем последовательность событий
# #         events = []
# #         event_list = list(event_types.keys())
# #
# #         for i in range(n_events):
# #             # Выбираем событие (с некоторыми весами)
# #             weights = {
# #                 'danger': 0.1,
# #                 'success': 0.15,
# #                 'failure': 0.1,
# #                 'injustice': 0.1,
# #                 'surprise': 0.1,
# #                 'love': 0.1,
# #                 'betrayal': 0.05,
# #                 'achievement': 0.15,
# #                 'loss': 0.05,
# #                 'gift': 0.1
# #             }
# #
# #             event_type = np.random.choice(list(weights.keys()), p=list(weights.values()))
# #             event_data = event_types[event_type]
# #
# #             event = {
# #                 'id': f"event_{i}",
# #                 'type': event_type,
# #                 'description': event_data['desc'],
# #                 'embedding': event_data['embedding'],
# #                 'context': {'type': event_type, 'step': i}
# #             }
# #             events.append(event)
# #
# #         return events
# #
# #     def _create_embedding(self, pattern: list) -> np.ndarray:
# #         """Создает эмбеддинг из паттерна."""
# #         embedding = np.zeros(128)
# #         embedding[:len(pattern)] = pattern
# #         # Добавляем небольшой шум для разнообразия
# #         embedding += np.random.randn(128) * 0.05
# #         # Нормализуем
# #         norm = np.linalg.norm(embedding) + 1e-8
# #         return embedding / norm
# #
# #     def run_emotion_simulation(self, n_events: int = 100):
# #         """
# #         Запускает симуляцию эмоциональных реакций.
# #         """
# #         print("🧠 Запуск симуляции эмоций...")
# #
# #         # Создаем события
# #         events = self.generate_test_events(n_events)
# #
# #         # Счетчики эмоций
# #         self.emotion_counts = {e.value: 0 for e in EmotionType}
# #
# #         for event in events:
# #             # Создаем объект события
# #             emotional_event = EmotionalEvent(
# #                 id=event['id'],
# #                 description=event['description'],
# #                 timestamp=0,
# #                 context=event['context'],
# #                 participants=['bot'],
# #                 embedding=event['embedding']
# #             )
# #
# #             # Обрабатываем событие
# #             responses = self.emotion_system.process_sensory_input({
# #                 'vision': event['embedding'][:64],
# #                 'sound': event['embedding'][64:96],
# #                 'smell': event['embedding'][96:128],
# #                 'context': event['context'],
# #                 'participants': ['bot']
# #             })
# #
# #             # Сохраняем историю
# #             state = self.emotion_system.get_emotional_state()
# #             self.emotion_history.append({
# #                 'step': len(self.emotion_history),
# #                 'event_type': event['type'],
# #                 'description': event['description'],
# #                 'responses': responses,
# #                 'dominant': state['dominant_emotion'],
# #                 'intensity': state['intensity'],
# #                 'valence': state['valence'],
# #                 'arousal': state['arousal']
# #             })
# #
# #             # Обновляем счетчики
# #             for response in responses:
# #                 self.emotion_counts[response.emotion_type.value] += 1
# #
# #             # Создаем цепочки эмоций
# #             if len(responses) > 1:
# #                 for j in range(len(responses) - 1):
# #                     self.emotion_chains.append((
# #                         responses[j].emotion_type.value,
# #                         responses[j + 1].emotion_type.value
# #                     ))
# #
# #         print(f"✅ Симуляция завершена. Обработано {n_events} событий.")
# #         self._print_statistics()
# #
# #     def _print_statistics(self):
# #         """Выводит статистику эмоций."""
# #         print("\n📊 СТАТИСТИКА ЭМОЦИЙ:")
# #         print("-" * 50)
# #         sorted_emotions = sorted(
# #             [(k, v) for k, v in self.emotion_counts.items() if v > 0],
# #             key=lambda x: x[1],
# #             reverse=True
# #         )
# #         total = sum(self.emotion_counts.values())
# #         for emotion, count in sorted_emotions[:15]:
# #             print(f"  {emotion:15} : {count:4} раз ({count / total * 100:5.1f}%)")
# #         print("-" * 50)
# #
# #         # Цепочки эмоций
# #         if self.emotion_chains:
# #             print("\n🔗 ПОПУЛЯРНЫЕ ЦЕПОЧКИ ЭМОЦИЙ:")
# #             from collections import Counter
# #             chain_counts = Counter(self.emotion_chains)
# #             for (e1, e2), count in chain_counts.most_common(5):
# #                 print(f"  {e1:12} → {e2:12} : {count:3} раз")
# #
# #     def visualize_emotions(self):
# #         """
# #         Визуализирует эмоциональную эволюцию.
# #         """
# #         if not self.emotion_history:
# #             print("⚠️ Нет данных. Сначала запустите run_emotion_simulation()")
# #             return
# #
# #         fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# #         fig.suptitle('Эмоциональная эволюция бота', fontsize=16)
# #
# #         # 1. Временная линия эмоций
# #         ax1 = axes[0, 0]
# #         self._plot_timeline(ax1)
# #
# #         # 2. Интенсивность и валентность
# #         ax2 = axes[0, 1]
# #         self._plot_intensity_valence(ax2)
# #
# #         # 3. Распределение эмоций
# #         ax3 = axes[0, 2]
# #         self._plot_distribution(ax3)
# #
# #         # 4. Матрица переходов
# #         ax4 = axes[1, 0]
# #         self._plot_transition_matrix(ax4)
# #
# #         # 5. Эволюция сложности
# #         ax5 = axes[1, 1]
# #         self._plot_complexity_evolution(ax5)
# #
# #         # 6. Статистика
# #         ax6 = axes[1, 2]
# #         self._plot_stats(ax6)
# #
# #         plt.tight_layout()
# #         plt.show()
# #
# #     def _plot_timeline(self, ax):
# #         """Рисует временную линию эмоций."""
# #         steps = [h['step'] for h in self.emotion_history]
# #         emotions = [h['dominant'] for h in self.emotion_history]
# #         intensities = [h['intensity'] for h in self.emotion_history]
# #
# #         # Цветовая карта для эмоций
# #         emotion_colors = {
# #             'joy': '#FFD700',
# #             'sadness': '#4169E1',
# #             'anger': '#FF0000',
# #             'fear': '#8B008B',
# #             'surprise': '#FFA500',
# #             'disgust': '#228B22',
# #             'trust': '#00CED1',
# #             'anticipation': '#FF69B4',
# #             'love': '#FF1493',
# #             'resentment': '#8B0000',
# #             'hatred': '#000000',
# #         }
# #
# #         # Рисуем точки
# #         for i, (step, emotion, intensity) in enumerate(zip(steps, emotions, intensities)):
# #             color = emotion_colors.get(emotion, '#808080')
# #             ax.scatter(step, 0, c=[color], s=intensity * 300 + 50, alpha=0.7)
# #
# #         # Добавляем названия событий
# #         for i, h in enumerate(self.emotion_history[::10]):
# #             ax.text(h['step'], -0.5, h['event_type'][:8],
# #                     rotation=45, ha='right', fontsize=7)
# #
# #         ax.set_title('Временная линия эмоций')
# #         ax.set_xlabel('Шаг')
# #         ax.set_yticks([])
# #         ax.set_ylim(-1, 1)
# #         ax.grid(True, alpha=0.3)
# #
# #         # Легенда
# #         patches = [plt.Line2D([0], [0], marker='o', color='w',
# #                               markerfacecolor=color, markersize=8, label=emotion)
# #                    for emotion, color in list(emotion_colors.items())[:8]]
# #         if patches:
# #             ax.legend(handles=patches, loc='upper right', fontsize=7, ncol=2)
# #
# #     def _plot_intensity_valence(self, ax):
# #         """Рисует интенсивность и валентность."""
# #         steps = [h['step'] for h in self.emotion_history]
# #         intensities = [h['intensity'] for h in self.emotion_history]
# #         valences = [h['valence'] for h in self.emotion_history]
# #
# #         ax.plot(steps, intensities, 'b-', label='Интенсивность', alpha=0.7)
# #         ax.plot(steps, valences, 'r-', label='Валентность', alpha=0.7)
# #         ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
# #         ax.set_title('Интенсивность и валентность эмоций')
# #         ax.set_xlabel('Шаг')
# #         ax.set_ylabel('Значение')
# #         ax.legend()
# #         ax.grid(True, alpha=0.3)
# #         ax.set_ylim(-1.1, 1.1)
# #
# #     def _plot_distribution(self, ax):
# #         """Рисует распределение эмоций."""
# #         emotions = [h['dominant'] for h in self.emotion_history]
# #
# #         unique, counts = np.unique(emotions, return_counts=True)
# #         colors = plt.cm.tab20(np.linspace(0, 1, len(unique)))
# #
# #         bars = ax.bar(range(len(unique)), counts, color=colors, alpha=0.7)
# #         ax.set_xticks(range(len(unique)))
# #         ax.set_xticklabels(unique, rotation=45, ha='right')
# #         ax.set_title('Распределение доминирующих эмоций')
# #         ax.set_xlabel('Эмоция')
# #         ax.set_ylabel('Частота')
# #
# #         # Добавляем значения
# #         for bar, count in zip(bars, counts):
# #             ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
# #                     str(count), ha='center', va='bottom', fontsize=9)
# #
# #     def _plot_transition_matrix(self, ax):
# #         """Рисует матрицу переходов между эмоциями."""
# #         transitions = {}
# #         for i in range(len(self.emotion_history) - 1):
# #             e1 = self.emotion_history[i]['dominant']
# #             e2 = self.emotion_history[i + 1]['dominant']
# #             if e1 != e2:
# #                 key = (e1, e2)
# #                 transitions[key] = transitions.get(key, 0) + 1
# #
# #         if not transitions:
# #             ax.text(0.5, 0.5, 'Нет переходов', ha='center', va='center')
# #             return
# #
# #         all_emotions = list(set([e for pair in transitions.keys() for e in pair]))
# #         n = len(all_emotions)
# #         matrix = np.zeros((n, n))
# #         emotion_to_idx = {e: i for i, e in enumerate(all_emotions)}
# #
# #         for (e1, e2), count in transitions.items():
# #             if e1 in emotion_to_idx and e2 in emotion_to_idx:
# #                 matrix[emotion_to_idx[e1], emotion_to_idx[e2]] = count
# #
# #         # Нормализуем
# #         row_sums = matrix.sum(axis=1, keepdims=True)
# #         matrix = np.divide(matrix, row_sums, where=row_sums != 0)
# #
# #         im = ax.imshow(matrix, cmap='Blues', aspect='auto')
# #         ax.set_xticks(range(n))
# #         ax.set_yticks(range(n))
# #         ax.set_xticklabels(all_emotions, rotation=45, ha='right', fontsize=7)
# #         ax.set_yticklabels(all_emotions, fontsize=7)
# #         ax.set_title('Матрица переходов между эмоциями')
# #         plt.colorbar(im, ax=ax)
# #
# #     def _plot_complexity_evolution(self, ax):
# #         """Рисует эволюцию сложности эмоциональных реакций."""
# #         steps = [h['step'] for h in self.emotion_history]
# #         complexity = [len(h['responses']) for h in self.emotion_history]
# #
# #         window = 10
# #         moving_avg = np.convolve(complexity, np.ones(window) / window, mode='valid')
# #
# #         ax.plot(steps, complexity, 'b.', alpha=0.3, label='Реакции')
# #         ax.plot(steps[window - 1:], moving_avg, 'r-', linewidth=2, label='Среднее')
# #         ax.set_title('Эволюция сложности эмоциональных реакций')
# #         ax.set_xlabel('Шаг')
# #         ax.set_ylabel('Количество эмоций в реакции')
# #         ax.legend()
# #         ax.grid(True, alpha=0.3)
# #
# #     def _plot_stats(self, ax):
# #         """Рисует статистику."""
# #         ax.axis('off')
# #
# #         all_emotions = [h['dominant'] for h in self.emotion_history]
# #         unique_emotions = len(set(all_emotions))
# #         event_types = [h['event_type'] for h in self.emotion_history]
# #         unique_events = len(set(event_types))
# #
# #         stats_text = f"""
# #         ╔══════════════════════════════════════════╗
# #         ║      СТАТИСТИКА ЭМОЦИОНАЛЬНОЙ СИСТЕМЫ    ║
# #         ╠══════════════════════════════════════════╣
# #         ║  Всего событий:        {len(self.emotion_history):<6}              ║
# #         ║  Уникальных эмоций:    {unique_emotions:<6}              ║
# #         ║  Уникальных событий:   {unique_events:<6}              ║
# #         ║                                        ║
# #         ║  Средняя интенсивность: {np.mean([h['intensity'] for h in self.emotion_history]):.3f}   ║
# #         ║  Средняя валентность:   {np.mean([h['valence'] for h in self.emotion_history]):.3f}   ║
# #         ║  Средняя сложность:     {np.mean([len(h['responses']) for h in self.emotion_history]):.3f}   ║
# #         ║                                        ║
# #         ║  Цепочек эмоций:       {len(self.emotion_chains):<6}              ║
# #         ║  Уникальных цепочек:   {len(set(self.emotion_chains)):<6}              ║
# #         ╚══════════════════════════════════════════╝
# #         """
# #
# #         ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
# #                 fontsize=9, verticalalignment='top', fontfamily='monospace')
# #
# #
# # def main():
# #     """
# #     Главная функция для визуализации реальной эмоциональной эволюции.
# #     """
# #     print("=" * 60)
# #     print("РЕАЛЬНАЯ ЭМОЦИОНАЛЬНАЯ ЭВОЛЮЦИЯ БОТА")
# #     print("=" * 60)
# #
# #     # Создаем визуализатор
# #     visualizer = RealEmotionVisualizer()
# #
# #     # Запускаем симуляцию
# #     visualizer.run_emotion_simulation(n_events=80)
# #
# #     # Визуализируем результаты
# #     visualizer.visualize_emotions()
# #
# #     print("\n✅ Визуализация завершена!")
# #
# #
# # if __name__ == "__main__":
# #     main()
# #
# #
# #
# # # # scripts/visualize_emotion_evolution_real.py
# # # """
# # # Визуализация эволюции эмоциональной подсистемы с реалистичными сценариями.
# # # """
# # #
# # # import sys
# # # import os
# # #
# # # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# # #
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # from typing import List, Dict, Any
# # # import warnings
# # #
# # # warnings.filterwarnings('ignore')
# # #
# # # from core.emotions.emotion_system import EmotionSystem
# # # from core.emotions.emotion_base import EmotionalEvent, EmotionalResponse, EmotionType
# # # from scripts.emotion_scenarios import EmotionScenarioGenerator
# # #
# # #
# # # class RealEmotionVisualizer:
# # #     """
# # #     Визуализатор эмоций с реалистичными сценариями.
# # #     """
# # #
# # #     def __init__(self):
# # #         self.emotion_system = EmotionSystem()
# # #         self.scenario_generator = EmotionScenarioGenerator()
# # #         self.emotion_history = []
# # #         self.emotion_counts = {}
# # #         self.emotion_chains = []
# # #
# # #     def train_emotion_system(self, n_events: int = 100):
# # #         """
# # #         Обучает эмоциональную систему на последовательности событий.
# # #         """
# # #         print("🧠 Обучение эмоциональной системы...")
# # #
# # #         # Генерируем последовательность событий
# # #         sequence = self.scenario_generator.generate_sequence(n_events)
# # #
# # #         # Счетчики эмоций
# # #         emotion_counts = {e.value: 0 for e in EmotionType}
# # #
# # #         for i, scenario in enumerate(sequence):
# # #             # Создаем событие
# # #             event = EmotionalEvent(
# # #                 id=f"train_event_{i}",
# # #                 description=scenario['description'],
# # #                 timestamp=i,
# # #                 context=scenario['context'],
# # #                 participants=scenario['participants'],
# # #                 embedding=scenario['embedding']
# # #             )
# # #
# # #             # Обрабатываем событие
# # #             responses = self.emotion_system.process_sensory_input({
# # #                 'vision': scenario['embedding'][:64],
# # #                 'sound': scenario['embedding'][64:96],
# # #                 'smell': scenario['embedding'][96:128],
# # #                 'context': scenario['context'],
# # #                 'participants': scenario['participants']
# # #             })
# # #
# # #             # Сохраняем историю
# # #             state = self.emotion_system.get_emotional_state()
# # #             self.emotion_history.append({
# # #                 'step': i,
# # #                 'event_type': scenario['type'],
# # #                 'description': scenario['description'],
# # #                 'expected_emotion': scenario['expected_emotion'].value,
# # #                 'responses': responses,
# # #                 'dominant': state['dominant_emotion'],
# # #                 'intensity': state['intensity'],
# # #                 'valence': state['valence'],
# # #                 'arousal': state['arousal']
# # #             })
# # #
# # #             # Обновляем счетчики
# # #             for response in responses:
# # #                 emotion_counts[response.emotion_type.value] += 1
# # #
# # #             # Создаем цепочки эмоций
# # #             if len(responses) > 1:
# # #                 for j in range(len(responses) - 1):
# # #                     self.emotion_chains.append((
# # #                         responses[j].emotion_type.value,
# # #                         responses[j + 1].emotion_type.value
# # #                     ))
# # #
# # #         self.emotion_counts = emotion_counts
# # #
# # #         print(f"✅ Обучение завершено. Обработано {n_events} событий.")
# # #         self._print_statistics()
# # #
# # #     def _print_statistics(self):
# # #         """Выводит статистику эмоций."""
# # #         print("\n📊 СТАТИСТИКА ЭМОЦИЙ:")
# # #         print("-" * 40)
# # #         sorted_emotions = sorted(
# # #             [(k, v) for k, v in self.emotion_counts.items() if v > 0],
# # #             key=lambda x: x[1],
# # #             reverse=True
# # #         )
# # #         for emotion, count in sorted_emotions[:10]:
# # #             print(f"  {emotion:15} : {count:4} раз ({count / len(self.emotion_history) * 100:.1f}%)")
# # #         print("-" * 40)
# # #
# # #         # Цепочки эмоций
# # #         if self.emotion_chains:
# # #             print("\n🔗 ПОПУЛЯРНЫЕ ЦЕПОЧКИ ЭМОЦИЙ:")
# # #             from collections import Counter
# # #             chain_counts = Counter(self.emotion_chains)
# # #             for (e1, e2), count in chain_counts.most_common(5):
# # #                 print(f"  {e1} → {e2} : {count} раз")
# # #
# # #     def visualize_emotions(self):
# # #         """
# # #         Визуализирует эмоциональную эволюцию.
# # #         """
# # #         if not self.emotion_history:
# # #             print("⚠️ Нет данных. Сначала запустите train_emotion_system()")
# # #             return
# # #
# # #         fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# # #         fig.suptitle('Реальная эмоциональная эволюция бота', fontsize=16)
# # #
# # #         # 1. Временная линия эмоций
# # #         ax1 = axes[0, 0]
# # #         self._plot_timeline(ax1)
# # #
# # #         # 2. Интенсивность и валентность
# # #         ax2 = axes[0, 1]
# # #         self._plot_intensity_valence(ax2)
# # #
# # #         # 3. Распределение эмоций
# # #         ax3 = axes[0, 2]
# # #         self._plot_distribution(ax3)
# # #
# # #         # 4. Матрица переходов
# # #         ax4 = axes[1, 0]
# # #         self._plot_transition_matrix(ax4)
# # #
# # #         # 5. Эволюция сложности
# # #         ax5 = axes[1, 1]
# # #         self._plot_complexity_evolution(ax5)
# # #
# # #         # 6. Статистика
# # #         ax6 = axes[1, 2]
# # #         self._plot_stats(ax6)
# # #
# # #         plt.tight_layout()
# # #         plt.show()
# # #
# # #     def _plot_timeline(self, ax):
# # #         """Рисует временную линию эмоций."""
# # #         steps = [h['step'] for h in self.emotion_history]
# # #         emotions = [h['dominant'] for h in self.emotion_history]
# # #         intensities = [h['intensity'] for h in self.emotion_history]
# # #
# # #         # Цветовая карта
# # #         unique_emotions = list(set(emotions))
# # #         colors = plt.cm.tab20(np.linspace(0, 1, len(unique_emotions)))
# # #         emotion_to_color = {e: colors[i] for i, e in enumerate(unique_emotions)}
# # #
# # #         # Рисуем точки
# # #         for i, (step, emotion, intensity) in enumerate(zip(steps, emotions, intensities)):
# # #             ax.scatter(step, 0, c=[emotion_to_color[emotion]],
# # #                        s=intensity * 200 + 50, alpha=0.7)
# # #
# # #         # Добавляем названия событий
# # #         for i, h in enumerate(self.emotion_history[::10]):
# # #             ax.text(h['step'], -0.5, h['event_type'][:8],
# # #                     rotation=45, ha='right', fontsize=7)
# # #
# # #         ax.set_title('Временная линия эмоций')
# # #         ax.set_xlabel('Шаг')
# # #         ax.set_yticks([])
# # #         ax.set_ylim(-1, 1)
# # #         ax.grid(True, alpha=0.3)
# # #
# # #         # Легенда
# # #         patches = [plt.Line2D([0], [0], marker='o', color='w',
# # #                               markerfacecolor=color, markersize=8, label=emotion)
# # #                    for emotion, color in list(emotion_to_color.items())[:8]]
# # #         if patches:
# # #             ax.legend(handles=patches, loc='upper right', fontsize=7, ncol=2)
# # #
# # #     def _plot_intensity_valence(self, ax):
# # #         """Рисует интенсивность и валентность."""
# # #         steps = [h['step'] for h in self.emotion_history]
# # #         intensities = [h['intensity'] for h in self.emotion_history]
# # #         valences = [h['valence'] for h in self.emotion_history]
# # #
# # #         ax.plot(steps, intensities, 'b-', label='Интенсивность', alpha=0.7)
# # #         ax.plot(steps, valences, 'r-', label='Валентность', alpha=0.7)
# # #         ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
# # #         ax.set_title('Интенсивность и валентность эмоций')
# # #         ax.set_xlabel('Шаг')
# # #         ax.set_ylabel('Значение')
# # #         ax.legend()
# # #         ax.grid(True, alpha=0.3)
# # #         ax.set_ylim(-1.1, 1.1)
# # #
# # #     def _plot_distribution(self, ax):
# # #         """Рисует распределение эмоций."""
# # #         emotions = [h['dominant'] for h in self.emotion_history]
# # #
# # #         unique, counts = np.unique(emotions, return_counts=True)
# # #         colors = plt.cm.tab20(np.linspace(0, 1, len(unique)))
# # #
# # #         bars = ax.bar(range(len(unique)), counts, color=colors, alpha=0.7)
# # #         ax.set_xticks(range(len(unique)))
# # #         ax.set_xticklabels(unique, rotation=45, ha='right')
# # #         ax.set_title('Распределение доминирующих эмоций')
# # #         ax.set_xlabel('Эмоция')
# # #         ax.set_ylabel('Частота')
# # #
# # #         # Добавляем значения
# # #         for bar, count in zip(bars, counts):
# # #             ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
# # #                     str(count), ha='center', va='bottom', fontsize=9)
# # #
# # #     def _plot_transition_matrix(self, ax):
# # #         """Рисует матрицу переходов между эмоциями."""
# # #         # Собираем переходы
# # #         transitions = {}
# # #         for i in range(len(self.emotion_history) - 1):
# # #             e1 = self.emotion_history[i]['dominant']
# # #             e2 = self.emotion_history[i + 1]['dominant']
# # #             if e1 != e2:
# # #                 key = (e1, e2)
# # #                 transitions[key] = transitions.get(key, 0) + 1
# # #
# # #         if not transitions:
# # #             ax.text(0.5, 0.5, 'Нет переходов', ha='center', va='center')
# # #             return
# # #
# # #         # Создаем матрицу
# # #         all_emotions = list(set([e for pair in transitions.keys() for e in pair]))
# # #         n = len(all_emotions)
# # #         matrix = np.zeros((n, n))
# # #         emotion_to_idx = {e: i for i, e in enumerate(all_emotions)}
# # #
# # #         for (e1, e2), count in transitions.items():
# # #             if e1 in emotion_to_idx and e2 in emotion_to_idx:
# # #                 matrix[emotion_to_idx[e1], emotion_to_idx[e2]] = count
# # #
# # #         # Нормализуем
# # #         row_sums = matrix.sum(axis=1, keepdims=True)
# # #         matrix = np.divide(matrix, row_sums, where=row_sums != 0)
# # #
# # #         # Рисуем
# # #         im = ax.imshow(matrix, cmap='Blues', aspect='auto')
# # #         ax.set_xticks(range(n))
# # #         ax.set_yticks(range(n))
# # #         ax.set_xticklabels(all_emotions, rotation=45, ha='right', fontsize=7)
# # #         ax.set_yticklabels(all_emotions, fontsize=7)
# # #         ax.set_title('Матрица переходов между эмоциями')
# # #         plt.colorbar(im, ax=ax)
# # #
# # #     def _plot_complexity_evolution(self, ax):
# # #         """Рисует эволюцию сложности эмоциональных реакций."""
# # #         steps = [h['step'] for h in self.emotion_history]
# # #         complexity = [len(h['responses']) for h in self.emotion_history]
# # #
# # #         # Скользящее среднее
# # #         window = 10
# # #         moving_avg = np.convolve(complexity, np.ones(window) / window, mode='valid')
# # #
# # #         ax.plot(steps, complexity, 'b.', alpha=0.3, label='Реакции')
# # #         ax.plot(steps[window - 1:], moving_avg, 'r-', linewidth=2, label='Среднее')
# # #         ax.set_title('Эволюция сложности эмоциональных реакций')
# # #         ax.set_xlabel('Шаг')
# # #         ax.set_ylabel('Количество эмоций в реакции')
# # #         ax.legend()
# # #         ax.grid(True, alpha=0.3)
# # #
# # #     def _plot_stats(self, ax):
# # #         """Рисует статистику."""
# # #         ax.axis('off')
# # #
# # #         # Подсчет уникальных эмоций
# # #         all_emotions = [h['dominant'] for h in self.emotion_history]
# # #         unique_emotions = len(set(all_emotions))
# # #
# # #         # Статистика по событиям
# # #         event_types = [h['event_type'] for h in self.emotion_history]
# # #         unique_events = len(set(event_types))
# # #
# # #         stats_text = f"""
# # #         ╔══════════════════════════════════════════╗
# # #         ║      СТАТИСТИКА ЭМОЦИОНАЛЬНОЙ СИСТЕМЫ    ║
# # #         ╠══════════════════════════════════════════╣
# # #         ║  Всего событий:        {len(self.emotion_history):<6}              ║
# # #         ║  Уникальных эмоций:    {unique_emotions:<6}              ║
# # #         ║  Уникальных событий:   {unique_events:<6}              ║
# # #         ║                                        ║
# # #         ║  Средняя интенсивность: {np.mean([h['intensity'] for h in self.emotion_history]):.3f}   ║
# # #         ║  Средняя валентность:   {np.mean([h['valence'] for h in self.emotion_history]):.3f}   ║
# # #         ║  Средняя сложность:     {np.mean([len(h['responses']) for h in self.emotion_history]):.3f}   ║
# # #         ║                                        ║
# # #         ║  Цепочек эмоций:       {len(self.emotion_chains):<6}              ║
# # #         ║  Уникальных цепочек:   {len(set(self.emotion_chains)):<6}              ║
# # #         ╚══════════════════════════════════════════╝
# # #         """
# # #
# # #         ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
# # #                 fontsize=9, verticalalignment='top', fontfamily='monospace')
# # #
# # #
# # # def main():
# # #     """
# # #     Главная функция для визуализации реальной эмоциональной эволюции.
# # #     """
# # #     print("=" * 60)
# # #     print("РЕАЛЬНАЯ ЭМОЦИОНАЛЬНАЯ ЭВОЛЮЦИЯ БОТА")
# # #     print("=" * 60)
# # #
# # #     # Создаем визуализатор
# # #     visualizer = RealEmotionVisualizer()
# # #
# # #     # Обучаем систему на реалистичных сценариях
# # #     visualizer.train_emotion_system(n_events=100)
# # #
# # #     # Визуализируем результаты
# # #     visualizer.visualize_emotions()
# # #
# # #     print("\n✅ Визуализация завершена!")
# # #
# # #
# # # if __name__ == "__main__":
# # #     main()