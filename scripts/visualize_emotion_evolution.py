# scripts/visualize_emotion_evolution.py (исправленный)
"""
Визуализация эволюции эмоциональной подсистемы бота.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
from typing import List, Dict, Any, Tuple
import networkx as nx
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')

from core.emotions.emotion_system import EmotionSystem
from core.emotions.emotion_graph import EmotionGraph
from core.emotions.emotion_base import EmotionalEvent, EmotionalResponse, EmotionType
from db.emotion_db import EmotionDB


class EmotionEvolutionVisualizer:
    """
    Визуализатор эволюции эмоциональной подсистемы.
    """

    def __init__(self):
        self.emotion_system = EmotionSystem()
        self.db = EmotionDB()
        self.emotion_history = []
        self.graph_snapshots = []

        # Цвета для разных эмоций
        self.emotion_colors = {
            'joy': '#FFD700',
            'sadness': '#4169E1',
            'anger': '#FF0000',
            'fear': '#8B008B',
            'surprise': '#FFA500',
            'disgust': '#228B22',
            'trust': '#00CED1',
            'anticipation': '#FF69B4',
            'love': '#FF1493',
            'optimism': '#FFD700',
            'awe': '#9370DB',
            'contempt': '#808080',
            'resentment': '#8B0000',
            'hatred': '#000000',
            'guilt': '#696969',
            'shame': '#A52A2A',
            'empathy': '#98FB98'
        }

    def simulate_emotional_evolution(self, n_events: int = 50):
        """
        Симулирует эволюцию эмоциональной системы.
        """
        print("🧠 Симуляция эволюции эмоциональной системы...")

        # Базовые сценарии
        scenarios = [
            {'description': 'Угроза', 'emotion': EmotionType.FEAR, 'intensity': 0.8},
            {'description': 'Успех', 'emotion': EmotionType.JOY, 'intensity': 0.9},
            {'description': 'Неудача', 'emotion': EmotionType.SADNESS, 'intensity': 0.7},
            {'description': 'Несправедливость', 'emotion': EmotionType.ANGER, 'intensity': 0.8},
            {'description': 'Неожиданность', 'emotion': EmotionType.SURPRISE, 'intensity': 0.6},
            {'description': 'Доверие', 'emotion': EmotionType.TRUST, 'intensity': 0.5},
            {'description': 'Ожидание', 'emotion': EmotionType.ANTICIPATION, 'intensity': 0.4},
            {'description': 'Любовь', 'emotion': EmotionType.LOVE, 'intensity': 0.7},
            {'description': 'Обида', 'emotion': EmotionType.RESENTMENT, 'intensity': 0.6},
            {'description': 'Ненависть', 'emotion': EmotionType.HATRED, 'intensity': 0.9},
        ]

        for i in range(n_events):
            # Выбираем сценарий
            scenario = scenarios[i % len(scenarios)]

            # Создаем сенсорные данные
            sensory_data = {
                'vision': np.random.randn(64),
                'sound': np.random.randn(32),
                'smell': np.random.randn(32),
                'context': {'event': scenario['description'], 'step': i},
                'participants': ['sim_bot']
            }

            # Обрабатываем событие
            responses = self.emotion_system.process_sensory_input(sensory_data)

            # Сохраняем историю
            self.emotion_history.append({
                'step': i,
                'event_description': scenario['description'],
                'responses': responses,
                'state': self.emotion_system.get_emotional_state()
            })

            # Сохраняем снимок графа каждые 10 шагов
            if i % 10 == 0:
                # Получаем статистику связей
                stats = self.emotion_system.engine.graph.get_links_statistics()
                self.graph_snapshots.append({
                    'step': i,
                    'stats': stats,
                    'emotions_count': len(self.emotion_system.engine.graph.emotions)
                })

            # Добавляем случайные связи между эмоциями (эволюция)
            if i > 0 and np.random.random() < 0.2:
                emotions = list(EmotionType)
                e1 = np.random.choice(emotions)
                e2 = np.random.choice(emotions)
                if e1 != e2:
                    try:
                        self.emotion_system.engine.graph.add_emotion_chain(
                            e1, e2,
                            weight=np.random.uniform(0.3, 0.9)
                        )
                    except Exception:
                        pass

        print(f"✅ Симуляция завершена. Обработано {n_events} событий.")
        return self.emotion_history

    def visualize_emotion_timeline(self):
        """
        Визуализирует временную линию эмоциональных состояний.
        """
        if not self.emotion_history:
            print("⚠️ Нет данных. Сначала запустите simulate_emotional_evolution()")
            return

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Эволюция эмоциональной подсистемы', fontsize=16)

        # 1. Временная линия эмоций
        ax1 = axes[0, 0]
        self._plot_emotion_timeline(ax1)

        # 2. Интенсивность эмоций
        ax2 = axes[0, 1]
        self._plot_emotion_intensity(ax2)

        # 3. Граф эмоций (текущее состояние)
        ax3 = axes[1, 0]
        self._plot_emotion_graph(ax3)

        # 4. Эволюция графа
        ax4 = axes[1, 1]
        self._plot_graph_evolution(ax4)

        # 5. Распределение эмоций
        ax5 = axes[2, 0]
        self._plot_emotion_distribution(ax5)

        # 6. Статистика
        ax6 = axes[2, 1]
        self._plot_emotion_stats(ax6)

        plt.tight_layout()
        plt.show()

    def _plot_emotion_timeline(self, ax):
        """Рисует временную линию эмоций."""
        steps = [h['step'] for h in self.emotion_history]
        emotions = []
        intensities = []

        for h in self.emotion_history:
            state = h['state']
            emotions.append(state['dominant_emotion'])
            intensities.append(state['intensity'])

        # Создаем цветовую карту
        unique_emotions = list(set(emotions))
        emotion_to_color = {e: self.emotion_colors.get(e, '#808080') for e in unique_emotions}
        colors = [emotion_to_color.get(e, '#808080') for e in emotions]

        # Рисуем точки
        ax.scatter(steps, [0] * len(steps), c=colors, s=np.array(intensities) * 200 + 50, alpha=0.7)
        ax.set_title('Временная линия эмоций')
        ax.set_xlabel('Шаг')
        ax.set_yticks([])
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3)

        # Добавляем легенду
        patches = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color, markersize=10, label=emotion)
                   for emotion, color in list(emotion_to_color.items())[:10]]
        if patches:
            ax.legend(handles=patches, loc='upper right', fontsize=8, ncol=2)

    def _plot_emotion_intensity(self, ax):
        """Рисует график интенсивности эмоций."""
        steps = [h['step'] for h in self.emotion_history]
        intensities = [h['state']['intensity'] for h in self.emotion_history]

        ax.plot(steps, intensities, 'b-', linewidth=2, alpha=0.7)
        ax.fill_between(steps, intensities, 0, alpha=0.3, color='blue')
        ax.set_title('Интенсивность доминирующей эмоции')
        ax.set_xlabel('Шаг')
        ax.set_ylabel('Интенсивность')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)

        # Добавляем скользящее среднее
        if len(intensities) > 10:
            window = 10
            moving_avg = np.convolve(intensities, np.ones(window) / window, mode='valid')
            ax.plot(steps[window - 1:], moving_avg, 'r-', linewidth=2, label='Скользящее среднее')
            ax.legend()

    def _plot_emotion_graph(self, ax):
        """Рисует текущий граф эмоций."""
        graph = self.emotion_system.engine.graph

        # Строим граф
        G = nx.DiGraph()

        # Добавляем узлы
        for emotion_type in graph.emotions.keys():
            G.add_node(emotion_type,
                       color=self.emotion_colors.get(emotion_type, '#808080'))

        # Добавляем ребра
        for link in graph.emotion_chain_links.values():
            G.add_edge(link.source_id, link.target_id, weight=link.weight)

        # Если узлов мало, добавляем фиктивные для наглядности
        if len(G.nodes()) < 3:
            # Добавляем базовые эмоции для демонстрации
            for emotion in ['joy', 'sadness', 'anger', 'fear']:
                if emotion not in G.nodes():
                    G.add_node(emotion, color=self.emotion_colors.get(emotion, '#808080'))

        # Рисуем
        if len(G.nodes()) > 0:
            pos = nx.spring_layout(G, k=1, iterations=50)
            colors = [G.nodes[node].get('color', '#808080') for node in G.nodes()]

            nx.draw(G, pos, ax=ax, node_color=colors, with_labels=True,
                    node_size=500, font_size=8, arrows=True,
                    arrowstyle='->', arrowsize=20)
        else:
            ax.text(0.5, 0.5, 'Нет связей', ha='center', va='center')

        ax.set_title('Граф эмоций (текущее состояние)')

    def _plot_graph_evolution(self, ax):
        """Рисует эволюцию графа."""
        if len(self.graph_snapshots) < 2:
            ax.text(0.5, 0.5, 'Недостаточно данных', ha='center', va='center')
            return

        steps = [s['step'] for s in self.graph_snapshots]
        emotion_counts = [s['emotions_count'] for s in self.graph_snapshots]

        ax.plot(steps, emotion_counts, 'b-o', label='Эмоций', linewidth=2)
        ax.set_title('Эволюция графа эмоций')
        ax.set_xlabel('Шаг')
        ax.set_ylabel('Количество эмоций')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_emotion_distribution(self, ax):
        """Рисует распределение эмоций."""
        emotions = [h['state']['dominant_emotion'] for h in self.emotion_history]

        unique, counts = np.unique(emotions, return_counts=True)
        colors = [self.emotion_colors.get(e, '#808080') for e in unique]

        if len(unique) > 0:
            bars = ax.bar(range(len(unique)), counts, color=colors, alpha=0.7)
            ax.set_xticks(range(len(unique)))
            ax.set_xticklabels(unique, rotation=45, ha='right')
            ax.set_title('Распределение эмоций')
            ax.set_xlabel('Эмоция')
            ax.set_ylabel('Частота')

            # Добавляем значения
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center')

    def _plot_emotion_stats(self, ax):
        """Рисует статистику эмоций."""
        ax.axis('off')

        try:
            stats = self.emotion_system.engine.graph.get_links_statistics()
        except Exception:
            stats = {
                'causal_links': 0,
                'emotion_chain_links': 0,
                'event_emotion_links': 0,
                'emotion_event_links': 0,
                'avg_causal_weight': 0,
                'avg_emotion_chain_weight': 0,
                'avg_event_emotion_weight': 0,
                'avg_emotion_event_weight': 0
            }

        stats_text = f"""
        ╔══════════════════════════════════════════╗
        ║      СТАТИСТИКА ЭМОЦИОНАЛЬНОЙ СИСТЕМЫ    ║
        ╠══════════════════════════════════════════╣
        ║  Всего эмоций:          {len(self.emotion_system.engine.graph.emotions):<6}    ║
        ║  Всего событий:         {len(self.emotion_system.engine.graph.events):<6}    ║
        ║                                        ║
        ║  СВЯЗИ:                                  ║
        ║    Причинные:           {stats.get('causal_links', 0):<6}              ║
        ║    Цепочки эмоций:      {stats.get('emotion_chain_links', 0):<6}              ║
        ║    Событие→Эмоция:      {stats.get('event_emotion_links', 0):<6}              ║
        ║    Эмоция→Событие:      {stats.get('emotion_event_links', 0):<6}              ║
        ║                                        ║
        ║  Средние веса:                          ║
        ║    Причинные:           {stats.get('avg_causal_weight', 0):.3f}   ║
        ║    Цепочки эмоций:      {stats.get('avg_emotion_chain_weight', 0):.3f}   ║
        ║    Событие→Эмоция:      {stats.get('avg_event_emotion_weight', 0):.3f}   ║
        ║    Эмоция→Событие:      {stats.get('avg_emotion_event_weight', 0):.3f}   ║
        ║                                        ║
        ║  История:                               ║
        ║    Шагов:              {len(self.emotion_history):<6}              ║
        ║    Снимков графа:      {len(self.graph_snapshots):<6}              ║
        ╚══════════════════════════════════════════╝
        """

        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')


class EmotionBotComparator:
    """
    Сравнивает эмоциональное поведение ботов до и после эволюции.
    """

    def __init__(self):
        self.emotion_system = EmotionSystem()

    def create_base_emotion_bot(self):
        """Создает базовую эмоциональную систему."""
        return EmotionSystem()

    def create_evolved_emotion_bot(self):
        """Создает эволюционировавшую эмоциональную систему."""
        system = EmotionSystem()

        # Добавляем сложные эмоциональные связи
        try:
            system.engine.graph.add_emotion_chain(
                EmotionType.RESENTMENT,
                EmotionType.HATRED,
                weight=0.8
            )
            system.engine.graph.add_emotion_chain(
                EmotionType.ANGER,
                EmotionType.HATRED,
                weight=0.6
            )
            system.engine.graph.add_emotion_chain(
                EmotionType.FEAR,
                EmotionType.AWE,
                weight=0.5
            )
            system.engine.graph.add_emotion_chain(
                EmotionType.JOY,
                EmotionType.LOVE,
                weight=0.7
            )
        except Exception:
            pass

        return system

    def compare_emotion_systems(self):
        """
        Сравнивает базовую и эволюционировавшую эмоциональные системы.
        """
        print("🧠 Сравнение эмоциональных систем...")

        base_system = self.create_base_emotion_bot()
        evolved_system = self.create_evolved_emotion_bot()

        test_events = [
            {'description': 'Несправедливость', 'context': {'type': 'social'}},
            {'description': 'Предательство', 'context': {'type': 'social'}},
            {'description': 'Успех', 'context': {'type': 'achievement'}},
            {'description': 'Угроза', 'context': {'type': 'danger'}},
        ]

        results = {
            'base': [],
            'evolved': []
        }

        for event_data in test_events:
            event = EmotionalEvent(
                id=f"test_{event_data['description']}",
                description=event_data['description'],
                timestamp=0,
                context=event_data['context'],
                participants=['test'],
                embedding=np.random.randn(128)
            )

            # Базовая система
            base_responses = base_system.engine.process_event(event)
            results['base'].append({
                'event': event_data['description'],
                'responses': [r.emotion_type.value for r in base_responses],
                'dominant': base_responses[0].emotion_type.value if base_responses else 'neutral'
            })

            # Эволюционировавшая система
            evolved_responses = evolved_system.engine.process_event(event)
            results['evolved'].append({
                'event': event_data['description'],
                'responses': [r.emotion_type.value for r in evolved_responses],
                'dominant': evolved_responses[0].emotion_type.value if evolved_responses else 'neutral'
            })

        return results

    def visualize_emotion_comparison(self, comparison_results):
        """
        Визуализирует сравнение эмоциональных систем.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Сравнение эмоциональных систем: Базовая vs Эволюционировавшая', fontsize=16)

        # 1. Сравнение реакций
        ax1 = axes[0, 0]
        events = [r['event'] for r in comparison_results['base']]
        base_emotions = [r['dominant'] for r in comparison_results['base']]
        evolved_emotions = [r['dominant'] for r in comparison_results['evolved']]

        x = np.arange(len(events))
        width = 0.35

        ax1.bar(x - width / 2, [1] * len(events), width,
                color='red', alpha=0.5, label='Базовая')
        ax1.bar(x + width / 2, [1] * len(events), width,
                color='blue', alpha=0.5, label='Эволюционировавшая')

        ax1.set_title('Доминирующие эмоции')
        ax1.set_xticks(x)
        ax1.set_xticklabels(events, rotation=45, ha='right')
        ax1.set_yticks([])
        ax1.legend()

        # Добавляем подписи
        for i, (b, e) in enumerate(zip(base_emotions, evolved_emotions)):
            ax1.text(i - width / 2, 1.05, b, ha='center', va='bottom', fontsize=8)
            ax1.text(i + width / 2, 1.05, e, ha='center', va='bottom', fontsize=8)

        # 2. Сложность реакций
        ax2 = axes[0, 1]
        base_complexity = [len(r['responses']) for r in comparison_results['base']]
        evolved_complexity = [len(r['responses']) for r in comparison_results['evolved']]

        ax2.plot(events, base_complexity, 'ro-', label='Базовая', linewidth=2, markersize=8)
        ax2.plot(events, evolved_complexity, 'bs-', label='Эволюционировавшая', linewidth=2, markersize=8)
        ax2.set_title('Сложность эмоциональных реакций')
        ax2.set_xlabel('Событие')
        ax2.set_ylabel('Количество реакций')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticklabels(events, rotation=45, ha='right')

        # 3. Распределение эмоций (базовая)
        ax3 = axes[1, 0]
        base_all_emotions = []
        for r in comparison_results['base']:
            base_all_emotions.extend(r['responses'])
        unique, counts = np.unique(base_all_emotions, return_counts=True)
        if len(unique) > 0:
            ax3.bar(unique, counts, color='red', alpha=0.7)
            ax3.set_xticklabels(unique, rotation=45, ha='right')
        ax3.set_title('Распределение эмоций (Базовая)')
        ax3.set_xlabel('Эмоция')
        ax3.set_ylabel('Частота')

        # 4. Распределение эмоций (эволюционировавшая)
        ax4 = axes[1, 1]
        evolved_all_emotions = []
        for r in comparison_results['evolved']:
            evolved_all_emotions.extend(r['responses'])
        unique, counts = np.unique(evolved_all_emotions, return_counts=True)
        if len(unique) > 0:
            ax4.bar(unique, counts, color='blue', alpha=0.7)
            ax4.set_xticklabels(unique, rotation=45, ha='right')
        ax4.set_title('Распределение эмоций (Эволюционировавшая)')
        ax4.set_xlabel('Эмоция')
        ax4.set_ylabel('Частота')

        plt.tight_layout()
        plt.show()

        # Выводим сравнение в консоль
        print("\n📊 СРАВНЕНИЕ ЭМОЦИОНАЛЬНЫХ РЕАКЦИЙ:")
        print("-" * 60)
        print(f"{'Событие':<20} | {'Базовая':<15} | {'Эволюционировавшая':<15}")
        print("-" * 60)
        for i, event in enumerate(events):
            print(f"{event:<20} | {base_emotions[i]:<15} | {evolved_emotions[i]:<15}")
        print("-" * 60)


def main():
    """
    Главная функция для визуализации эмоциональной эволюции.
    """
    print("=" * 60)
    print("ВИЗУАЛИЗАЦИЯ ЭВОЛЮЦИИ ЭМОЦИОНАЛЬНОЙ ПОДСИСТЕМЫ")
    print("=" * 60)

    # 1. Симуляция эволюции эмоций
    visualizer = EmotionEvolutionVisualizer()
    visualizer.simulate_emotional_evolution(n_events=100)
    visualizer.visualize_emotion_timeline()

    # 2. Сравнение систем
    comparator = EmotionBotComparator()
    comparison_results = comparator.compare_emotion_systems()
    comparator.visualize_emotion_comparison(comparison_results)

    print("\n✅ Визуализация эмоциональной эволюции завершена!")


if __name__ == "__main__":
    main()


# # # scripts/visualize_emotion_evolution.py
# # """
# # Визуализация эволюции эмоциональной подсистемы бота.
# # """
# #
# # import sys
# # import os
# #
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# #
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import matplotlib.animation as animation
# # from matplotlib.patches import Circle, FancyBboxPatch
# # from typing import List, Dict, Any, Tuple
# # import networkx as nx
# # from datetime import datetime
# # import json
# #
# # from core.emotions.emotion_system import EmotionSystem
# # from core.emotions.emotion_graph import EmotionGraph
# # from core.emotions.emotion_base import EmotionalEvent, EmotionalResponse, EmotionType
# # from db.emotion_db import EmotionDB
# #
# #
# # class EmotionEvolutionVisualizer:
# #     """
# #     Визуализатор эволюции эмоциональной подсистемы.
# #     """
# #
# #     def __init__(self):
# #         self.emotion_system = EmotionSystem()
# #         self.db = EmotionDB()
# #         self.emotion_history = []
# #         self.graph_snapshots = []
# #
# #         # Цвета для разных эмоций
# #         self.emotion_colors = {
# #             'joy': '#FFD700',  # Золотой
# #             'sadness': '#4169E1',  # Синий
# #             'anger': '#FF0000',  # Красный
# #             'fear': '#8B008B',  # Темно-фиолетовый
# #             'surprise': '#FFA500',  # Оранжевый
# #             'disgust': '#228B22',  # Зеленый
# #             'trust': '#00CED1',  # Бирюзовый
# #             'anticipation': '#FF69B4',  # Розовый
# #             'love': '#FF1493',  # Горячий розовый
# #             'optimism': '#FFD700',  # Золотой
# #             'awe': '#9370DB',  # Средний фиолетовый
# #             'contempt': '#808080',  # Серый
# #             'resentment': '#8B0000',  # Темно-красный
# #             'hatred': '#000000',  # Черный
# #             'guilt': '#696969',  # Темно-серый
# #             'shame': '#A52A2A',  # Коричневый
# #             'empathy': '#98FB98'  # Светло-зеленый
# #         }
# #
# #     def simulate_emotional_evolution(self, n_events: int = 50):
# #         """
# #         Симулирует эволюцию эмоциональной системы.
# #         Генерирует последовательность событий и эмоциональных реакций.
# #         """
# #         print("🧠 Симуляция эволюции эмоциональной системы...")
# #
# #         # Базовые сценарии
# #         scenarios = [
# #             {'description': 'Угроза', 'emotion': EmotionType.FEAR, 'intensity': 0.8},
# #             {'description': 'Успех', 'emotion': EmotionType.JOY, 'intensity': 0.9},
# #             {'description': 'Неудача', 'emotion': EmotionType.SADNESS, 'intensity': 0.7},
# #             {'description': 'Несправедливость', 'emotion': EmotionType.ANGER, 'intensity': 0.8},
# #             {'description': 'Неожиданность', 'emotion': EmotionType.SURPRISE, 'intensity': 0.6},
# #             {'description': 'Доверие', 'emotion': EmotionType.TRUST, 'intensity': 0.5},
# #             {'description': 'Ожидание', 'emotion': EmotionType.ANTICIPATION, 'intensity': 0.4},
# #             {'description': 'Любовь', 'emotion': EmotionType.LOVE, 'intensity': 0.7},
# #             {'description': 'Обида', 'emotion': EmotionType.RESENTMENT, 'intensity': 0.6},
# #             {'description': 'Ненависть', 'emotion': EmotionType.HATRED, 'intensity': 0.9},
# #         ]
# #
# #         # Добавляем цепочки событий
# #         chains = [
# #             ['Успех', 'Успех', 'Неудача', 'Обида'],
# #             ['Угроза', 'Угроза', 'Страх', 'Бегство'],
# #             ['Несправедливость', 'Несправедливость', 'Гнев', 'Месть'],
# #             ['Доверие', 'Доверие', 'Предательство', 'Обида'],
# #         ]
# #
# #         for i in range(n_events):
# #             # Выбираем сценарий
# #             if i < len(scenarios):
# #                 scenario = scenarios[i % len(scenarios)]
# #             else:
# #                 scenario = scenarios[np.random.randint(0, len(scenarios))]
# #
# #             # Создаем событие
# #             event = EmotionalEvent(
# #                 id=f"sim_event_{i}",
# #                 description=scenario['description'],
# #                 timestamp=i,
# #                 context={'simulation': True, 'step': i},
# #                 participants=['sim_bot'],
# #                 embedding=np.random.randn(128)
# #             )
# #
# #             # Обрабатываем событие
# #             responses = self.emotion_system.process_sensory_input({
# #                 'vision': np.random.randn(64),
# #                 'sound': np.random.randn(32),
# #                 'smell': np.random.randn(32),
# #                 'context': {'event': scenario['description']}
# #             })
# #
# #             # Сохраняем историю
# #             self.emotion_history.append({
# #                 'step': i,
# #                 'event': event,
# #                 'responses': responses,
# #                 'state': self.emotion_system.get_emotional_state()
# #             })
# #
# #             # Сохраняем снимок графа каждые 10 шагов
# #             if i % 10 == 0:
# #                 self.graph_snapshots.append({
# #                     'step': i,
# #                     'graph': self.emotion_system.engine.graph,
# #                     'stats': self.emotion_system.engine.graph.get_links_statistics()
# #                 })
# #
# #             # Добавляем случайные связи между эмоциями (эволюция)
# #             if i > 0 and np.random.random() < 0.2:
# #                 emotions = list(EmotionType)
# #                 e1 = np.random.choice(emotions)
# #                 e2 = np.random.choice(emotions)
# #                 if e1 != e2:
# #                     self.emotion_system.engine.graph.add_emotion_chain(
# #                         e1, e2,
# #                         weight=np.random.uniform(0.3, 0.9)
# #                     )
# #
# #         print(f"✅ Симуляция завершена. Обработано {n_events} событий.")
# #         return self.emotion_history
# #
# #     def visualize_emotion_timeline(self):
# #         """
# #         Визуализирует временную линию эмоциональных состояний.
# #         """
# #         if not self.emotion_history:
# #             print("⚠️ Нет данных. Сначала запустите simulate_emotional_evolution()")
# #             return
# #
# #         fig, axes = plt.subplots(3, 2, figsize=(16, 12))
# #         fig.suptitle('Эволюция эмоциональной подсистемы', fontsize=16)
# #
# #         # 1. Временная линия эмоций
# #         ax1 = axes[0, 0]
# #         self._plot_emotion_timeline(ax1)
# #
# #         # 2. Интенсивность эмоций
# #         ax2 = axes[0, 1]
# #         self._plot_emotion_intensity(ax2)
# #
# #         # 3. Граф эмоций (текущее состояние)
# #         ax3 = axes[1, 0]
# #         self._plot_emotion_graph(ax3)
# #
# #         # 4. Эволюция графа
# #         ax4 = axes[1, 1]
# #         self._plot_graph_evolution(ax4)
# #
# #         # 5. Распределение эмоций
# #         ax5 = axes[2, 0]
# #         self._plot_emotion_distribution(ax5)
# #
# #         # 6. Статистика
# #         ax6 = axes[2, 1]
# #         self._plot_emotion_stats(ax6)
# #
# #         plt.tight_layout()
# #         plt.show()
# #
# #     def _plot_emotion_timeline(self, ax):
# #         """Рисует временную линию эмоций."""
# #         steps = [h['step'] for h in self.emotion_history]
# #         emotions = []
# #         intensities = []
# #
# #         for h in self.emotion_history:
# #             state = h['state']
# #             emotions.append(state['dominant_emotion'])
# #             intensities.append(state['intensity'])
# #
# #         # Создаем цветовую карту
# #         unique_emotions = list(set(emotions))
# #         emotion_to_color = {e: self.emotion_colors.get(e, '#808080') for e in unique_emotions}
# #         colors = [emotion_to_color[e] for e in emotions]
# #
# #         # Рисуем точки
# #         ax.scatter(steps, [0] * len(steps), c=colors, s=intensities * 200, alpha=0.7)
# #         ax.set_title('Временная линия эмоций')
# #         ax.set_xlabel('Шаг')
# #         ax.set_yticks([])
# #         ax.set_ylim(-1, 1)
# #         ax.grid(True, alpha=0.3)
# #
# #         # Добавляем легенду
# #         patches = [plt.Line2D([0], [0], marker='o', color='w',
# #                               markerfacecolor=color, markersize=10, label=emotion)
# #                    for emotion, color in emotion_to_color.items()]
# #         ax.legend(handles=patches, loc='upper right', fontsize=8, ncol=2)
# #
# #     def _plot_emotion_intensity(self, ax):
# #         """Рисует график интенсивности эмоций."""
# #         steps = [h['step'] for h in self.emotion_history]
# #         intensities = [h['state']['intensity'] for h in self.emotion_history]
# #
# #         ax.plot(steps, intensities, 'b-', linewidth=2, alpha=0.7)
# #         ax.fill_between(steps, intensities, 0, alpha=0.3, color='blue')
# #         ax.set_title('Интенсивность доминирующей эмоции')
# #         ax.set_xlabel('Шаг')
# #         ax.set_ylabel('Интенсивность')
# #         ax.set_ylim(0, 1.1)
# #         ax.grid(True, alpha=0.3)
# #
# #         # Добавляем скользящее среднее
# #         if len(intensities) > 10:
# #             window = 10
# #             moving_avg = np.convolve(intensities, np.ones(window) / window, mode='valid')
# #             ax.plot(steps[window - 1:], moving_avg, 'r-', linewidth=2, label='Скользящее среднее')
# #             ax.legend()
# #
# #     def _plot_emotion_graph(self, ax):
# #         """Рисует текущий граф эмоций."""
# #         graph = self.emotion_system.engine.graph
# #
# #         # Строим граф
# #         G = nx.DiGraph()
# #
# #         # Добавляем узлы
# #         for emotion_type in graph.emotions.keys():
# #             G.add_node(emotion_type,
# #                        color=self.emotion_colors.get(emotion_type, '#808080'))
# #
# #         # Добавляем ребра
# #         for link in graph.emotion_chain_links.values():
# #             G.add_edge(link.source_id, link.target_id, weight=link.weight)
# #
# #         # Рисуем
# #         pos = nx.spring_layout(G, k=1, iterations=50)
# #         colors = [G.nodes[node].get('color', '#808080') for node in G.nodes()]
# #
# #         nx.draw(G, pos, ax=ax, node_color=colors, with_labels=True,
# #                 node_size=500, font_size=8, arrows=True,
# #                 arrowstyle='->', arrowsize=20)
# #
# #         ax.set_title('Граф эмоций (текущее состояние)')
# #
# #     def _plot_graph_evolution(self, ax):
# #         """Рисует эволюцию графа."""
# #         if len(self.graph_snapshots) < 2:
# #             ax.text(0.5, 0.5, 'Недостаточно данных', ha='center', va='center')
# #             return
# #
# #         steps = [s['step'] for s in self.graph_snapshots]
# #         edge_counts = [len(s['graph'].emotion_chain_links) for s in self.graph_snapshots]
# #         node_counts = [len(s['graph'].emotions) for s in self.graph_snapshots]
# #
# #         ax.plot(steps, edge_counts, 'b-o', label='Связей', linewidth=2)
# #         ax.plot(steps, node_counts, 'r-s', label='Эмоций', linewidth=2)
# #         ax.set_title('Эволюция графа эмоций')
# #         ax.set_xlabel('Шаг')
# #         ax.set_ylabel('Количество')
# #         ax.legend()
# #         ax.grid(True, alpha=0.3)
# #
# #     def _plot_emotion_distribution(self, ax):
# #         """Рисует распределение эмоций."""
# #         emotions = [h['state']['dominant_emotion'] for h in self.emotion_history]
# #
# #         unique, counts = np.unique(emotions, return_counts=True)
# #         colors = [self.emotion_colors.get(e, '#808080') for e in unique]
# #
# #         bars = ax.bar(unique, counts, color=colors, alpha=0.7)
# #         ax.set_title('Распределение эмоций')
# #         ax.set_xlabel('Эмоция')
# #         ax.set_ylabel('Частота')
# #         ax.set_xticklabels(unique, rotation=45, ha='right')
# #
# #         # Добавляем значения
# #         for bar, count in zip(bars, counts):
# #             ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
# #                     str(count), ha='center', va='bottom', fontsize=9)
# #
# #     def _plot_emotion_stats(self, ax):
# #         """Рисует статистику эмоций."""
# #         ax.axis('off')
# #
# #         stats = self.emotion_system.engine.graph.get_links_statistics()
# #
# #         stats_text = f"""
# #         ╔══════════════════════════════════════════╗
# #         ║      СТАТИСТИКА ЭМОЦИОНАЛЬНОЙ СИСТЕМЫ    ║
# #         ╠══════════════════════════════════════════╣
# #         ║  Всего эмоций:          {len(self.emotion_system.engine.graph.emotions):<6}    ║
# #         ║  Всего событий:         {len(self.emotion_system.engine.graph.events):<6}    ║
# #         ║                                        ║
# #         ║  СВЯЗИ:                                  ║
# #         ║    Причинные:           {stats['causal_links']:<6}              ║
# #         ║    Цепочки эмоций:      {stats['emotion_chain_links']:<6}              ║
# #         ║    Событие→Эмоция:      {stats['event_emotion_links']:<6}              ║
# #         ║    Эмоция→Событие:      {stats['emotion_event_links']:<6}              ║
# #         ║                                        ║
# #         ║  Средние веса:                          ║
# #         ║    Причинные:           {stats['avg_causal_weight']:.3f}   ║
# #         ║    Цепочки эмоций:      {stats['avg_emotion_chain_weight']:.3f}   ║
# #         ║    Событие→Эмоция:      {stats['avg_event_emotion_weight']:.3f}   ║
# #         ║    Эмоция→Событие:      {stats['avg_emotion_event_weight']:.3f}   ║
# #         ║                                        ║
# #         ║  История:                               ║
# #         ║    Шагов:              {len(self.emotion_history):<6}              ║
# #         ║    Снимков графа:      {len(self.graph_snapshots):<6}              ║
# #         ╚══════════════════════════════════════════╝
# #         """
# #
# #         ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
# #                 fontsize=9, verticalalignment='top', fontfamily='monospace')
# #
# #
# # class EmotionBotComparator:
# #     """
# #     Сравнивает эмоциональное поведение ботов до и после эволюции.
# #     """
# #
# #     def __init__(self):
# #         self.emotion_system = EmotionSystem()
# #
# #     def create_base_emotion_bot(self):
# #         """Создает базовую эмоциональную систему."""
# #         # Используем только базовые эмоции
# #         return EmotionSystem()
# #
# #     def create_evolved_emotion_bot(self):
# #         """Создает эволюционировавшую эмоциональную систему."""
# #         system = EmotionSystem()
# #
# #         # Добавляем сложные эмоциональные связи
# #         # Обида → Ненависть
# #         system.engine.graph.add_emotion_chain(
# #             EmotionType.RESENTMENT,
# #             EmotionType.HATRED,
# #             weight=0.8
# #         )
# #
# #         # Гнев → Ненависть
# #         system.engine.graph.add_emotion_chain(
# #             EmotionType.ANGER,
# #             EmotionType.HATRED,
# #             weight=0.6
# #         )
# #
# #         # Страх → Трепет
# #         system.engine.graph.add_emotion_chain(
# #             EmotionType.FEAR,
# #             EmotionType.AWE,
# #             weight=0.5
# #         )
# #
# #         # Радость → Любовь
# #         system.engine.graph.add_emotion_chain(
# #             EmotionType.JOY,
# #             EmotionType.LOVE,
# #             weight=0.7
# #         )
# #
# #         # Добавляем связи с событиями
# #         system.engine.graph.add_event_emotion_link(
# #             'injustice',
# #             EmotionType.RESENTMENT,
# #             probability=0.8,
# #             intensity_factor=1.2
# #         )
# #
# #         system.engine.graph.add_event_emotion_link(
# #             'betrayal',
# #             EmotionType.HATRED,
# #             probability=0.9,
# #             intensity_factor=1.3
# #         )
# #
# #         return system
# #
# #     def compare_emotion_systems(self):
# #         """
# #         Сравнивает базовую и эволюционировавшую эмоциональные системы.
# #         """
# #         print("🧠 Сравнение эмоциональных систем...")
# #
# #         # Создаем системы
# #         base_system = self.create_base_emotion_bot()
# #         evolved_system = self.create_evolved_emotion_bot()
# #
# #         # Тестовые события
# #         test_events = [
# #             {'description': 'Несправедливость', 'context': {'type': 'social'}},
# #             {'description': 'Предательство', 'context': {'type': 'social'}},
# #             {'description': 'Успех', 'context': {'type': 'achievement'}},
# #             {'description': 'Угроза', 'context': {'type': 'danger'}},
# #         ]
# #
# #         results = {
# #             'base': [],
# #             'evolved': []
# #         }
# #
# #         for event_data in test_events:
# #             # Создаем событие
# #             event = EmotionalEvent(
# #                 id=f"test_{event_data['description']}",
# #                 description=event_data['description'],
# #                 timestamp=0,
# #                 context=event_data['context'],
# #                 participants=['test'],
# #                 embedding=np.random.randn(128)
# #             )
# #
# #             # Обрабатываем в базовой системе
# #             base_responses = base_system.engine.process_event(event)
# #             results['base'].append({
# #                 'event': event_data['description'],
# #                 'responses': [r.emotion_type.value for r in base_responses],
# #                 'dominant': base_responses[0].emotion_type.value if base_responses else 'neutral'
# #             })
# #
# #             # Обрабатываем в эволюционировавшей системе
# #             evolved_responses = evolved_system.engine.process_event(event)
# #             results['evolved'].append({
# #                 'event': event_data['description'],
# #                 'responses': [r.emotion_type.value for r in evolved_responses],
# #                 'dominant': evolved_responses[0].emotion_type.value if evolved_responses else 'neutral'
# #             })
# #
# #         return results
# #
# #     def visualize_emotion_comparison(self, comparison_results):
# #         """
# #         Визуализирует сравнение эмоциональных систем.
# #         """
# #         fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# #         fig.suptitle('Сравнение эмоциональных систем: Базовая vs Эволюционировавшая', fontsize=16)
# #
# #         # 1. Сравнение реакций
# #         ax1 = axes[0, 0]
# #         events = [r['event'] for r in comparison_results['base']]
# #         base_emotions = [r['dominant'] for r in comparison_results['base']]
# #         evolved_emotions = [r['dominant'] for r in comparison_results['evolved']]
# #
# #         x = np.arange(len(events))
# #         width = 0.35
# #
# #         # Создаем цветовую карту
# #         all_emotions = list(set(base_emotions + evolved_emotions))
# #         colors = {e: plt.cm.tab20(i) for i, e in enumerate(all_emotions)}
# #
# #         ax1.bar(x - width / 2, [1] * len(events), width,
# #                 color=[colors.get(e, 'gray') for e in base_emotions],
# #                 label='Базовая', alpha=0.7)
# #         ax1.bar(x + width / 2, [1] * len(events), width,
# #                 color=[colors.get(e, 'gray') for e in evolved_emotions],
# #                 label='Эволюционировавшая', alpha=0.7)
# #
# #         ax1.set_title('Доминирующие эмоции')
# #         ax1.set_xticks(x)
# #         ax1.set_xticklabels(events, rotation=45, ha='right')
# #         ax1.set_yticks([])
# #         ax1.legend()
# #
# #         # Добавляем подписи
# #         for i, (b, e) in enumerate(zip(base_emotions, evolved_emotions)):
# #             ax1.text(i - width / 2, 1.05, b, ha='center', va='bottom', fontsize=8, rotation=90)
# #             ax1.text(i + width / 2, 1.05, e, ha='center', va='bottom', fontsize=8, rotation=90)
# #
# #         # 2. Сложность реакций
# #         ax2 = axes[0, 1]
# #         base_complexity = [len(r['responses']) for r in comparison_results['base']]
# #         evolved_complexity = [len(r['responses']) for r in comparison_results['evolved']]
# #
# #         ax2.plot(events, base_complexity, 'ro-', label='Базовая', linewidth=2, markersize=8)
# #         ax2.plot(events, evolved_complexity, 'bs-', label='Эволюционировавшая', linewidth=2, markersize=8)
# #         ax2.set_title('Сложность эмоциональных реакций')
# #         ax2.set_xlabel('Событие')
# #         ax2.set_ylabel('Количество реакций')
# #         ax2.legend()
# #         ax2.grid(True, alpha=0.3)
# #         ax2.set_xticklabels(events, rotation=45, ha='right')
# #
# #         # 3. Распределение эмоций (базовая)
# #         ax3 = axes[1, 0]
# #         base_all_emotions = []
# #         for r in comparison_results['base']:
# #             base_all_emotions.extend(r['responses'])
# #         unique, counts = np.unique(base_all_emotions, return_counts=True)
# #         ax3.bar(unique, counts, color='red', alpha=0.7)
# #         ax3.set_title('Распределение эмоций (Базовая)')
# #         ax3.set_xlabel('Эмоция')
# #         ax3.set_ylabel('Частота')
# #         ax3.set_xticklabels(unique, rotation=45, ha='right')
# #
# #         # 4. Распределение эмоций (эволюционировавшая)
# #         ax4 = axes[1, 1]
# #         evolved_all_emotions = []
# #         for r in comparison_results['evolved']:
# #             evolved_all_emotions.extend(r['responses'])
# #         unique, counts = np.unique(evolved_all_emotions, return_counts=True)
# #         ax4.bar(unique, counts, color='blue', alpha=0.7)
# #         ax4.set_title('Распределение эмоций (Эволюционировавшая)')
# #         ax4.set_xlabel('Эмоция')
# #         ax4.set_ylabel('Частота')
# #         ax4.set_xticklabels(unique, rotation=45, ha='right')
# #
# #         plt.tight_layout()
# #         plt.show()
# #
# #         # Выводим сравнение в консоль
# #         print("\n📊 СРАВНЕНИЕ ЭМОЦИОНАЛЬНЫХ РЕАКЦИЙ:")
# #         print("-" * 60)
# #         print(f"{'Событие':<20} | {'Базовая':<15} | {'Эволюционировавшая':<15}")
# #         print("-" * 60)
# #         for i, event in enumerate(events):
# #             print(f"{event:<20} | {base_emotions[i]:<15} | {evolved_emotions[i]:<15}")
# #         print("-" * 60)
# #
# #
# # def main():
# #     """
# #     Главная функция для визуализации эмоциональной эволюции.
# #     """
# #     print("=" * 60)
# #     print("ВИЗУАЛИЗАЦИЯ ЭВОЛЮЦИИ ЭМОЦИОНАЛЬНОЙ ПОДСИСТЕМЫ")
# #     print("=" * 60)
# #
# #     # 1. Симуляция эволюции эмоций
# #     visualizer = EmotionEvolutionVisualizer()
# #     visualizer.simulate_emotional_evolution(n_events=100)
# #     visualizer.visualize_emotion_timeline()
# #
# #     # 2. Сравнение систем
# #     comparator = EmotionBotComparator()
# #     comparison_results = comparator.compare_emotion_systems()
# #     comparator.visualize_emotion_comparison(comparison_results)
# #
# #     print("\n✅ Визуализация эмоциональной эволюции завершена!")
# #
# #
# # if __name__ == "__main__":
# #     main()