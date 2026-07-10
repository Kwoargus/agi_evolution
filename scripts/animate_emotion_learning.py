# scripts/animate_emotion_learning.py
"""
Анимация обучения эмоциональной системы в реальном времени.
Показывает, как бот учится распознавать эмоции и как заполняются диаграммы.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
from typing import List, Dict, Any
import warnings

warnings.filterwarnings('ignore')

from core.emotions.emotion_system import EmotionSystem
from core.emotions.emotion_base import EmotionalEvent, EmotionalResponse, EmotionType


class EmotionLearningAnimator:
    """
    Анимированная визуализация обучения эмоциональной системы.
    """

    def __init__(self, n_events: int = 100, interval: int = 500):
        """
        Args:
            n_events: Количество событий для симуляции
            interval: Интервал обновления в мс
        """
        self.n_events = n_events
        self.interval = interval

        self.emotion_system = EmotionSystem()
        self.emotion_history = []
        self.emotion_counts = {}
        self.emotion_chains = []

        # Цвета для эмоций
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
            'resentment': '#8B0000',
            'hatred': '#000000',
        }

        # Создаем события и связи
        self._create_events_and_links()

        # Данные для анимации
        self.current_step = 0
        self.event_types = ['danger', 'success', 'failure', 'injustice',
                            'surprise', 'love', 'betrayal', 'achievement',
                            'loss', 'gift']
        self.weights = [0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05, 0.15, 0.05, 0.1]

        # Инициализируем фигуру
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
        self.fig.suptitle('Обучение эмоциональной системы', fontsize=16)

        # Счетчики
        self.emotion_counts = {e.value: 0 for e in EmotionType}

        print("✅ Аниматор инициализирован")

    def _create_events_and_links(self):
        """Создает события и связи с эмоциями."""
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

        for event_id, data in events_data.items():
            embedding = np.zeros(128)
            embedding[:len(data['pattern'])] = data['pattern']
            embedding += np.random.randn(128) * 0.05
            norm = np.linalg.norm(embedding) + 1e-8
            embedding = embedding / norm

            event = EmotionalEvent(
                id=event_id,
                description=data['desc'],
                timestamp=0,
                context={'type': event_id},
                participants=['system'],
                embedding=embedding
            )

            self.emotion_system.engine.graph.add_event(event)

            main_emotion = event_emotion_map[event_id]
            self.emotion_system.engine.graph.add_event_emotion_link(
                event_id, main_emotion, probability=0.9, intensity_factor=1.2
            )

            for extra_emotion in additional_emotions.get(event_id, []):
                self.emotion_system.engine.graph.add_event_emotion_link(
                    event_id, extra_emotion, probability=0.5, intensity_factor=0.8
                )

    def _process_one_step(self):
        """Обрабатывает один шаг обучения."""
        if self.current_step >= self.n_events:
            return

        # Выбираем событие
        event_type = np.random.choice(self.event_types, p=self.weights)
        event = self.emotion_system.engine.graph.events.get(event_type)
        if not event:
            return

        # Обрабатываем
        responses = self.emotion_system.process_sensory_input({
            'vision': event.embedding[:64],
            'sound': event.embedding[64:96],
            'smell': event.embedding[96:128],
            'context': {'type': event_type},
            'participants': ['bot']
        })

        state = self.emotion_system.get_emotional_state()

        # Сохраняем историю
        self.emotion_history.append({
            'step': self.current_step,
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

        # Цепочки
        if len(responses) > 1:
            for j in range(len(responses) - 1):
                self.emotion_chains.append((
                    responses[j].emotion_type.value,
                    responses[j + 1].emotion_type.value
                ))

        # Добавляем новые связи (эволюция)
        if self.current_step > 0 and self.current_step % 20 == 0:
            random_event = np.random.choice(self.event_types)
            random_emotion = np.random.choice(list(EmotionType))
            if random_event in self.emotion_system.engine.graph.events:
                self.emotion_system.engine.graph.add_event_emotion_link(
                    random_event, random_emotion,
                    probability=np.random.uniform(0.3, 0.7),
                    intensity_factor=np.random.uniform(0.5, 1.0)
                )

        self.current_step += 1

    def _update_plot(self, frame):
        """Обновляет графики."""
        # Обрабатываем шаг
        self._process_one_step()

        # Очищаем все оси
        for ax in self.axes.flat:
            ax.clear()

        # 1. Текущее состояние бота
        ax1 = self.axes[0, 0]
        self._plot_bot_state(ax1)

        # 2. Временная линия эмоций
        ax2 = self.axes[0, 1]
        self._plot_timeline(ax2)

        # 3. Распределение эмоций (обновляемое)
        ax3 = self.axes[0, 2]
        self._plot_distribution(ax3)

        # 4. Интенсивность
        ax4 = self.axes[1, 0]
        self._plot_intensity(ax4)

        # 5. Матрица переходов
        ax5 = self.axes[1, 1]
        self._plot_transition_matrix(ax5)

        # 6. Статистика
        ax6 = self.axes[1, 2]
        self._plot_stats(ax6)

        plt.tight_layout()

        # Заголовок с прогрессом
        self.fig.suptitle(f'Обучение эмоциональной системы: шаг {self.current_step}/{self.n_events}',
                          fontsize=14)

        return self.axes.flat

    def _plot_bot_state(self, ax):
        """Показывает текущее состояние бота."""
        if not self.emotion_history:
            ax.text(0.5, 0.5, 'Ожидание...', ha='center', va='center')
            return

        last = self.emotion_history[-1]

        # Рисуем "бота"
        circle = Circle((0.3, 0.5), 0.2,
                        color=self.emotion_colors.get(last['dominant'], 'gray'),
                        alpha=0.8)
        ax.add_patch(circle)

        # Эмоция
        ax.text(0.3, 0.8, last['dominant'], ha='center', fontsize=12, fontweight='bold')
        ax.text(0.3, 0.3, f'Интенсивность: {last["intensity"]:.2f}', ha='center', fontsize=9)

        # Событие
        ax.text(0.7, 0.7, f'Событие:', ha='center', fontsize=10, fontweight='bold')
        ax.text(0.7, 0.5, last['event_type'], ha='center', fontsize=10)
        ax.text(0.7, 0.3, f'Шаг: {last["step"]}', ha='center', fontsize=9)

        ax.set_title('Состояние бота', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def _plot_timeline(self, ax):
        """Рисует временную линию."""
        if len(self.emotion_history) < 2:
            ax.text(0.5, 0.5, 'Сбор данных...', ha='center', va='center')
            return

        steps = [h['step'] for h in self.emotion_history]
        emotions = [h['dominant'] for h in self.emotion_history]
        intensities = [h['intensity'] for h in self.emotion_history]

        # Показываем последние 30 шагов
        n = min(30, len(steps))
        steps = steps[-n:]
        emotions = emotions[-n:]
        intensities = intensities[-n:]

        # Рисуем точки
        for i, (step, emotion, intensity) in enumerate(zip(steps, emotions, intensities)):
            color = self.emotion_colors.get(emotion, '#808080')
            ax.scatter(i, 0, c=[color], s=intensity * 200 + 20, alpha=0.7)

        ax.set_title('Последние 30 эмоций', fontsize=12)
        ax.set_xlabel('Шаг в окне')
        ax.set_yticks([])
        ax.set_ylim(-0.5, 0.5)
        ax.grid(True, alpha=0.3)

    def _plot_distribution(self, ax):
        """Рисует распределение эмоций."""
        if not self.emotion_history:
            ax.text(0.5, 0.5, 'Сбор данных...', ha='center', va='center')
            return

        # Берем последние 50 шагов для динамики
        n = min(50, len(self.emotion_history))
        recent = self.emotion_history[-n:]
        emotions = [h['dominant'] for h in recent]

        unique, counts = np.unique(emotions, return_counts=True)

        if len(unique) > 0:
            colors = [self.emotion_colors.get(e, '#808080') for e in unique]
            bars = ax.bar(range(len(unique)), counts, color=colors, alpha=0.7)
            ax.set_xticks(range(len(unique)))
            ax.set_xticklabels(unique, rotation=45, ha='right', fontsize=8)

            # Добавляем значения
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        str(count), ha='center', va='bottom', fontsize=8)

        ax.set_title(f'Распределение эмоций (последние {n} шагов)', fontsize=12)
        ax.set_ylabel('Частота')

    def _plot_intensity(self, ax):
        """Рисует интенсивность."""
        if len(self.emotion_history) < 2:
            ax.text(0.5, 0.5, 'Сбор данных...', ha='center', va='center')
            return

        steps = [h['step'] for h in self.emotion_history]
        intensities = [h['intensity'] for h in self.emotion_history]
        valences = [h['valence'] for h in self.emotion_history]

        # Показываем все данные
        ax.plot(steps, intensities, 'b-', alpha=0.7, label='Интенсивность')
        ax.plot(steps, valences, 'r-', alpha=0.5, label='Валентность')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

        ax.set_title('Интенсивность и валентность', fontsize=12)
        ax.set_xlabel('Шаг')
        ax.set_ylabel('Значение')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)

    def _plot_transition_matrix(self, ax):
        """Рисует матрицу переходов."""
        if len(self.emotion_history) < 3:
            ax.text(0.5, 0.5, 'Сбор данных...', ha='center', va='center')
            return

        # Берем последние 50 переходов
        n = min(50, len(self.emotion_history))
        recent = self.emotion_history[-n:]

        transitions = {}
        for i in range(len(recent) - 1):
            e1 = recent[i]['dominant']
            e2 = recent[i + 1]['dominant']
            if e1 != e2:
                key = (e1, e2)
                transitions[key] = transitions.get(key, 0) + 1

        if not transitions:
            ax.text(0.5, 0.5, 'Нет переходов', ha='center', va='center')
            return

        all_emotions = list(set([e for pair in transitions.keys() for e in pair]))
        n_emotions = len(all_emotions)
        matrix = np.zeros((n_emotions, n_emotions))
        emotion_to_idx = {e: i for i, e in enumerate(all_emotions)}

        for (e1, e2), count in transitions.items():
            if e1 in emotion_to_idx and e2 in emotion_to_idx:
                matrix[emotion_to_idx[e1], emotion_to_idx[e2]] = count

        # Нормализуем
        row_sums = matrix.sum(axis=1, keepdims=True)
        if row_sums.sum() > 0:
            matrix = np.divide(matrix, row_sums, where=row_sums != 0)

        im = ax.imshow(matrix, cmap='Blues', aspect='auto')
        ax.set_xticks(range(n_emotions))
        ax.set_yticks(range(n_emotions))
        if n_emotions <= 8:
            ax.set_xticklabels(all_emotions, rotation=45, ha='right', fontsize=7)
            ax.set_yticklabels(all_emotions, fontsize=7)

        ax.set_title(f'Матрица переходов (последние {n} шагов)', fontsize=12)
        plt.colorbar(im, ax=ax)

    def _plot_stats(self, ax):
        """Рисует статистику."""
        ax.axis('off')

        if not self.emotion_history:
            ax.text(0.5, 0.5, 'Сбор данных...', ha='center', va='center')
            return

        all_emotions = [h['dominant'] for h in self.emotion_history]
        unique_emotions = len(set(all_emotions))

        stats_text = f"""
        ╔══════════════════════════════════════════╗
        ║      СТАТИСТИКА ОБУЧЕНИЯ                ║
        ╠══════════════════════════════════════════╣
        ║  Шаг:              {self.current_step:>4}/{self.n_events}        ║
        ║  Уникальных эмоций: {unique_emotions:>4}                    ║
        ║  Всего реакций:     {len(self.emotion_history):>4}                    ║
        ║                                        ║
        ║  Текущая эмоция:    {self.emotion_history[-1]['dominant']:>12}  ║
        ║  Интенсивность:     {self.emotion_history[-1]['intensity']:>6.2f}   ║
        ║  Валентность:       {self.emotion_history[-1]['valence']:>6.2f}   ║
        ║                                        ║
        ║  Цепочек эмоций:    {len(self.emotion_chains):>4}                    ║
        ║  Прогресс:          {'█' * (self.current_step // 5)}{' ' * (20 - self.current_step // 5)}  ║
        ╚══════════════════════════════════════════╝
        """

        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')

    def run(self):
        """
        Запускает анимацию обучения.
        """
        print("🚀 Запуск анимации обучения...")
        print("Нажмите Ctrl+C для остановки")

        # Создаем анимацию
        anim = animation.FuncAnimation(
            self.fig,
            self._update_plot,
            frames=self.n_events,
            interval=self.interval,
            blit=False,
            repeat=False
        )

        plt.show()

        print("✅ Анимация завершена")


def main():
    """
    Главная функция для запуска анимации.
    """
    print("=" * 60)
    print("АНИМАЦИЯ ОБУЧЕНИЯ ЭМОЦИОНАЛЬНОЙ СИСТЕМЫ")
    print("=" * 60)

    # Создаем и запускаем аниматор
    animator = EmotionLearningAnimator(n_events=100, interval=400)
    animator.run()

    print("\n✅ Программа завершена")


if __name__ == "__main__":
    main()