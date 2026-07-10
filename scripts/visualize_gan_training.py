# scripts/visualize_gan_training.py (исправленный)
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from models.gan import GAN
import torch
from typing import List, Dict, Optional, Tuple, Any, Union


class GANTrainingVisualizer:
    """
    Визуализатор процесса обучения GAN в реальном времени.
    """

    def __init__(self, gan: GAN, save_path: str = 'gan_training.gif'):
        self.gan = gan
        self.save_path = save_path
        self.fig, self.axes = None, None
        self.frame_count = 0

        # Храним данные для графиков
        self.g_losses = []
        self.d_losses = []
        self.scores = []

    def setup_plot(self):
        """Настраивает окна для визуализации."""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('GAN Training Visualization', fontsize=16)

        # 1. Потери - настройка
        ax_loss = self.axes[0, 0]
        ax_loss.set_title('Generator & Discriminator Losses')
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss')
        ax_loss.grid(True)
        ax_loss.set_xlim(0, 100)
        ax_loss.set_ylim(0, 2.5)

        # 2. Оценки паттернов
        ax_scores = self.axes[0, 1]
        ax_scores.set_title('Pattern Quality Scores')
        ax_scores.set_xlabel('Pattern #')
        ax_scores.set_ylabel('Score')
        ax_scores.set_ylim(0, 1)
        ax_scores.grid(True)

        # 3. Тепловая карта паттернов
        self.axes[1, 0].set_title('Generated Patterns (Heatmap)')
        self.axes[1, 0].set_xlabel('Pattern Dimension')
        self.axes[1, 0].set_ylabel('Pattern #')

        # 4. Статистика
        self.axes[1, 1].set_title('GAN Statistics')
        self.axes[1, 1].axis('off')

        plt.tight_layout()

    def update_loss_plot(self):
        """Обновляет график потерь."""
        ax = self.axes[0, 0]
        ax.clear()

        # Получаем потери из GAN
        g_losses = self.gan.generator_losses
        d_losses = self.gan.discriminator_losses

        # Настройка осей
        ax.set_title('Generator & Discriminator Losses')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.grid(True)

        # Проверяем, есть ли данные
        if g_losses and len(g_losses) > 0:
            print(f"  [DEBUG] Обновление графика: G_losses={len(g_losses)}, D_losses={len(d_losses) if d_losses else 0}")

            # Показываем последние 200 точек для производительности
            if len(g_losses) > 200:
                x_data = list(range(len(g_losses) - 200, len(g_losses)))
                g_show = g_losses[-200:]
                d_show = d_losses[-200:] if d_losses else []
            else:
                x_data = list(range(len(g_losses)))
                g_show = g_losses
                d_show = d_losses if d_losses else []

            # Рисуем линии
            ax.plot(x_data, g_show, label='Generator', color='blue', alpha=0.7, linewidth=1.5)
            if d_show:
                ax.plot(x_data, d_show, label='Discriminator', color='red', alpha=0.7, linewidth=1.5)

            # Настраиваем масштаб
            if g_show:
                ax.set_xlim(0, max(len(g_show) + 10, 50))
                all_values = g_show + (d_show if d_show else [])
                if all_values:
                    max_val = max(all_values)
                    min_val = min(all_values)
                    y_max = max(max_val * 1.2, 0.5)
                    y_min = max(min_val * 0.8, 0)
                    ax.set_ylim(y_min, min(y_max, 5.0))

            # Легенда
            if d_show:
                ax.legend(loc='upper right')
            else:
                ax.legend(loc='upper right')

            # Добавляем текущие значения
            if g_show:
                last_g = g_show[-1]
                last_d = d_show[-1] if d_show else 0
                ax.text(0.02, 0.95, f'G: {last_g:.4f} | D: {last_d:.4f} | Steps: {len(g_losses)}',
                        transform=ax.transAxes, verticalalignment='top',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        else:
            # Если данных нет, показываем сообщение
            ax.text(0.5, 0.5, 'Waiting for data...\n(Need at least 1 training step)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='gray')

    def update_scores_plot(self):
        """Обновляет график оценок паттернов."""
        ax = self.axes[0, 1]
        ax.clear()

        # Получаем оценки из репозитория
        scores = self.gan.pattern_repository.scores if hasattr(self.gan, 'pattern_repository') else []

        ax.set_title('Pattern Quality Scores')
        ax.set_xlabel('Pattern #')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.grid(True)

        if scores:
            # Показываем последние 50 оценок
            scores_show = scores[-50:]
            x_data = list(range(len(scores_show)))

            ax.bar(x_data, scores_show, color='green', alpha=0.7, width=0.8)
            ax.axhline(y=0.7, color='red', linestyle='--', label='Quality threshold (0.7)', linewidth=1.5)
            ax.axhline(y=0.5, color='orange', linestyle='--', label='Min threshold (0.5)', alpha=0.5, linewidth=1)

            # Статистика
            avg_score = np.mean(scores_show)
            best_score = max(scores_show)
            ax.text(0.02, 0.95, f'Avg: {avg_score:.3f} | Best: {best_score:.3f} | Count: {len(scores)}',
                    transform=ax.transAxes, verticalalignment='top',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            ax.legend(loc='lower right', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No patterns yet...',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='gray')

    def update_heatmap(self):
        """Обновляет тепловую карту паттернов."""
        ax = self.axes[1, 0]
        ax.clear()
        ax.set_title('Generated Patterns (Heatmap)')
        ax.set_xlabel('Pattern Dimension')
        ax.set_ylabel('Pattern #')

        try:
            patterns = self.gan.generate_batch(10)
            im = ax.imshow(patterns, aspect='auto', cmap='viridis')
            plt.colorbar(im, ax=ax)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}',
                    ha='center', va='center', transform=ax.transAxes)

    def update_stats(self):
        """Обновляет панель статистики."""
        ax = self.axes[1, 1]
        ax.clear()
        ax.axis('off')

        stats = self.gan.get_training_stats() if hasattr(self.gan, 'get_training_stats') else {}
        scores = self.gan.pattern_repository.scores if hasattr(self.gan, 'pattern_repository') else []

        # Вычисляем скользящее среднее потерь
        g_losses = self.gan.generator_losses
        d_losses = self.gan.discriminator_losses

        g_avg = np.mean(g_losses[-50:]) if len(g_losses) >= 50 else np.mean(g_losses) if g_losses else 0
        d_avg = np.mean(d_losses[-50:]) if len(d_losses) >= 50 else np.mean(d_losses) if d_losses else 0

        stats_text = f"""
╔══════════════════════════════════════╗
║        GAN TRAINING STATISTICS       ║
╠══════════════════════════════════════╣
║  Buffer Size:        {stats.get('buffer_size', 0):<6}       ║
║  Total Generations:  {stats.get('total_generations', 0):<6}       ║
║  Pattern Dim:        {stats.get('pattern_dim', 0):<6}       ║
║  Latent Dim:         {stats.get('latent_dim', 0):<6}       ║
║                                    ║
║  Generator Loss:     {g_avg:.4f}  ║
║  Discriminator Loss: {d_avg:.4f}  ║
║                                    ║
║  Pattern Scores:                   ║
║    Best:   {max(scores) if scores else 0:.3f}           ║
║    Avg:    {np.mean(scores) if scores else 0:.3f}           ║
║    Count:  {len(scores):<6}           ║
║                                    ║
║  Steps: {len(g_losses):<6}              ║
╚══════════════════════════════════════╝
        """

        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')

    def update(self, frame):
        """Обновляет все графики."""
        self.frame_count += 1

        # Обновляем каждый график
        self.update_loss_plot()
        self.update_scores_plot()
        self.update_heatmap()
        self.update_stats()

        # Обновляем заголовок
        self.fig.suptitle(f'GAN Training Visualization - Step {self.frame_count}', fontsize=14)

        plt.tight_layout()
        return self.axes


def train_gan_with_visualization():
    """
    Обучает GAN с визуализацией процесса в реальном времени.
    """
    print("Инициализация GAN...")
    gan = GAN(
        latent_dim=64,
        pattern_dim=None,
        batch_size=16,
        state_dim=8,
        action_dim=4
    )
    print(f"GAN создан: pattern_dim={gan.pattern_dim}")

    # Генерируем синтетические данные с правильной размерностью
    print("Генерация обучающих данных...")
    experiences = []

    # Разные типы опыта для обучения
    for _ in range(200):
        state = np.array([0.8, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        action = 0
        reward = 10.0
        next_state = np.array([0.9, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        experiences.append((state, action, reward, next_state))

    for _ in range(200):
        state = np.array([0.1, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
        action = 2
        reward = 8.0
        next_state = np.array([0.1, 0.9, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
        experiences.append((state, action, reward, next_state))

    for _ in range(200):
        state = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        action = 3
        reward = 2.0
        next_state = np.array([0.6, 0.6, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        experiences.append((state, action, reward, next_state))

    gan.add_experiences(experiences)
    print(f"Добавлено {len(experiences)} переходов в буфер")

    # Создаём визуализатор
    visualizer = GANTrainingVisualizer(gan)
    visualizer.setup_plot()

    print("Начинаем обучение с визуализацией...")
    print("Нажмите Ctrl+C для остановки...")

    # Счетчик для вывода прогресса
    step_counter = 0

    # Функция для обновления графиков
    def update_plot(frame):
        nonlocal step_counter

        # Проверяем, что есть данные
        if len(gan.experience_buffer) < gan.batch_size:
            return visualizer.axes

        # Выполняем несколько шагов обучения за один кадр
        steps_per_frame = 5
        for _ in range(steps_per_frame):
            try:
                patterns_tensor = gan.prepare_training_data()
                if patterns_tensor is not None and patterns_tensor.size(0) >= gan.batch_size:
                    g_loss, d_loss = gan.train_step(patterns_tensor)
                    step_counter += 1

                    # Выводим прогресс каждые 5 шагов
                    if step_counter % 5 == 0:
                        print(f"  Шаг {step_counter}: G_loss={g_loss:.4f}, D_loss={d_loss:.4f}")
                        print(f"    [DEBUG] Всего сохранено G_loss: {len(gan.generator_losses)}")
            except Exception as e:
                print(f"Ошибка в шаге {frame}: {e}")
                return visualizer.axes

        # Генерируем и оцениваем паттерны каждые 3 кадра
        if frame % 3 == 0:
            try:
                test_patterns = gan.generate_batch(10)
                for pattern in test_patterns:
                    score = gan.discriminator.evaluate_pattern(pattern)
                    gan.pattern_repository.add_pattern(pattern, score)
            except Exception as e:
                print(f"Ошибка генерации паттернов: {e}")

        # Обновляем визуализацию
        return visualizer.update(frame)

    # Создаём анимацию
    try:
        anim = animation.FuncAnimation(
            visualizer.fig,
            update_plot,
            frames=None,
            interval=100,  # Обновление каждые 100 мс
            blit=False,
            cache_frame_data=False
        )

        # Показываем анимацию
        plt.show()

        # Сохраняем анимацию в GIF
        try:
            anim.save(visualizer.save_path, writer='pillow', fps=10)
            print(f"Анимация сохранена в {visualizer.save_path}")
        except Exception as e:
            print(f"Не удалось сохранить анимацию: {e}")

    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем")
    except Exception as e:
        print(f"Ошибка при визуализации: {e}")

    return gan


if __name__ == "__main__":
    gan = train_gan_with_visualization()
    print("\nОбучение завершено!")

    # Выводим финальную статистику
    print("\nФинальная статистика:")
    print(f"  Всего шагов: {len(gan.generator_losses)}")
    if gan.generator_losses:
        print(f"  Средняя потеря G: {np.mean(gan.generator_losses):.4f}")
        print(f"  Средняя потеря D: {np.mean(gan.discriminator_losses):.4f}")
        print(f"  Min G: {min(gan.generator_losses):.4f}")
        print(f"  Max G: {max(gan.generator_losses):.4f}")
    print(f"  Паттернов в репозитории: {len(gan.pattern_repository.patterns)}")



# # scripts/visualize_gan_training.py (упрощенная и исправленная)
# import sys
# import os
#
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import numpy as np
# from models.gan import GAN
# import torch
#
#
# class GANTrainingVisualizer:
#     """
#     Визуализатор процесса обучения GAN в реальном времени.
#     """
#
#     def __init__(self, gan: GAN, save_path: str = 'gan_training.gif'):
#         self.gan = gan
#         self.save_path = save_path
#         self.fig, self.axes = None, None
#         self.frame_count = 0
#
#         # Храним данные для графиков
#         self.g_losses = []
#         self.d_losses = []
#         self.scores = []
#
#         # Линии графиков
#         self.g_line = None
#         self.d_line = None
#         self.score_bars = None
#
#     def setup_plot(self):
#         """Настраивает окна для визуализации."""
#         self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
#         self.fig.suptitle('GAN Training Visualization', fontsize=16)
#
#         # 1. Потери
#         ax_loss = self.axes[0, 0]
#         ax_loss.set_title('Generator & Discriminator Losses')
#         ax_loss.set_xlabel('Iteration')
#         ax_loss.set_ylabel('Loss')
#         ax_loss.grid(True)
#         ax_loss.set_xlim(0, 100)
#         ax_loss.set_ylim(0, 2.5)
#
#         # Создаем пустые линии
#         self.g_line, = ax_loss.plot([], [], label='Generator', color='blue', alpha=0.7, linewidth=1.5)
#         self.d_line, = ax_loss.plot([], [], label='Discriminator', color='red', alpha=0.7, linewidth=1.5)
#         ax_loss.legend(loc='upper right')
#
#         # 2. Оценки паттернов
#         ax_scores = self.axes[0, 1]
#         ax_scores.set_title('Pattern Quality Scores')
#         ax_scores.set_xlabel('Pattern #')
#         ax_scores.set_ylabel('Score')
#         ax_scores.set_ylim(0, 1)
#         ax_scores.grid(True)
#         self.score_bars = ax_scores.bar([], [], color='green', alpha=0.7)
#
#         # 3. Тепловая карта паттернов
#         self.axes[1, 0].set_title('Generated Patterns (Heatmap)')
#         self.axes[1, 0].set_xlabel('Pattern Dimension')
#         self.axes[1, 0].set_ylabel('Pattern #')
#
#         # 4. Статистика
#         self.axes[1, 1].set_title('GAN Statistics')
#         self.axes[1, 1].axis('off')
#
#         plt.tight_layout()
#
#     def update_loss_plot(self):
#         """Обновляет график потерь."""
#         ax = self.axes[0, 0]
#
#         # Получаем потери из GAN
#         g_losses = self.gan.generator_losses
#         d_losses = self.gan.discriminator_losses
#
#         if g_losses:
#             # Обновляем данные линий
#             x_data = list(range(len(g_losses)))
#
#             # Показываем все данные, но ограничиваем количество для производительности
#             if len(g_losses) > 200:
#                 # Берем последние 200 точек
#                 x_data = x_data[-200:]
#                 g_show = g_losses[-200:]
#                 d_show = d_losses[-200:] if d_losses else []
#             else:
#                 g_show = g_losses
#                 d_show = d_losses if d_losses else []
#
#             # Обновляем линии
#             self.g_line.set_data(x_data, g_show)
#             if d_show:
#                 self.d_line.set_data(x_data, d_show)
#             else:
#                 self.d_line.set_data([], [])
#
#             # Настраиваем масштаб
#             if g_show:
#                 ax.set_xlim(0, max(len(g_show), 10))
#                 all_values = g_show + (d_show if d_show else [])
#                 if all_values:
#                     max_val = max(all_values)
#                     min_val = min(all_values)
#                     y_max = max(max_val * 1.2, 0.5)
#                     y_min = max(min_val * 0.8, 0)
#                     ax.set_ylim(y_min, min(y_max, 5.0))
#
#             # Добавляем текущие значения
#             if g_show:
#                 last_g = g_show[-1]
#                 last_d = d_show[-1] if d_show else 0
#                 # Удаляем старые тексты
#                 for text in ax.texts:
#                     text.remove()
#                 ax.text(0.02, 0.98, f'G: {last_g:.4f} | D: {last_d:.4f}',
#                         transform=ax.transAxes, verticalalignment='top',
#                         fontsize=10, fontweight='bold',
#                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
#
#     def update_scores_plot(self):
#         """Обновляет график оценок паттернов."""
#         ax = self.axes[0, 1]
#
#         # Получаем оценки из репозитория
#         scores = self.gan.pattern_repository.scores if hasattr(self.gan, 'pattern_repository') else []
#
#         # Обновляем данные
#         if scores:
#             # Показываем последние 50 оценок
#             scores_show = scores[-50:]
#             x_data = list(range(len(scores_show)))
#
#             # Обновляем бары
#             ax.clear()
#             ax.set_title('Pattern Quality Scores')
#             ax.set_xlabel('Pattern #')
#             ax.set_ylabel('Score')
#             ax.set_ylim(0, 1)
#             ax.grid(True)
#
#             ax.bar(x_data, scores_show, color='green', alpha=0.7, width=0.8)
#             ax.axhline(y=0.7, color='red', linestyle='--', label='Quality threshold (0.7)', linewidth=1.5)
#             ax.axhline(y=0.5, color='orange', linestyle='--', label='Min threshold (0.5)', alpha=0.5, linewidth=1)
#
#             # Статистика
#             avg_score = np.mean(scores_show)
#             best_score = max(scores_show)
#             ax.text(0.02, 0.98, f'Avg: {avg_score:.3f} | Best: {best_score:.3f}',
#                     transform=ax.transAxes, verticalalignment='top',
#                     fontsize=9, fontweight='bold',
#                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
#             ax.legend(loc='lower right', fontsize=8)
#
#     def update_heatmap(self):
#         """Обновляет тепловую карту паттернов."""
#         ax = self.axes[1, 0]
#         ax.clear()
#         ax.set_title('Generated Patterns (Heatmap)')
#         ax.set_xlabel('Pattern Dimension')
#         ax.set_ylabel('Pattern #')
#
#         try:
#             patterns = self.gan.generate_batch(10)
#             im = ax.imshow(patterns, aspect='auto', cmap='viridis')
#             plt.colorbar(im, ax=ax)
#         except Exception as e:
#             ax.text(0.5, 0.5, f'Error: {str(e)[:50]}',
#                     ha='center', va='center', transform=ax.transAxes)
#
#     def update_stats(self):
#         """Обновляет панель статистики."""
#         ax = self.axes[1, 1]
#         ax.clear()
#         ax.axis('off')
#
#         stats = self.gan.get_training_stats() if hasattr(self.gan, 'get_training_stats') else {}
#         scores = self.gan.pattern_repository.scores if hasattr(self.gan, 'pattern_repository') else []
#
#         # Вычисляем скользящее среднее потерь
#         g_losses = self.gan.generator_losses
#         d_losses = self.gan.discriminator_losses
#
#         g_avg = np.mean(g_losses[-50:]) if len(g_losses) >= 50 else np.mean(g_losses) if g_losses else 0
#         d_avg = np.mean(d_losses[-50:]) if len(d_losses) >= 50 else np.mean(d_losses) if d_losses else 0
#
#         stats_text = f"""
# ╔══════════════════════════════════════╗
# ║        GAN TRAINING STATISTICS       ║
# ╠══════════════════════════════════════╣
# ║  Buffer Size:        {stats.get('buffer_size', 0):<6}       ║
# ║  Total Generations:  {stats.get('total_generations', 0):<6}       ║
# ║  Pattern Dim:        {stats.get('pattern_dim', 0):<6}       ║
# ║  Latent Dim:         {stats.get('latent_dim', 0):<6}       ║
# ║                                    ║
# ║  Generator Loss:     {g_avg:.4f}  ║
# ║  Discriminator Loss: {d_avg:.4f}  ║
# ║                                    ║
# ║  Pattern Scores:                   ║
# ║    Best:   {max(scores) if scores else 0:.3f}           ║
# ║    Avg:    {np.mean(scores) if scores else 0:.3f}           ║
# ║    Count:  {len(scores):<6}           ║
# ║                                    ║
# ║  Steps: {stats.get('steps', 0):<6}              ║
# ╚══════════════════════════════════════╝
#         """
#
#         ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
#                 fontsize=9, verticalalignment='top', fontfamily='monospace')
#
#     def update(self, frame):
#         """Обновляет все графики."""
#         self.frame_count += 1
#
#         # Обновляем каждый график
#         self.update_loss_plot()
#         self.update_scores_plot()
#         self.update_heatmap()
#         self.update_stats()
#
#         # Обновляем заголовок
#         self.fig.suptitle(f'GAN Training Visualization - Step {self.frame_count}', fontsize=14)
#
#         plt.tight_layout()
#         return self.axes
#
#
# def train_gan_with_visualization():
#     """
#     Обучает GAN с визуализацией процесса в реальном времени.
#     """
#     print("Инициализация GAN...")
#     gan = GAN(
#         latent_dim=64,
#         pattern_dim=None,
#         batch_size=16,
#         state_dim=8,
#         action_dim=4
#     )
#     print(f"GAN создан: pattern_dim={gan.pattern_dim}")
#
#     # Генерируем синтетические данные с правильной размерностью
#     print("Генерация обучающих данных...")
#     experiences = []
#
#     # Разные типы опыта для обучения
#     for _ in range(200):
#         state = np.array([0.8, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
#         action = 0
#         reward = 10.0
#         next_state = np.array([0.9, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
#         experiences.append((state, action, reward, next_state))
#
#     for _ in range(200):
#         state = np.array([0.1, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
#         action = 2
#         reward = 8.0
#         next_state = np.array([0.1, 0.9, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
#         experiences.append((state, action, reward, next_state))
#
#     for _ in range(200):
#         state = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
#         action = 3
#         reward = 2.0
#         next_state = np.array([0.6, 0.6, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
#         experiences.append((state, action, reward, next_state))
#
#     gan.add_experiences(experiences)
#     print(f"Добавлено {len(experiences)} переходов в буфер")
#
#     # Создаём визуализатор
#     visualizer = GANTrainingVisualizer(gan)
#     visualizer.setup_plot()
#
#     print("Начинаем обучение с визуализацией...")
#     print("Нажмите Ctrl+C для остановки...")
#
#     # Функция для обновления графиков
#     def update_plot(frame):
#         # Проверяем, что есть данные
#         if len(gan.experience_buffer) < gan.batch_size:
#             return visualizer.axes
#
#         # Выполняем несколько шагов обучения за один кадр
#         steps_per_frame = 3
#         for _ in range(steps_per_frame):
#             try:
#                 patterns_tensor = gan.prepare_training_data()
#                 if patterns_tensor is not None and patterns_tensor.size(0) >= gan.batch_size:
#                     g_loss, d_loss = gan.train_step(patterns_tensor)
#
#                     # Выводим прогресс каждые 10 шагов
#                     if len(gan.generator_losses) % 10 == 0:
#                         print(f"  Шаг {len(gan.generator_losses)}: G_loss={g_loss:.4f}, D_loss={d_loss:.4f}")
#             except Exception as e:
#                 print(f"Ошибка в шаге {frame}: {e}")
#                 return visualizer.axes
#
#         # Генерируем и оцениваем паттерны каждые 3 кадра
#         if frame % 3 == 0:
#             try:
#                 test_patterns = gan.generate_batch(10)
#                 for pattern in test_patterns:
#                     score = gan.discriminator.evaluate_pattern(pattern)
#                     gan.pattern_repository.add_pattern(pattern, score)
#             except Exception as e:
#                 print(f"Ошибка генерации паттернов: {e}")
#
#         # Обновляем визуализацию
#         return visualizer.update(frame)
#
#     # Создаём анимацию
#     try:
#         anim = animation.FuncAnimation(
#             visualizer.fig,
#             update_plot,
#             frames=None,
#             interval=150,  # Обновление каждые 150 мс
#             blit=False,
#             cache_frame_data=False
#         )
#
#         # Показываем анимацию
#         plt.show()
#
#         # Сохраняем анимацию в GIF
#         try:
#             anim.save(visualizer.save_path, writer='pillow', fps=5)
#             print(f"Анимация сохранена в {visualizer.save_path}")
#         except Exception as e:
#             print(f"Не удалось сохранить анимацию: {e}")
#
#     except KeyboardInterrupt:
#         print("\nОбучение прервано пользователем")
#     except Exception as e:
#         print(f"Ошибка при визуализации: {e}")
#
#     return gan
#
#
# if __name__ == "__main__":
#     gan = train_gan_with_visualization()
#     print("\nОбучение завершено!")
#
#     # Выводим финальную статистику
#     print("\nФинальная статистика:")
#     print(f"  Всего шагов: {len(gan.generator_losses)}")
#     if gan.generator_losses:
#         print(f"  Средняя потеря G: {np.mean(gan.generator_losses):.4f}")
#         print(f"  Средняя потеря D: {np.mean(gan.discriminator_losses):.4f}")
#     print(f"  Паттернов в репозитории: {len(gan.pattern_repository.patterns)}")
#
#
#
# # # scripts/visualize_gan_training.py (полностью исправленный)
# # import sys
# # import os
# #
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# #
# # import matplotlib.pyplot as plt
# # import matplotlib.animation as animation
# # import numpy as np
# # from models.gan import GAN
# # import torch
# # from collections import deque
# # import time
# #
# #
# # class GANTrainingVisualizer:
# #     """
# #     Визуализатор процесса обучения GAN в реальном времени.
# #     """
# #
# #     def __init__(self, gan: GAN, save_path: str = 'gan_training.gif'):
# #         self.gan = gan
# #         self.save_path = save_path
# #         self.fig, self.axes = None, None
# #         self.loss_history = {'g_loss': [], 'd_loss': []}
# #         self.score_history = []
# #         self.pattern_history = []
# #         self.frame_count = 0
# #
# #         # Для хранения линий графиков
# #         self.g_line = None
# #         self.d_line = None
# #
# #     def setup_plot(self):
# #         """Настраивает окна для визуализации."""
# #         self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
# #         self.fig.suptitle('GAN Training Visualization', fontsize=16)
# #
# #         # Потери
# #         ax_loss = self.axes[0, 0]
# #         ax_loss.set_title('Generator & Discriminator Losses')
# #         ax_loss.set_xlabel('Iteration')
# #         ax_loss.set_ylabel('Loss')
# #         ax_loss.grid(True)
# #         ax_loss.set_xlim(0, 200)
# #         ax_loss.set_ylim(0, 2.5)
# #
# #         # Оценки паттернов
# #         ax_scores = self.axes[0, 1]
# #         ax_scores.set_title('Pattern Quality Scores')
# #         ax_scores.set_xlabel('Pattern #')
# #         ax_scores.set_ylabel('Score')
# #         ax_scores.set_ylim(0, 1)
# #         ax_scores.grid(True)
# #
# #         # Тепловая карта паттернов
# #         self.axes[1, 0].set_title('Generated Patterns (Heatmap)')
# #
# #         # Статистика
# #         self.axes[1, 1].set_title('GAN Statistics')
# #         self.axes[1, 1].axis('off')
# #
# #         plt.tight_layout()
# #
# #     def update_loss_plot(self):
# #         """Обновляет график потерь."""
# #         ax = self.axes[0, 0]
# #         ax.clear()
# #         ax.set_title('Generator & Discriminator Losses')
# #         ax.set_xlabel('Iteration')
# #         ax.set_ylabel('Loss')
# #         ax.grid(True)
# #
# #         # Получаем потери из GAN
# #         g_losses = self.gan.generator_losses
# #         d_losses = self.gan.discriminator_losses
# #
# #         if g_losses:
# #             # Показываем последние 200 значений
# #             g_show = g_losses[-200:]
# #             d_show = d_losses[-200:] if d_losses else []
# #
# #             # Рисуем линии
# #             ax.plot(g_show, label='Generator', color='blue', alpha=0.7, linewidth=1.5)
# #             if d_show:
# #                 ax.plot(d_show, label='Discriminator', color='red', alpha=0.7, linewidth=1.5)
# #
# #             # Настраиваем масштаб
# #             if g_show:
# #                 ax.set_xlim(0, max(len(g_show), 10))
# #                 all_values = g_show + (d_show if d_show else [])
# #                 if all_values:
# #                     max_val = max(all_values)
# #                     min_val = min(all_values)
# #                     # Добавляем отступы
# #                     y_max = max(max_val * 1.2, 0.5)
# #                     y_min = max(min_val * 0.8, 0)
# #                     ax.set_ylim(y_min, min(y_max, 5.0))
# #
# #             ax.legend(loc='upper right')
# #
# #             # Добавляем текст с текущими значениями
# #             if g_show:
# #                 last_g = g_show[-1]
# #                 last_d = d_show[-1] if d_show else 0
# #                 ax.text(0.02, 0.98, f'G: {last_g:.4f} | D: {last_d:.4f}',
# #                         transform=ax.transAxes, verticalalignment='top',
# #                         fontsize=10, fontweight='bold',
# #                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
# #
# #     def update_scores_plot(self):
# #         """Обновляет график оценок паттернов."""
# #         ax = self.axes[0, 1]
# #         ax.clear()
# #         ax.set_title('Pattern Quality Scores')
# #         ax.set_xlabel('Pattern #')
# #         ax.set_ylabel('Score')
# #         ax.set_ylim(0, 1)
# #         ax.grid(True)
# #
# #         scores = self.gan.pattern_repository.scores if hasattr(self.gan, 'pattern_repository') else []
# #
# #         if scores:
# #             # Показываем последние 50 оценок
# #             scores_show = scores[-50:]
# #             ax.bar(range(len(scores_show)), scores_show, color='green', alpha=0.7, width=0.8)
# #             ax.axhline(y=0.7, color='red', linestyle='--', label='Quality threshold (0.7)', linewidth=1.5)
# #             ax.axhline(y=0.5, color='orange', linestyle='--', label='Min threshold (0.5)', alpha=0.5, linewidth=1)
# #
# #             # Статистика
# #             avg_score = np.mean(scores_show)
# #             best_score = max(scores_show)
# #             ax.text(0.02, 0.98, f'Avg: {avg_score:.3f} | Best: {best_score:.3f}',
# #                     transform=ax.transAxes, verticalalignment='top',
# #                     fontsize=9, fontweight='bold',
# #                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
# #             ax.legend(loc='lower right', fontsize=8)
# #
# #     def update_heatmap(self):
# #         """Обновляет тепловую карту паттернов."""
# #         ax = self.axes[1, 0]
# #         ax.clear()
# #         ax.set_title('Generated Patterns (Heatmap)')
# #         ax.set_xlabel('Pattern Dimension')
# #         ax.set_ylabel('Pattern #')
# #
# #         try:
# #             patterns = self.gan.generate_batch(10)
# #             im = ax.imshow(patterns, aspect='auto', cmap='viridis')
# #             plt.colorbar(im, ax=ax)
# #         except Exception as e:
# #             ax.text(0.5, 0.5, f'Error: {str(e)[:50]}',
# #                     ha='center', va='center', transform=ax.transAxes)
# #
# #     def update_stats(self):
# #         """Обновляет панель статистики."""
# #         ax = self.axes[1, 1]
# #         ax.clear()
# #         ax.axis('off')
# #
# #         stats = self.gan.get_training_stats() if hasattr(self.gan, 'get_training_stats') else {}
# #         scores = self.gan.pattern_repository.scores if hasattr(self.gan, 'pattern_repository') else []
# #
# #         # Вычисляем скользящее среднее потерь
# #         g_losses = self.gan.generator_losses
# #         d_losses = self.gan.discriminator_losses
# #
# #         g_avg = np.mean(g_losses[-50:]) if len(g_losses) >= 50 else np.mean(g_losses) if g_losses else 0
# #         d_avg = np.mean(d_losses[-50:]) if len(d_losses) >= 50 else np.mean(d_losses) if d_losses else 0
# #
# #         stats_text = f"""
# # ╔══════════════════════════════════════╗
# # ║        GAN TRAINING STATISTICS       ║
# # ╠══════════════════════════════════════╣
# # ║  Buffer Size:        {stats.get('buffer_size', 0):<6}       ║
# # ║  Total Generations:  {stats.get('total_generations', 0):<6}       ║
# # ║  Pattern Dim:        {stats.get('pattern_dim', 0):<6}       ║
# # ║  Latent Dim:         {stats.get('latent_dim', 0):<6}       ║
# # ║                                    ║
# # ║  Generator Loss:     {g_avg:.4f}  ║
# # ║  Discriminator Loss: {d_avg:.4f}  ║
# # ║                                    ║
# # ║  Pattern Scores:                   ║
# # ║    Best:   {max(scores) if scores else 0:.3f}           ║
# # ║    Avg:    {np.mean(scores) if scores else 0:.3f}           ║
# # ║    Count:  {len(scores):<6}           ║
# # ║                                    ║
# # ║  Steps: {stats.get('steps', 0):<6}              ║
# # ╚══════════════════════════════════════╝
# #         """
# #
# #         ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
# #                 fontsize=9, verticalalignment='top', fontfamily='monospace')
# #
# #     def update(self, frame):
# #         """Обновляет все графики."""
# #         self.frame_count += 1
# #
# #         # Обновляем каждый график
# #         self.update_loss_plot()
# #         self.update_scores_plot()
# #         self.update_heatmap()
# #         self.update_stats()
# #
# #         # Обновляем заголовок
# #         self.fig.suptitle(f'GAN Training Visualization - Step {self.frame_count}', fontsize=14)
# #
# #         plt.tight_layout()
# #         return self.axes
# #
# #
# # def train_gan_with_visualization():
# #     """
# #     Обучает GAN с визуализацией процесса в реальном времени.
# #     """
# #     print("Инициализация GAN...")
# #     gan = GAN(
# #         latent_dim=64,
# #         pattern_dim=None,
# #         batch_size=16,
# #         state_dim=8,
# #         action_dim=4
# #     )
# #     print(f"GAN создан: pattern_dim={gan.pattern_dim}")
# #
# #     # Генерируем синтетические данные с правильной размерностью
# #     print("Генерация обучающих данных...")
# #     experiences = []
# #
# #     # Разные типы опыта для обучения
# #     for _ in range(200):
# #         state = np.array([0.8, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
# #         action = 0
# #         reward = 10.0
# #         next_state = np.array([0.9, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
# #         experiences.append((state, action, reward, next_state))
# #
# #     for _ in range(200):
# #         state = np.array([0.1, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
# #         action = 2
# #         reward = 8.0
# #         next_state = np.array([0.1, 0.9, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
# #         experiences.append((state, action, reward, next_state))
# #
# #     for _ in range(200):
# #         state = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
# #         action = 3
# #         reward = 2.0
# #         next_state = np.array([0.6, 0.6, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
# #         experiences.append((state, action, reward, next_state))
# #
# #     gan.add_experiences(experiences)
# #     print(f"Добавлено {len(experiences)} переходов в буфер")
# #
# #     # Создаём визуализатор
# #     visualizer = GANTrainingVisualizer(gan)
# #     visualizer.setup_plot()
# #
# #     print("Начинаем обучение с визуализацией...")
# #     print("Нажмите Ctrl+C для остановки...")
# #
# #     # Функция для обновления графиков
# #     def update_plot(frame):
# #         # Проверяем, что есть данные
# #         if len(gan.experience_buffer) < gan.batch_size:
# #             return visualizer.axes
# #
# #         # Выполняем несколько шагов обучения за один кадр для ускорения
# #         steps_per_frame = 2
# #         for _ in range(steps_per_frame):
# #             try:
# #                 patterns_tensor = gan.prepare_training_data()
# #                 if patterns_tensor is not None and patterns_tensor.size(0) >= gan.batch_size:
# #                     g_loss, d_loss = gan.train_step(patterns_tensor)
# #
# #                     # Сохраняем потери в локальный список для отображения
# #                     visualizer.loss_history['g_loss'].append(g_loss)
# #                     visualizer.loss_history['d_loss'].append(d_loss)
# #             except Exception as e:
# #                 print(f"Ошибка в шаге {frame}: {e}")
# #                 return visualizer.axes
# #
# #         # Генерируем и оцениваем паттерны каждые 3 кадра
# #         if frame % 3 == 0:
# #             try:
# #                 test_patterns = gan.generate_batch(10)
# #                 for pattern in test_patterns:
# #                     score = gan.discriminator.evaluate_pattern(pattern)
# #                     visualizer.score_history.append(score)
# #                     gan.pattern_repository.add_pattern(pattern, score)
# #
# #                 if len(visualizer.pattern_history) < 20:
# #                     visualizer.pattern_history.append(test_patterns[0])
# #             except Exception as e:
# #                 print(f"Ошибка генерации паттернов: {e}")
# #
# #         # Обновляем визуализацию
# #         return visualizer.update(frame)
# #
# #     # Создаём анимацию
# #     try:
# #         anim = animation.FuncAnimation(
# #             visualizer.fig,
# #             update_plot,
# #             frames=None,
# #             interval=200,  # Обновление каждые 200 мс
# #             blit=False,
# #             cache_frame_data=False
# #         )
# #
# #         # Показываем анимацию
# #         plt.show()
# #
# #         # Сохраняем анимацию в GIF
# #         try:
# #             anim.save(visualizer.save_path, writer='pillow', fps=5)
# #             print(f"Анимация сохранена в {visualizer.save_path}")
# #         except Exception as e:
# #             print(f"Не удалось сохранить анимацию: {e}")
# #
# #     except KeyboardInterrupt:
# #         print("\nОбучение прервано пользователем")
# #     except Exception as e:
# #         print(f"Ошибка при визуализации: {e}")
# #
# #     return gan
# #
# #
# # if __name__ == "__main__":
# #     gan = train_gan_with_visualization()
# #     print("\nОбучение завершено!")
# #
# #     # Выводим финальную статистику
# #     print("\nФинальная статистика:")
# #     stats = gan.get_training_stats()
# #     print(f"  Всего шагов: {len(gan.generator_losses)}")
# #     print(f"  Средняя потеря G: {np.mean(gan.generator_losses):.4f}")
# #     print(f"  Средняя потеря D: {np.mean(gan.discriminator_losses):.4f}")
# #     print(f"  Паттернов в репозитории: {len(gan.pattern_repository.patterns)}")
# #
# #
#
#
#
# # # scripts/visualize_gan_training.py (полностью исправленный)
# # import sys
# # import os
# #
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# #
# # import matplotlib.pyplot as plt
# # import matplotlib.animation as animation
# # import numpy as np
# # from models.gan import GAN
# # import torch
# # from collections import deque
# # import time
# #
# #
# # class GANTrainingVisualizer:
# #     """
# #     Визуализатор процесса обучения GAN в реальном времени.
# #     """
# #
# #     def __init__(self, gan: GAN, save_path: str = 'gan_training.gif'):
# #         self.gan = gan
# #         self.save_path = save_path
# #         self.fig, self.axes = None, None
# #         self.loss_history = {'g_loss': [], 'd_loss': []}
# #         self.score_history = []
# #         self.pattern_history = []
# #         self.frame_count = 0
# #
# #         # Для хранения линий графиков
# #         self.g_line = None
# #         self.d_line = None
# #
# #     def setup_plot(self):
# #         """Настраивает окна для визуализации."""
# #         self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
# #         self.fig.suptitle('GAN Training Visualization', fontsize=16)
# #
# #         # Потери
# #         ax_loss = self.axes[0, 0]
# #         ax_loss.set_title('Generator & Discriminator Losses')
# #         ax_loss.set_xlabel('Iteration')
# #         ax_loss.set_ylabel('Loss')
# #         ax_loss.grid(True)
# #         ax_loss.set_xlim(0, 200)
# #         ax_loss.set_ylim(0, 2.5)
# #
# #         # Оценки паттернов
# #         ax_scores = self.axes[0, 1]
# #         ax_scores.set_title('Pattern Quality Scores')
# #         ax_scores.set_xlabel('Pattern #')
# #         ax_scores.set_ylabel('Score')
# #         ax_scores.set_ylim(0, 1)
# #         ax_scores.grid(True)
# #
# #         # Тепловая карта паттернов
# #         self.axes[1, 0].set_title('Generated Patterns (Heatmap)')
# #
# #         # Статистика
# #         self.axes[1, 1].set_title('GAN Statistics')
# #         self.axes[1, 1].axis('off')
# #
# #         plt.tight_layout()
# #
# #     def update_loss_plot(self):
# #         """Обновляет график потерь."""
# #         ax = self.axes[0, 0]
# #         ax.clear()
# #         ax.set_title('Generator & Discriminator Losses')
# #         ax.set_xlabel('Iteration')
# #         ax.set_ylabel('Loss')
# #         ax.grid(True)
# #
# #         # Получаем потери из GAN
# #         g_losses = self.gan.generator_losses
# #         d_losses = self.gan.discriminator_losses
# #
# #         if g_losses:
# #             # Показываем последние 200 значений
# #             g_show = g_losses[-200:]
# #             d_show = d_losses[-200:] if d_losses else []
# #
# #             # Рисуем линии
# #             ax.plot(g_show, label='Generator', color='blue', alpha=0.7, linewidth=1.5)
# #             if d_show:
# #                 ax.plot(d_show, label='Discriminator', color='red', alpha=0.7, linewidth=1.5)
# #
# #             # Настраиваем масштаб
# #             if g_show:
# #                 ax.set_xlim(0, max(len(g_show), 10))
# #                 all_values = g_show + (d_show if d_show else [])
# #                 if all_values:
# #                     max_val = max(all_values)
# #                     min_val = min(all_values)
# #                     # Добавляем отступы
# #                     y_max = max(max_val * 1.2, 0.5)
# #                     y_min = max(min_val * 0.8, 0)
# #                     ax.set_ylim(y_min, min(y_max, 5.0))
# #
# #             ax.legend(loc='upper right')
# #
# #             # Добавляем текст с текущими значениями
# #             if g_show:
# #                 last_g = g_show[-1]
# #                 last_d = d_show[-1] if d_show else 0
# #                 ax.text(0.02, 0.98, f'G: {last_g:.4f} | D: {last_d:.4f}',
# #                         transform=ax.transAxes, verticalalignment='top',
# #                         fontsize=10, fontweight='bold',
# #                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
# #
# #     def update_scores_plot(self):
# #         """Обновляет график оценок паттернов."""
# #         ax = self.axes[0, 1]
# #         ax.clear()
# #         ax.set_title('Pattern Quality Scores')
# #         ax.set_xlabel('Pattern #')
# #         ax.set_ylabel('Score')
# #         ax.set_ylim(0, 1)
# #         ax.grid(True)
# #
# #         scores = self.gan.pattern_repository.scores if hasattr(self.gan, 'pattern_repository') else []
# #
# #         if scores:
# #             # Показываем последние 50 оценок
# #             scores_show = scores[-50:]
# #             ax.bar(range(len(scores_show)), scores_show, color='green', alpha=0.7, width=0.8)
# #             ax.axhline(y=0.7, color='red', linestyle='--', label='Quality threshold (0.7)', linewidth=1.5)
# #             ax.axhline(y=0.5, color='orange', linestyle='--', label='Min threshold (0.5)', alpha=0.5, linewidth=1)
# #
# #             # Статистика
# #             avg_score = np.mean(scores_show)
# #             best_score = max(scores_show)
# #             ax.text(0.02, 0.98, f'Avg: {avg_score:.3f} | Best: {best_score:.3f}',
# #                     transform=ax.transAxes, verticalalignment='top',
# #                     fontsize=9, fontweight='bold',
# #                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
# #             ax.legend(loc='lower right', fontsize=8)
# #
# #     def update_heatmap(self):
# #         """Обновляет тепловую карту паттернов."""
# #         ax = self.axes[1, 0]
# #         ax.clear()
# #         ax.set_title('Generated Patterns (Heatmap)')
# #         ax.set_xlabel('Pattern Dimension')
# #         ax.set_ylabel('Pattern #')
# #
# #         try:
# #             patterns = self.gan.generate_batch(10)
# #             im = ax.imshow(patterns, aspect='auto', cmap='viridis')
# #             plt.colorbar(im, ax=ax)
# #         except Exception as e:
# #             ax.text(0.5, 0.5, f'Error: {str(e)[:50]}',
# #                     ha='center', va='center', transform=ax.transAxes)
# #
# #     def update_stats(self):
# #         """Обновляет панель статистики."""
# #         ax = self.axes[1, 1]
# #         ax.clear()
# #         ax.axis('off')
# #
# #         stats = self.gan.get_training_stats() if hasattr(self.gan, 'get_training_stats') else {}
# #         scores = self.gan.pattern_repository.scores if hasattr(self.gan, 'pattern_repository') else []
# #
# #         # Вычисляем скользящее среднее потерь
# #         g_losses = self.gan.generator_losses
# #         d_losses = self.gan.discriminator_losses
# #
# #         g_avg = np.mean(g_losses[-50:]) if len(g_losses) >= 50 else np.mean(g_losses) if g_losses else 0
# #         d_avg = np.mean(d_losses[-50:]) if len(d_losses) >= 50 else np.mean(d_losses) if d_losses else 0
# #
# #         stats_text = f"""
# # ╔══════════════════════════════════════╗
# # ║        GAN TRAINING STATISTICS       ║
# # ╠══════════════════════════════════════╣
# # ║  Buffer Size:        {stats.get('buffer_size', 0):<6}       ║
# # ║  Total Generations:  {stats.get('total_generations', 0):<6}       ║
# # ║  Pattern Dim:        {stats.get('pattern_dim', 0):<6}       ║
# # ║  Latent Dim:         {stats.get('latent_dim', 0):<6}       ║
# # ║                                    ║
# # ║  Generator Loss:     {g_avg:.4f}  ║
# # ║  Discriminator Loss: {d_avg:.4f}  ║
# # ║                                    ║
# # ║  Pattern Scores:                   ║
# # ║    Best:   {max(scores) if scores else 0:.3f}           ║
# # ║    Avg:    {np.mean(scores) if scores else 0:.3f}           ║
# # ║    Count:  {len(scores):<6}           ║
# # ║                                    ║
# # ║  Steps: {stats.get('steps', 0):<6}              ║
# # ╚══════════════════════════════════════╝
# #         """
# #
# #         ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
# #                 fontsize=9, verticalalignment='top', fontfamily='monospace')
# #
# #     def update(self, frame):
# #         """Обновляет все графики."""
# #         self.frame_count += 1
# #
# #         # Обновляем каждый график
# #         self.update_loss_plot()
# #         self.update_scores_plot()
# #         self.update_heatmap()
# #         self.update_stats()
# #
# #         # Обновляем заголовок
# #         self.fig.suptitle(f'GAN Training Visualization - Step {self.frame_count}', fontsize=14)
# #
# #         plt.tight_layout()
# #         return self.axes
# #
# #
# # def train_gan_with_visualization():
# #     """
# #     Обучает GAN с визуализацией процесса в реальном времени.
# #     """
# #     print("Инициализация GAN...")
# #     gan = GAN(
# #         latent_dim=64,
# #         pattern_dim=None,
# #         batch_size=16,
# #         state_dim=8,
# #         action_dim=4
# #     )
# #     print(f"GAN создан: pattern_dim={gan.pattern_dim}")
# #
# #     # Генерируем синтетические данные с правильной размерностью
# #     print("Генерация обучающих данных...")
# #     experiences = []
# #
# #     # Разные типы опыта для обучения
# #     for _ in range(200):
# #         state = np.array([0.8, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
# #         action = 0
# #         reward = 10.0
# #         next_state = np.array([0.9, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
# #         experiences.append((state, action, reward, next_state))
# #
# #     for _ in range(200):
# #         state = np.array([0.1, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
# #         action = 2
# #         reward = 8.0
# #         next_state = np.array([0.1, 0.9, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
# #         experiences.append((state, action, reward, next_state))
# #
# #     for _ in range(200):
# #         state = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
# #         action = 3
# #         reward = 2.0
# #         next_state = np.array([0.6, 0.6, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
# #         experiences.append((state, action, reward, next_state))
# #
# #     gan.add_experiences(experiences)
# #     print(f"Добавлено {len(experiences)} переходов в буфер")
# #
# #     # Создаём визуализатор
# #     visualizer = GANTrainingVisualizer(gan)
# #     visualizer.setup_plot()
# #
# #     print("Начинаем обучение с визуализацией...")
# #     print("Нажмите Ctrl+C для остановки...")
# #
# #     # Функция для обновления графиков
# #     def update_plot(frame):
# #         # Проверяем, что есть данные
# #         if len(gan.experience_buffer) < gan.batch_size:
# #             return visualizer.axes
# #
# #         # Выполняем несколько шагов обучения за один кадр для ускорения
# #         steps_per_frame = 2
# #         for _ in range(steps_per_frame):
# #             try:
# #                 patterns_tensor = gan.prepare_training_data()
# #                 if patterns_tensor is not None and patterns_tensor.size(0) >= gan.batch_size:
# #                     g_loss, d_loss = gan.train_step(patterns_tensor)
# #
# #                     # Сохраняем потери в локальный список для отображения
# #                     visualizer.loss_history['g_loss'].append(g_loss)
# #                     visualizer.loss_history['d_loss'].append(d_loss)
# #             except Exception as e:
# #                 print(f"Ошибка в шаге {frame}: {e}")
# #                 return visualizer.axes
# #
# #         # Генерируем и оцениваем паттерны каждые 3 кадра
# #         if frame % 3 == 0:
# #             try:
# #                 test_patterns = gan.generate_batch(10)
# #                 for pattern in test_patterns:
# #                     score = gan.discriminator.evaluate_pattern(pattern)
# #                     visualizer.score_history.append(score)
# #                     gan.pattern_repository.add_pattern(pattern, score)
# #
# #                 if len(visualizer.pattern_history) < 20:
# #                     visualizer.pattern_history.append(test_patterns[0])
# #             except Exception as e:
# #                 print(f"Ошибка генерации паттернов: {e}")
# #
# #         # Обновляем визуализацию
# #         return visualizer.update(frame)
# #
# #     # Создаём анимацию
# #     try:
# #         anim = animation.FuncAnimation(
# #             visualizer.fig,
# #             update_plot,
# #             frames=None,
# #             interval=200,  # Обновление каждые 200 мс
# #             blit=False,
# #             cache_frame_data=False
# #         )
# #
# #         # Показываем анимацию
# #         plt.show()
# #
# #         # Сохраняем анимацию в GIF
# #         try:
# #             anim.save(visualizer.save_path, writer='pillow', fps=5)
# #             print(f"Анимация сохранена в {visualizer.save_path}")
# #         except Exception as e:
# #             print(f"Не удалось сохранить анимацию: {e}")
# #
# #     except KeyboardInterrupt:
# #         print("\nОбучение прервано пользователем")
# #     except Exception as e:
# #         print(f"Ошибка при визуализации: {e}")
# #
# #     return gan
# #
# #
# # if __name__ == "__main__":
# #     gan = train_gan_with_visualization()
# #     print("\nОбучение завершено!")
# #
# #     # Выводим финальную статистику
# #     print("\nФинальная статистика:")
# #     stats = gan.get_training_stats()
# #     print(f"  Всего шагов: {len(gan.generator_losses)}")
# #     print(f"  Средняя потеря G: {np.mean(gan.generator_losses):.4f}")
# #     print(f"  Средняя потеря D: {np.mean(gan.discriminator_losses):.4f}")
# #     print(f"  Паттернов в репозитории: {len(gan.pattern_repository.patterns)}")
#
#
#
# # # scripts/visualize_gan_training.py (исправленный)
# # import sys
# # import os
# #
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# #
# # import matplotlib.pyplot as plt
# # import matplotlib.animation as animation
# # import numpy as np
# # from models.gan import GAN
# # import torch
# # from collections import deque
# # import time
# #
# #
# # class GANTrainingVisualizer:
# #     """
# #     Визуализатор процесса обучения GAN в реальном времени.
# #     """
# #
# #     def __init__(self, gan: GAN, save_path: str = 'gan_training.gif'):
# #         self.gan = gan
# #         self.save_path = save_path
# #         self.fig, self.axes = None, None
# #         self.loss_history = {'g_loss': [], 'd_loss': []}
# #         self.score_history = []
# #         self.pattern_history = []
# #         self.frame_count = 0
# #
# #     def setup_plot(self):
# #         """Настраивает окна для визуализации."""
# #         self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
# #         self.fig.suptitle('GAN Training Visualization', fontsize=16)
# #
# #         # Потери
# #         self.axes[0, 0].set_title('Generator & Discriminator Losses')
# #         self.axes[0, 0].set_xlabel('Iteration')
# #         self.axes[0, 0].set_ylabel('Loss')
# #         self.axes[0, 0].grid(True)
# #
# #         # Оценки паттернов
# #         self.axes[0, 1].set_title('Pattern Quality Scores')
# #         self.axes[0, 1].set_xlabel('Generation')
# #         self.axes[0, 1].set_ylabel('Score')
# #         self.axes[0, 1].set_ylim(0, 1)
# #         self.axes[0, 1].grid(True)
# #
# #         # Тепловая карта паттернов
# #         self.axes[1, 0].set_title('Generated Patterns (Heatmap)')
# #
# #         # Статистика
# #         self.axes[1, 1].set_title('GAN Statistics')
# #         self.axes[1, 1].axis('off')
# #
# #         plt.tight_layout()
# #
# #     def update(self, frame):
# #         """Обновляет графики на каждом шаге."""
# #         self.frame_count += 1
# #
# #         # Обновляем потери
# #         if self.gan.generator_losses and self.gan.discriminator_losses:
# #             g_losses = self.gan.generator_losses[-200:]
# #             d_losses = self.gan.discriminator_losses[-200:]
# #
# #             self.axes[0, 0].clear()
# #             self.axes[0, 0].plot(g_losses, label='Generator', color='blue', alpha=0.7)
# #             self.axes[0, 0].plot(d_losses, label='Discriminator', color='red', alpha=0.7)
# #             self.axes[0, 0].set_title('Generator & Discriminator Losses')
# #             self.axes[0, 0].set_xlabel('Iteration')
# #             self.axes[0, 0].set_ylabel('Loss')
# #             self.axes[0, 0].legend()
# #             self.axes[0, 0].grid(True)
# #
# #         # Обновляем оценки паттернов
# #         if hasattr(self.gan, 'pattern_repository') and self.gan.pattern_repository:
# #             scores = self.gan.pattern_repository.scores[-50:]
# #             self.axes[0, 1].clear()
# #             if scores:
# #                 self.axes[0, 1].bar(range(len(scores)), scores, color='green', alpha=0.7)
# #                 self.axes[0, 1].axhline(y=0.7, color='red', linestyle='--', label='Quality threshold')
# #             self.axes[0, 1].set_title('Pattern Quality Scores')
# #             self.axes[0, 1].set_xlabel('Pattern #')
# #             self.axes[0, 1].set_ylabel('Score')
# #             self.axes[0, 1].set_ylim(0, 1)
# #             if scores:
# #                 self.axes[0, 1].legend()
# #             self.axes[0, 1].grid(True)
# #
# #         # Обновляем тепловую карту паттернов
# #         if hasattr(self.gan, 'generator'):
# #             try:
# #                 patterns = self.gan.generate_batch(10)
# #                 self.axes[1, 0].clear()
# #                 im = self.axes[1, 0].imshow(patterns, aspect='auto', cmap='viridis')
# #                 self.axes[1, 0].set_title('Generated Patterns (Heatmap)')
# #                 self.axes[1, 0].set_xlabel('Pattern Dimension')
# #                 self.axes[1, 0].set_ylabel('Pattern #')
# #                 plt.colorbar(im, ax=self.axes[1, 0])
# #             except Exception as e:
# #                 self.axes[1, 0].clear()
# #                 self.axes[1, 0].text(0.5, 0.5, f'Error: {str(e)[:50]}',
# #                                      ha='center', va='center', transform=self.axes[1, 0].transAxes)
# #
# #         # Обновляем статистику
# #         self.axes[1, 1].clear()
# #         self.axes[1, 1].axis('off')
# #         stats = self.gan.get_training_stats() if hasattr(self.gan, 'get_training_stats') else {}
# #
# #         scores = self.gan.pattern_repository.scores if hasattr(self.gan, 'pattern_repository') else []
# #
# #         stats_text = f"""
# #         GAN Training Statistics:
# #
# #         Buffer Size: {stats.get('buffer_size', 0)}
# #         Total Generations: {stats.get('total_generations', 0)}
# #         Pattern Dim: {stats.get('pattern_dim', 0)}
# #         Latent Dim: {stats.get('latent_dim', 0)}
# #
# #         Avg G Loss: {stats.get('avg_g_loss', 0):.4f}
# #         Avg D Loss: {stats.get('avg_d_loss', 0):.4f}
# #
# #         Recent Pattern Scores:
# #         Best: {max(scores) if scores else 'N/A'}
# #         Avg: {np.mean(scores) if scores else 'N/A'}
# #         Count: {len(scores)}
# #         """
# #
# #         self.axes[1, 1].text(0.1, 0.9, stats_text, transform=self.axes[1, 1].transAxes,
# #                              fontsize=10, verticalalignment='top')
# #
# #         return self.axes
# #
# #
# # def train_gan_with_visualization():
# #     """
# #     Обучает GAN с визуализацией процесса в реальном времени.
# #     """
# #     print("Инициализация GAN...")
# #     gan = GAN(
# #         latent_dim=64,
# #         pattern_dim=None,  # <-- ИСПРАВЛЕНО: автоматическое определение
# #         batch_size=16,
# #         state_dim=8,
# #         action_dim=4
# #     )
# #     print(f"GAN создан: pattern_dim={gan.pattern_dim}")
# #
# #     # Генерируем синтетические данные с правильной размерностью
# #     print("Генерация обучающих данных...")
# #     experiences = []
# #
# #     # Разные типы опыта для обучения
# #     for _ in range(200):
# #         # Паттерн 1: Поиск еды (высокий reward)
# #         state = np.array([0.8, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
# #         action = 0  # grab
# #         reward = 10.0
# #         next_state = np.array([0.9, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
# #         experiences.append((state, action, reward, next_state))
# #
# #     for _ in range(200):
# #         # Паттерн 2: Убегание от опасности (высокий reward)
# #         state = np.array([0.1, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
# #         action = 2  # run_away
# #         reward = 8.0
# #         next_state = np.array([0.1, 0.9, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
# #         experiences.append((state, action, reward, next_state))
# #
# #     for _ in range(200):
# #         # Паттерн 3: Исследование (средний reward)
# #         state = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
# #         action = 3  # move_on
# #         reward = 2.0
# #         next_state = np.array([0.6, 0.6, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
# #         experiences.append((state, action, reward, next_state))
# #
# #     gan.add_experiences(experiences)
# #     print(f"Добавлено {len(experiences)} переходов в буфер")
# #
# #     # Создаём визуализатор
# #     visualizer = GANTrainingVisualizer(gan)
# #     visualizer.setup_plot()
# #
# #     print("Начинаем обучение с визуализацией...")
# #     print("Нажмите Ctrl+C для остановки...")
# #
# #     # Функция для обновления графиков
# #     def update_plot(frame):
# #         # Проверяем, что есть данные
# #         if len(gan.experience_buffer) < gan.batch_size:
# #             return visualizer.axes
# #
# #         try:
# #             # Подготавливаем данные
# #             patterns_tensor = gan.prepare_training_data()
# #             if patterns_tensor is not None and patterns_tensor.size(0) >= gan.batch_size:
# #                 # Тренируем GAN
# #                 g_loss, d_loss = gan.train_step(patterns_tensor)
# #                 visualizer.loss_history['g_loss'].append(g_loss)
# #                 visualizer.loss_history['d_loss'].append(d_loss)
# #         except Exception as e:
# #             print(f"Ошибка в шаге {frame}: {e}")
# #             return visualizer.axes
# #
# #         # Генерируем и оцениваем паттерны каждые 5 шагов
# #         if frame % 5 == 0:
# #             try:
# #                 test_patterns = gan.generate_batch(10)
# #                 for pattern in test_patterns:
# #                     score = gan.discriminator.evaluate_pattern(pattern)
# #                     visualizer.score_history.append(score)
# #
# #                 # Сохраняем паттерны для истории
# #                 if len(visualizer.pattern_history) < 20:
# #                     visualizer.pattern_history.append(test_patterns[0])
# #             except Exception as e:
# #                 print(f"Ошибка генерации паттернов: {e}")
# #
# #         return visualizer.update(frame)
# #
# #     # Создаём анимацию
# #     try:
# #         anim = animation.FuncAnimation(
# #             visualizer.fig,
# #             update_plot,
# #             frames=100,
# #             interval=500,
# #             blit=False,
# #             cache_frame_data=False  # <-- ДОБАВЛЕНО ДЛЯ СОВМЕСТИМОСТИ
# #         )
# #
# #         # Показываем анимацию
# #         plt.show()
# #
# #         # Сохраняем анимацию в GIF
# #         try:
# #             anim.save(visualizer.save_path, writer='pillow', fps=2)
# #             print(f"Анимация сохранена в {visualizer.save_path}")
# #         except Exception as e:
# #             print(f"Не удалось сохранить анимацию: {e}")
# #
# #     except KeyboardInterrupt:
# #         print("\nОбучение прервано пользователем")
# #     except Exception as e:
# #         print(f"Ошибка при визуализации: {e}")
# #
# #     return gan
# #
# #
# # if __name__ == "__main__":
# #     gan = train_gan_with_visualization()
# #     print("\nОбучение завершено!")
# #
# #
# #
# # # # scripts/visualize_gan_training.py
# # # import sys
# # # import os
# # #
# # # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# # #
# # # import matplotlib.pyplot as plt
# # # import matplotlib.animation as animation
# # # import numpy as np
# # # from models.gan import GAN
# # # import torch
# # # from collections import deque
# # # import time
# # #
# # #
# # # class GANTrainingVisualizer:
# # #     """
# # #     Визуализатор процесса обучения GAN в реальном времени.
# # #     """
# # #
# # #     def __init__(self, gan: GAN, save_path: str = 'gan_training.gif'):
# # #         self.gan = gan
# # #         self.save_path = save_path
# # #         self.fig, self.axes = None, None
# # #         self.loss_history = {'g_loss': [], 'd_loss': []}
# # #         self.score_history = []
# # #         self.pattern_history = []
# # #
# # #     def setup_plot(self):
# # #         """Настраивает окна для визуализации."""
# # #         self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
# # #         self.fig.suptitle('GAN Training Visualization', fontsize=16)
# # #
# # #         # Потери
# # #         self.axes[0, 0].set_title('Generator & Discriminator Losses')
# # #         self.axes[0, 0].set_xlabel('Iteration')
# # #         self.axes[0, 0].set_ylabel('Loss')
# # #         self.axes[0, 0].grid(True)
# # #
# # #         # Оценки паттернов
# # #         self.axes[0, 1].set_title('Pattern Quality Scores')
# # #         self.axes[0, 1].set_xlabel('Generation')
# # #         self.axes[0, 1].set_ylabel('Score')
# # #         self.axes[0, 1].set_ylim(0, 1)
# # #         self.axes[0, 1].grid(True)
# # #
# # #         # Тепловая карта паттернов
# # #         self.axes[1, 0].set_title('Generated Patterns (Heatmap)')
# # #
# # #         # Статистика
# # #         self.axes[1, 1].set_title('GAN Statistics')
# # #         self.axes[1, 1].axis('off')
# # #
# # #         plt.tight_layout()
# # #
# # #     def update(self, frame):
# # #         """Обновляет графики на каждом шаге."""
# # #         # Обновляем потери
# # #         if self.gan.generator_losses and self.gan.discriminator_losses:
# # #             g_losses = self.gan.generator_losses[-200:]
# # #             d_losses = self.gan.discriminator_losses[-200:]
# # #
# # #             self.axes[0, 0].clear()
# # #             self.axes[0, 0].plot(g_losses, label='Generator', color='blue', alpha=0.7)
# # #             self.axes[0, 0].plot(d_losses, label='Discriminator', color='red', alpha=0.7)
# # #             self.axes[0, 0].set_title('Generator & Discriminator Losses')
# # #             self.axes[0, 0].set_xlabel('Iteration')
# # #             self.axes[0, 0].set_ylabel('Loss')
# # #             self.axes[0, 0].legend()
# # #             self.axes[0, 0].grid(True)
# # #
# # #         # Обновляем оценки паттернов
# # #         if hasattr(self.gan, 'pattern_repository') and self.gan.pattern_repository:
# # #             scores = self.gan.pattern_repository.scores[-50:]
# # #             self.axes[0, 1].clear()
# # #             self.axes[0, 1].bar(range(len(scores)), scores, color='green', alpha=0.7)
# # #             self.axes[0, 1].axhline(y=0.7, color='red', linestyle='--', label='Quality threshold')
# # #             self.axes[0, 1].set_title('Pattern Quality Scores')
# # #             self.axes[0, 1].set_xlabel('Pattern #')
# # #             self.axes[0, 1].set_ylabel('Score')
# # #             self.axes[0, 1].set_ylim(0, 1)
# # #             self.axes[0, 1].legend()
# # #             self.axes[0, 1].grid(True)
# # #
# # #         # Обновляем тепловую карту паттернов
# # #         if hasattr(self.gan, 'generator'):
# # #             patterns = self.gan.generate_batch(10)
# # #             self.axes[1, 0].clear()
# # #             im = self.axes[1, 0].imshow(patterns, aspect='auto', cmap='viridis')
# # #             self.axes[1, 0].set_title('Generated Patterns (Heatmap)')
# # #             self.axes[1, 0].set_xlabel('Pattern Dimension')
# # #             self.axes[1, 0].set_ylabel('Pattern #')
# # #             plt.colorbar(im, ax=self.axes[1, 0])
# # #
# # #         # Обновляем статистику
# # #         self.axes[1, 1].clear()
# # #         self.axes[1, 1].axis('off')
# # #         stats = self.gan.get_training_stats() if hasattr(self.gan, 'get_training_stats') else {}
# # #
# # #         stats_text = f"""
# # #         GAN Training Statistics:
# # #
# # #         Buffer Size: {stats.get('buffer_size', 0)}
# # #         Total Generations: {stats.get('total_generations', 0)}
# # #         Pattern Dim: {stats.get('pattern_dim', 0)}
# # #         Latent Dim: {stats.get('latent_dim', 0)}
# # #
# # #         Avg G Loss: {stats.get('avg_g_loss', 0):.4f}
# # #         Avg D Loss: {stats.get('avg_d_loss', 0):.4f}
# # #
# # #         Recent Pattern Scores:
# # #         Best: {max(scores) if scores else 'N/A'}
# # #         Avg: {np.mean(scores) if scores else 'N/A'}
# # #         """
# # #
# # #         self.axes[1, 1].text(0.1, 0.9, stats_text, transform=self.axes[1, 1].transAxes,
# # #                              fontsize=10, verticalalignment='top')
# # #
# # #         return self.axes
# # #
# # #
# # # def train_gan_with_visualization():
# # #     """
# # #     Обучает GAN с визуализацией процесса в реальном времени.
# # #     """
# # #     print("Инициализация GAN...")
# # #     gan = GAN(
# # #         latent_dim=64,
# # #         pattern_dim=32,
# # #         batch_size=16,
# # #         lr_generator=0.0002,
# # #         lr_discriminator=0.0002
# # #     )
# # #
# # #     # Генерируем синтетические данные с разными паттернами
# # #     print("Генерация обучающих данных...")
# # #     experiences = []
# # #
# # #     # Разные типы опыта для обучения
# # #     for _ in range(200):
# # #         # Паттерн 1: Поиск еды (высокий reward)
# # #         state = np.array([0.8, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
# # #         action = 0  # grab
# # #         reward = 10.0
# # #         next_state = np.array([0.9, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
# # #         experiences.append((state, action, reward, next_state))
# # #
# # #     for _ in range(200):
# # #         # Паттерн 2: Убегание от опасности (высокий reward)
# # #         state = np.array([0.1, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
# # #         action = 2  # run_away
# # #         reward = 8.0
# # #         next_state = np.array([0.1, 0.9, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
# # #         experiences.append((state, action, reward, next_state))
# # #
# # #     for _ in range(200):
# # #         # Паттерн 3: Исследование (средний reward)
# # #         state = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
# # #         action = 3  # move_on
# # #         reward = 2.0
# # #         next_state = np.array([0.6, 0.6, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
# # #         experiences.append((state, action, reward, next_state))
# # #
# # #     gan.add_experiences(experiences)
# # #     print(f"Добавлено {len(experiences)} переходов в буфер")
# # #
# # #     # Создаём визуализатор
# # #     visualizer = GANTrainingVisualizer(gan)
# # #     visualizer.setup_plot()
# # #
# # #     print("Начинаем обучение с визуализацией...")
# # #
# # #     # Функция для обновления графиков
# # #     def update_plot(frame):
# # #         # Тренируем GAN
# # #         patterns = gan.prepare_training_data()
# # #         if patterns is not None and len(patterns) > 0:
# # #             g_loss, d_loss = gan.train_step(patterns)
# # #             visualizer.loss_history['g_loss'].append(g_loss)
# # #             visualizer.loss_history['d_loss'].append(d_loss)
# # #
# # #         # Генерируем и оцениваем паттерны
# # #         if frame % 5 == 0:
# # #             test_patterns = gan.generate_batch(10)
# # #             for pattern in test_patterns:
# # #                 score = gan.discriminator.evaluate_pattern(pattern)
# # #                 visualizer.score_history.append(score)
# # #
# # #             # Сохраняем паттерны для истории
# # #             if len(visualizer.pattern_history) < 20:
# # #                 visualizer.pattern_history.append(test_patterns[0])
# # #
# # #         return visualizer.update(frame)
# # #
# # #     # Создаём анимацию
# # #     anim = animation.FuncAnimation(
# # #         visualizer.fig,
# # #         update_plot,
# # #         frames=100,
# # #         interval=500,
# # #         blit=False
# # #     )
# # #
# # #     # Показываем анимацию
# # #     plt.show()
# # #
# # #     # Сохраняем анимацию в GIF
# # #     try:
# # #         anim.save(visualizer.save_path, writer='pillow', fps=2)
# # #         print(f"Анимация сохранена в {visualizer.save_path}")
# # #     except Exception as e:
# # #         print(f"Не удалось сохранить анимацию: {e}")
# # #
# # #     return gan
# # #
# # #
# # # if __name__ == "__main__":
# # #     gan = train_gan_with_visualization()