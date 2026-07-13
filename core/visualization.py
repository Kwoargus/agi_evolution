# core/visualization.py - УПРОЩЁННАЯ ВЕРСИЯ

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict, Optional


class TrainingVisualizer:
    """
    Визуализация прогресса тренировки ботов с использованием GAN.
    """

    def __init__(self, save_path: str = "training_plots/"):
        self.save_path = save_path
        # Создаём папку, если её нет
        os.makedirs(save_path, exist_ok=True)
        print(f"📁 Папка для визуализаций: {save_path}")

    def plot_fitness_progress(self, generations: List[int],
                              best_fitness: List[float],
                              avg_fitness: List[float]):
        """График прогресса фитнеса."""
        if not generations:
            print("⚠️ Нет данных для графика фитнеса")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(generations, best_fitness, 'g-', linewidth=2, label='Лучший фитнес')
        ax.plot(generations, avg_fitness, 'b--', linewidth=1.5, label='Средний фитнес')
        ax.fill_between(generations, avg_fitness, best_fitness, alpha=0.2, color='green')

        ax.set_xlabel('Поколение')
        ax.set_ylabel('Фитнес')
        ax.set_title('Прогресс эволюции ботов')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.save_path}fitness_progress.png", dpi=150)
        plt.close()
        print(f"✅ График фитнеса сохранён")

    def plot_gan_progress(self, generations: List[int],
                          g_loss: List[float],
                          d_loss: List[float]):
        """График потерь GAN."""
        if not generations or not any(g_loss) or not any(d_loss):
            print("⚠️ Нет данных для графика GAN")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(generations, g_loss, 'r-', linewidth=2, label='Потери генератора')
        ax.plot(generations, d_loss, 'b-', linewidth=2, label='Потери дискриминатора')

        ax.set_xlabel('Поколение')
        ax.set_ylabel('Потери')
        ax.set_title('Обучение GAN')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.save_path}gan_progress.png", dpi=150)
        plt.close()
        print(f"✅ График GAN сохранён")

    def plot_pattern_diversity(self, generations: List[int],
                               diversity: List[float]):
        """График разнообразия паттернов."""
        if not generations or not any(diversity):
            print("⚠️ Нет данных для графика разнообразия")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.bar(generations, diversity, color='purple', alpha=0.7)
        ax.set_xlabel('Поколение')
        ax.set_ylabel('Разнообразие паттернов')
        ax.set_title('Разнообразие сгенерированных паттернов')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f"{self.save_path}pattern_diversity.png", dpi=150)
        plt.close()
        print(f"✅ График разнообразия сохранён")

    def plot_reflex_success(self, reflex_data: Dict[str, List[float]]):
        """График успешности рефлексов."""
        if not reflex_data:
            print("⚠️ Нет данных для графика рефлексов")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        for reflex_name, rates in reflex_data.items():
            if rates:
                ax.plot(rates, label=reflex_name, linewidth=2)

        ax.set_xlabel('Поколение')
        ax.set_ylabel('Успешность')
        ax.set_title('Успешность рефлексов')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(f"{self.save_path}reflex_success.png", dpi=150)
        plt.close()
        print(f"✅ График рефлексов сохранён")

    def create_dashboard(self, generation_data: Dict):
        """Создаёт полную панель мониторинга."""
        if not generation_data.get('generations'):
            print("⚠️ Нет данных для панели мониторинга")
            return

        fig = plt.figure(figsize=(20, 16))

        gens = generation_data['generations']
        best = generation_data['best_fitness']
        avg = generation_data['avg_fitness']
        g_loss = generation_data.get('g_loss', [0] * len(gens))
        d_loss = generation_data.get('d_loss', [0] * len(gens))
        diversity = generation_data.get('diversity', [0] * len(gens))

        # 1. Фитнес (верхний левый)
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(gens, best, 'g-', linewidth=2, label='Лучший')
        ax1.plot(gens, avg, 'b--', linewidth=1.5, label='Средний')
        ax1.set_xlabel('Поколение')
        ax1.set_ylabel('Фитнес')
        ax1.set_title('Прогресс эволюции')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. GAN (верхний правый)
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(gens, g_loss, 'r-', linewidth=2, label='G потери')
        ax2.plot(gens, d_loss, 'b-', linewidth=2, label='D потери')
        ax2.set_xlabel('Поколение')
        ax2.set_ylabel('Потери')
        ax2.set_title('Обучение GAN')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Разнообразие (средний левый)
        ax3 = plt.subplot(3, 2, 3)
        if any(diversity):
            ax3.bar(gens, diversity, color='purple', alpha=0.7)
        else:
            ax3.text(0.5, 0.5, 'Нет данных о разнообразии', ha='center', va='center')
        ax3.set_xlabel('Поколение')
        ax3.set_ylabel('Разнообразие')
        ax3.set_title('Разнообразие паттернов')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Успешность рефлексов (средний правый)
        ax4 = plt.subplot(3, 2, 4)
        reflex_data = generation_data.get('reflex_success', {})
        if reflex_data:
            for reflex, rates in reflex_data.items():
                if rates:
                    ax4.plot(rates, label=reflex, linewidth=2)
            ax4.set_ylim(0, 1)
        else:
            ax4.text(0.5, 0.5, 'Нет данных о рефлексах', ha='center', va='center')
        ax4.set_xlabel('Поколение')
        ax4.set_ylabel('Успешность')
        ax4.set_title('Успешность рефлексов')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Статистика популяции (нижний левый)
        ax5 = plt.subplot(3, 2, 5)
        population_stats = generation_data.get('population_stats', {})
        if population_stats and isinstance(population_stats, dict):
            labels = list(population_stats.keys())
            values = [float(v) if isinstance(v, (int, float)) else 0 for v in population_stats.values()]
            if any(values):
                ax5.bar(labels, values, alpha=0.7)
        else:
            ax5.text(0.5, 0.5, 'Нет статистики популяции', ha='center', va='center')
        ax5.set_title('Статистика популяции')
        ax5.grid(True, alpha=0.3, axis='y')

        # 6. Информация о тренировке (нижний правый)
        ax6 = plt.subplot(3, 2, 6)
        ax6.axis('off')
        info_text = f"""
        Тренировка AGI

        Поколений: {len(gens)}
        Лучший фитнес: {best[-1] if best else 0:.2f}
        Средний фитнес: {avg[-1] if avg else 0:.2f}
        Потери G: {g_loss[-1] if g_loss else 0:.4f}
        Потери D: {d_loss[-1] if d_loss else 0:.4f}
        Разнообразие: {diversity[-1] if diversity else 0:.3f}

        Паттернов сгенерировано: {generation_data.get('patterns_generated', 0)}
        Паттернов сохранено: {generation_data.get('patterns_stored', 0)}
        """
        ax6.text(0.1, 0.9, info_text, fontsize=12, verticalalignment='top')

        plt.suptitle('Панель мониторинга AGI Evolution', fontsize=18)
        plt.tight_layout()
        plt.savefig(f"{self.save_path}dashboard.png", dpi=200)
        plt.close()
        print(f"✅ Панель мониторинга сохранена")

    def create_animation(self, generation_data: Dict, interval: int = 200):
        """Создаёт анимацию прогресса тренировки."""
        try:
            import matplotlib.animation as animation

            gens = generation_data['generations']
            best = generation_data['best_fitness']
            avg = generation_data['avg_fitness']
            g_loss = generation_data.get('g_loss', [0] * len(gens))
            d_loss = generation_data.get('d_loss', [0] * len(gens))

            if not gens:
                print("⚠️ Нет данных для анимации")
                return

            fig, ax = plt.subplots(figsize=(12, 8))

            def update(frame):
                ax.clear()
                limit = frame + 1

                # Фитнес
                ax.plot(gens[:limit], best[:limit], 'g-', linewidth=2, label='Лучший')
                ax.plot(gens[:limit], avg[:limit], 'b--', linewidth=1.5, label='Средний')

                # GAN потери (масштабируем для видимости)
                g_scaled = [x * 10 for x in g_loss[:limit]]
                d_scaled = [x * 10 for x in d_loss[:limit]]
                if any(g_scaled) or any(d_scaled):
                    ax.plot(gens[:limit], g_scaled, 'r-', linewidth=1, alpha=0.5, label='G потери (x10)')
                    ax.plot(gens[:limit], d_scaled, 'orange', linewidth=1, alpha=0.5, label='D потери (x10)')

                ax.set_xlabel('Поколение')
                ax.set_ylabel('Фитнес / Потери')
                ax.set_title(f'Прогресс тренировки (поколение {limit})')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Добавляем текущие значения
                if limit > 0:
                    best_val = best[limit - 1] if best else 0
                    avg_val = avg[limit - 1] if avg else 0
                    ax.text(0.02, 0.98, f'Лучший: {best_val:.2f}', transform=ax.transAxes,
                            fontsize=10, verticalalignment='top')
                    ax.text(0.02, 0.93, f'Средний: {avg_val:.2f}', transform=ax.transAxes,
                            fontsize=10, verticalalignment='top')

            ani = animation.FuncAnimation(fig, update,
                                          frames=len(gens),
                                          interval=interval, repeat=False)

            ani.save(f"{self.save_path}training_animation.gif", writer='pillow', fps=5)
            print(f"✅ Анимация сохранена")

        except Exception as e:
            print(f"⚠️ Не удалось создать анимацию: {e}")


# # core/visualization.py - НОВЫЙ ФАЙЛ
#
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import numpy as np
# from typing import List, Dict, Optional
# from db.connector import get_connection
# import json
#
#
# class TrainingVisualizer:
#     """
#     Визуализация прогресса тренировки ботов с использованием GAN.
#     """
#
#     def __init__(self, save_path: str = "training_plots/"):
#         self.save_path = save_path
#         self.fig = None
#         self.axes = None
#
#     def plot_fitness_progress(self, generations: List[int],
#                               best_fitness: List[float],
#                               avg_fitness: List[float]):
#         """График прогресса фитнеса."""
#         fig, ax = plt.subplots(figsize=(12, 6))
#
#         ax.plot(generations, best_fitness, 'g-', linewidth=2, label='Лучший фитнес')
#         ax.plot(generations, avg_fitness, 'b--', linewidth=1.5, label='Средний фитнес')
#         ax.fill_between(generations, avg_fitness, best_fitness, alpha=0.2, color='green')
#
#         ax.set_xlabel('Поколение')
#         ax.set_ylabel('Фитнес')
#         ax.set_title('Прогресс эволюции ботов')
#         ax.legend()
#         ax.grid(True, alpha=0.3)
#
#         plt.tight_layout()
#         plt.savefig(f"{self.save_path}fitness_progress.png", dpi=150)
#         plt.close()
#
#         print(f"✅ График фитнеса сохранён: {self.save_path}fitness_progress.png")
#
#     def plot_gan_progress(self, generations: List[int],
#                           g_loss: List[float],
#                           d_loss: List[float]):
#         """График потерь GAN."""
#         fig, ax = plt.subplots(figsize=(12, 6))
#
#         ax.plot(generations, g_loss, 'r-', linewidth=2, label='Потери генератора')
#         ax.plot(generations, d_loss, 'b-', linewidth=2, label='Потери дискриминатора')
#
#         ax.set_xlabel('Поколение')
#         ax.set_ylabel('Потери')
#         ax.set_title('Обучение GAN')
#         ax.legend()
#         ax.grid(True, alpha=0.3)
#
#         plt.tight_layout()
#         plt.savefig(f"{self.save_path}gan_progress.png", dpi=150)
#         plt.close()
#
#         print(f"✅ График GAN сохранён: {self.save_path}gan_progress.png")
#
#     def plot_pattern_diversity(self, generations: List[int],
#                                diversity: List[float]):
#         """График разнообразия паттернов."""
#         fig, ax = plt.subplots(figsize=(12, 6))
#
#         ax.bar(generations, diversity, color='purple', alpha=0.7)
#         ax.set_xlabel('Поколение')
#         ax.set_ylabel('Разнообразие паттернов')
#         ax.set_title('Разнообразие сгенерированных паттернов')
#         ax.grid(True, alpha=0.3, axis='y')
#
#         plt.tight_layout()
#         plt.savefig(f"{self.save_path}pattern_diversity.png", dpi=150)
#         plt.close()
#
#         print(f"✅ График разнообразия сохранён: {self.save_path}pattern_diversity.png")
#
#     def plot_reflex_success(self, reflex_data: Dict[str, List[float]]):
#         """График успешности рефлексов."""
#         fig, ax = plt.subplots(figsize=(12, 6))
#
#         for reflex_name, rates in reflex_data.items():
#             ax.plot(rates, label=reflex_name, linewidth=2)
#
#         ax.set_xlabel('Поколение')
#         ax.set_ylabel('Успешность')
#         ax.set_title('Успешность рефлексов')
#         ax.legend()
#         ax.grid(True, alpha=0.3)
#         ax.set_ylim(0, 1)
#
#         plt.tight_layout()
#         plt.savefig(f"{self.save_path}reflex_success.png", dpi=150)
#         plt.close()
#
#         print(f"✅ График рефлексов сохранён: {self.save_path}reflex_success.png")
#
#     def create_dashboard(self, generation_data: Dict):
#         """Создаёт полную панель мониторинга."""
#         fig = plt.figure(figsize=(20, 16))
#
#         # 1. Фитнес (верхний левый)
#         ax1 = plt.subplot(3, 2, 1)
#         ax1.plot(generation_data['generations'], generation_data['best_fitness'],
#                  'g-', linewidth=2, label='Лучший')
#         ax1.plot(generation_data['generations'], generation_data['avg_fitness'],
#                  'b--', linewidth=1.5, label='Средний')
#         ax1.set_xlabel('Поколение')
#         ax1.set_ylabel('Фитнес')
#         ax1.set_title('Прогресс эволюции')
#         ax1.legend()
#         ax1.grid(True, alpha=0.3)
#
#         # 2. GAN (верхний правый)
#         ax2 = plt.subplot(3, 2, 2)
#         ax2.plot(generation_data['generations'], generation_data['g_loss'],
#                  'r-', linewidth=2, label='G потери')
#         ax2.plot(generation_data['generations'], generation_data['d_loss'],
#                  'b-', linewidth=2, label='D потери')
#         ax2.set_xlabel('Поколение')
#         ax2.set_ylabel('Потери')
#         ax2.set_title('Обучение GAN')
#         ax2.legend()
#         ax2.grid(True, alpha=0.3)
#
#         # 3. Разнообразие (средний левый)
#         ax3 = plt.subplot(3, 2, 3)
#         ax3.bar(generation_data['generations'], generation_data['diversity'],
#                 color='purple', alpha=0.7)
#         ax3.set_xlabel('Поколение')
#         ax3.set_ylabel('Разнообразие')
#         ax3.set_title('Разнообразие паттернов')
#         ax3.grid(True, alpha=0.3, axis='y')
#
#         # 4. Успешность рефлексов (средний правый)
#         ax4 = plt.subplot(3, 2, 4)
#         for reflex, rates in generation_data.get('reflex_success', {}).items():
#             ax4.plot(rates, label=reflex, linewidth=2)
#         ax4.set_xlabel('Поколение')
#         ax4.set_ylabel('Успешность')
#         ax4.set_title('Успешность рефлексов')
#         ax4.legend()
#         ax4.grid(True, alpha=0.3)
#         ax4.set_ylim(0, 1)
#
#         # 5. Статистика популяции (нижний левый)
#         ax5 = plt.subplot(3, 2, 5)
#         population_stats = generation_data.get('population_stats', {})
#         if population_stats:
#             labels = list(population_stats.keys())
#             values = list(population_stats.values())
#             ax5.bar(labels, values, alpha=0.7)
#         ax5.set_title('Статистика популяции')
#         ax5.grid(True, alpha=0.3, axis='y')
#
#         # 6. Информация о тренировке (нижний правый)
#         ax6 = plt.subplot(3, 2, 6)
#         ax6.axis('off')
#         info_text = f"""
#         Тренировка AGI
#
#         Поколений: {len(generation_data['generations'])}
#         Лучший фитнес: {generation_data['best_fitness'][-1]:.2f}
#         Средний фитнес: {generation_data['avg_fitness'][-1]:.2f}
#         Потери G: {generation_data['g_loss'][-1]:.4f}
#         Потери D: {generation_data['d_loss'][-1]:.4f}
#         Разнообразие: {generation_data['diversity'][-1]:.3f}
#
#         Паттернов сгенерировано: {generation_data.get('patterns_generated', 0)}
#         Паттернов сохранено: {generation_data.get('patterns_stored', 0)}
#         """
#         ax6.text(0.1, 0.9, info_text, fontsize=12, verticalalignment='top')
#
#         plt.suptitle('Панель мониторинга AGI Evolution', fontsize=18)
#         plt.tight_layout()
#         plt.savefig(f"{self.save_path}dashboard.png", dpi=200)
#         plt.close()
#
#         print(f"✅ Панель мониторинга сохранена: {self.save_path}dashboard.png")
#
#     def create_animation(self, generation_data: Dict, interval: int = 200):
#         """Создаёт анимацию прогресса тренировки."""
#         fig, ax = plt.subplots(figsize=(12, 8))
#
#         def update(frame):
#             ax.clear()
#             limit = frame + 1
#
#             # Фитнес
#             ax.plot(generation_data['generations'][:limit],
#                     generation_data['best_fitness'][:limit],
#                     'g-', linewidth=2, label='Лучший')
#             ax.plot(generation_data['generations'][:limit],
#                     generation_data['avg_fitness'][:limit],
#                     'b--', linewidth=1.5, label='Средний')
#
#             # GAN потери
#             ax.plot(generation_data['generations'][:limit],
#                     [g * 10 for g in generation_data['g_loss'][:limit]],
#                     'r-', linewidth=1, alpha=0.5, label='G потери (x10)')
#
#             ax.set_xlabel('Поколение')
#             ax.set_ylabel('Фитнес / Потери')
#             ax.set_title(f'Прогресс тренировки (поколение {limit})')
#             ax.legend()
#             ax.grid(True, alpha=0.3)
#
#             # Добавляем текущие значения
#             if limit > 0:
#                 best = generation_data['best_fitness'][limit - 1]
#                 avg = generation_data['avg_fitness'][limit - 1]
#                 ax.text(0.02, 0.98, f'Лучший: {best:.2f}', transform=ax.transAxes,
#                         fontsize=10, verticalalignment='top')
#                 ax.text(0.02, 0.93, f'Средний: {avg:.2f}', transform=ax.transAxes,
#                         fontsize=10, verticalalignment='top')
#
#         ani = animation.FuncAnimation(fig, update,
#                                       frames=len(generation_data['generations']),
#                                       interval=interval, repeat=False)
#
#         ani.save(f"{self.save_path}training_animation.gif", writer='pillow', fps=5)
#         print(f"✅ Анимация сохранена: {self.save_path}training_animation.gif")
#
#         return ani