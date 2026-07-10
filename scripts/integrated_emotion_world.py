# scripts/integrated_emotion_world.py
"""
Интегрированная визуализация: мир с объектами + эмоциональные реакции бота.
Показывает движение бота в ландшафте и его эмоциональное состояние в реальном времени.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import List, Dict, Any, Tuple
import warnings

warnings.filterwarnings('ignore')

from core.world import World
from core.individual import Individual
from core.genome import Genome
from core.objects import Food, Predator, GameObject
from core.emotions.emotion_system import EmotionSystem
from core.emotions.emotion_base import EmotionalEvent, EmotionalResponse, EmotionType


class IntegratedEmotionWorld:
    """
    Интегрированная визуализация: мир + эмоции бота.
    """

    def __init__(self, world_size: int = 20, cell_size: int = 40):
        # Инициализируем мир
        self.world = World(width=1200, height=800, world_size=world_size, cell_size=cell_size)
        self.world_size = world_size
        self.cell_size = cell_size

        # Создаем бота
        self.bot = Individual(x=0, z=0)

        # Создаем эмоциональную систему
        self.emotion_system = EmotionSystem()
        self.bot.emotion_system = self.emotion_system

        # Создаем события и связи для эмоций
        self._create_emotion_events()

        # История эмоций для графиков
        self.emotion_history = []
        self.emotion_counts = {}
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

        # Инициализируем Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((1600, 900))
        pygame.display.set_caption("Интегрированный мир: Ландшафт + Эмоции бота")
        self.clock = pygame.time.Clock()

        # Создаем объекты в мире
        self._populate_world()

        print("✅ Интегрированная визуализация инициализирована")

    def _create_emotion_events(self):
        """Создает события и связи для эмоциональной системы."""
        events_data = {
            'danger': {'desc': 'Опасность!', 'pattern': [0.9, 0.8, 0.1, 0.0, 0.0]},
            'success': {'desc': 'Успех!', 'pattern': [0.1, 0.0, 0.9, 0.8, 0.0]},
            'failure': {'desc': 'Неудача...', 'pattern': [0.0, 0.9, 0.8, 0.1, 0.0]},
            'food': {'desc': 'Еда!', 'pattern': [0.2, 0.0, 0.7, 0.6, 0.1]},
            'predator': {'desc': 'Хищник!', 'pattern': [0.9, 0.7, 0.1, 0.0, 0.2]},
            'fire': {'desc': 'Огонь!', 'pattern': [0.8, 0.9, 0.1, 0.0, 0.0]},
        }

        event_emotion_map = {
            'danger': EmotionType.FEAR,
            'success': EmotionType.JOY,
            'failure': EmotionType.SADNESS,
            'food': EmotionType.JOY,
            'predator': EmotionType.FEAR,
            'fire': EmotionType.FEAR,
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
                participants=['bot'],
                embedding=embedding
            )

            self.emotion_system.engine.graph.add_event(event)

            main_emotion = event_emotion_map[event_id]
            self.emotion_system.engine.graph.add_event_emotion_link(
                event_id, main_emotion, probability=0.8, intensity_factor=1.0
            )

    def _populate_world(self):
        """Заполняет мир объектами."""
        # Еда
        food_positions = [(-5, 5), (5, -5), (-7, -7), (7, 7), (0, 10), (10, 0)]
        for x, z in food_positions[:4]:
            food = Food(x, z, name="apple", obj_type="food", smell="food_smell")
            food.active = True  # Добавляем
            self.world.add_object(food)

        # Хищники
        predator_positions = [(6, 6), (-6, -6)]
        for x, z in predator_positions:
            predator = Predator(x, z, name="wolf", obj_type="predator", smell="predator_smell")
            predator.active = True  # Добавляем
            self.world.add_object(predator)

        # Огонь
        fire_positions = [(-8, 8), (8, -8)]
        for x, z in fire_positions:
            fire = GameObject(x, z, obj_type="fire", temperature=800, size=1.0)
            fire.active = True  # Добавляем
            self.world.add_object(fire)

    # def _populate_world(self):
    #     """Заполняет мир объектами."""
    #     # Еда
    #     food_positions = [(-5, 5), (5, -5), (-7, -7), (7, 7), (0, 10), (10, 0)]
    #     for x, z in food_positions[:4]:
    #         self.world.add_object(Food(x, z, name="apple", obj_type="food", smell="food_smell"))
    #
    #     # Хищники
    #     predator_positions = [(6, 6), (-6, -6)]
    #     for x, z in predator_positions:
    #         self.world.add_object(Predator(x, z, name="wolf", obj_type="predator", smell="predator_smell"))
    #
    #     # Огонь
    #     fire_positions = [(-8, 8), (8, -8)]
    #     for x, z in fire_positions:
    #         self.world.add_object(GameObject(x, z, obj_type="fire", temperature=800, size=1.0))

    def _get_event_from_position(self, x: float, z: float) -> str:
        """Определяет событие на основе позиции бота."""
        for obj in self.world.objects:
            if hasattr(obj, 'active') and not obj.active:
                continue  # Пропускаем неактивные объекты
            dist = np.sqrt((obj.x - x) ** 2 + (obj.z - z) ** 2)
            if dist < 2.0:
                if hasattr(obj, 'type') and obj.type == 'food':
                    return 'food'
                elif hasattr(obj, 'type') and obj.type == 'predator':
                    return 'predator'
                elif hasattr(obj, 'type') and obj.type == 'fire':
                    return 'fire'
        # # Проверяем объекты рядом
        # for obj in self.world.objects:
        #     dist = np.sqrt((obj.x - x) ** 2 + (obj.z - z) ** 2)
        #     if dist < 2.0:
        #         if hasattr(obj, 'type') and obj.type == 'food':
        #             return 'food'
        #         elif hasattr(obj, 'type') and obj.type == 'predator':
        #             return 'predator'
        #         elif hasattr(obj, 'type') and obj.type == 'fire':
        #             return 'fire'

        # Проверяем границы мира (опасность)
        half = self.world_size // 2
        if abs(x) > half - 2 or abs(z) > half - 2:
            return 'danger'

        # Если ничего не найдено
        return None

    def _process_emotion(self):
        """Обрабатывает эмоциональную реакцию на текущее положение бота."""
        event_type = self._get_event_from_position(self.bot.x, self.bot.z)

        if event_type:
            event = self.emotion_system.engine.graph.events.get(event_type)
            if event:
                responses = self.emotion_system.process_sensory_input({
                    'vision': event.embedding[:64],
                    'sound': event.embedding[64:96],
                    'smell': event.embedding[96:128],
                    'context': {'type': event_type},
                    'participants': ['bot']
                })

                state = self.emotion_system.get_emotional_state()

                self.emotion_history.append({
                    'step': len(self.emotion_history),
                    'event_type': event_type,
                    'dominant': state['dominant_emotion'],
                    'intensity': state['intensity'],
                    'valence': state['valence'],
                    'arousal': state['arousal']
                })

                if state['dominant_emotion'] not in self.emotion_counts:
                    self.emotion_counts[state['dominant_emotion']] = 0
                self.emotion_counts[state['dominant_emotion']] += 1

                # ========== ДОБАВЛЯЕМ СБОР ЕДЫ ==========
                if event_type == 'food':
                    for obj in self.world.objects:
                        if hasattr(obj, 'type') and obj.type == 'food':
                            if hasattr(obj, 'active') and obj.active:
                                dist = np.sqrt((obj.x - self.bot.x) ** 2 + (obj.z - self.bot.z) ** 2)
                                if dist < 1.5:
                                    obj.active = False
                                    print(f"🍎 Бот собрал еду в ({self.bot.x}, {self.bot.z})!")
                                    break

    # def _process_emotion(self):
    #     """Обрабатывает эмоциональную реакцию на текущее положение бота."""
    #     event_type = self._get_event_from_position(self.bot.x, self.bot.z)
    #
    #     if event_type:
    #         event = self.emotion_system.engine.graph.events.get(event_type)
    #         if event:
    #             responses = self.emotion_system.process_sensory_input({
    #                 'vision': event.embedding[:64],
    #                 'sound': event.embedding[64:96],
    #                 'smell': event.embedding[96:128],
    #                 'context': {'type': event_type},
    #                 'participants': ['bot']
    #             })
    #
    #             state = self.emotion_system.get_emotional_state()
    #
    #             self.emotion_history.append({
    #                 'step': len(self.emotion_history),
    #                 'event_type': event_type,
    #                 'dominant': state['dominant_emotion'],
    #                 'intensity': state['intensity'],
    #                 'valence': state['valence'],
    #                 'arousal': state['arousal']
    #             })
    #
    #             # Обновляем счетчики
    #             if state['dominant_emotion'] not in self.emotion_counts:
    #                 self.emotion_counts[state['dominant_emotion']] = 0
    #             self.emotion_counts[state['dominant_emotion']] += 1

    def _draw_emotion_dashboard(self, screen, x_offset: int):
        """Рисует панель эмоций справа от мира."""
        font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 18)

        # Фон панели
        panel_rect = pygame.Rect(x_offset, 0, 400, 900)
        pygame.draw.rect(screen, (40, 40, 50), panel_rect)

        # Заголовок
        title = font.render("ЭМОЦИИ БОТА", True, (255, 255, 255))
        screen.blit(title, (x_offset + 10, 10))

        # Текущее состояние
        if self.emotion_history:
            last = self.emotion_history[-1]

            # Текущая эмоция (крупно)
            emotion_text = font.render(f"Текущая эмоция:", True, (200, 200, 200))
            screen.blit(emotion_text, (x_offset + 10, 50))

            color = self.emotion_colors.get(last['dominant'], (128, 128, 128))
            emotion_name = font.render(last['dominant'].upper(), True, color)
            screen.blit(emotion_name, (x_offset + 10, 80))

            # Интенсивность
            intensity_text = small_font.render(f"Интенсивность: {last['intensity']:.2f}", True, (200, 200, 200))
            screen.blit(intensity_text, (x_offset + 10, 115))

            # Валентность
            valence_text = small_font.render(f"Валентность: {last['valence']:.2f}", True, (200, 200, 200))
            screen.blit(valence_text, (x_offset + 10, 140))

            # Активация
            arousal_text = small_font.render(f"Активация: {last['arousal']:.2f}", True, (200, 200, 200))
            screen.blit(arousal_text, (x_offset + 10, 165))

            # Полоса интенсивности
            bar_rect = pygame.Rect(x_offset + 10, 190, 200, 20)
            pygame.draw.rect(screen, (60, 60, 70), bar_rect)
            fill_rect = pygame.Rect(x_offset + 10, 190, int(200 * last['intensity']), 20)
            pygame.draw.rect(screen, color, fill_rect)

        # График распределения эмоций (pygame-версия)
        if self.emotion_counts:
            y_start = 230
            y_step = 25
            count_text = small_font.render("Распределение эмоций:", True, (200, 200, 200))
            screen.blit(count_text, (x_offset + 10, y_start))

            sorted_emotions = sorted(self.emotion_counts.items(), key=lambda x: x[1], reverse=True)
            for i, (emotion, count) in enumerate(sorted_emotions[:10]):
                y = y_start + 25 + i * y_step
                color = self.emotion_colors.get(emotion, (128, 128, 128))

                # Название эмоции
                name_text = small_font.render(f"{emotion}:", True, (200, 200, 200))
                screen.blit(name_text, (x_offset + 10, y))

                # Полоса
                bar_rect = pygame.Rect(x_offset + 100, y + 5, 150, 15)
                pygame.draw.rect(screen, (60, 60, 70), bar_rect)
                max_count = max(self.emotion_counts.values()) if self.emotion_counts else 1
                fill_rect = pygame.Rect(x_offset + 100, y + 5, int(150 * count / max_count), 15)
                pygame.draw.rect(screen, color, fill_rect)

                # Количество
                count_text = small_font.render(str(count), True, (200, 200, 200))
                screen.blit(count_text, (x_offset + 260, y))

        # Общая статистика
        y = 550
        stats_text = small_font.render(f"Всего реакций: {len(self.emotion_history)}", True, (200, 200, 200))
        screen.blit(stats_text, (x_offset + 10, y))

        y += 25
        if self.emotion_history:
            unique = len(set([h['dominant'] for h in self.emotion_history]))
            stats_text = small_font.render(f"Уникальных эмоций: {unique}", True, (200, 200, 200))
            screen.blit(stats_text, (x_offset + 10, y))

        y += 25
        stats_text = small_font.render(f"Шаг: {len(self.emotion_history)}", True, (200, 200, 200))
        screen.blit(stats_text, (x_offset + 10, y))

        # Легенда объектов
        y = 630
        legend_text = small_font.render("Объекты в мире:", True, (200, 200, 200))
        screen.blit(legend_text, (x_offset + 10, y))

        objects_info = [
            ("🍎 Еда", (100, 200, 100)),
            ("🐺 Хищник", (200, 100, 100)),
            ("🔥 Огонь", (255, 150, 0)),
        ]

        for i, (name, color) in enumerate(objects_info):
            y_pos = y + 25 + i * 25
            color_rect = pygame.Rect(x_offset + 10, y_pos, 15, 15)
            pygame.draw.rect(screen, color, color_rect)
            text = small_font.render(name, True, (200, 200, 200))
            screen.blit(text, (x_offset + 30, y_pos))

        # Подсказка по управлению
        y = 740
        help_text = small_font.render("Управление:", True, (200, 200, 200))
        screen.blit(help_text, (x_offset + 10, y))

        controls = [
            "Стрелки - движение бота",
            "Space - сброс позиции",
            "Esc - выход"
        ]
        for i, ctrl in enumerate(controls):
            y_pos = y + 20 + i * 20
            text = small_font.render(ctrl, True, (150, 150, 150))
            screen.blit(text, (x_offset + 10, y_pos))

        # Интуитивные подсказки
        y = 800
        intuition_text = small_font.render("🧠 Интуиция:", True, (200, 200, 200))
        screen.blit(intuition_text, (x_offset + 10, y))

        # Проверяем интуицию для текущей позиции
        warning = self._check_intuition(self.bot.x, self.bot.z)
        if warning:
            y += 25
            warning_text = small_font.render(warning, True, (255, 200, 0))
            screen.blit(warning_text, (x_offset + 10, y))
        else:
            y += 25
            safe_text = small_font.render("✅ Путь безопасен", True, (100, 200, 100))
            screen.blit(safe_text, (x_offset + 10, y))


    def run(self):
        """Запускает интегрированную визуализацию."""
        running = True

        # Смещение для панели эмоций
        panel_offset = 1200

        while running:
            # Обработка событий
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_SPACE:
                        self.bot.x = 0
                        self.bot.z = 0

            # Управление ботом (стрелки)
            keys = pygame.key.get_pressed()
            speed = 0.3
            dx, dz = 0, 0
            if keys[pygame.K_UP]:
                dz = -speed
            if keys[pygame.K_DOWN]:
                dz = speed
            if keys[pygame.K_LEFT]:
                dx = -speed
            if keys[pygame.K_RIGHT]:
                dx = speed

            if dx != 0 or dz != 0:
                # Двигаем бота
                new_x = self.bot.x + dx
                new_z = self.bot.z + dz
                half = self.world.world_size // 2
                if -half <= new_x <= half and -half <= new_z <= half:
                    self.bot.x = new_x
                    self.bot.z = new_z

                    # Обрабатываем эмоции
                    self._process_emotion()

            # Отрисовка
            self.screen.fill((30, 30, 30))

            # Рисуем мир (используем метод world.draw, но передаем бота)
            self.world.draw(self.screen, self.bot)

            # Рисуем панель эмоций
            self._draw_emotion_dashboard(self.screen, panel_offset)

            # Обновляем дисплей
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

    def _check_intuition(self, target_x: float, target_z: float) -> Optional[str]:
        """
        Проверяет интуицию перед движением к цели.
        Возвращает предупреждение, если цель опасна.
        """
        # Проверяем, есть ли рядом опасные объекты
        danger_distance = 3.0
        for obj in self.world.objects:
            if hasattr(obj, 'active') and not obj.active:
                continue

            dist_to_target = np.sqrt((obj.x - target_x) ** 2 + (obj.z - target_z) ** 2)

            if dist_to_target < danger_distance:
                # Создаем описание ситуации
                situation = f"вижу {obj.type if hasattr(obj, 'type') else 'объект'} рядом с целью"
                action = "пойти к цели"

                # Определяем последствие
                if obj.type == 'predator':
                    consequence = "волк напал"
                    emotion = "страх"
                elif obj.type == 'fire':
                    consequence = "обжёгся"
                    emotion = "страх"
                else:
                    continue

                # Проверяем интуицию
                prediction = self.emotion_system.intuition.predict_consequence(situation, action)

                if prediction and prediction['probability'] > 0.5:
                    return f"⚠️ Интуиция предупреждает: {prediction['consequence']}! (уверенность: {prediction['confidence']:.2f})"

        return None

    def _process_emotion(self):
        """Обрабатывает эмоциональную реакцию и обучает интуицию."""
        event_type = self._get_event_from_position(self.bot.x, self.bot.z)

        if event_type:
            event = self.emotion_system.engine.graph.events.get(event_type)
            if event:
                responses = self.emotion_system.process_sensory_input({...})
                state = self.emotion_system.get_emotional_state()

                # ... сохраняем эмоции ...

                # ========== ОБУЧАЕМ ИНТУИЦИЮ ==========
                if event_type == 'predator':
                    # Бот встретил хищника - запоминаем
                    situation = "вижу хищника"
                    action = "подойти близко"
                    consequence = "хищник напал"
                    emotion = state['dominant_emotion']
                    self.emotion_system.intuition.learn_from_experience(
                        situation, action, consequence, emotion,
                        success=False, intensity=state['intensity']
                    )

                elif event_type == 'fire':
                    # Бот подошел к огню - запоминаем
                    situation = "вижу огонь"
                    action = "подойти близко"
                    consequence = "обжёгся"
                    emotion = state['dominant_emotion']
                    self.emotion_system.intuition.learn_from_experience(
                        situation, action, consequence, emotion,
                        success=False, intensity=state['intensity']
                    )

                elif event_type == 'food':
                    # Бот собрал еду - запоминаем успешный опыт
                    situation = "вижу еду"
                    action = "пойти за едой"
                    consequence = "собрал еду"
                    emotion = state['dominant_emotion']
                    self.emotion_system.intuition.learn_from_experience(
                        situation, action, consequence, emotion,
                        success=True, intensity=state['intensity']
                    )



def main():
    """
    Главная функция для запуска интегрированной визуализации.
    """
    print("=" * 60)
    print("ИНТЕГРИРОВАННАЯ ВИЗУАЛИЗАЦИЯ: МИР + ЭМОЦИИ")
    print("=" * 60)
    print("\nУправление:")
    print("  Стрелки - движение бота")
    print("  Space - сброс позиции")
    print("  Esc - выход")
    print("\nНаблюдайте, как эмоции бота меняются при взаимодействии с объектами!")

    # Запускаем визуализацию
    visualizer = IntegratedEmotionWorld(world_size=20, cell_size=40)
    visualizer.run()

    print("\n✅ Визуализация завершена")


if __name__ == "__main__":
    main()