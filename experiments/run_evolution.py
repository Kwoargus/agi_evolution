# experiments/run_evolution.py (исправленный)
import sys
import os

# Добавляем корневую директорию проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.world import World
from core.evolution import EvolutionEngine
from models.gan import GAN

if __name__ == "__main__":
    # Создаём мир
    world = World(width=800, height=600, world_size=20, cell_size=40)

    # Добавляем объекты в мир
    from core.objects import GameObject, Predator, Food

    # Добавляем огонь
    fire = GameObject(-5, -5, obj_type="fire", temperature=800, size=1.0)
    world.add_object(fire)

    # Добавляем хищника
    predator = Predator(5, 5, name="wolf", obj_type="predator", smell="predator_smell")
    world.add_object(predator)

    # Добавляем еду
    food1 = Food(-6, 6, name="apple", obj_type="food", smell="food_smell")
    food2 = Food(6, -6, name="apple", obj_type="food", smell="food_smell")
    world.add_object(food1)
    world.add_object(food2)

    # Инициализация движка с GAN
    engine = EvolutionEngine(
        world=world,
        population_size=15,
        generations=30,
        steps_per_episode=300,
        elite_count=3,
        mutation_rate=0.15,
        use_gan=True,  # Включаем GAN
        gan_training_epochs=3  # Количество эпох обучения GAN за поколение
    )

    # Запускаем эволюцию
    history = engine.run(save_to_db=True)
    print("Эволюция с GAN завершена.")
    print("История лучших фитнесов:", history)

    # Сохраняем состояние GAN
    if engine.gan:
        engine.gan.save_models('models/gan_evolution')
        print("GAN сохранён в models/gan_evolution_checkpoint.pth")



# # experiments/run_evolution.py (исправленный)
# from core.world import World
# from core.evolution import EvolutionEngine
# from models.gan import GAN
# import sys
# import os
#
# # Добавляем импорт функции визуализации
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from scripts.plot_gan_losses import plot_gan_losses
# from scripts.plot_gan_losses_seperated import plot_fitness_history
#
# # Добавляем путь к проекту
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# if __name__ == "__main__":
#     # Создаём мир
#     world = World(width=800, height=600, world_size=20, cell_size=40)
#
#     # Добавляем объекты в мир
#     from core.objects import GameObject, Predator, Food
#
#     # Добавляем огонь
#     fire = GameObject(-5, -5, obj_type="fire", temperature=800, size=1.0)
#     world.add_object(fire)
#
#     # Добавляем хищника
#     predator = Predator(5, 5, name="wolf", obj_type="predator", smell="predator_smell")
#     world.add_object(predator)
#
#     # Добавляем еду
#     food1 = Food(-6, 6, name="apple", obj_type="food", smell="food_smell")
#     food2 = Food(6, -6, name="apple", obj_type="food", smell="food_smell")
#     world.add_object(food1)
#     world.add_object(food2)
#
#     # Инициализация движка с GAN
#     engine = EvolutionEngine(
#         world=world,
#         population_size=15,
#         generations=30,
#         steps_per_episode=300,
#         elite_count=3,
#         mutation_rate=0.15,
#         use_gan=True,  # Включаем GAN
#         gan_training_epochs=3  # Количество эпох обучения GAN за поколение
#     )
#
#     # Запускаем эволюцию
#     history = engine.run(save_to_db=True)
#     print("Эволюция с GAN завершена.")
#     print("История лучших фитнесов:", history)
#
#     # Сохраняем состояние GAN
#     if engine.gan:
#         engine.gan.save_models('models/gan_evolution')
#         print("GAN сохранён в models/gan_evolution_checkpoint.pth")
#
#     # ВИЗУАЛИЗИРУЕМ РЕЗУЛЬТАТЫ
#     if engine.gan and engine.gan.generator_losses:
#         plot_gan_losses(
#             engine.gan.generator_losses,
#             engine.gan.discriminator_losses,
#             f"GAN Losses - {engine.generations} поколений",
#             save_path='gan_losses_plot.png'
#         )
#
#     plot_fitness_history(
#         history,
#         title=f"Эволюция - Лучший фитнес ({engine.generations} поколений)",
#         save_path='fitness_history.png'
#     )
#
#     # Сохраняем GAN
#     if engine.gan:
#         engine.gan.save_models('models/gan_evolution')
#         print("GAN сохранён в models/gan_evolution_checkpoint.pth")
#
# # # run_evolution_with_gan.py
# # from core.world import World
# # from core.evolution import EvolutionEngine
# # from models.gan import GAN
# #
# # if __name__ == "__main__":
# #     # Создаём мир
# #     world = World(width=800, height=600, world_size=20, cell_size=40)
# #
# #     # Инициализация движка с GAN
# #     engine = EvolutionEngine(
# #         world=world,
# #         population_size=15,
# #         generations=30,
# #         steps_per_episode=300,
# #         elite_count=3,
# #         mutation_rate=0.15,
# #         use_gan=True,  # Включаем GAN
# #         gan_training_epochs=3  # Количество эпох обучения GAN за поколение
# #     )
# #
# #     # Запускаем эволюцию
# #     history = engine.run(save_to_db=True)
# #     print("Эволюция с GAN завершена.")
# #     print("История лучших фитнесов:", history)
# #
# #     # Сохраняем состояние GAN
# #     if engine.gan:
# #         engine.gan.save_models('models/gan_evolution')
# #         print("GAN сохранён в models/gan_evolution_checkpoint.pth")
# #
# #
# # # from core.world import World
# # # from core.evolution import EvolutionEngine
# # #
# # # if __name__ == "__main__":
# # #     # Создаём мир без визуализации (можно передать dummy screen)
# # #     world = World(width=800, height=600, world_size=20, cell_size=40)
# # #     # Можно добавить объекты в мир (они будут общими для всех ботов)
# # #     # но для эволюции лучше создавать объекты в мире и не менять их
# # #     # (или сбрасывать после каждого бота, но тогда объекты должны быть статичны)
# # #
# # #     # Инициализация движка
# # #     engine = EvolutionEngine(
# # #         world=world,
# # #         population_size=10,
# # #         generations=20,
# # #         steps_per_episode=200,
# # #         elite_count=2,
# # #         mutation_rate=0.2
# # #     )
# # #
# # #     history = engine.run(save_to_db=True)
# # #     print("Эволюция завершена.")
# # #     print("История лучших фитнесов:", history)