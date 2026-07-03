from core.world import World
from core.evolution import EvolutionEngine

if __name__ == "__main__":
    # Создаём мир без визуализации (можно передать dummy screen)
    world = World(width=800, height=600, world_size=20, cell_size=40)
    # Можно добавить объекты в мир (они будут общими для всех ботов)
    # но для эволюции лучше создавать объекты в мире и не менять их
    # (или сбрасывать после каждого бота, но тогда объекты должны быть статичны)

    # Инициализация движка
    engine = EvolutionEngine(
        world=world,
        population_size=10,
        generations=20,
        steps_per_episode=200,
        elite_count=2,
        mutation_rate=0.2
    )

    history = engine.run(save_to_db=True)
    print("Эволюция завершена.")
    print("История лучших фитнесов:", history)