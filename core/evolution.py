from core.population import Population
from db.connector import save_generation, save_genome
import time

class EvolutionEngine:
    def __init__(self, world, population_size=20, generations=50,
                 steps_per_episode=500, elite_count=2, mutation_rate=0.1):
        self.world = world
        self.population = Population(population_size, world)
        self.generations = generations
        self.steps_per_episode = steps_per_episode
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.best_fitness_history = []

    def run(self, save_to_db=False):
        for gen in range(self.generations):
            start_time = time.time()
            print(f"=== Поколение {gen+1}/{self.generations} ===")

            # Оцениваем всех особей
            fitnesses = self.population.evaluate_all(self.steps_per_episode)
            best_fit = max(fitnesses)
            avg_fit = sum(fitnesses) / len(fitnesses)
            self.best_fitness_history.append(best_fit)

            print(f"Лучший фитнес: {best_fit:.2f}, Средний: {avg_fit:.2f}")

            # Сохраняем статистику в БД (опционально)
            if save_to_db:
                gen_id = save_generation(gen+1, best_fit, avg_fit)
                # Сохраняем геном лучшей особи
                best_ind = self.population.select_best(1)[0]
                if gen_id:
                    save_genome(gen_id, best_ind.genome.to_dict(), best_ind.fitness)

            # Создаём следующее поколение (кроме последнего)
            if gen < self.generations - 1:
                self.population.next_generation(
                    elite_count=self.elite_count,
                    mutation_rate=self.mutation_rate
                )

            elapsed = time.time() - start_time
            print(f"Время поколения: {elapsed:.2f} сек.")

        return self.best_fitness_history