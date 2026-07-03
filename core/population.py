import random
from core.individual import Individual
from core.genome import Genome

class Population:
    def __init__(self, size, world, genome_template=None, start_pos=(0,0)):
        self.size = size
        self.world = world
        self.individuals = []
        self.generation = 0

        for _ in range(size):
            if genome_template:
                genome = genome_template.mutate(0.3)  # начальное разнообразие
            else:
                genome = Genome()
            ind = Individual(x=start_pos[0], z=start_pos[1], genome=genome)
            self.individuals.append(ind)

    def evaluate_all(self, steps_per_episode=500):
        """Запускает всех особей на эпизод и возвращает список фитнесов."""
        fitnesses = []
        for ind in self.individuals:
            # Сброс мира (объекты остаются, но бот сбрасывается)
            self.world.reset()
            fit = ind.evaluate(self.world, steps_per_episode)
            fitnesses.append(fit)
            ind.get_experiences()
            all_experiences = []
            for ind in self.individuals:
                ind.evaluate(...)
                exps = ind.get_experiences()
                all_experiences.extend(exps)
            # Теперь all_experiences можно передать в GAN для обучения

        return fitnesses

    def _generate_random_objects(self):
        pass

    def select_best(self, top_k):
        """Возвращает top_k особей по фитнесу."""
        sorted_inds = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
        return sorted_inds[:top_k]

    def next_generation(self, elite_count=2, mutation_rate=0.1):
        """Создаёт следующее поколение."""
        best = self.select_best(elite_count)
        new_individuals = []
        # Элита сохраняется без изменений
        new_individuals.extend(best)

        # Заполняем остальных потомками от элиты
        while len(new_individuals) < self.size:
            parent1 = random.choice(best)
            parent2 = random.choice(best)
            child_genome = parent1.genome.crossover(parent2.genome)
            child_genome = child_genome.mutate(mutation_rate)
            child = Individual(x=0, z=0, genome=child_genome)
            new_individuals.append(child)

        self.individuals = new_individuals
        self.generation += 1