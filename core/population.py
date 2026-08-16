# core/population.py
import random
from core.individual import Individual
from core.genome import Genome
from typing import List, Dict, Optional, Tuple, Any, Union

class Population:
    def __init__(self, size, world, genome_template=None, start_pos=(0,0)):
        self.size = size
        self.world = world
        self.individuals = []
        self.generation = 0

        for _ in range(size):
            if genome_template:
                genome = genome_template.mutate(0.3)  # [ru] начальное разнообразие
                genome = genome_template.mutate(0.3)  # [en] initial diversity
            else:
                genome = Genome()
            ind = Individual(x=start_pos[0], z=start_pos[1], genome=genome)
            self.individuals.append(ind)

        for ind in self.individuals:
            world.add_bot(ind)

    def evaluate_all(self, steps_per_episode=500):
        fitnesses = []
        all_experiences = []  # [ru] список всех переходов за поколение [en] list of all transitions per generation
        for ind in self.individuals:
            self.world.reset()
            fit = ind.evaluate(self.world, steps_per_episode)
            fitnesses.append(fit)
            # [ru] Получаем и очищаем буфер бота
            # [en] Get and clear bot's buffer
            exps = ind.get_experiences()
            all_experiences.extend(exps)
        # [ru] Теперь all_experiences можно передать в GAN
        # [en] Now all_experiences can be passed to GAN
        return fitnesses, all_experiences

    def _generate_random_objects(self):
        pass

    def select_best(self, top_k):
        """
        [ru] Возвращает top_k особей по фитнесу.
        [en] Returns the top_k individuals by fitness.
        """
        sorted_inds = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
        return sorted_inds[:top_k]

    def next_generation(self, elite_count=2, mutation_rate=0.1):
        """
        [ru] Создаёт следующее поколение с учётом разнообразия.
        [en] Creates the next generation taking diversity into account.
        """
        # [ru] Используем отбор с разнообразием
        # [en] Use selection with diversity
        best = self.select_best_with_diversity(elite_count * 2, diversity_weight=0.3)
        best = best[:elite_count]  # [ru] Оставляем только elite_count
        best = best[:elite_count]  # [en] Keep only elite_count

        new_individuals = []
        new_individuals.extend(best)

        while len(new_individuals) < self.size:
            parent1 = random.choice(best)
            parent2 = random.choice(best)
            child_genome = parent1.genome.crossover(parent2.genome)
            child_genome = child_genome.mutate(mutation_rate)
            child = Individual(x=0, z=0, genome=child_genome)
            new_individuals.append(child)

        self.individuals = new_individuals
        self.generation += 1


    def get_statistics(self) -> Dict:
        """
        [ru] Возвращает статистику популяции.
        [en] Returns population statistics.
        """
        if not self.individuals:
            return {'size': 0}

        fitnesses = [ind.fitness for ind in self.individuals if hasattr(ind, 'fitness')]

        return {
            'size': len(self.individuals),
            'max_fitness': max(fitnesses) if fitnesses else 0,
            'min_fitness': min(fitnesses) if fitnesses else 0,
            'avg_fitness': sum(fitnesses) / len(fitnesses) if fitnesses else 0,
            'alive': sum(1 for ind in self.individuals if ind.alive),
            'food_collected': sum(ind.food_collected for ind in self.individuals if hasattr(ind, 'food_collected'))
        }

    def select_best_with_diversity(self, top_k: int, diversity_weight: float = 0.3):
        """
        [ru] Выбирает лучших с учётом разнообразия.
        [en] Selects the best taking diversity into account.
        """
        if len(self.individuals) <= top_k:
            return self.individuals

        # [ru] Сортируем по фитнесу
        # [en] Sort by fitness
        sorted_inds = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)

        # [ru] Берём top_k * 2 кандидатов
        # [en] Take top_k * 2 candidates
        candidates = sorted_inds[:top_k * 2]

        # [ru] Выбираем top_k с учётом разнообразия
        # [en] Select top_k taking diversity into account
        selected = []
        for i in range(top_k):
            if not candidates:
                break

            # [ru] Берём лучшего из оставшихся
            # [en] Take the best from the remaining
            best = candidates[0]
            selected.append(best)
            candidates.pop(0)

            # [ru] Если остались кандидаты, применяем штраф за схожесть
            # [en] If candidates remain, apply a penalty for similarity
            if candidates:
                # [ru] Удаляем слишком похожих на выбранного
                # [en] Remove those too similar to the selected one
                candidates = [ind for ind in candidates
                              if abs(ind.fitness - best.fitness) > 0.1]

        # [ru] Если не хватило, добираем из оставшихся
        # [en] If not enough, pick from the remaining
        while len(selected) < top_k and sorted_inds:
            ind = sorted_inds.pop(0)
            if ind not in selected:
                selected.append(ind)

        return selected[:top_k]



# import random
# from core.individual import Individual
# from core.genome import Genome
# from typing import List, Dict, Optional, Tuple, Any, Union
#
# class Population:
#     def __init__(self, size, world, genome_template=None, start_pos=(0,0)):
#         self.size = size
#         self.world = world
#         self.individuals = []
#         self.generation = 0
#
#         for _ in range(size):
#             if genome_template:
#                 genome = genome_template.mutate(0.3)  # начальное разнообразие
#             else:
#                 genome = Genome()
#             ind = Individual(x=start_pos[0], z=start_pos[1], genome=genome)
#             self.individuals.append(ind)
#
#     def evaluate_all(self, steps_per_episode=500):
#         fitnesses = []
#         all_experiences = []  # список всех переходов за поколение
#         for ind in self.individuals:
#             self.world.reset()
#             fit = ind.evaluate(self.world, steps_per_episode)
#             fitnesses.append(fit)
#             # Получаем и очищаем буфер бота
#             exps = ind.get_experiences()
#             all_experiences.extend(exps)
#         # Теперь all_experiences можно передать в GAN
#         return fitnesses, all_experiences
#
#     def _generate_random_objects(self):
#         pass
#
#     def select_best(self, top_k):
#         """
#         Возвращает top_k особей по фитнесу.
#         """
#         sorted_inds = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
#         return sorted_inds[:top_k]
#
#     def next_generation(self, elite_count=2, mutation_rate=0.1):
#         """
#         Создаёт следующее поколение с учётом разнообразия.
#         """
#         # Используем отбор с разнообразием
#         best = self.select_best_with_diversity(elite_count * 2, diversity_weight=0.3)
#         best = best[:elite_count]  # Оставляем только elite_count
#
#         new_individuals = []
#         new_individuals.extend(best)
#
#         while len(new_individuals) < self.size:
#             parent1 = random.choice(best)
#             parent2 = random.choice(best)
#             child_genome = parent1.genome.crossover(parent2.genome)
#             child_genome = child_genome.mutate(mutation_rate)
#             child = Individual(x=0, z=0, genome=child_genome)
#             new_individuals.append(child)
#
#         self.individuals = new_individuals
#         self.generation += 1
#
#
#     def get_statistics(self) -> Dict:
#         """Возвращает статистику популяции."""
#         if not self.individuals:
#             return {'size': 0}
#
#         fitnesses = [ind.fitness for ind in self.individuals if hasattr(ind, 'fitness')]
#
#         return {
#             'size': len(self.individuals),
#             'max_fitness': max(fitnesses) if fitnesses else 0,
#             'min_fitness': min(fitnesses) if fitnesses else 0,
#             'avg_fitness': sum(fitnesses) / len(fitnesses) if fitnesses else 0,
#             'alive': sum(1 for ind in self.individuals if ind.alive),
#             'food_collected': sum(ind.food_collected for ind in self.individuals if hasattr(ind, 'food_collected'))
#         }
#
#     def select_best_with_diversity(self, top_k: int, diversity_weight: float = 0.3):
#         """
#         Выбирает лучших с учётом разнообразия.
#         """
#         if len(self.individuals) <= top_k:
#             return self.individuals
#
#         # Сортируем по фитнесу
#         sorted_inds = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
#
#         # Берём top_k * 2 кандидатов
#         candidates = sorted_inds[:top_k * 2]
#
#         # Выбираем top_k с учётом разнообразия
#         selected = []
#         for i in range(top_k):
#             if not candidates:
#                 break
#
#             # Берём лучшего из оставшихся
#             best = candidates[0]
#             selected.append(best)
#             candidates.pop(0)
#
#             # Если остались кандидаты, применяем штраф за схожесть
#             if candidates:
#                 # Удаляем слишком похожих на выбранного
#                 candidates = [ind for ind in candidates
#                               if abs(ind.fitness - best.fitness) > 0.1]
#
#         # Если не хватило, добираем из оставшихся
#         while len(selected) < top_k and sorted_inds:
#             ind = sorted_inds.pop(0)
#             if ind not in selected:
#                 selected.append(ind)
#
#         return selected[:top_k]