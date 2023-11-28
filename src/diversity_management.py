import numpy as np

from src.initial_population import initial_population_random


class DiversityManagement:
    """
    Serves as a local search mechanism superior to random mutation
    """

    def __init__(self, ga: "GA"):
        self.ga = ga

    # TODO: same values same ranks?
    def calculate_biased_fitness(self):
        self.calculate_diversity_contribution()

        fitness_values = self.ga.population["fitness"]
        diversity_control_values = self.ga.population["diversity_contribution"]

        # Calculate the ranks
        fitness_indexes = np.argsort(fitness_values)
        # Higher distance measurement for diversity contributes to lower ranks. Therefore, reverse array
        diversity_contribution_indexes = np.argsort(diversity_control_values)[::-1]

        # Initialize arrays to store the ranked values
        fitness_ranked = np.zeros_like(fitness_indexes)
        diversity_contribution_ranked = np.zeros_like(diversity_contribution_indexes)

        # Assign ranks to fitness_ranked and diversity_contribution_ranked. Start ranks from 1
        fitness_ranked[fitness_indexes] = np.arange(1, len(fitness_indexes) + 1)
        diversity_contribution_ranked[diversity_contribution_indexes] = np.arange(1, len(diversity_contribution_indexes) + 1)

        # Now you can use fitness_ranked and diversity_contribution_ranked to calculate biased_fitness
        biased_fitness = fitness_ranked + self.ga.diversity_weight * diversity_contribution_ranked

        # Update the population array with the new values
        self.ga.population["fitness_ranked"] = fitness_ranked
        self.ga.population["diversity_contribution_ranked"] = diversity_contribution_ranked
        self.ga.population["biased_fitness"] = biased_fitness

    def calculate_diversity_contribution(self):
        for i, chromosome_a in enumerate(self.ga.population["chromosome"]):
            distances = []
            for j, chromosome_b in enumerate(self.ga.population["chromosome"]):
                # Avoid calculating distance with the same chromosome
                if i != j:
                    distance = self.ga.distance_method(chromosome_a, chromosome_b, self.ga.vrp_instance.n_depots)
                    distances.append((j, distance))

            # Sort distances and pick n_closest_neighbors
            distances.sort(key=lambda x: x[1])
            n_closest_neighbors = distances[:self.ga.n_closest_neighbors]
            # Calculate the average distance of the n_closest_neighbors
            avg_distance = np.mean([dist for _, dist in n_closest_neighbors])
            # Set diversity_contribution
            self.ga.population[i]["diversity_contribution"] = avg_distance

    def diversity_procedure(self):
        print("DIVERSITY PROCEDURE")
        self.ga.diversify_counter = 0
        self.ga.fitness_evaluation()
        self.ga.population.sort(order='fitness')
        # Determine the number of individuals to keep
        num_to_keep = int(self.ga.p_diversify_survival * len(self.ga.population))
        initial_population_random(self.ga, num_to_keep, self.ga.population_size)
        self.ga.fitness_evaluation()
        self.calculate_biased_fitness()

    def survivor_selection(self):
        print("SURVIVOR SELECTION")
        unique_fitness = set()
        unique_individuals = []
        clones = []

        for individual in self.ga.population:
            fitness_value = individual["fitness"]
            diversity_contribution = individual["diversity_contribution"]

            if diversity_contribution != 0 and fitness_value not in unique_fitness:
                unique_individuals.append(individual)
                unique_fitness.add(fitness_value)
            else:
                clones.append(individual.copy())

        clones = sorted(clones, key=lambda x: x["biased_fitness"])
        unique_individuals = sorted(unique_individuals, key=lambda x: x["biased_fitness"])
        # List of individuals sorted by biased_fitness, clones at the end
        unique_individuals.extend(clones)

        num_to_keep = int(self.ga.p_selection_survival * len(self.ga.population))
        # ga.population will be replaced with random individuals
        initial_population_random(self.ga, num_to_keep, self.ga.population_size)

        # Keep certain percentage of current generation individuals. Clones will be removed first then the rest according to there biased fitness
        self.ga.population[:num_to_keep] = np.array(unique_individuals[:num_to_keep], dtype=self.ga.population.dtype)

        # Fitness evaluation updating for the randoms
        self.ga.fitness_evaluation()
        self.calculate_biased_fitness()
