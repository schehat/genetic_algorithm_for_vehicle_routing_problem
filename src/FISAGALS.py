from typing import Callable

import numpy as np
from numpy import ndarray

from crossover import Crossover
from customer import Customer
from depot import Depot
from mutation import Mutation
from vrp_instance import VRPInstance


class FISAGALS:
    """
    - Fitness-scaling adaptive genetic algorithm with local search
    - Chromosome representation specific integer string consisting of three parts:
        1. Number of vehicles for each depot
        1. Number of customers for each vehicle to serve
        2. The order of customers for each vehicle to serve
        E.g. for 2 depots with 3 vehicles and 7 customers (2, 1, 2, 3, 2, 1, 2, 3, 4, 5, 6, 7)
        => first depot (index 0) has 2 vehicles, first vehicle (index 2) serves customer 1 and 2 (index 5 and 6)
        => second depot (Index 1) has 1 vehicles (index 2), serving customer 6 and 7 (index 10, 11)
    """

    def __init__(self, vrp_instance: VRPInstance, population_size: int, crossover_rate: float, mutation_rate: float,
                 max_generations: int, fitness_scaling: Callable[[ndarray], ndarray], selection_method: Callable[[ndarray, ndarray, int], ndarray]):
        self.vrp_instance: VRPInstance = vrp_instance
        self.population_size = population_size
        self.crossover = Crossover(self.vrp_instance, crossover_rate)
        self.mutation = Mutation(self.vrp_instance, mutation_rate)
        self.max_generations = max_generations
        self.fitness_scaling: Callable[[ndarray], ndarray] = fitness_scaling
        self.selection_method: Callable[[ndarray, ndarray, int], ndarray] = selection_method

    def generate_initial_population(self) -> ndarray:
        """
        Random initial population
        return: 2D array with all chromosome in the population
        """

        initial_population = []
        for _ in range(self.population_size):
            # Part 1: Number of vehicles for each depot
            depot_vehicle_count = np.zeros(self.vrp_instance.n_depots, dtype=int)
            for _ in range(self.vrp_instance.n_vehicles):
                depot_index = np.random.randint(self.vrp_instance.n_depots)
                depot_vehicle_count[depot_index] += 1

            # Part 2: Number of customers for each vehicle
            vehicle_customer_count = np.zeros(self.vrp_instance.n_vehicles, dtype=int)
            total_customers_assigned = 0
            avg_customers_per_vehicle = self.vrp_instance.n_customers / self.vrp_instance.n_vehicles
            std_deviation = 1.0

            # One additional loop to guarantee all customers are assigned to vehicles
            for i in range(self.vrp_instance.n_vehicles + 1):
                # Calculate the maximum number of customers that can be assigned to this vehicle
                max_customers = self.vrp_instance.n_customers - total_customers_assigned
                if max_customers < 1:
                    break

                # Excluding the additional loop
                if i < self.vrp_instance.n_vehicles:
                    # Generate a random number of customers for this vehicle using a Gaussian distribution
                    # centered around the avg_customers_per_vehicle
                    num_customers = int(np.random.normal(loc=avg_customers_per_vehicle, scale=std_deviation))
                    # Ensure it's within valid bounds
                    num_customers = max(1, min(max_customers, num_customers))
                    vehicle_customer_count[i] = num_customers
                else:
                    # If all vehicles assigned and customers remain, assign the rest to random vehicle
                    num_customers = max_customers
                    i = np.random.randint(self.vrp_instance.n_vehicles)
                    vehicle_customer_count[i] += num_customers

                total_customers_assigned += num_customers

            # Part 3: Random order of customers for each vehicle
            order_of_customers = np.random.permutation(np.arange(1, self.vrp_instance.n_customers + 1))

            # Combine the three parts to form a chromosome
            chromosome = np.concatenate((depot_vehicle_count, vehicle_customer_count, order_of_customers))
            initial_population.append(chromosome)

        return np.array(initial_population, dtype=int)

    def evaluate_fitness(self, chromosome: ndarray) -> float:
        """
        Fitness evaluation of a single chromosome
        param: chromosome 1D array
        return: fitness value
        """

        # fitness: total distance
        fitness = 0.0
        depot_index = 0
        vehicle_index = self.vrp_instance.n_depots
        customer_index = self.vrp_instance.n_depots + self.vrp_instance.n_vehicles
        # keep track of iterations of a depot
        depot_value_counter = 1

        for i in range(self.vrp_instance.n_vehicles):
            vehicle_i_n_customers = chromosome[vehicle_index + i]
            # Capacity for every vehicle the same at the moment. TODO dynamic capacity which vehicle class
            vehicle_i_capacity = 0

            # Check if all iterations for vehicles of current depot are done. Then continue with next depot
            if depot_value_counter > chromosome[depot_index]:
                depot_value_counter = 0
                depot_index += 1

            vehicle_i_depot: Depot = self.vrp_instance.depots[depot_index]

            for j in range(vehicle_i_n_customers):
                customer_value1 = chromosome[customer_index + j]
                # Indexing of customers starts with 1 not 0, so -1 necessary
                customer_1: Customer = self.vrp_instance.customers[customer_value1 - 1]

                # First iteration in loop: first trip 
                if j == 0:
                    # Add distance from depot to customer with the euclidean distance.
                    # Assuming single customer demand <= vehicle max capacity
                    fitness += np.linalg.norm(
                        np.array([vehicle_i_depot.x, vehicle_i_depot.y]) - np.array([customer_1.x, customer_1.y]))

                    # TODO add capacity constraint meaning vehicles with different capacity
                    # Thus customer demand > vehicle max capacity possible but at least 1 vehicle exists with greater capacity
                    vehicle_i_capacity += customer_1.demand

                # Check if next customer exists in route exists
                if j < vehicle_i_n_customers - 1:
                    customer_value2 = chromosome[customer_index + j + 1]
                    customer_2: Customer = self.vrp_instance.customers[customer_value2 - 1]

                    # Check customer_2 demand exceeds vehicle capacity limit
                    # TODO Add heterogeneous capacity for vehicles
                    if vehicle_i_capacity + customer_2.demand > self.vrp_instance.max_capacity:
                        # Trip back to depot necessary. Assuming heading back to same depot it came from
                        # TODO visit different depot if possible e.g. AF-VRP charging points for robots
                        fitness += np.linalg.norm(
                            np.array([customer_1.x, customer_1.y]) - np.array([vehicle_i_depot.x, vehicle_i_depot.y]))

                        # from depot to next customer
                        fitness += np.linalg.norm(
                            np.array([vehicle_i_depot.x, vehicle_i_depot.y]) - np.array([customer_2.x, customer_2.y]))
                        vehicle_i_capacity = 0
                    else:
                        # Add distance between customers
                        fitness += np.linalg.norm(
                            np.array([customer_1.x, customer_1.y]) - np.array([customer_2.x, customer_2.y]))

                    vehicle_i_capacity += customer_2.demand

                # Last iteration in loop, add trip from last customer to depot
                if j >= vehicle_i_n_customers - 1:
                    fitness += np.linalg.norm(
                        np.array([customer_1.x, customer_1.y]) - np.array([vehicle_i_depot.x, vehicle_i_depot.y]))

            customer_index += vehicle_i_n_customers
            depot_value_counter += 1

        return fitness

    def run(self):
        """
        Execution of FISAGALS
        """

        population = self.generate_initial_population()

        for generation in range(self.max_generations):
            # Fitness evaluation and scaling
            fitness_scores = np.array([(int(i), self.evaluate_fitness(chromosome)) for i, chromosome in enumerate(population)],
                                      dtype=np.dtype([('index', int), ('fitness', float)]))
            self.fitness_scaling(fitness_scores)

            # Parent selection
            selected_parents = self.selection_method(population, fitness_scores, 5)

            # Crossover
            children = np.empty((self.population_size, self.vrp_instance.n_depots + self.vrp_instance.n_vehicles + self.vrp_instance.n_customers), dtype=population.dtype)
            for i in range(0, self.population_size, 2):
                # Generate children, second child by swapping parents
                children[i] = self.crossover.order(self.crossover.uniform(selected_parents[i], selected_parents[i + 1]), selected_parents[i + 1])
                children[i + 1] = self.crossover.order(self.crossover.uniform(selected_parents[i + 1], selected_parents[i]), selected_parents[i])

            # Mutation TODO check if mutation works
            for i in range(0, self.population_size):
                children[i] = self.mutation.uniform(children[i])
                children[i] = self.mutation.insertion(self.mutation.inversion(self.mutation.swap(children[i])))

            # population = np.copy(children)

        # print(selected_parents)

        #     # Replace old generation with new generation
        #     population = children

        #     # Termination criteria (you can customize this)
        #     if self.convergence_criteria_met():
        #         break

        # # Return the best solution found
        # best_solution = min(population, key=self.evaluate_fitness)
        # return best_solution

