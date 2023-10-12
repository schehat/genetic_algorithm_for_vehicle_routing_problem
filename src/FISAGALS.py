from typing import Callable

import numpy as np

from crossover import Crossover
from customer import Customer
from depot import Depot
from fitness_scaling import FitnessScaling
from mutation import Mutation
from vrp_instance import VRPInstance

"""
    TODO updaten
    - Fitness-scaling adaptive genetic algorithm with local search
    - Chromosome representation specific integer string consisting of two parts:
        1. Number of vehicles for each depot
        1. Number of customers for each vehicle to serve
        2. The order of customers for each vehicle to serve
        E.g. for 2 depots with 3 vehicles and 7 customers (2, 1, 2, 3, 2, 1, 2, 3, 4, 5, 6, 7)
        => first depot (index 0) has 2 vehicles, first vehicle (index 2) serves customer 1 and 2 (index 5 and 6)
        => second depot (Index 1) has 1 vehicles (index 2), serving customer 6 and 7 (index 10, 11)
"""
class FISAGALS:
    def __init__(self, vrp_instance: VRPInstance, population_size: int, crossover_rate: float, mutation_rate: float, max_generations: int, fitness_scaling):
        self.vrp_instance: VRPInstance = vrp_instance
        self.population_size = population_size
        self.crossover_rate = Crossover(self.vrp_instance, crossover_rate)
        self.mutation = Mutation(self.vrp_instance, mutation_rate)
        self.max_generations = max_generations
        self.fitness_scaling: FitnessScaling = fitness_scaling

    # Random initial population
    def generate_initial_population(self):
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

        return np.array(initial_population, dtype=object)
    
    def evaluate_fitness(self, chromosome):
        # Initialize variables to keep track of fitness (total distance)
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

            print("====================")
            print(f"i: {i}")        
            # print(f"depot_value_counter: {depot_value_counter}")
            # print(f"vehicle_index + i: {vehicle_index + i}")
            # print(f"vehicle_i_n_customers: {vehicle_i_n_customers}")

            # Check if all iterations for vehicles of current depot are done. Then continue with next depot
            if (depot_value_counter > chromosome[depot_index]):
                depot_value_counter = 0
                depot_index += 1

            vehicle_i_depot: Depot = self.vrp_instance.depots[depot_index]

            # print(f"vehicle_i_depot: {vehicle_i_depot.id}")
 
            # -1 in range because of special treatment for last customer
            for j in range(vehicle_i_n_customers - 1):
                print(f"j: {j}")
                print(f"customer_index: {customer_index + j}")
                customer_value1 = chromosome[customer_index + j]
                customer_value2 = chromosome[customer_index + j + 1]
                # Using customer_value - 1 as index to pick the right customer, due to ids start by 1 not 0
                customer_1: Customer = self.vrp_instance.customers[customer_value1 - 1]  
                customer_2: Customer = self.vrp_instance.customers[customer_value2 - 1]  

                # First departure of vehicle distance from depot to customer needed 
                if (j == 0):
                    # Add distance from depot to customer with the euclidean distance
                    # Assuming for now that customer demand is never more than vehicle capacity
                    fitness += np.linalg.norm(np.array([vehicle_i_depot.x, vehicle_i_depot.y]) - np.array([customer_1.x, customer_1.y]))
                    # TODO add capacity constraint meaning vehicles with different capacity can have customer demand which is greater then their full capacity
                    # Assuming for now that single customer demand is never more than vehicle capacity so check if capacity needed not necessary for now
                    vehicle_i_capacity += customer_1.demand

                # Check if additional customer demand exceeds vehicle capacity limit
                # TODO Add heterogenous capacity for vehicles
                if (vehicle_i_capacity + customer_2.demand > self.vrp_instance.max_capacity):
                    # if capacity limit reached then trip back to depot necessary. Assuming for now heading back to same depot it came from
                    # TODO visit different depot if possible e.g. AF-VRP charging points for robots
                    fitness += np.linalg.norm(np.array([customer_1.x, customer_1.y]) - np.array([vehicle_i_depot.x, vehicle_i_depot.y]))
                    
                    # from depot to next customer
                    fitness += np.linalg.norm(np.array([vehicle_i_depot.x, vehicle_i_depot.y]) - np.array([customer_2.x, customer_2.y]))
                    vehicle_i_capacity = 0
                else:
                    # Add distance between customers
                    fitness += np.linalg.norm(np.array([customer_1.x, customer_1.y]) - np.array([customer_2.x, customer_2.y]))

                vehicle_i_capacity += customer_2.demand

                # Last iteration in loop, add trip from last customer to depot
                if (j == vehicle_i_n_customers - 1):
                    fitness += np.linalg.norm(np.array([customer_2.x, customer_2.y]) - np.array([vehicle_i_depot.x, vehicle_i_depot.y]))
                                
                print(f"customer 1: {customer_1.id}")
                print(f"fitness: {fitness}")
                print(f"Demands customer: {customer_1.demand} und {customer_2.demand} \n")
                print(f"vehicle_i_capacity: {vehicle_i_capacity} \n")

            customer_index += vehicle_i_n_customers
            depot_value_counter += 1

        return fitness

    def run(self):
        population = self.generate_initial_population()
        print(population[0])

        #for generation in range(self.max_generations):
            # Evaluate fitness of the population
        fitness_scores = self.evaluate_fitness(population[0])
        print(fitness_scores)

        #     # Selection
        #     selected_parents = self.selection(population, fitness_scores)

        #     # Crossover
        #     children = []
        #     for i in range(0, self.population_size, 2):
        #         child1 = self.crossover(selected_parents[i], selected_parents[i + 1])
        #         child2 = self.crossover(selected_parents[i + 1], selected_parents[i])
        #         children.extend([child1, child2])

        #     # Mutation
        #     for child in children:
        #         self.mutation(child)

        #     # Replace old generation with new generation
        #     population = children

        #     # Termination criteria (you can customize this)
        #     if self.convergence_criteria_met():
        #         break

        # # Return the best solution found
        # best_solution = min(population, key=self.evaluate_fitness)
        # return best_solution