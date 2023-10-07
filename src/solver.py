#!/usr/bin/env python3
#Todo
import math
import random


class Customer:
    def __init__(self, customer_id, x, y, demand):
        self.customer_id = customer_id
        self.x = x
        self.y = y
        self.demand = demand

    def distance_to(self, other_customer):
        return math.sqrt((self.x - other_customer.x)**2 + (self.y - other_customer.y)**2)

class Vehicle:
    def __init__(self, capacity):
        self.capacity = capacity
        self.route = []

class VRPInstance:
    def __init__(self, num_vehicles, customers, vehicle_capacity):
        self.num_vehicles = num_vehicles
        self.customers = customers
        self.vehicle_capacity = vehicle_capacity

    def calculate_distance(self, customer1, customer2):
        return customer1.distance_to(customer2)

class VRPSolver:
    def __init__(self, vrp_instance, population_size, num_generations):
        self.vrp_instance = vrp_instance
        self.population_size = population_size
        self.num_generations = num_generations
        self.population = []

    def generate_initial_population(self):
        for _ in range(self.population_size):
            random.shuffle(self.vrp_instance.customers)
            vehicles = [Vehicle(self.vrp_instance.vehicle_capacity) for _ in range(self.vrp_instance.num_vehicles)]
            for customer in self.vrp_instance.customers:
                added = False
                for vehicle in vehicles:
                    if self._can_add_customer(vehicle, customer):
                        vehicle.route.append(customer)
                        added = True
                        break
                if not added:
                    print("Warning: Customer could not be added to any vehicle")

            self.population.append(vehicles)

    def _can_add_customer(self, vehicle, customer):
        return sum(c.demand for c in vehicle.route) + customer.demand <= vehicle.capacity

    def fitness(self, solution):
        total_distance = 0
        for vehicle in solution:
            if vehicle.route:
                total_distance += self.vrp_instance.calculate_distance(
                    self.vrp_instance.customers[0], vehicle.route[0]
                )  # Distance from depot to first customer
                for i in range(len(vehicle.route) - 1):
                    total_distance += self.vrp_instance.calculate_distance(
                        vehicle.route[i], vehicle.route[i + 1]
                    )
                total_distance += self.vrp_instance.calculate_distance(
                    vehicle.route[-1], self.vrp_instance.customers[0]
                )  # Distance from last customer to depot
        return total_distance

    def select_parents(self):
        # Tournament selection to select two parents
        tournament_size = 3
        selected_parents = []
        for _ in range(2):
            participants = random.sample(self.population, tournament_size)
            selected = min(participants, key=lambda x: self.fitness(x))
            selected_parents.append(selected)
        return selected_parents

    def crossover(self, parent1, parent2):
        # Apply ordered crossover to create a child
        # (You can use other crossover methods as well)
        child1 = []
        child2 = []
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child1.extend(parent1[start:end + 1])
        child2.extend(parent2[start:end + 1])
        
        # Add missing customers from parent2 to child1 and vice versa
        for customer in parent2:
            if customer not in child1:
                child1.append(customer)
        for customer in parent1:
            if customer not in child2:
                child2.append(customer)

        return [child1, child2]


    def mutate(self, solution):
        # Apply mutation (swap two customers in a route)
        if not solution:
            return solution
        vehicle = random.choice(solution)
        if len(vehicle.route) < 2:
            return solution
        i, j = random.sample(range(len(vehicle.route)), 2)
        vehicle.route[i], vehicle.route[j] = vehicle.route[j], vehicle.route[i]
        return solution

    def genetic_algorithm(self):
        self.generate_initial_population()
        for _ in range(self.num_generations):
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents()
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            self.population = new_population
        best_solution = min(self.population, key=lambda x: self.fitness(x))
        return best_solution

if __name__ == "__main__":
    # Create a VRP instance
    num_customers = 20
    num_vehicles = 5
    max_demand = 10
    capacity = 50
    customers = [
        Customer(i, random.uniform(0, 100), random.uniform(0, 100), random.randint(1, max_demand))
        for i in range(num_customers)
    ]
    vrp_instance = VRPInstance(num_vehicles, customers, capacity)

    # Run the genetic algorithm
    population_size = 50
    num_generations = 100
    solver = VRPSolver(vrp_instance, population_size, num_generations)
    best_solution = solver.genetic_algorithm()

    # Print the best solution found
    total_distance = solver.fitness(best_solution)
    print(f"Total Distance: {total_distance}")
    for i, vehicle in enumerate(best_solution):
        if vehicle.route:
            route_distance = solver.fitness([vehicle])
            print(f"Route {i + 1}: {', '.join(str(customer.customer_id) for customer in vehicle.route)} (Distance: {route_distance})")
