def FISAGALS(population_size, max_generations):
    # Initialize population
    population = [Chromosome(np.random.permutation(np.arange(1, Chromosome.N_CUSTOMERS + 1))) for _ in range(population_size)]

    for generation in range(max_generations):
        # Evaluate fitness of the population
        fitness_scores = [evaluate_fitness(chromosome) for chromosome in population]

        # Selection
        selected_parents = selection(population, fitness_scores)

        # Crossover
        children = []
        for i in range(0, population_size, 2):
            child1 = crossover.uniform(selected_parents[i], selected_parents[i + 1])
            child2 = crossover.uniform(selected_parents[i + 1], selected_parents[i])
            children.extend([child1, child2])

        # Mutation
        for child in children:
            mutation.uniform(child)

        # Repair procedure
        for child in children:
            mutation.repair_procedure(child)

        # Replace old generation with new generation
        population = children

        # Termination criteria (you can add your own)
        if convergence_criteria_met():
            break

    # Return the best solution found
    best_solution = min(population, key=evaluate_fitness)
    return best_solution