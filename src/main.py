#!/usr/bin/env python3

from timeit import default_timer as timer

import numpy as np

from chromosome import Chromosome
from crossover import Crossover
from mutation import Mutation

if __name__ == "__main__":
    mutation_rate = 1.0
    crossover_rate = 1.0
    mutation = Mutation(mutation_rate)
    crossover = Crossover(crossover_rate)
    for i in range(100):
        genes1 = np.array([2, 4, 3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])      
        chromosome1 = Chromosome(genes1)
        genes2 = np.array([3, 3, 4, 2, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])      
        chromosome2 = Chromosome(genes2)

        start = timer()
        # mutation.insertion(chromosome1)
        crossover_new1 = crossover.order(chromosome1, chromosome2)
        end = timer()
        
        #print(end - start)
        print(f"{crossover_new1.genes}")
