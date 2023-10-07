import numpy as np


class Chromosome:
    N_DEPOTS = 2
    N_VEHICLES_PER_DEPOT = 2
    N_CUSTOMERS = 12

    def __init__(self, genes: np.ndarray):
        self.genes = genes
