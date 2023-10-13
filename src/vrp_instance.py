from numpy import ndarray


class VRPInstance:
    """
    VRP Instance representation for testing the genetic algorithm by providing all the necessary information excluding
    specific parameters for the genetic algorithm
    """

    def __init__(self, n_vehicles: int, n_customers: int, n_depots: int, max_capacity: int, customers: ndarray, depots: ndarray):
        self.n_vehicles = n_vehicles
        self.n_customers = n_customers
        self.n_depots = n_depots
        self.max_capacity = max_capacity
        self.customers = customers
        self.depots = depots
