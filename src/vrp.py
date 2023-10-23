from numpy import ndarray


class Customer:
    """
    Customer information
    """

    def __init__(self, id: int, x: float, y: float, demand: int):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand


class Depot:
    """
    Depot information
    """

    def __init__(self, id: int, x: float, y: float):
        self.id = id
        self.x = x
        self.y = y


class VRPInstance:
    """
    VRP Instance representation providing all the necessary information excluding specific parameters for the genetic algorithm
    """

    def __init__(self, n_vehicles: int, n_customers: int, n_depots: int, max_capacity: int, customers: ndarray,
                 depots: ndarray):
        self.n_vehicles = n_vehicles
        self.n_customers = n_customers
        self.n_depots = n_depots
        self.max_capacity = max_capacity
        self.customers = customers
        self.depots = depots
