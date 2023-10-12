from numpy import ndarray

from customer import Customer
from depot import Depot


class VRPInstance:
    def __init__(self, n_vehicles: int, n_customers: int, n_depots: int, max_capacity: int, customers: ndarray, depots: ndarray):
        self.n_vehicles = n_vehicles
        self.n_customers = n_customers
        self.n_depots = n_depots
        self.max_capacity = max_capacity
        self.customers = customers
        self.depots = depots
