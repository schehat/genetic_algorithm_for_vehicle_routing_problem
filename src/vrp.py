from numpy import ndarray


class Customer:
    """
    Customer information
    """

    def __init__(self, id: int, x: float, y: float, demand: int, service_duration: int = 0,
                 start_time_window: int = 0, end_time_window: int = 0):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.service_duration = service_duration
        self.start_time_window = start_time_window
        self.end_time_window = end_time_window


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

    def __init__(self, n_vehicles: int, n_customers: int, n_depots: int, max_capacity: int,
                 customers: ndarray, depots: ndarray, max_duration_of_a_route: int = 0):
        self.n_vehicles = n_vehicles
        self.n_customers = n_customers
        self.n_depots = n_depots
        self.max_capacity = max_capacity
        self.customers = customers
        self.depots = depots
        self.max_duration_of_a_route = max_duration_of_a_route
