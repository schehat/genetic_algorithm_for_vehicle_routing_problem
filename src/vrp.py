from numpy import ndarray

from src.graph import Graph


class Customer:
    """
    Customer information
    """

    def __init__(self, id: int, x: float, y: float, service_duration: int, demand: int,
                 start_time_window: int, end_time_window: int, label: int = 0):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.service_duration = service_duration
        self.start_time_window = start_time_window
        self.end_time_window = end_time_window
        self.label = label


class Depot:
    """
    Depot information
    """

    def __init__(self, id: int, x: float, y: float, start_time_window: int, end_time_window: int, label: int = 0):
        self.id = id
        self.x = x
        self.y = y
        self.start_time_window = start_time_window
        self.end_time_window = end_time_window
        self.label = label


class ChargingStation:
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
                 customers: ndarray, depots: ndarray, max_duration_of_a_route: int, graph: Graph, charging_stations: ndarray = None, max_distance:int = 200):
        self.n_vehicles = n_vehicles
        self.n_customers = n_customers
        self.n_depots = n_depots
        self.max_capacity = max_capacity
        self.customers = customers
        self.depots = depots
        self.max_duration_of_a_route = max_duration_of_a_route
        self.graph = graph
        self.charging_stations = charging_stations
        self.max_distance = max_distance
