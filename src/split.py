from typing import Tuple

import numpy as np
from numpy import ndarray

from src.enums import Problem
from src.utility import set_customer_index_list
from src.vrp import Depot, Customer


class Split:
    def __init__(self, ga: "GA"):
        self.ga = ga
        if self.ga.problem_type == Problem.MDVRPTW:
            self.split_depot = self.split_single_depot
        elif self.ga.problem_type == Problem.AFVRP:
            self.split_depot = self.split_single_depot_afvrp

    def split(self, chromosome: ndarray, ) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
        # Determine indices for chromosome "splitting"
        customer_index_list = set_customer_index_list(self.ga.vrp_instance.n_depots, chromosome)

        # Initial list gets appended with lists of single depot split
        p_complete = []
        pred_complete = []
        distance_complete = []
        capacity_complete = []
        time_complete = []
        time_warp_complete = []
        duration_complete = []

        for x, depot_i in enumerate(range(self.ga.vrp_instance.n_depots)):
            p, pred, distance_list, capacity_list, time_list, time_warp_list, duration_list = self.split_depot(chromosome, depot_i, customer_index_list[x])
            p_complete += p
            pred_complete += pred
            distance_complete += distance_list
            capacity_complete += capacity_list
            time_complete += time_list
            time_warp_complete += time_warp_list
            duration_complete += duration_list

        return np.array(p_complete), np.array(pred_complete), np.array(distance_complete), np.array(capacity_complete), np.array(time_complete), np.array(time_warp_complete), np.array(duration_complete)

    def split_single_depot(self, chromosome: ndarray, depot_i: int, customer_offset: int, depot_i_vehicle=-1) -> Tuple[
        list, list, list, list, list, list, list]:
        depot_i_n_customers = chromosome[depot_i]
        if depot_i_vehicle == -1:
            depot_i_vehicle = depot_i
        vehicle_i_depot: Depot = self.ga.vrp_instance.depots[depot_i_vehicle]

        # Shortest path containing cost
        p1 = [float('inf') if i > 0 else 0 for i in range(depot_i_n_customers + 1)]
        # Note from which node path comes from to build path
        pred = [0] * (depot_i_n_customers + 1)
        # Accumulating values for every depot. Resetting value to 0 for every next depot
        distance_list = [0] * (depot_i_n_customers + 1)
        capacity_list = [0] * (depot_i_n_customers + 1)
        time_list = [0] * (depot_i_n_customers + 1)
        time_warp_list = [0] * (depot_i_n_customers + 1)
        duration_list = [0] * (depot_i_n_customers + 1)

        for t in range(depot_i_n_customers):
            distance = 0
            current_capacity = 0

            distance_depot_start = 0
            first_start_window = 0

            time_i = 0
            sum_time_warp = 0

            i = t + 1

            try:
                customer_value_i = chromosome[customer_offset + (i - 1)]
            except:
                print("SPLIT ERROR")
                break
            customer_i: Customer = self.ga.vrp_instance.customers[customer_value_i - 1]

            # 2 * Capacity to allow infeasible solution for better space search
            while i <= depot_i_n_customers and current_capacity + customer_i.demand <= 2 * self.ga.vrp_instance.max_capacity:

                current_capacity += customer_i.demand
                if i == t + 1:
                    distance_to_customer = self.ga.euclidean_distance.euclidean_distance((vehicle_i_depot.x, vehicle_i_depot.y), (customer_i.x, customer_i.y))

                    distance = distance_to_customer
                    time_i = customer_i.start_time_window

                    distance_depot_start = distance
                    first_start_window = customer_i.start_time_window
                else:
                    customer_value_pre_i = chromosome[customer_offset + (i - 1 - 1)]
                    customer_pre_i: Customer = self.ga.vrp_instance.customers[customer_value_pre_i - 1]

                    distance_to_customer = self.ga.euclidean_distance.euclidean_distance((customer_pre_i.x, customer_pre_i.y), (customer_i.x, customer_i.y))

                    distance += distance_to_customer

                    # Late arrival => time warp
                    if time_i + customer_pre_i.service_duration + distance_to_customer > customer_i.end_time_window:
                        sum_time_warp += max(
                            time_i + customer_pre_i.service_duration + distance_to_customer - customer_i.end_time_window,
                            0)
                        time_i = customer_i.end_time_window
                    # Early arrival => wait
                    elif time_i + customer_pre_i.service_duration + distance_to_customer < customer_i.start_time_window:
                        time_i = customer_i.start_time_window
                    # In time window
                    else:
                        time_i += customer_pre_i.service_duration + distance_to_customer

                distance_to_depot = self.ga.euclidean_distance.euclidean_distance((customer_i.x, customer_i.y), (vehicle_i_depot.x, vehicle_i_depot.y))

                duration = distance_depot_start + time_i + sum_time_warp + customer_i.service_duration - first_start_window
                cost = distance \
                       + self.ga.duration_penalty_factor * max(0,
                                                               duration + distance_to_depot - self.ga.vrp_instance.max_duration_of_a_route) \
                       + self.ga.capacity_penalty_factor * max(0,
                                                               current_capacity - self.ga.vrp_instance.max_capacity) \
                       + self.ga.time_window_penalty * sum_time_warp
                # if new solution better than current then update labels
                if p1[t] + cost + distance_to_depot < p1[i]:
                    p1[i] = p1[t] + cost + distance_to_depot
                    pred[i] = t
                    distance_list[i] = distance + distance_to_depot
                    capacity_list[i] = current_capacity
                    time_list[i] = time_i
                    time_warp_list[i] = sum_time_warp
                    duration_list[i] = duration + distance_to_depot

                i += 1

                # Bounds check
                if customer_offset + (i - 1) < self.ga.vrp_instance.n_depots + self.ga.vrp_instance.n_customers:
                    try:
                        customer_value_i = chromosome[customer_offset + (i - 1)]
                        customer_i: Customer = self.ga.vrp_instance.customers[customer_value_i - 1]
                    except IndexError:
                        break
                else:
                    break

        return p1, pred, distance_list, capacity_list, time_list, time_warp_list, duration_list

    def split_single_depot_afvrp(self, chromosome: ndarray, depot_i: int, customer_offset: int, depot_i_vehicle=-1) -> Tuple[
        list, list, list, list, list, list, list]:
        depot_i_n_customers = chromosome[depot_i]
        if depot_i_vehicle == -1:
            depot_i_vehicle = depot_i
        vehicle_i_depot: Depot = self.ga.vrp_instance.depots[depot_i_vehicle]

        # Shortest path containing cost
        p1 = [float('inf') if i > 0 else 0 for i in range(depot_i_n_customers + 1)]
        # Note from which node path comes from to build path
        pred = [0] * (depot_i_n_customers + 1)
        # Accumulating values for every depot. Resetting value to 0 for every next depot
        distance_list = [0] * (depot_i_n_customers + 1)
        capacity_list = [0] * (depot_i_n_customers + 1)
        time_list = [0] * (depot_i_n_customers + 1)
        time_warp_list = [0] * (depot_i_n_customers + 1)
        duration_list = [0] * (depot_i_n_customers + 1)

        for t in range(depot_i_n_customers):
            distance = 0
            current_capacity = 0

            distance_depot_start = 0
            first_start_window = 0

            time_i = 0
            sum_time_warp = 0

            i = t + 1

            try:
                customer_value_i = chromosome[customer_offset + (i - 1)]
            except:
                print("SPLIT ERROR")
                break
            customer_i: Customer = self.ga.vrp_instance.customers[customer_value_i - 1]

            # 2 * Capacity to allow infeasible solution for better space search
            while i <= depot_i_n_customers and current_capacity + customer_i.demand <= 2 * self.ga.vrp_instance.max_capacity:

                current_capacity += customer_i.demand
                if i == t + 1:
                    distance_to_customer = self.ga.vrp_instance.graph.shortest_path_between_two_nodes((vehicle_i_depot.x, vehicle_i_depot.y), (customer_i.x, customer_i.y))

                    distance = distance_to_customer
                    time_i = customer_i.start_time_window

                    distance_depot_start = distance
                    first_start_window = customer_i.start_time_window
                else:
                    customer_value_pre_i = chromosome[customer_offset + (i - 1 - 1)]
                    customer_pre_i: Customer = self.ga.vrp_instance.customers[customer_value_pre_i - 1]

                    distance_to_customer = self.ga.vrp_instance.graph.shortest_path_between_two_nodes((customer_pre_i.x, customer_pre_i.y), (customer_i.x, customer_i.y))

                    distance += distance_to_customer

                    # Late arrival => time warp
                    if time_i + customer_pre_i.service_duration + distance_to_customer > customer_i.end_time_window:
                        sum_time_warp += max(
                            time_i + customer_pre_i.service_duration + distance_to_customer - customer_i.end_time_window,
                            0)
                        time_i = customer_i.end_time_window
                    # Early arrival => wait
                    elif time_i + customer_pre_i.service_duration + distance_to_customer < customer_i.start_time_window:
                        time_i = customer_i.start_time_window
                    # In time window
                    else:
                        time_i += customer_pre_i.service_duration + distance_to_customer

                distance_to_depot = self.ga.vrp_instance.graph.shortest_path_between_two_nodes((customer_i.x, customer_i.y), (vehicle_i_depot.x, vehicle_i_depot.y))

                duration = distance_depot_start + time_i + sum_time_warp + customer_i.service_duration - first_start_window
                cost = distance \
                       + self.ga.duration_penalty_factor * max(0,
                                                               duration + distance_to_depot - self.ga.vrp_instance.max_duration_of_a_route) \
                       + self.ga.capacity_penalty_factor * max(0,
                                                               current_capacity - self.ga.vrp_instance.max_capacity) \
                       + self.ga.time_window_penalty * sum_time_warp
                # if new solution better than current then update labels
                if p1[t] + cost + distance_to_depot < p1[i]:
                    p1[i] = p1[t] + cost + distance_to_depot
                    pred[i] = t
                    distance_list[i] = distance + distance_to_depot
                    capacity_list[i] = current_capacity
                    time_list[i] = time_i
                    time_warp_list[i] = sum_time_warp
                    duration_list[i] = duration + distance_to_depot

                i += 1

                # Bounds check
                if customer_offset + (i - 1) < self.ga.vrp_instance.n_depots + self.ga.vrp_instance.n_customers:
                    try:
                        customer_value_i = chromosome[customer_offset + (i - 1)]
                        customer_i: Customer = self.ga.vrp_instance.customers[customer_value_i - 1]
                    except IndexError:
                        break
                else:
                    break

        return p1, pred, distance_list, capacity_list, time_list, time_warp_list, duration_list
