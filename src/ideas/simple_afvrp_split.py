 # def split_single_depot_afvrp_simple(self, chromosome: ndarray, depot_i: int, customer_offset: int, depot_i_vehicle=-1) -> \
    #         Tuple[
    #             list, list, list, list, list, list, list]:
    #     depot_i_n_customers = chromosome[depot_i]
    #     if depot_i_vehicle == -1:
    #         depot_i_vehicle = depot_i
    #     vehicle_i_depot: Depot = self.ga.vrp_instance.depots[depot_i_vehicle]
    #
    #     # Shortest path containing cost
    #     p1 = [float('inf') if i > 0 else 0 for i in range(depot_i_n_customers + 1)]
    #     # Note from which node path comes from to build path
    #     pred = [0] * (depot_i_n_customers + 1)
    #     # Accumulating values for every depot. Resetting value to 0 for every next depot
    #     distance_list = [0] * (depot_i_n_customers + 1)
    #     capacity_list = [0] * (depot_i_n_customers + 1)
    #     time_list = [0] * (depot_i_n_customers + 1)
    #     time_warp_list = [0] * (depot_i_n_customers + 1)
    #     duration_list = [0] * (depot_i_n_customers + 1)
    #
    #     t_charging_or_equipment_switch = 2
    #
    #     for t in range(depot_i_n_customers):
    #         distance = 0
    #         current_capacity = 0
    #
    #         distance_depot_start = 0
    #         first_start_window = 0
    #
    #         time_i = 0
    #         sum_time_warp = 0
    #
    #         number_of_recharges = 0
    #
    #         i = t + 1
    #
    #         try:
    #             customer_value_i = chromosome[customer_offset + (i - 1)]
    #         except:
    #             print("SPLIT ERROR")
    #             break
    #         customer_i: Customer = self.ga.vrp_instance.customers[customer_value_i - 1]
    #
    #         # 2 * Capacity to allow infeasible solution for better space search
    #         while i <= depot_i_n_customers and current_capacity + customer_i.demand <= 2 * self.ga.vrp_instance.max_capacity:
    #
    #             current_capacity += customer_i.demand
    #             if i == t + 1:
    #                 distance_to_customer = self.ga.vrp_instance.graph.shortest_path_between_two_nodes(
    #                     (vehicle_i_depot.x, vehicle_i_depot.y), (customer_i.x, customer_i.y))
    #
    #                 distance = distance_to_customer
    #                 time_i = customer_i.start_time_window
    #
    #                 distance_depot_start = distance
    #                 first_start_window = customer_i.start_time_window
    #             else:
    #                 customer_value_pre_i = chromosome[customer_offset + (i - 1 - 1)]
    #                 customer_pre_i: Customer = self.ga.vrp_instance.customers[customer_value_pre_i - 1]
    #
    #                 distance_to_customer = self.ga.vrp_instance.graph.shortest_path_between_two_nodes(
    #                     (customer_pre_i.x, customer_pre_i.y), (customer_i.x, customer_i.y))
    #
    #                 # Distance from customer_i to depot or closest charging station to recharge
    #                 distance_to_recharge, charging_x, charging_y = self.get_closest_charging_point(customer_i,
    #                                                                                                vehicle_i_depot)
    #
    #                 # Check if trip to depot or charging station is not in reach and no equipment violation
    #                 if (
    #                         distance + distance_to_customer + distance_to_recharge) // self.ga.vrp_instance.max_distance != number_of_recharges and customer_pre_i.equipment == customer_i.equipment:
    #                     distance_customer_pre_to_charging = self.ga.vrp_instance.graph.shortest_path_between_two_nodes(
    #                         (customer_pre_i.x, customer_pre_i.y), (charging_x, charging_y))
    #                     distance_charging_to_customer = self.ga.vrp_instance.graph.shortest_path_between_two_nodes(
    #                         (charging_x, charging_y), (customer_i.x, customer_i.y))
    #                     distance_to_customer = distance_customer_pre_to_charging + distance_charging_to_customer
    #                     distance += distance_to_customer
    #                     number_of_recharges += 1
    #                     time_i += t_charging_or_equipment_switch
    #                     # In this condition not certain if charging in depot, so check if depot then capacity is zero
    #                     if charging_x == vehicle_i_depot.x and charging_y == vehicle_i_depot.y:
    #                         current_capacity = 0
    #                 elif customer_pre_i.equipment != customer_i.equipment:
    #                     distance_customer_pre_to_depot = self.ga.vrp_instance.graph.shortest_path_between_two_nodes(
    #                         (customer_pre_i.x, customer_pre_i.y), (vehicle_i_depot.x, vehicle_i_depot.y))
    #                     distance_depot_to_customer = self.ga.vrp_instance.graph.shortest_path_between_two_nodes(
    #                         (vehicle_i_depot.x, vehicle_i_depot.y), (customer_i.x, customer_i.y))
    #                     distance_to_customer = distance_customer_pre_to_depot + distance_depot_to_customer
    #                     distance += distance_to_customer
    #                     # Check if charging also necessary
    #                     if (
    #                             distance + distance_to_customer + distance_to_recharge) // self.ga.vrp_instance.max_distance != number_of_recharges:
    #                         number_of_recharges += 1
    #                     time_i += t_charging_or_equipment_switch
    #                     current_capacity = 0
    #                 else:
    #                     # Does not need charging and has correct equipment
    #                     distance += distance_to_customer
    #
    #                 # Late arrival => time warp
    #                 if time_i + customer_pre_i.service_duration + distance_to_customer > customer_i.end_time_window:
    #                     sum_time_warp += max(
    #                         time_i + customer_pre_i.service_duration + distance_to_customer - customer_i.end_time_window,
    #                         0)
    #                     time_i = customer_i.end_time_window
    #                 # Early arrival => wait
    #                 elif time_i + customer_pre_i.service_duration + distance_to_customer < customer_i.start_time_window:
    #                     time_i = customer_i.start_time_window
    #                 # In time window
    #                 else:
    #                     time_i += customer_pre_i.service_duration + distance_to_customer
    #
    #             distance_to_depot = self.ga.vrp_instance.graph.shortest_path_between_two_nodes(
    #                 (customer_i.x, customer_i.y), (vehicle_i_depot.x, vehicle_i_depot.y))
    #
    #             duration = distance_depot_start + time_i + sum_time_warp + customer_i.service_duration - first_start_window
    #             cost = distance \
    #                    + self.ga.duration_penalty_factor * max(0,
    #                                                            duration + distance_to_depot - self.ga.vrp_instance.max_duration_of_a_route) \
    #                    + self.ga.capacity_penalty_factor * max(0,
    #                                                            current_capacity - self.ga.vrp_instance.max_capacity) \
    #                    + self.ga.time_window_penalty * sum_time_warp
    #             # if new solution better than current then update labels
    #             if p1[t] + cost + distance_to_depot < p1[i]:
    #                 p1[i] = p1[t] + cost + distance_to_depot
    #                 pred[i] = t
    #                 distance_list[i] = distance + distance_to_depot
    #                 capacity_list[i] = current_capacity
    #                 time_list[i] = time_i
    #                 time_warp_list[i] = sum_time_warp
    #                 duration_list[i] = duration + distance_to_depot
    #
    #             i += 1
    #
    #             # Bounds check
    #             if customer_offset + (i - 1) < self.ga.vrp_instance.n_depots + self.ga.vrp_instance.n_customers:
    #                 try:
    #                     customer_value_i = chromosome[customer_offset + (i - 1)]
    #                     customer_i: Customer = self.ga.vrp_instance.customers[customer_value_i - 1]
    #                 except IndexError:
    #                     break
    #             else:
    #                 break
    #
    #     return p1, pred, distance_list, capacity_list, time_list, time_warp_list, duration_list

    # def get_closest_charging_point(self, customer_i, vehicle_i_depot):
    #     # Check if depot is in reach of charging
    #     distance_to_recharge = self.ga.vrp_instance.graph.shortest_path_between_two_nodes((customer_i.x, customer_i.y),
    #                                                                                       (vehicle_i_depot.x,
    #                                                                                        vehicle_i_depot.y))
    #     charging_x, charging_y = vehicle_i_depot.x, vehicle_i_depot.y
    #     # Check if charging station is closer than depot
    #     for charging_station in self.ga.vrp_instance.charging_stations:
    #         distance_to_charging_station = self.ga.vrp_instance.graph.shortest_path_between_two_nodes(
    #             (customer_i.x, customer_i.y), (charging_station.x, charging_station.y))
    #         if distance_to_charging_station < distance_to_recharge:
    #             distance_to_recharge = distance_to_charging_station
    #             charging_x, charging_y = charging_station.x, charging_station.y
    #
    #     return distance_to_recharge, charging_x, charging_y
    #
    # def get_closest_depot_and_charging_stations(self, customer_i):
    #     if customer_i in self.closest_depot_and_charging_stations:
    #         # If the result is already cached, return it
    #         return self.closest_depot_and_charging_stations[customer_i]
    #
    #     # set distance to first depot
    #     first_depot = self.ga.vrp_instance.depots[0]
    #     distance_recharge_to_depot = self.ga.vrp_instance.graph.shortest_path_between_two_nodes(
    #         (customer_i.x, customer_i.y), (first_depot.x, first_depot.y))
    #     depot_x, depot_y = first_depot.x, first_depot.y
    #
    #     # iterate trough rest of depot
    #     for depot in self.ga.vrp_instance.depots[1:]:
    #         distance_to_depot = self.ga.vrp_instance.graph.shortest_path_between_two_nodes(
    #             (customer_i.x, customer_i.y),
    #             (depot.x, depot.y))
    #         if distance_to_depot < distance_recharge_to_depot:
    #             distance_recharge_to_depot = distance_to_depot
    #             depot_x, depot_y = depot.x, depot.y
    #
    #     distance_to_recharge = distance_recharge_to_depot
    #     charging_x, charging_y = depot_x, depot_y
    #     # Check if charging station is closer than depot
    #     for charging_station in self.ga.vrp_instance.charging_stations:
    #         distance_to_charging_station = self.ga.vrp_instance.graph.shortest_path_between_two_nodes(
    #             (customer_i.x, customer_i.y), (charging_station.x, charging_station.y))
    #         if distance_to_charging_station < distance_to_recharge:
    #             distance_to_recharge = distance_to_charging_station
    #             charging_x, charging_y = charging_station.x, charging_station.y
    #
    #     # Cache the result before returning
    #     result = (distance_recharge_to_depot, depot_x, depot_y, distance_to_recharge, charging_x, charging_y)
    #     self.closest_depot_and_charging_stations[customer_i] = result
    #
    #     return result
