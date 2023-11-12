# for depot_index in range(self.vrp_instance.n_depots):
#     depot_i_n_customers = chromosome[depot_index]
#     # Capacity for every vehicle the same at the moment. TODO dynamic capacity with vehicle class
#     vehicle_i_capacity = 0
#     # TODO route duration is not travelled distance!!!
#     vehicle_i_travelled_distance = 0
#     vehicle_i_current_time = 0
#     vehicle_i_depot: Depot = self.vrp_instance.depots[depot_index]
#
#     for j in range(depot_i_n_customers):
#         customer_value1 = chromosome[customer_index + j]
#         # Indexing of customers starts with 1 not 0, so -1 necessary
#         customer1: Customer = self.vrp_instance.customers[customer_value1 - 1]
#
#         # First iteration in loop: first trip
#         if j == 0:
#             # Add distance from depot to customer with the euclidean distance
#             # Assuming single customer demand <= vehicle max capacity
#             # TODO add capacity constraint meaning vehicles with different capacity
#             # Thus customer demand > vehicle max capacity possible but at least 1 vehicle exists with greater capacity
#             vehicle_i_capacity += customer1.demand
#
#             # Track travelled distance in total and per vehicle to check route duration constraint
#             distance = self.euclidean_distance(vehicle_i_depot, customer1)
#             self.total_distance += distance
#             vehicle_i_travelled_distance += distance
#
#             # At the beginning vehicle time starts always with the first customer start window
#             vehicle_i_current_time = customer1.start_time_window
#
#             if purpose == purpose.PLOTTING:
#                 self.collect_routes(vehicle_i_depot, depot_index)
#                 self.collect_routes(customer1, depot_index)
#
#         # Check if next customer exists in route
#         if j < depot_i_n_customers - 1:
#             customer_value2 = chromosome[customer_index + j + 1]
#             customer2: Customer = self.vrp_instance.customers[customer_value2 - 1]
#
#             # Check customer 2 demand exceeds vehicle capacity limit
#             # TODO Add heterogeneous capacity for vehicles
#             if vehicle_i_capacity + customer2.demand > self.vrp_instance.max_capacity:
#                 # Trip back to depot necessary. Assuming heading back to same depot it came from
#                 # TODO visit different depot if possible e.g. AF-VRP charging points for robots
#
#                 # From customer 1 to depot
#                 distance1 = self.euclidean_distance(customer1, vehicle_i_depot)
#                 self.total_distance += distance1
#                 vehicle_i_travelled_distance += distance1
#
#                 # TODO LOG info vehicle_i travel and capacity... Learn to use all vehicles
#
#                 # New vehicle from depot to customer 2
#                 distance2 = self.euclidean_distance(vehicle_i_depot, customer2)
#                 self.total_distance += distance2
#                 # Reset values. Capacity for customer2.demand added later
#                 vehicle_i_capacity = 0
#                 vehicle_i_travelled_distance = distance2
#                 vehicle_i_current_time = customer2.start_time_window
#
#                 if purpose == purpose.PLOTTING:
#                     self.collect_routes(vehicle_i_depot, depot_index)
#                     self.collect_routes(customer2, depot_index)
#             else:
#                 # Add distance between customer 1 and customer 2
#                 distance = self.euclidean_distance(customer1, customer2)
#                 self.total_distance += distance
#                 vehicle_i_travelled_distance += distance
#
#                 vehicle_i_current_time += customer1.service_duration + distance
#
#                 if purpose == purpose.PLOTTING:
#                     self.collect_routes(customer2, depot_index)
#
#             vehicle_i_capacity += customer2.demand
#             # Check if vehicle reaches customer 2 before start window then needs to wait
#             if vehicle_i_current_time < customer2.start_time_window:
#                 vehicle_i_current_time = customer2.start_time_window
#             # Check if vehicle reaches customer 2 later than end time window then penalty
#             elif vehicle_i_current_time > customer2.end_time_window:
#                 self.total_timeout += vehicle_i_current_time - customer2.start_time_window
#
#         # Last iteration in loop, add trip from last customer to depot
#         if j >= depot_i_n_customers - 1:
#             # Add distance between customer 1 and customer 2
#             distance = self.euclidean_distance(customer1, vehicle_i_depot)
#             self.total_distance += distance
#             vehicle_i_travelled_distance += distance
#
#             if purpose == purpose.PLOTTING:
#                 self.collect_routes(vehicle_i_depot, depot_index)
#
#     customer_index += depot_i_n_customers


################
# p01
# self.population[0]["chromosome"] = [14, 19, 8, 9,
#                                     44, 45, 33, 15, 37, 17,
#                                     42, 19, 40, 41, 13,
#                                     25, 18, 4,
#
#                                     6, 27, 1, 32, 11, 46,
#                                     48, 8, 26, 31, 28, 22,
#                                     23, 7, 43, 24, 14,
#                                     12, 47,
#
#                                     9, 34, 30, 39, 10,
#                                     49, 5, 38,
#
#                                     35, 36, 3, 20,
#                                     21, 50, 16, 2, 29
#                                     ]

# self.population[0]["chromosome"] = [50, 50,
#                                     41, 23, 67, 39, 56,
#                                     96, 6, 89, 27, 28, 53, 58,
#                                     92, 61, 16, 86, 38, 43, 15,
#                                     57, 42, 14, 44, 91, 100, 98, 37, 97,
#                                     21, 72, 74, 75, 22, 2,
#                                     73, 4, 25, 55, 54, 12, 26, 40,
#                                     59, 99, 93, 85,
#                                     87, 95, 94, 13,
#
#                                     69, 76, 77, 68, 80, 24, 29, 3,
#                                     51, 9, 35, 71, 65, 66, 70,
#                                     8, 45, 17, 84, 5, 60, 83, 18, 52,
#                                     63, 64, 49, 36, 47, 46, 82,
#                                     33, 81, 34, 78, 79, 50, 1,
#                                     88, 7, 48, 19, 11, 62,
#                                     31, 10, 90, 32, 20, 30
#                                     ]


# def split(self, chromosome: ndarray) -> Tuple[ndarray, ndarray]:
#         # TODO parallel
#         customer_index = self.vrp_instance.n_depots
#         customer_index_list = []
#         for depot_i in range(self.vrp_instance.n_depots):
#             customer_index += chromosome[depot_i]
#             customer_index_list.append(customer_index)
#
#         p_complete = np.array([], dtype=int)
#         pred_complete = np.array([], dtype=int)
#
#         for x, depot_i in enumerate(range(self.vrp_instance.n_depots)):
#             depot_i_n_customers = chromosome[depot_i]
#             vehicle_i_depot: Depot = self.vrp_instance.depots[depot_i]
#
#             p = np.full(depot_i_n_customers + 1, np.inf)
#             p[0] = 0
#             pred = np.zeros(depot_i_n_customers + 1, dtype=int)
#
#             for t in range(depot_i_n_customers):
#                 load = 0
#                 i = t + 1
#
#                 customer_value_i = chromosome[customer_index_list[x] + (i - 1)]
#                 # Indexing of customers starts with 1 not 0, so -1 necessary
#                 customer_i: Customer = self.vrp_instance.customers[customer_value_i - 1]
#
#                 while i <= depot_i_n_customers and load + customer_i.demand <= self.vrp_instance.max_capacity:
#
#                     load += customer_i.demand
#                     if i == t + 1:
#                         distance = self.euclidean_distance(vehicle_i_depot, customer_i)
#                         cost = distance
#                     else:
#                         customer_value_pre_i = chromosome[customer_index_list[x] + (i - 1 - 1)]
#                         customer_pre_i: Customer = self.vrp_instance.customers[customer_value_pre_i - 1]
#                         distance = self.euclidean_distance(customer_pre_i, customer_i)
#                         cost += distance
#
#                     distance = self.euclidean_distance(customer_i, vehicle_i_depot)
#                     if p[t] + cost + distance < p[i]:
#                         p[i] = p[t] + cost + distance
#                         pred[i] = t
#
#                     i += 1
#
#                     # Bounds check
#                     if customer_index_list[x] + (i - 1) < self.vrp_instance.n_depots + self.vrp_instance.n_customers:
#                         customer_value_i = chromosome[customer_index_list[x] + (i - 1)]
#                         customer_i: Customer = self.vrp_instance.customers[customer_value_i - 1]
#                     else:
#                         break
#
#             p_complete = np.concatenate((p_complete, p))
#             pred_complete = np.concatenate((pred_complete, pred))
#
#         return p_complete, pred_complete
