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
