import numpy as np

arr = [None] * 10
arr[0] = 1114.9845679
arr[1] = 1129.75813254
arr[2] = 1124.36140449
arr[3] = 1129.75813254
arr[4] = 1113.95374479
arr[5] = 1113.95374479
arr[6] = 1124.36140449
arr[7] = 1113.95374479
arr[8] = 1129.94849918
arr[9] = 1129.75813254

avg = sum(arr) / len(arr)
print(avg)

best = 1083.98
# Prozentuale Abweichung vom Durchschnitt berechnen
percentage_deviation = (avg - best) / best * 100
print(percentage_deviation)

# Standardabweichung berechnen und ausgeben
std_deviation = np.std(arr)
print(std_deviation)
