import numpy as np

arr = [None] * 10
arr[0] = 1113.95374479
arr[1] = 1113.95374479
arr[2] = 1120.7693138
arr[3] = 1107.98939354
arr[4] = 1129.75813254
arr[5] = 1122.17768447
arr[6] = 1108.55701674
arr[7] = 1108.55701674
arr[8] = 1113.95374479
arr[9] = 1129.75813254

avg = sum(arr) / len(arr)
print(avg)

best1 = 1083.98
best2 = 1763.07
best7 = 1423.35
best = best1
# Prozentuale Abweichung vom Durchschnitt berechnen
percentage_deviation = (avg - best) / best * 100
print(percentage_deviation)

# Standardabweichung berechnen und ausgeben
std_deviation = np.std(arr)
print(std_deviation)
