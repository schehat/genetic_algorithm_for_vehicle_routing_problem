import numpy as np

arr = [None] * 10
arr[0] = 1464.2383740558225
arr[1] = 1489.69758456
arr[2] = 1478.57117585
arr[3] = 1483.25535365
arr[4] = 1491.58412016
arr[5] = 1488.67612537
arr[6] = 1491.58412016
arr[7] = 1496.19996868
arr[8] = 1650.61469588
arr[9] = 1490.73463477

avg = sum(arr) / len(arr)
print(avg)

best1 = 1083.98
best2 = 1763.07
best7 = 1423.35
best = best7
# Prozentuale Abweichung vom Durchschnitt berechnen
percentage_deviation = (avg - best) / best * 100
print(percentage_deviation)

# Standardabweichung berechnen und ausgeben
std_deviation = np.std(arr)
print(std_deviation)
