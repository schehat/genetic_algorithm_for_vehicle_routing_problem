import matplotlib.pyplot as plt
import numpy as np

arr = [None] * 10
arr[0] = 1472.56294013
arr[1] = 1475.53559363
arr[2] = 1453.90161343
arr[3] = 1499.69080581
arr[4] = 1477.17878866
arr[5] = 1489.35883264
arr[6] = 1493.41699829
arr[7] = 1517.47887938
arr[8] = 1501.87861121
arr[9] = 1525.52776411

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

# Data for the table
# values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
# ranks_pr01 = [2, 9, 4, 5, 1, 7, 8, 3, 6]
# ranks_pr02 = [5, 1, 2, 7, 4, 6, 2, 8, 9]
# ranks_pr07 = [7, 4, 8, 9, 3, 6, 1, 5, 2]

values = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ranks_pr01 = [6, 7, 3, 6, 2, 4, 1]
ranks_pr02 = [7, 4, 1, 5, 3, 6, 2]
ranks_pr07 = [7, 1, 3, 5, 4, 2, 6]
#
# Calculate the average ranks and the range (min and max) for each p_c
average_ranks = np.mean([ranks_pr01, ranks_pr02, ranks_pr07], axis=0)
min_ranks = np.min([ranks_pr01, ranks_pr02, ranks_pr07], axis=0)
max_ranks = np.max([ranks_pr01, ranks_pr02, ranks_pr07], axis=0)

# Create a bar chart with error bars
plt.bar(np.arange(len(values)), average_ranks, color='skyblue', width=0.2,
        yerr=[average_ranks - min_ranks, max_ranks - average_ranks], capsize=5)

# Set x-axis ticks and labels
plt.xticks(np.arange(len(values)), values)

# Adjust the line properties of the error bars
plt.errorbar(np.arange(len(values)), average_ranks, yerr=[average_ranks - min_ranks, max_ranks - average_ranks],
             fmt='none', ecolor='black', capsize=1, elinewidth=1)

# Add labels, title, and legend
plt.xlabel('$p_c$')
plt.ylabel('Durchschnittlicher Rang')
plt.title('Average Ranks of $p_c$ Values for pr02 and pr07')

# Show the plot
plt.show()

