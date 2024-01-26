import matplotlib.pyplot as plt
import numpy as np

arr = [None] * 10
arr[0] = 5447.1405722
arr[1] = 5655.11691442
arr[2] = 5287.7333795
arr[3] = 5719.32055613
arr[4] = 5411.07339547
arr[5] = 5329.97186455
arr[6] = 5773.34734881
arr[7] = 5412.43067332
arr[8] = 5416.50703002
arr[9] = 5311.74882989

avg = sum(arr) / len(arr)
print(avg)

best1 = 1083.98
best2 = 1763.07
best3 = 2408.42
best = best3
# Prozentuale Abweichung vom Durchschnitt berechnen
percentage_deviation = (avg - best) / best * 100
print(percentage_deviation)

# Standardabweichung berechnen und ausgeben
std_deviation = np.std(arr)
print(std_deviation)

# crossover
# values = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
# ranks_pr01 = [6, 7, 2, 5, 4, 3, 1]
# ranks_pr02 = [6, 5, 7, 1, 2, 3, 4]
# ranks_pr03 = [6, 5, 7, 2, 1, 4, 3]

# mutation
# values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# ranks_pr01 = [10, 9, 8, 6, 5, 7, 3, 4, 2, 1]
# ranks_pr02 = [10, 9, 7, 8, 3, 5, 4, 1, 2, 6]
# ranks_pr03 = [10, 9, 8, 5, 1, 3, 4, 2, 6, 7]

# elitism
values = [1, 2, 4, 6, 8, 10, 12, 15]
ranks_pr01 = [8, 6, 1, 4, 2, 5, 7, 3]
ranks_pr02 = [8, 7, 6, 4, 3, 1, 5, 2]
ranks_pr03 = [8, 7, 6, 5, 4, 1, 3, 2]

# Calculate the average ranks and the range (min and max) for each p_c
average_ranks = np.mean([ranks_pr01, ranks_pr02, ranks_pr03], axis=0)
min_ranks = np.min([ranks_pr01, ranks_pr02, ranks_pr03], axis=0)
max_ranks = np.max([ranks_pr01, ranks_pr02, ranks_pr03], axis=0)

# Create a bar chart with error bars
plt.bar(np.arange(len(values)), average_ranks, color='skyblue', width=0.2,
        yerr=[average_ranks - min_ranks, max_ranks - average_ranks], capsize=5)

# Set x-axis ticks and labels
plt.xticks(np.arange(len(values)), values)

# Adjust the line properties of the error bars
plt.errorbar(np.arange(len(values)), average_ranks, yerr=[average_ranks - min_ranks, max_ranks - average_ranks],
             fmt='none', ecolor='black', capsize=1, elinewidth=1)

# Add labels, title, and legend
plt.xlabel('$n_{elite}$')
plt.ylabel('Durchschnittlicher Rang')
plt.title('Average Ranks of $p_c$ Values for pr02 and pr07')

# Show the plot
plt.show()

