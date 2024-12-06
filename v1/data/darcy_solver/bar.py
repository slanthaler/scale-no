import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Data for 6 models and 5 problems
# models = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6']
models = ["fno", "fno_normalized"]
problems = ['64', '96', '128', '256', '512']

# Sample values: each model has a value for each problem
values = np.zeros((2,5))
values[0] = [0.06408, 0.06403, 0.06703, 0.06855, 0.06529]
values[1] = [0.04981, 0.05216, 0.05410, 0.05102, 0.04934]

# Set up the bar plot
bar_width = 0.15
fig, ax = plt.subplots(figsize=(12, 8))

# Create bars for each model
for i in range(len(models)):
    ax.bar(np.arange(len(problems)) + i * bar_width, values[:, i], width=bar_width, label=models[i])

# Set the position of bar on X axis
ax.set_xticks(np.arange(len(problems)) + bar_width * (len(models) - 1) / 2)
ax.set_xticklabels(problems)

# Adding labels and title
plt.xlabel('Problems')
plt.ylabel('Values')
plt.title('Comparison of 6 Models Across 5 Problems')
plt.legend()

# Show the plot
plt.show()
