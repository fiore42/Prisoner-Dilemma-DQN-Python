import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Load the CSV data into a pandas DataFrame
# df = pd.read_csv('./lr_gamma_delta.csv')
# df = pd.read_csv('./exp_lr_0_0.2_gamma_0_1_temp.out', delimiter=', ')
# df = pd.read_csv('./exp_lr_0_0.04_gamma_0_1_temp.csv', delimiter=', ')
df = pd.read_csv('./exp_lr_0.005_0.015_gamma_0.75_1.00_temp.csv', delimiter=', ')

# Create a pivot table. The rows are LR, the columns are GAMMA, and the values are the mean of DELTA.
pivot_table = df.pivot_table(index='LR', columns='GAMMA', values='DELTA', aggfunc=np.mean)

# Calculate mean and standard deviation for rows and columns
row_means = pivot_table.mean(axis=1)
col_means = pivot_table.mean(axis=0)
row_stdev = pivot_table.std(axis=1)
col_stdev = pivot_table.std(axis=0)


# Create the heatmap
#
# # this is ok for 20x20 values
#
# plt.figure(figsize=(14, 8)) 
# sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt=".2f")

# sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt=".2f", annot_kws={'rotation': 90})  # Rotate annotations

# plt.title('Performance Decrease (%) by Learning Rate and Gamma')
# plt.xlabel('Gamma')
# plt.ylabel('Learning Rate')

# # Adjust the colorbar width to make columns adapt to content
# cbar = plt.gcf().axes[-1]
# cbar.set_xlabel('Colorbar Label', labelpad=15)  # You can customize the colorbar label here
# cbar.set_position([0.92, 0.15, 0.02, 0.7])  # Adjust the position and width as needed

# plt.show()

# Initialize figure with GridSpec
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(3, 3, fig)

# Create the heatmap in the top-left 2x2 slots
heatmap_ax = fig.add_subplot(gs[0:2, 0:2])
sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt=".2f", annot_kws={'rotation': 90}, ax=heatmap_ax)
heatmap_ax.set_title('Heatmap')
heatmap_ax.set_xlabel('Gamma')
heatmap_ax.set_ylabel('Learning Rate')

# Create the mean by row plot in the bottom left slot
mean_row_ax = fig.add_subplot(gs[2, 0])
row_bar = sns.barplot(x=row_means.index, y=row_means.values, ax=mean_row_ax)
mean_row_ax.set_title('Mean by Row')
mean_row_ax.set_xlabel('Learning Rate')
mean_row_ax.set_ylabel('Mean Value')
row_bar.set_xticklabels(row_bar.get_xticklabels(), rotation=90)

# Create the stddev by row plot in the bottom middle slot
stddev_row_ax = fig.add_subplot(gs[2, 1])
stddev_row_bar = sns.barplot(x=row_stdev.index, y=row_stdev.values, ax=stddev_row_ax)
stddev_row_ax.set_title('StdDev by Row')
stddev_row_ax.set_xlabel('Learning Rate')
stddev_row_ax.set_ylabel('StdDev Value')
stddev_row_bar.set_xticklabels(stddev_row_bar.get_xticklabels(), rotation=90)

# Create the mean by column plot in the top right slot
mean_col_ax = fig.add_subplot(gs[0, 2])
col_bar = sns.barplot(x=col_means.index, y=col_means.values, ax=mean_col_ax)
mean_col_ax.set_title('Mean by Column')
mean_col_ax.set_xlabel('Gamma')
mean_col_ax.set_ylabel('Mean Value')
col_bar.set_xticklabels(col_bar.get_xticklabels(), rotation=90)

# Create the stddev by column plot in the middle middle slot
stddev_col_ax = fig.add_subplot(gs[1, 2])
stddev_col_bar = sns.barplot(x=col_stdev.index, y=col_stdev.values, ax=stddev_col_ax)
stddev_col_ax.set_title('StdDev by Column')
stddev_col_ax.set_xlabel('Gamma')
stddev_col_ax.set_ylabel('StdDev Value')
stddev_col_bar.set_xticklabels(stddev_col_bar.get_xticklabels(), rotation=90)

# Adjust the layout and spacing
plt.tight_layout()

plt.show()