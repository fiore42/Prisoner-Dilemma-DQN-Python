import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV data into a pandas DataFrame
df = pd.read_csv('./lr_gamma_delta.csv')

# Create a pivot table. The rows are LR, the columns are GAMMA, and the values are the mean of DELTA.
pivot_table = df.pivot_table(index='LR', columns='GAMMA', values='DELTA', aggfunc=np.mean)

# Create the heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt=".2f")
plt.title('Performance Decrease (%) by Learning Rate and Gamma')
plt.xlabel('Gamma')
plt.ylabel('Learning Rate')

# Adjust the colorbar width to make columns adapt to content
cbar = plt.gcf().axes[-1]
cbar.set_xlabel('Colorbar Label', labelpad=15)  # You can customize the colorbar label here
cbar.set_position([0.92, 0.15, 0.02, 0.7])  # Adjust the position and width as needed

plt.show()
