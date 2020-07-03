# Plot datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

x = list(range(1, 145))
fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True,
                         sharey=True, figsize=(12, 6.75))
dates = ["1st", "2nd", "7th"]
dataset = []
for ii in range(0, 3):
    dataset.append(pd.read_csv('IMF-day{}.csv'.format(ii+1))['Power'])
    dataset[ii].rename("Dataset {}".format(ii+1), inplace=True)
    axes[ii].plot(x, dataset[ii])
    axes[ii].set_title('Dataset {} (August {})'.format(
        ii+1, dates[ii]), fontsize=16)
    axes[ii].set_xlim(0, 145)
    if ii == 1:
        axes[ii].set_ylabel('Wind Power ($kW$)', fontsize=16)
    axes[ii].grid(color='gainsboro')
    axes[ii].tick_params(axis='x', labelsize=14)
    axes[ii].tick_params(axis='y', labelsize=14)
plt.xlabel('Samples (10 minutes)', fontsize=16)
plt.tight_layout()
plt.savefig('dataset.eps', dpi=300)
# plt.show()
