import pandas as pd
import matplotlib.pyplot as plt

# Plot IMFs

IMF = []
for ii in range(3):
    IMF.append(pd.read_csv('IMF-day{}.csv'.format(ii+1)))
    IMF[ii].rename(columns={'Power': 'Data'}, inplace=True)
    # IMF[ii].drop('Data', axis=1, inplace=True)
    x = list(range(1, 145))

    # print(IMF[ii].head())

    fig, axes = plt.subplots(
        nrows=IMF[ii].shape[1], ncols=1, figsize=(4, 6.75), sharex='col')
    for jj in range(IMF[ii].shape[1]):
        axes[jj].plot(x, IMF[ii][IMF[ii].columns[jj]])
        axes[jj].set_ylabel((IMF[ii].columns[jj]), fontsize=14)
        axes[jj].grid(color='gainsboro')
        axes[jj].tick_params(axis='x', labelsize=12)
        axes[jj].tick_params(axis='y', labelsize=12)
        axes[jj].set_xlim(0, 145)
        if jj != (IMF[ii].shape[1]-1):
            axes[jj].set_xticks([])
            axes[jj].xaxis.set_ticks_position('none')
        else:
            axes[jj].set_xticks([0, 70, 140])
    plt.grid(True)
    plt.xlabel('Samples (10 minutes)', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.2)
    # plt.savefig('IMF-plot-day{}-thin.eps'.format(ii+1), dpi=300)
    # plt.savefig('IMF-plot-day{}-thin.pdf'.format(ii+1), dpi=300)
    plt.show()
