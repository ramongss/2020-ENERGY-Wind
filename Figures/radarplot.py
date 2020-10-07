# Libraries
import matplotlib.pyplot as plt
#from matplotlib import PolarAxesSubplot
import pandas as pd
from math import pi

months = ['August', 'September', 'October']

fig = plt.figure(figsize=(20.25,6.75))
fig, axs = plt.subplots(nrows=1, ncols=3, subplot_kw=dict(projection='polar'))
plt.subplots_adjust(wspace=0.2)

for ii in range(1,4):
    # Set data
    df = pd.read_csv('dataset{}_std.csv'.format(ii))
    
    # create a color palette
    palette = plt.get_cmap('Set1')

    # ------- PART 1: Create background

    # number of variable
    categories=list(df)
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    #fig = plt.figure(figsize=(6.75,6.75))
    #ax = plt.subplot(111, polar=True)
    ax = axs[ii - 1]

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    ax.set_xticks(ticks=angles[:-1])
    ax.set_xticklabels(categories, fontdict={'fontsize':12})

    ax.set_rlabel_position(0)
    ticks=ax.yaxis.get_major_ticks()
    for tick in ticks:
        tick.label1.set_color('grey')
        tick.label1.set_size(10)
    ax.set_ylim(df.values.min()-10,df.values.max()+10)


    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't do a loop, because plotting more than 3 groups makes the chart unreadable

    # Ind1
    values=df.loc[0].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=palette(1), linewidth=2, linestyle='solid', label="10 minutes")
    ax.fill(angles, values, color=palette(1), alpha=0.1)

    # Ind2
    values=df.loc[1].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=palette(0), linewidth=2, linestyle='solid', label="20 minutes")
    ax.fill(angles, values, color=palette(0), alpha=0.1)
    
    # Ind3
    values=df.loc[2].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=palette(4), linewidth=2, linestyle='solid', label="30 minutes")
    ax.fill(angles, values, color=palette(4), alpha=0.1)

    # Set title
    ax.set_title(months[ii-1], fontsize=20, pad=20)

# Add legend
# plt.legend(loc='lower left', bbox_to_anchor=(-1.5, -0.2), fontsize=16, ncol=3) # legend at up left
handles, labels = ax.get_legend_handles_labels()
# #fig.legend(handles, labels, bbox_to_anchor=(0.8,0.9), loc='lower left', fontsize=6)
fig.legend(handles, labels, ncol = 3, loc='center', bbox_to_anchor=(0.15, -0.2, 0.5, 0.5), fontsize=16)

fig.set_size_inches(20.25, 6.75)
# fig.savefig('radaplot.png',dpi=1200, bbox_inches='tight', pad_inches=0)
fig.savefig('radaplot.pdf',dpi=1200, bbox_inches='tight', pad_inches=0)
# plt.show()
