# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

months = ['August', 'September', 'October']

for ii in range(1, 4):
    # Set data
    df = pd.read_csv('dataset{}_std.csv'.format(ii))

    # create a color palette
    palette = plt.get_cmap('Set1')

    # ------- PART 1: Create background

    # number of variable
    categories = list(df)
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    fig = plt.figure(figsize=(6.75, 6.75))
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks(color="grey", size=8)
    plt.ylim(df.values.min()-10, df.values.max()+10)

    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't do a loop, because plotting more than 3 groups makes the chart unreadable

    # Ind1
    values = df.loc[0].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=palette(1), linewidth=1,
            linestyle='solid', label="10 minutes")
    ax.fill(angles, values, color=palette(1), alpha=0.1)

    # Ind2
    values = df.loc[1].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=palette(0), linewidth=1,
            linestyle='solid', label="20 minutes")
    ax.fill(angles, values, color=palette(0), alpha=0.1)

    # Ind3
    values = df.loc[2].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=palette(4), linewidth=1,
            linestyle='solid', label="30 minutes")
    ax.fill(angles, values, color=palette(4), alpha=0.1)

    # Add legend
    # plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12) # legend at bottom left
    plt.legend(loc='lower left', bbox_to_anchor=(0.8, 0.9),
               fancybox=True, shadow=True, fontsize=12)  # legend at up left
    plt.title(months[ii-1], fontsize=20)

    # plt.savefig('radaplot_dataset{}.png'.format(ii), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig('radaplot_dataset{}.pdf'.format(ii),
                dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.show()
