# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

for ii in range(1,4):
    # Set data
    df = pd.read_csv('dataset{}_std.csv'.format(ii))

    df.columns = ['Model','10 minutes', '30 minutes']

    df = df.T

    df.columns = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
    df.drop('Model', axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ------- PART 1: Create background

    # number of variable
    categories=list(df)
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    fig = plt.figure(figsize=(6.75,6.75))
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks(color="grey", size=8)
    plt.ylim(df.values.min()-10,df.values.max()+10)

    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't do a loop, because plotting more than 3 groups makes the chart unreadable

    # Ind1
    values=df.loc[0].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="10 minutes ahead")
    ax.fill(angles, values, 'b', alpha=0.1)

    # Ind2
    values=df.loc[1].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="30 minutes ahead")
    ax.fill(angles, values, 'r', alpha=0.1)

    # Add legend
    # plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12) # legend at bottom left
    plt.legend(loc='lower right', bbox_to_anchor=(0.1, 0.9), fontsize=12) # legend at up left
    # plt.title('Dataset {}'.format(ii))

    # plt.savefig('radaplot_dataset{}.png'.format(ii), dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.savefig('radaplot_dataset{}.pdf'.format(ii), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
