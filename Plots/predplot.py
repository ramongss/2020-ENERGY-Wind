# -*- coding: utf-8 -*-
"""predplot.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gU4uAKa7To3rBkRHzqRcXF38oxACq7OT
"""

# libraries and data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for day in range(3): 
    # Make a data frame
    df=pd.read_csv('day{}.csv'.format(day+1))
    x = list(range(0,df.shape[0]))
    
    # create a color palette
    palette = plt.get_cmap('Set1')

    # set figure size
    fig, axs = plt.subplots(3,1,figsize=(6.75, 6.75),sharey=True)
    
    # multiple line plot
    num=0
    for column in df.drop('Observed', axis=1):
        num+=1
        
        # Find the right spot on the plot
        axs[num-1] = plt.subplot(3,1,num)
    
        # plot Observed line
        axs[num-1].plot(x, df['Observed'], color=palette(1), label='Observed')
    
        # Plot the lineplot
        axs[num-1].plot(x, df[column], marker='', color=palette(0), label="Predicted")
    
        # Not ticks everywhere
        if num in range(3) :
            plt.tick_params(labelbottom=False)
        if num not in range(3):
            # Add X label in the last
            plt.xlabel('Samples (10 minutes)', fontsize=14)
            # Add legend in the last plot and outside plot
            plt.legend(loc='center', fontsize=12, ncol=2,
                    bbox_to_anchor=(0.5, -0.5),
                    fancybox=True, shadow=True)
            # Set x ticks
            plt.xticks([0,35,70,105,140])
    
        # Add title
        plt.title(column, fontsize=11, rotation=270, x=1.02, y=0.05)

        # Add grid
        plt.grid(color='gainsboro')

        # Set ticks size
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)
        
        # Add Y label 
        if num == 2:
            plt.ylabel('Wind Power ($kW$)', fontsize=14)

        # Add vertical line
        plt.vlines(x=98,
                ymin=df['Observed'].values.min()-100,
                ymax=df['Observed'].values.max()+100)
        plt.ylim(df['Observed'].values.min()-100,
                df['Observed'].values.max()+100)
    
        # Add comments
        plt.text(10, df['Observed'].values.min()-50, 'Training', fontsize=14)
        plt.text(110, df['Observed'].values.min()-50, 'Test', fontsize=14)
        plt.tight_layout()
    fig.savefig('pred_day{}.png'.format(day+1), dpi=300)
    fig.savefig('pred_day{}.pdf'.format(day+1), dpi=300)
