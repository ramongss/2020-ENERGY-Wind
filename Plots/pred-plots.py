import pandas as pd
import matplotlib.pyplot as plt

# plt.rc('font', family='Times New Roman')

# Plot Predictions
df = []
for day in range(0, 3):
    df.append(pd.read_csv('day{}.csv'.format(day+1)))
    x = list(range(1, 141))

    plt.figure(figsize=(6.75, 6.75))

    for cc in range(0, df[day].shape[1]):
        plt.plot(x, df[day][df[day].columns[cc]], label=df[day].columns[cc])

    plt.legend(loc='upper left', fontsize=12)
    plt.text(x=10, y=(df[day].values.min()-100),
             s='Training set', fontsize=16)
    plt.text(x=110, y=(df[day].values.min()-100),
             s='Test set', fontsize=16)
    plt.vlines(x=98,
               ymin=df[day].values.min()-200,
               ymax=df[day].values.max()+200)
    plt.ylabel('Wind Power ($kW$)', fontsize=16)
    plt.xlabel('Samples (10 minutes)', fontsize=16)
    plt.ylim(df[day].values.min()-200,
             df[day].values.max()+200)
    plt.xlim(0, None)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.grid(color='gainsboro')
    # plt.savefig('Pred_day{}.pdf'.format(day+1))
    plt.savefig('Pred_day{}_squared.eps'.format(day+1), dpi=300)
    # plt.show()
