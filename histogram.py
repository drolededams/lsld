import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def get_data(args):
    file = "data.csv"
    if len(args) < 2:
        print("No arguments given. Try with \"data.csv\".")
    else:
        file = args[1]
    try:
        data = pd.read_csv(file)
    except Exception as e:
        print("Can't extract data from {}.".format(file))
        print(e.__doc__)
        sys.exit(0)
    return data


if __name__ == '__main__':
    df = get_data(sys.argv)
    houses = df['Hogwarts House'].unique().tolist()
    print(houses)
# Create a histogram. replace Herbology by good subject
    for house in houses:
        grade = df.loc[df['Hogwarts House'] == house, 'Arithmancy'].dropna()
        plt.hist(grade, alpha=0.5, label=house)
    plt.legend(loc='upper right')
    plt.show()
# Create a boxplot as bonus
    # plt.figure(1)
    # df.boxplot('Arithmancy', by='Hogwarts House', figsize=(12, 8))
    # plt.figure(2)
    # df.boxplot('Astronomy', by='Hogwarts House', figsize=(12, 8))
    # plt.show()


    '''
        to do : 
            - understand ANOVA
            - find selection criteria
            - list of subjects (Arithmancy, Herbology, etc...)
            - loop stats.f_oneway
            - selection
            - plot hist
            - bonus : plot boxplot
    '''
    print(stats.f_oneway(df.loc[df['Hogwarts House'] == 'Hufflepuff', 'Herbology'].dropna(), df.loc[df['Hogwarts House'] == 'Gryffindor', 'Herbology'].dropna(), df.loc[df['Hogwarts House'] == 'Slytherin', 'Herbology'].dropna()))
    print(stats.f_oneway(df.loc[df['Hogwarts House'] == 'Hufflepuff', 'Arithmancy'].dropna(), df.loc[df['Hogwarts House'] == 'Gryffindor', 'Arithmancy'].dropna(), df.loc[df['Hogwarts House'] == 'Slytherin', 'Arithmancy'].dropna()))
    print(stats.f_oneway(df.loc[df['Hogwarts House'] == 'Hufflepuff', 'Astronomy'].dropna(), df.loc[df['Hogwarts House'] == 'Gryffindor', 'Astronomy'].dropna(), df.loc[df['Hogwarts House'] == 'Slytherin', 'Astronomy'].dropna()))
    print(stats.f_oneway(df.loc[df['Hogwarts House'] == 'Hufflepuff', 'Defense Against the Dark Arts'].dropna(), df.loc[df['Hogwarts House'] == 'Gryffindor', 'Defense Against the Dark Arts'].dropna(), df.loc[df['Hogwarts House'] == 'Slytherin', 'Defense Against the Dark Arts'].dropna()))
