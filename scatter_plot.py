import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def get_data(args):
    file = "resources/dataset_train.csv"
    if len(args) < 2:
        print("No arguments given. Try with \"resources/dataset_train.csv\".")
    else:
        file = args[1]
    try:
        data = pd.read_csv(file)
    except Exception as e:
        print("Can't extract data from {}.".format(file))
        print(e.__doc__)
        sys.exit(0)
    return data


def var_init(df):
    houses = df['Hogwarts House'].values
    colors = {
                'Gryffindor': 'Red',
                'Hufflepuff': 'Yellow',
                'Ravenclaw': 'DarkBlue',
                'Slytherin': 'Green'}
    return houses, colors


def manual_selection(df, houses, colors):
    scatter_x = df['Arithmancy'].values
    scatter_y = df['Care of Magical Creatures'].values
    fig, ax = plt.subplots()
    for house in np.unique(houses):
        ix = np.where(houses == house)
        ax.scatter(
                    scatter_x[ix], scatter_y[ix],
                    c=colors[house], label=house, alpha=0.5)
    plt.xlabel('Arithmancy')
    plt.ylabel('Care of Magical Creatures')
    plt.title('Similar Features')
    ax.legend()


def auto_selection(df, houses, colors, absolute=False):
    corrs = df.corr().abs() if absolute else df.corr()
    np.fill_diagonal(corrs.values, 0)
    l_cor = list(corrs.unstack().sort_values(ascending=False).to_dict().keys())
    x = l_cor[0][0]
    y = l_cor[0][1]
    scatter_x = df[x].values
    scatter_y = df[y].values
    fig, ax = plt.subplots()
    for house in np.unique(houses):
        ix = np.where(houses == house)
        ax.scatter(
                    scatter_x[ix], scatter_y[ix],
                    c=colors[house], label=house, alpha=0.5)
    plt.xlabel(x)
    plt.ylabel(y)
    sign = "Absolute" if absolute else "Positive"
    plt.title('Strongest Correlation (' + sign + ' Value)')
    ax.legend()


if __name__ == '__main__':
    df = get_data(sys.argv)
    houses, colors = var_init(df)
    manual_selection(df, houses, colors)
    auto_selection(df, houses, colors)
    auto_selection(df, houses, colors, absolute=True)
    plt.show()
