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
    corrs = df.corr()
    #corrs = df.corr().abs()
    np.fill_diagonal(corrs.values, 0)
    l_cor = list(corrs.unstack().sort_values(ascending=False).to_dict().keys())
    x = l_cor[0][0]
    y = l_cor[0][1]
    colors = {'Gryffindor': 'Red', 'Hufflepuff': 'Yellow', 'Ravenclaw': 'DarkBlue', 'Slytherin': 'Green'}
    scatter_x = df[x].values
    scatter_y = df[y].values
    houses = df['Hogwarts House'].values
    fig, ax = plt.subplots()
    for house in np.unique(houses):
        ix = np.where(houses == house)
        ax.scatter(scatter_x[ix], scatter_y[ix], c=colors[house], label=house, alpha = 0.5)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Strongest positive correlation')
    ax.legend()
    plt.show()
