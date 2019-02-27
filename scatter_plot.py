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
    pd.set_option('display.expand_frame_repr', False)
    corrs = df.corr().abs()
    print(corrs)
    np.fill_diagonal(corrs.values, -2)
    print(list(corrs.unstack().sort_values(ascending=False).to_dict().keys())[0][0])
    print(list(corrs.unstack().sort_values(ascending=False).to_dict().keys())[0][1])
    colors = {'Gryffindor': 'Red', 'Hufflepuff': 'Yellow', 'Ravenclaw': 'DarkBlue', 'Slytherin': 'Green'}
    scatter_x = df['Defense Against the Dark Arts'].values
    scatter_y = df['Astronomy'].values
    # scatter_x = df['History of Magic'].values
    # scatter_y = df['Transfiguration'].values
    houses = df['Hogwarts House'].values
    fig, ax = plt.subplots()
    for house in np.unique(houses):
        ix = np.where(houses == house)
        ax.scatter(scatter_x[ix], scatter_y[ix], c=colors[house], label=house, alpha = 0.5)
    ax.legend()
    plt.show()
