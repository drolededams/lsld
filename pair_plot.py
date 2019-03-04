import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    sns.set(font_scale=0.5)
    sns.pairplot(df.drop(columns='Index').dropna(), hue='Hogwarts House', height=2, aspect=1)
    plt.show()
