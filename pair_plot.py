import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def main():
    df = get_data(sys.argv)
    df = df.reindex(sorted(df.columns), axis=1)
    sns.set(font_scale=0.5)
    sns.pairplot(
                    df.drop(columns='Index').dropna(),
                    hue='Hogwarts House',
                    height=2,
                    aspect=1)
    plt.subplots_adjust(left=0.04, bottom=0.04)
    plt.show()


if __name__ == '__main__':
    main()
