import sys
import math
import numpy as np
import pandas as pd


def get_data(args):
    file = "data.csv"
    if len(args) < 2:
        print("No arguments given. Try with \"data.csv\".")
    else:
        file = args[1]
    try:
        data = pd.read_csv(file)
        # data = np.genfromtxt(file, delimiter=",", names=True)
    except Exception as e:
        print("Can't extract data from {}.".format(file))
        print(e.__doc__)
        sys.exit(0)
    return data


def describe(data):
    clean_data = {k: data[k] for k in data if not np.isnan(data[k])}
    values = np.sort(np.array(list(clean_data.values()), dtype=object))
    count = len(clean_data)
    stats = {'count': count}
    stats['mean'] = sum(clean_data.values()) / count
    stats['var'] = 1 / (count - 1) * np.sum(np.power(np.subtract(values, stats['mean']), 2))
    stats['std'] = np.sqrt(stats['var'])
    if isinstance(count * 0.25, int):
        stats['25%'] = values[int(count * 0.25) - 1]
    else:
        stats['25%'] = values[math.floor(count * 0.25) - 1] + (values[math.floor(count * 0.25)] - values[math.floor(count * 0.25) - 1]) * ((count * 0.25) - math.floor(count * 0.25))
    if count % 2 != 0:
        stats['50%'] = values[int((count + 1) / 2 - 1)]
    else:
        stats['50%'] = (values[int(count / 2 - 1)] + values[int(count / 2)]) / 2
    stats['75%'] = values[math.ceil((3 * count) / 4)]
    return stats


if __name__ == '__main__':
    df = get_data(sys.argv)
    df = df.select_dtypes('number')
    print(df.columns)
    dd = df.to_dict()
    dict_desc = {column: describe(sub_dict) for column, sub_dict in dd.items()}
    # if 'Index' in df.columns:
    #     print("Index column present")
    pd.set_option('display.expand_frame_repr', False)
    print(df.describe())
    print(dict_desc)
