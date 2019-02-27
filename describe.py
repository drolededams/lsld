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
    except Exception as e:
        print("Can't extract data from {}.".format(file))
        print(e.__doc__)
        sys.exit(0)
    return data


def percentile(percent, count, values):
    x = percent * (count - 1)
    return values[math.floor(x)] + (values[math.floor(x) + 1] - values[math.floor(x)]) * (x % 1)


def describe(data):
    clean_data = {k: data[k] for k in data if not np.isnan(data[k])}
    values = np.sort(np.array(list(clean_data.values()), dtype=object))
    count = len(clean_data)
    stats = {'count': count}
    stats['mean'] = sum(clean_data.values()) / count
    stats['var'] = 1 / (count - 1) * np.sum(np.power(np.subtract(values, stats['mean']), 2))
    stats['std'] = np.sqrt(stats['var'])
    stats['min'] = values[0]
    stats['max'] = values[count - 1]
    stats['range'] = stats['max'] - stats['min']
    stats['25%'] = percentile(0.25, count, values)
    stats['75%'] = percentile(0.75, count, values)
    if count % 2 == 0:
        stats['50%'] = (values[int(count / 2 - 1)] + values[int(count / 2)]) / 2
    else:
        stats['50%'] = values[int((count + 1) / 2 - 1)]
    return stats


if __name__ == '__main__':
    df = get_data(sys.argv)
    df = df.select_dtypes('number')
    dd = df.to_dict()
    dict_desc = {column: describe(sub_dict) for column, sub_dict in dd.items()}
    pd.set_option('display.expand_frame_repr', False)
    res = pd.DataFrame.from_dict(dict_desc)
    res = res.reindex(['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'var', 'range'])
    print(res)
