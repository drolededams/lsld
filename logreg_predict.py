import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inspect
import re
from sklearn.metrics import accuracy_score


def describe(arg):
    frame = inspect.currentframe()
    callerframeinfo = inspect.getframeinfo(frame.f_back)
    try:
        context = inspect.getframeinfo(frame.f_back).code_context
        caller_lines = ''.join([line.strip() for line in context])
        m = re.search(r'describe\s*\((.+?)\)$', caller_lines)
        if m:
            caller_lines = m.group(1)
            position = str(callerframeinfo.filename) + "@" + str(callerframeinfo.lineno)

            # Add additional info such as array shape or string length
            additional = ''
            if hasattr(arg, "shape"):
                additional += "[shape={}]".format(arg.shape)
            elif hasattr(arg, "__len__"):  # shape includes length information
                additional += "[len={}]".format(len(arg))

            # Use str() representation if it is printable
            str_arg = str(arg)
            str_arg = str_arg if str_arg.isprintable() else repr(arg)

            print(position, "describe(" + caller_lines + ") = ", end='')
            print(arg.__class__.__name__ + "(" + str_arg + ")", additional)
        else:
            print("Describe: couldn't find caller context")

    finally:
        del frame
        del callerframeinfo


def get_data(args):
    dataset_f = "resources/dataset_test.csv"
    weights_f = "weights.csv"
    if len(args) < 3:
        print("2 arguments needed. Try with \"resources/dataset_test.csv\" and \"weights.csv\".")
    else:
        dataset_f = args[1]
        weights_f = args[2]
    try:
        dataset = pd.read_csv(dataset_f)
    except Exception as e:
        print("Can't extract data from {}.".format(dataset_f))
        print(e.__doc__)
        sys.exit(0)
    try:
        weights = pd.read_csv(weights_f, index_col='Subject')
    except Exception as e:
        print("Can't extract data from {}.".format(weights_f))
        print(e.__doc__)
        sys.exit(0)
    return dataset, weights


def feature_scaling(x, stats):
    subjects = list(x.columns.values)
    for subj in subjects:
        x[subj] = (x[subj] - stats['mean'][subj]) / stats['std'][subj]
    return x


if __name__ == '__main__':
    #np.set_printoptions(threshold=np.inf) 
    #pd.set_option('display.expand_frame_repr', False)

    # Get Data
    dataset, weights = get_data(sys.argv)

    # Data Preprocessing
    thetas = weights.drop(columns=['mean', 'std'])
    houses = list(thetas.columns.values)
    x = dataset.drop(columns=[
        'Index', 'Arithmancy', 'Potions', 'Care of Magical Creatures',
        'First Name', 'Last Name', 'Birthday', 'Best Hand', 'Hogwarts House'])
    x = x.reindex(sorted(x.columns), axis=1)
    mean = weights.drop('Theta 0')['mean']
    x.fillna(mean, inplace=True)
    xScaled = feature_scaling(x, weights)
    xScaled = xScaled.to_numpy()
    xScaled = np.insert(xScaled, 0, 1.0, axis=1)
    thetas = thetas.to_numpy()

    # Prediction Processing
    results = 1 / (1 + np.exp(xScaled.dot(thetas) * -1))
    results = results.argmax(axis=1).tolist()
    for index, v in enumerate(results):
        results[index] = houses[v]

    # Results Generation
    df = pd.DataFrame(data={
        "Index": list(range(0, len(results))), "Hogwarts House": results})
    df.to_csv("./houses.csv", sep=',', index=False)
    print(df, "\nFull results in houses.csv.")

    #df = pd.read_csv('resources/dataset_train.csv')
    #print(accuracy_score(df['Hogwarts House'].tolist(), results))
