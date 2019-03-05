import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import inspect
import re


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


def df_classification(df, house):
    df_copy = df.copy()
    df_copy['Hogwarts House'] = np.where(df_copy['Hogwarts House'] == house, 1, 0)
    df_copy = df_copy.select_dtypes('number')
    df_copy = df_copy.drop(columns=['Index', 'Arithmancy', 'Potions', 'Care of Magical Creatures']).dropna() #drpna ? Really ?
    return df_copy


def theta_calc(theta, xScaled, y_class, lRate):
    hypothesis = np.divide(1, np.add(1, np.exp(np.multiply(np.dot(theta, np.transpose(xScaled)), -1))))
    size = np.size(xScaled, 0)
    return theta - (lRate / size) * np.dot(np.subtract(hypothesis, y_class), xScaled)


def cost(theta, xScaled, y_class):
    hypothesis = np.divide(1, np.add(1, np.exp(np.multiply(np.dot(theta, np.transpose(xScaled)), -1))))
    size = np.size(xScaled, 0)
    #return np.dot(y_class, np.transpose(np.log(hypothesis)))
    return (1 / size) * np.subtract(np.multiply(-1, np.dot(y_class, np.transpose(np.log(hypothesis)))), np.dot(np.subtract(1, y_class), np.transpose(np.log(np.subtract(1, hypothesis)))))


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf) 
    pd.set_option('display.expand_frame_repr', False)
    df = get_data(sys.argv)
    df_raven = df_classification(df, 'Ravenclaw')
    y_class = df_raven['Hogwarts House'].to_numpy()
    y_class = np.reshape(y_class, (1, np.size(y_class)))
    x = df_raven.drop(columns='Hogwarts House')
    names = x.columns
    scaler = preprocessing.StandardScaler()
    xScaled = scaler.fit_transform(x)
    xScaled = pd.DataFrame(xScaled, columns=names)
    xScaled = xScaled.to_numpy()
    xScaled = np.insert(xScaled, 0, 1.0, axis=1)
    theta = np.zeros((1, 11))
    converge = 10000000
    lRate = 1
    turn = 0
    tmp_cost = 0
    new_cost = 0
    costs = []
    tmp_cost = float(cost(theta, xScaled, y_class)[0])
    costs.append(tmp_cost)
    while converge > 0.0001:
        tmp_theta = theta
        theta = theta_calc(tmp_theta, xScaled, y_class, lRate)
        new_cost = float(cost(theta, xScaled, y_class)[0])
        costs.append(new_cost)
        converge = np.abs(tmp_cost - new_cost)
        tmp_cost = new_cost
        turn += 1
    print("theta final = ", theta)
    print("turn = ", turn)
    plt.figure(1)
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost Function')
    plt.title('Data Visualization')
    plt.plot(np.arange(turn + 1), costs)
    plt.show()
