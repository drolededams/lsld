import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def y_classification(df, house):
    df_copy = df.copy()
    df_copy['Hogwarts House'] = np.where(df_copy['Hogwarts House'] == house, 1, 0)
    y_class = df_copy['Hogwarts House'].to_numpy()
    y_class = np.reshape(y_class, (1, np.size(y_class)))
    return y_class


def preprocessing_feature_scaling(data):
    clean_data = {k: data[k] for k in data if not np.isnan(data[k])}
    values = np.sort(np.array(list(clean_data.values()), dtype=object))
    count = len(clean_data)
    stats = {'mean': sum(clean_data.values()) / count}
    stats['var'] = 1 / (count - 1) * np.sum(np.power(np.subtract(values, stats['mean']), 2))
    stats['std'] = np.sqrt(stats['var'])
    return stats


def feature_scaling(x, stats):
    for subject in stats:
        x[subject] = np.divide(np.subtract(x[subject], stats[subject]['mean']), stats[subject]['std'])
    return x


def theta_calc(theta, xScaled, y_class, lRate):
    hypothesis = np.divide(1, np.add(1, np.exp(np.multiply(np.dot(theta, np.transpose(xScaled)), -1))))
    size = np.size(xScaled, 0)
    return theta - (lRate / size) * np.dot(np.subtract(hypothesis, y_class), xScaled)


def cost(theta, xScaled, y_class):
    hypothesis = np.divide(1, np.add(1, np.exp(np.multiply(np.dot(theta, np.transpose(xScaled)), -1))))
    size = np.size(xScaled, 0)
    return (1 / size) * np.subtract(np.multiply(-1, np.dot(y_class, np.transpose(np.log(hypothesis)))), np.dot(np.subtract(1, y_class), np.transpose(np.log(np.subtract(1, hypothesis)))))


def gradient_descent(x, y, house):
    y_class = y_classification(y, house)
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
    results = {'theta': theta}
    results['turn'] = turn
    results['costs'] = costs
    return results


def display_results(results, house):
    i = 1
    for house in houses:
        print(house + "'s results:")
        print("finals thetas= ", results[house]['theta'])
        print("turns = ", results[house]['turn'])
        # plt.figure(i)
        # plt.xlabel('No. of iterations')
        # plt.ylabel('Cost Function')
        # plt.title(house + "'s Cost Function Evolution")
        # plt.plot(np.arange(results[house]['turn'] + 1), results[house]['costs'])
        i += 1
    #plt.show()


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf) 
    pd.set_option('display.expand_frame_repr', False)

    df = get_data(sys.argv)
    df = df.drop(columns=['Index', 'Arithmancy', 'Potions', 'Care of Magical Creatures', 'First Name', 'Last Name', 'Birthday', 'Best Hand']) #drpna ? Really ?
    x = df.drop(columns='Hogwarts House')
    x.fillna(x.mean(), inplace=True) # avoir
    x = x.reindex(sorted(x.columns), axis=1)
    index = list(x.columns.values)
    index.insert(0, 'Theta 0')
    y = df[['Hogwarts House']]
    houses = df['Hogwarts House'].unique().tolist()
    dd = x.to_dict()
    stats = {column: preprocessing_feature_scaling(sub_dict) for column, sub_dict in dd.items()}
    xScaled = feature_scaling(x, stats)
    xScaled = xScaled.to_numpy()
    xScaled = np.insert(xScaled, 0, 1.0, axis=1)

    results = {}
    for house in houses:
        results[house] = gradient_descent(x, y, house)
    display_results(results, houses)
    results_serialization = {}
    for house in houses:
        results_serialization[house] = results[house]['theta'].tolist()[0]
    mean = list()
    std = list()
    for subject in stats:
        mean.append(stats[subject]['mean'])
        std.append(stats[subject]['std'])
    mean.insert(0, 1)
    std.insert(0, 1)
    results_serialization['mean'] = mean
    results_serialization['std'] = std
    describe(results_serialization)
    describe(index)
    pd.DataFrame(results_serialization, index=index).to_csv('weight.csv', index_label='Subject')








