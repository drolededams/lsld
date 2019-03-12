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


def x_corr_fillna(x):
    defense = x['Defense Against the Dark Arts'].index[
                x['Defense Against the Dark Arts'].apply(np.isnan)].values
    astro = x['Astronomy'].index[x['Astronomy'].apply(np.isnan)].values
    i_common = set(astro).intersection(set(defense))
    astro = list(set(astro).difference(i_common))
    defense = list(set(defense).difference(i_common))
    i_notnull = x.dropna().index.values.tolist()[1]
    coeff = (
            x['Astronomy'][i_notnull]
            / x['Defense Against the Dark Arts'][i_notnull])
    for i in astro:
        x['Astronomy'][i] = x['Defense Against the Dark Arts'][i] * coeff
    for i in defense:
        x['Defense Against the Dark Arts'][i] = x['Astronomy'][i] / coeff


def preprocessing(df):
    # Select Revelant Features
    droped = [
        'Index',
        'Arithmancy',
        'Potions',
        'Care of Magical Creatures',
        'First Name',
        'Last Name',
        'Birthday',
        'Best Hand']
    df = df.drop(columns=droped)

    # Select x Features
    x = df.drop(columns='Hogwarts House')
    x = x.reindex(sorted(x.columns), axis=1)

    # Get Features List
    index = list(x.columns.values)
    index.insert(0, 'Theta 0')

    # Get Class list
    houses = df['Hogwarts House'].unique().tolist()

    # Get Class Values
    y = df[['Hogwarts House']]

    # Get Features's Stats (mean & std)
    stats = {
            column: get_stats(sub_dict)
            for column, sub_dict in x.to_dict().items()}
    mean = list()
    std = list()
    mean_fill = {}
    for subject in stats:
        mean_fill[subject] = stats[subject]['mean']
        mean.append(stats[subject]['mean'])
        std.append(stats[subject]['std'])
    mean.insert(0, 1)
    std.insert(0, 1)

    # Replace NaN values and Features Scaling
    x_corr_fillna(x)
    x.fillna(mean_fill, inplace=True)
    xScaled = feature_scaling(x, stats)
    xScaled = xScaled.to_numpy()
    xScaled = np.insert(xScaled, 0, 1.0, axis=1)

    return xScaled, y, mean, std, index, houses


def get_stats(data):
    clean_data = {k: data[k] for k in data if not np.isnan(data[k])}
    values = np.sort(np.array(list(clean_data.values()), dtype=object))
    count = len(clean_data)
    stats = {'mean': sum(clean_data.values()) / count}
    stats['var'] = (
            1 / (count - 1) * np.sum(np.power(values - stats['mean'], 2)))
    stats['std'] = np.sqrt(stats['var'])
    return stats


def feature_scaling(x, stats):
    for subj in stats:
        x[subj] = (x[subj] - stats[subj]['mean']) / stats[subj]['std']
    return x


def gradient_descent_loop(x, y, houses):
    results = {}
    for house in houses:
        results[house] = gradient_descent(x, y, house)
    return results


def gradient_descent(x, y, house):
    y_class = y_classification(y, house)
    theta = np.zeros((1, np.size(x, 1)))
    converge = 10000000
    lRate = 1
    turn = 0
    costs = []
    tmp_cost = cost(theta, x, y_class)[0]
    costs.append(tmp_cost)
    while converge > 0.0001:
        tmp_theta = theta
        theta = theta_calc(tmp_theta, x, y_class, lRate)
        new_cost = cost(theta, x, y_class)[0]
        costs.append(new_cost)
        converge = np.abs(tmp_cost - new_cost)
        tmp_cost = new_cost
        turn += 1
    results = {'theta': theta}
    results['turn'] = turn
    results['costs'] = costs
    return results


def y_classification(df, house):
    df_copy = df.copy()
    df_copy['Hogwarts House'] = np.where(
            df_copy['Hogwarts House'] == house, 1, 0)
    y_class = df_copy['Hogwarts House'].to_numpy()
    y_class = np.reshape(y_class, (1, np.size(y_class)))
    return y_class


def theta_calc(theta, xScaled, y_class, lRate):
    hypothesis = 1 / (1 + np.exp(-1 * theta.dot(np.transpose(xScaled))))
    size = np.size(xScaled, 0)
    return theta - (lRate / size) * (hypothesis - y_class).dot(xScaled)


def cost(theta, xScaled, y_class):
    hypothesis = 1 / (1 + np.exp(-1 * theta.dot(np.transpose(xScaled))))
    size = np.size(xScaled, 0)
    return ((1 / size)
            * (-1 * y_class.dot(np.transpose(np.log(hypothesis)))
            - (1 - y_class).dot(np.transpose(np.log(1 - hypothesis)))))


def display_results(results, houses):
    i = 1
    for house in houses:
        print(house + "'s results:")
        print("finals thetas= ", results[house]['theta'])
        print("turns = ", results[house]['turn'])
        plt.figure(i)
        plt.xlabel('No. of iterations')
        plt.ylabel('Cost Function')
        plt.title(house + "'s Cost Function Evolution")
        plt.plot(
                np.arange(results[house]['turn'] + 1),
                results[house]['costs'])
        i += 1
    plt.show()


def results_file(results, houses, index, mean, std):
    results_serialization = {}
    for house in houses:
        results_serialization[house] = results[house]['theta'].tolist()[0]
    results_serialization['mean'] = mean
    results_serialization['std'] = std
    pd.DataFrame(results_serialization, index=index).to_csv(
            'weights.csv', index_label='Subject')


def main():
    #np.set_printoptions(threshold=np.inf) 
    #pd.set_option('display.expand_frame_repr', False)

    # Get Data
    df = get_data(sys.argv)

    # Data Preprocessing
    x, y, mean, std, index, houses = preprocessing(df)

    # Gradient Descent Process
    results = gradient_descent_loop(x, y, houses)

    # Displaying Results
    display_results(results, houses)

    # Results Generation
    results_file(results, houses, index, mean, std)


if __name__ == '__main__':
    main()
