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


def get_subject_grade_house(df, subject, house):
    return df.loc[df['Hogwarts House'] == house, subject].dropna()


if __name__ == '__main__':
    df = get_data(sys.argv)
    houses = df['Hogwarts House'].unique().tolist()
    subjects = list(df.select_dtypes('number').to_dict().keys())
    subjects.remove('Index')
    p_values = {}
    for subject in subjects:
        F, p_values[subject] = stats.f_oneway(get_subject_grade_house(df, subject, 'Hufflepuff'), get_subject_grade_house(df, subject, 'Gryffindor'), get_subject_grade_house(df, subject, 'Ravenclaw'), get_subject_grade_house(df, subject, 'Slytherin'))
    p_values = {k: v for k, v in sorted(p_values.items(), key=lambda x: x[1], reverse=True)}
    homogen_subject = list(p_values.keys())[0]
    plt.figure(1)
    for house in houses:
        grade = df.loc[df['Hogwarts House'] == house, homogen_subject].dropna()
        plt.hist(grade, alpha=0.5, label=house)
    plt.legend(loc='upper right')
    plt.xlabel(homogen_subject + ' Grades')
    plt.ylabel('Frequency')
    plt.title("Most homogeneous distribution: " + homogen_subject)
    df.boxplot(homogen_subject, by='Hogwarts House', figsize=(12, 8))
    plt.ylabel(homogen_subject + ' Grades')
    plt.show()
