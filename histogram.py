import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


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


def get_subject_grade_house(df, subject, house):
    return df.loc[df['Hogwarts House'] == house, subject].dropna()


def auto_selection(df, houses, subjects):
    p_values = {}
    for subject in subjects:
        F, p_values[subject] = stats.f_oneway(
                get_subject_grade_house(df, subject, 'Hufflepuff'),
                get_subject_grade_house(df, subject, 'Gryffindor'),
                get_subject_grade_house(df, subject, 'Ravenclaw'),
                get_subject_grade_house(df, subject, 'Slytherin'))
    p_values = {k: v for k, v in sorted(
        p_values.items(), key=lambda x: x[1], reverse=True)}
    homogen_subject = list(p_values.keys())[0]
    plt.figure(3)
    for house in houses:
        grade = df.loc[df['Hogwarts House'] == house, homogen_subject].dropna()
        plt.hist(grade, alpha=0.5, label=house)
    plt.legend(loc='upper right')
    plt.xlabel(homogen_subject + ' Grades')
    plt.ylabel('Frequency')
    plt.title(
                "Most homogeneous distribution: "
                + homogen_subject
                + " (Auto Selection)")
    df.boxplot(homogen_subject, by='Hogwarts House', figsize=(12, 8))
    plt.ylabel(homogen_subject + ' Grades')
    plt.show()


def manual_selection(df, houses):
    plt.figure(1)
    for house in houses:
        grade = df.loc[
                df['Hogwarts House'] == house, 'Care of Magical Creatures'
                ].dropna()
        plt.hist(grade, alpha=0.5, label=house)
    plt.legend(loc='upper right')
    plt.xlabel('Care of Magical Creatures Grades')
    plt.ylabel('Frequency')
    plt.title(
                "Most homogeneous distribution: "
                + "Care of Magical Creatures (Manual Selection)")
    df.boxplot(
                'Care of Magical Creatures',
                by='Hogwarts House',
                figsize=(12, 8))
    plt.ylabel('Care of Magical Creatures Grades')
    plt.show()


def select_features(df):
    houses = df['Hogwarts House'].unique().tolist()
    subjects = list(df.select_dtypes('number').to_dict().keys())
    subjects.remove('Index')
    return houses, subjects


if __name__ == '__main__':
    df = get_data(sys.argv)
    houses, subjects = select_features(df)
    manual_selection(df, houses)
    auto_selection(df, houses, subjects)
