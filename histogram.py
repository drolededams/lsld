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
    print(houses)
# Create a histogram. replace Herbology by good subject
    # for house in houses:
    #     grade = df.loc[df['Hogwarts House'] == house, 'Arithmancy'].dropna()
    #     plt.hist(grade, alpha=0.5, label=house)
    # plt.legend(loc='upper right')
    # plt.show()
# Create a boxplot as bonus
    # plt.figure(1)
    # df.boxplot('Arithmancy', by='Hogwarts House', figsize=(12, 8))
    # plt.figure(2)
    # df.boxplot('Astronomy', by='Hogwarts House', figsize=(12, 8))
    # plt.show()
    '''
        to do : 
            - understand ANOVA. Choose if fit (other possibility : Bertlett, Leven)
            - find selection criteria -> p value highest.
            - list of subjects (Arithmancy, Herbology, etc...)
            - loop stats.f_oneway
            - selection
            - plot hist
            - bonus : plot boxplot
    '''
    subjects = list(df.select_dtypes('number').to_dict().keys())
    subjects.remove('Index')
    print(subjects)
    p_values = {}
    for subject in subjects:
        F, p_values[subject] = stats.f_oneway(get_subject_grade_house(df, subject, 'Hufflepuff'), get_subject_grade_house(df, subject, 'Gryffindor'), get_subject_grade_house(df, subject, 'Ravenclaw'), get_subject_grade_house(df, subject, 'Slytherin'))
    p_values = {k: v for k, v in sorted(p_values.items(), key=lambda x: x[1], reverse=True)}
    print(p_values)
    homogen_subject = list(p_values.keys())[0]
    for house in houses:
        grade = df.loc[df['Hogwarts House'] == house, homogen_subject].dropna()
        plt.hist(grade, alpha=0.5, label=house)
    plt.legend(loc='upper right')
    plt.show()
