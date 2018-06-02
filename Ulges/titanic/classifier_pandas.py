#!/usr/bin/python
# -*- coding: utf-8 -*-


################################################
#####   EMPOLIS MACHINE LEARNING WORKSHOP  #####
###########     - MAR 22 2016 -    #############
################################################

"""
This code showcases basic machine learning 
techniques as presented in the workshop.

Contact: Adrian Ulges (adrian.ulges@hs-rm.de)
"""

# some useful imports
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


# import training and test data
data      = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")


# impute some missing values!
def impute_missing_values(data, data_test):

    # impute Fare
    mean_fare = np.mean(data[data['Fare']!=0]['Fare'])
    data.loc[data['Fare']!=0,'Fare'] = mean_fare
    data_test.loc[data_test['Fare']!=0,'Fare'] = mean_fare

    # impute Embarked
    mode_embarked = mode(data['Embarked'])[0][0]
    data['Embarked'] = data['Embarked'].fillna(mode_embarked)
    data_test['Embarked'] = data_test['Embarked'].fillna(mode_embarked)

    # impute Age
    data.loc[np.isnan(data['Age']),'Age'] = -1
    data_test.loc[np.isnan(data_test['Age']),'Age'] = -1

    return data, data_test


# export results to csv (which could be uploaded
# to kaggle.com for evaluation)
def result_to_csv(result, filename):
    result.to_csv(filename, 
                  index=False,
                  index_label=['PassengerId','Survived'],
                  header=True)


# divide fare into categories (0-10, 10-20, 20-30, ...)
def farecat(x):
    return min(int(x/10), 3) if not np.isnan(x) else 0


# a first shot at classification deriving a simple decision
# rule from a few useful features (sex, fare, age)
def model_first_shot(data, data_test):

    def decision(sex, farecat, child, pivot):
        return int( pivot[1][sex,farecat,child] >= pivot[0][sex,farecat,child] )

    pivot = data.pivot_table(values=[],
                             rows=['Sex','Farecat','Child'],
                             cols=['Survived'],
                             aggfunc=len)
    data_test['Survived'] = data_test.apply(lambda row: decision(row['Sex'],
                                                                 row['Farecat'],
                                                                 row['Child'],
                                                                 pivot),
                                            axis=1)
    result = data_test[['PassengerId','Survived']]
    return result


# extracts person title of person and categorizes people
# into six 6 categories according to sex, wealth, and age. 
def name2title(name):
    title = name.split(".")[0].split(",")[-1].strip()
    if title in ['Capt', 'Col', 'Dr', 'Major', 'Don', 'Rev', 'Sir']:
        return 'male_upper'
    elif title in ['Lady', 'the Countess']:
        return 'female_upper'
    elif title in ['Master', 'Jonkheer']:
        return 'male_young'
    elif title in ['Mlle', 'Miss']:
        return 'female_young'
    elif title in ['Mr']:
        return 'male'
    elif title in ['Mrs', 'Mme', 'Ms', 'Dona']:
        return 'female'
    raise ValueError("BAD TITLE! " + name)


# selects the most useful features and converts them
# into sklearn format (using 'dummy variables')
def data2samples_dectree(data):
    data['Title'] = data['Name'].map(name2title).astype('category',
                                                        categories=['male','female',
                                                                    'male_upper','female_upper',
                                                                    'male_young','female_young'])

    # dummy variable for fare title, fare, sex, pclass
    title_dummies    = pd.get_dummies(data['Title'], prefix='Title')
    farecat_dummies  = pd.get_dummies(data['Farecat'], prefix='Farecat')
    sex_dummies      = pd.get_dummies(data['Sex'], prefix='Sex')
    pclass_dummies   = pd.get_dummies(data['Pclass'], prefix='Pclass')

    samples = pd.concat([data[['Child']], farecat_dummies, sex_dummies, pclass_dummies, title_dummies], axis=1)

    return samples


# transforms training and test data into sklearn format
# and trains, exports, and applies a decision tree classifier.
def model_dectree(data, data_test):

    samples      = data2samples_dectree(data)
    samples_test = data2samples_dectree(data_test)
    labels = data['Survived']

    model = tree.DecisionTreeClassifier()
    model = model.fit(samples, labels)

    with open('tree.dot', 'w') as f:
        f = tree.export_graphviz(model, out_file=f)

    result = data_test['PassengerId'].copy().to_frame()
    result['Survived'] = model.predict(samples_test)

    return result


if __name__ == "__main__":

    impute_missing_values(data, data_test)

    # add some more features
    data['Farecat'] = data['Fare'].map(farecat)
    data['Child']   = data ['Age'].map(lambda x: int(x<18 and x>0))
    data_test['Farecat'] = data_test['Fare'].map(farecat)
    data_test['Child']   = data_test['Age'].map(lambda x: int(x<18 and x>0))

    # first shot model
    # result = model_first_shot(data, data_test)
    # result_to_csv(result, 'pred_pandas_fareclass_sex_child.csv')

    # decision tree model
    result = model_dectree(data, data_test)
    result_to_csv(result, 'pred_pandas_dectree.csv')
