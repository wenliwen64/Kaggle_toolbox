#!/usr/bin/python
'''
feature1: no onehotencoding
'''
import argparse
import numpy as np
import pandas as pd 
from sklearn.feature_extraction import DictVectorizer as DV
import re

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
	return title_search.group(1)
    else:
	return ''
 
def get_family_id(row):
    last_name = row['Name'].split(',')[0]
    family_id = '{0}{1}'.format(last_name, row['FamilySize'])
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
	    current_id = 1
        else:
            current_id = (max(family_id_mapping.items(), key=itemgetter(1))[1]+1)
            family_id_mapping[family_id] = current_id

        family_id_mapping[family_id] = current_id

def generate_new_features(dataset=None):  # DONE!!!

    ds = dataset
    family_id_mapping = {}
    ds['FamilySize'] = ds['SibSp'] 
    ds['NameLength'] = ds['Name'].apply(lambda x: len(x))
    titles = ds['Name'].apply(get_title)
    print('titles:')
    
    title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5, 'Rev': 6, 'Col': 7, 'Major': 8, 'Mlle': 9, 'Countess': 10, 'Ms': 11, 'Lady': 12, 'Jonkheer': 13, 'Don': 14, 'Mme': 15, 'Capt': 16, 'Sir': 17, 'Dona':18}

    for k, v in title_mapping.items():
	titles[titles == k] = v 

    ds['Titles'] = titles

    return dataset.copy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate feature0...')
    parser.add_argument('--train-file', action='store', dest='train_file', help='trainning_file')
    parser.add_argument('--test-file', action='store', dest='test_file', help='testing_file')
    parser.add_argument('--train-feature-file', action='store', dest='train_feature_file', help='trainning_feature_file')
    parser.add_argument('--test-feature-file', action='store', dest='test_feature_file', help='testing_feature_file')

    args = parser.parse_args()

    df_train = pd.read_csv(args.train_file)
    df_test = pd.read_csv(args.test_file)

#======================
    for ds in [df_train, df_test]:
        ds.loc[(ds['Age'].isnull()) & (ds['SibSp']>=2), 'Age'] = 11 #median
        ds.loc[(ds['Age'].isnull()) & (ds['SibSp']<2) & (ds['Sex']=='male'), 'Age'] =30 
        ds.loc[(ds['Age'].isnull()) & (ds['SibSp']<2) & (ds['Sex']=='female'), 'Age'] =29 

	ds.loc[ds['Sex']=='male', 'Sex'] = 0 
	ds.loc[ds['Sex']=='female', 'Sex'] = 1 
        ds['Embarked'] = ds['Embarked'].fillna('S')

        ds.loc[ds['Embarked']=='S', 'Embarked'] = 0 
        ds.loc[ds['Embarked']=='C', 'Embarked'] = 1 
        ds.loc[ds['Embarked']=='Q', 'Embarked'] = 2 

	ds['Fare'] = ds['Fare'].fillna(ds['Fare'].median())  # only for test dataset in this case, or we can use kde 


#========================
    df_train = generate_new_features(df_train)
    df_test = generate_new_features(df_test)

    predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Titles"]
    df_train = df_train[predictors]
    df_test = df_test[predictors]
    df_train.to_csv(args.train_feature_file, index=False)
    df_test.to_csv(args.test_feature_file, index=False)
