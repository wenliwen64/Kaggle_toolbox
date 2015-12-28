#!/usr/bin/python
import argparse
import numpy as np
import pandas as pd 
from sklearn.feature_extraction import DictVectorizer as DV
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate feature0...')
    parser.add_argument('--train-input-file', action='store', dest='train_input_file', help='trainning_file')
    parser.add_argument('--test-input-file', action='store', dest='test_input_file', help='testing_file')
    parser.add_argument('--train-output-file', action='store', dest='train_output_file', help='trainning_feature_file')
    parser.add_argument('--test-output-file', action='store', dest='test_output_file', help='testing_feature_file')

    args = parser.parse_args()

    #predictors = self.predictors_after_encoding
    #if dataset == 'train':
    df_train = pd.read_csv(args.train_input_file)
    df_test = pd.read_csv(args.test_input_file)
    predictors = df_test.keys().tolist()
    print(predictors)

    ftrain = open(args.train_output_file, 'w')
    ftest = open(args.test_output_file, 'w')

    for i in range(df_train.shape[0]):
        ftrain.write('{} '.format(df_train['Survived'].iloc[i])) # target variable
        for j in range(len(predictors)):
            ftrain.write('{0}:{1} '.format(j, df_train[predictors[j]].iloc[i]))
        ftrain.write('\n')

    ftrain.close()

    for i in range(df_test.shape[0]):
        ftest.write('-1 ')
        for j in range(len(predictors)):
            ftest.write('{0}:{1} '.format(j, df_test[predictors[j]].iloc[i]))
        ftest.write('\n')

    ftest.close()
