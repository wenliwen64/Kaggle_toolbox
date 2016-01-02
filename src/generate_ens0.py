#!/usr/bin/python #TODO: test with two predictions
import pandas as pd
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-ens-files', action='store', nargs='*', dest='train_ens_files', help='folowed by a series of predictions to be ensembled for train dataset')
    parser.add_argument('--test-ens-files', action='store', nargs='*', dest='test_ens_files', help='folowed by a series of predictions to be ensembled for test dataset')
    parser.add_argument('--val-ext-feature-file', action='store', nargs='?', default=None, dest= 'train_ext_feature_file', help='raw feature to be included in the ensemble prediction from train dataset')
    parser.add_argument('--test-ext-feature-file', action='store', nargs='?', default=None,dest= 'test_ext_feature_file', help='raw feature to be included in the ensemble prediction from test dataset')
    parser.add_argument('--train-ens-feature-file', action='store', dest= 'train_ens_feature_file', help='output file of new features for train dataset')
    parser.add_argument('--test-ens-feature-file', action='store', dest= 'test_ens_feature_file', help='output file of new features for test dataset')
    args = parser.parse_args()

    train_ens_feature_list = []
    for f in args.train_ens_files:
        train_ens_feature_list.append(pd.read_csv(f).as_matrix())

    if args.train_ext_feature_file:
        train_ext_feature_matrix = pd.read_csv(args.train_ext_feature_file).as_matrix()
	train_ens_feature_list.append(train_ext_feature_matrix)

    train_ens_feature_df = pd.DataFrame(np.hstack(train_ens_feature_list))
    train_ens_feature_df.to_csv(args.train_ens_feature_file, index=False)

    test_ens_feature_list = []
    for f in args.test_ens_files:
        test_ens_feature_list.append(pd.read_csv(f).as_matrix())

    if args.test_ext_feature_file:
        test_ext_feature_matrix = pd.read_csv(args.test_ext_feature_file).as_matrix()
	test_ens_feature_list.append(test_ext_feature_matrix)

    test_ens_feature_df = pd.DataFrame(np.hstack(test_ens_feature_list))
    test_ens_feature_df.to_csv(args.test_ens_feature_file, index=False)
