#!/usr/bin/python

import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.cross_validation import KFold
from operator import itemgetter

def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', action='store', dest='train_file', help='train dataset')
    parser.add_argument('--train-y-file', action='store', dest='train_y_file', help='train target dataset in regular csv format')
    parser.add_argument('--test-file', action='store', dest='test_file', help='test dataset')
    parser.add_argument('--train-dmatrix', action='store', dest='train_dmatrix', help='train_dmatrix')
    parser.add_argument('--test-dmatrix', action='store', dest='test_dmatrix', help='test_dmatrix')
    parser.add_argument('--val-predict-file', action='store', dest='val_predict_file', help='predictitions on cross-validation dataset(training data)')
    parser.add_argument('--val-soft-predict-file', action='store', dest='val_soft_predict_file', help='soft predictitions on cross-validation dataset(training data)')
    parser.add_argument('--test-predict-file', action='store', dest='test_predict_file', help='predictions on test dataset')
    parser.add_argument('--test-soft-predict-file', action='store', dest='test_soft_predict_file', help='soft predictions on test dataset')
    parser.add_argument('--train-id-file', action='store', dest='train_id_file', help='id for train samples')
    parser.add_argument('--test-id-file', action='store', dest='test_id_file', help='id for test samples')
    parser.add_argument('--grid-search', action='store_true', dest='grid_search', help='turn on to do grid search for optimized hyperparameters')
    parser.add_argument('--rand-search', action='store_true', dest='rand_search', help='turn on to do randomized search for optimized hyperparameters')
    parser.add_argument('--cv', action='store', dest='n_cv', type=int, help='number of cross-validation folds')
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_file)
    train_y_df = pd.read_csv(args.train_y_file)
    test_df = pd.read_csv(args.test_file)
    train_id_df = pd.read_csv(args.train_id_file)
    test_id_df = pd.read_csv(args.test_id_file)
    train_dmatrix = xgb.DMatrix(args.train_dmatrix)
    test_dmatrix = xgb.DMatrix(args.test_dmatrix)

    predictors = test_df.keys().tolist()

    if args.rand_search: 
        param_dist = {
	    'max_depth': sp_randint(3, 10),
	    'learning_rate': [0.01, 0.03, 0.1, 0.3, 1.0],
	    'gamma': [0, 0.1, 0.2, 0.3],
	    'subsample': [.1, .2, .3, .4, 0.5],
	    'colsample_bytree': [.4, .5],
	    'objective': ['binary:logistic'],
	    'n_estimators': sp_randint(20, 150),
	}

        clf = xgb.XGBClassifier()
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=500, cv=args.n_cv)
	random_search.fit(train_df[predictors], train_y_df['Survived'])

        print('========xgb random search results============')
	report(random_search.grid_scores_)

    elif args.grid_search:
        param_grid = {
	    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
	    'learning_rate': [0.01, 0.03, 0.1, 0.3, 1.0],
	    'gamma': [0, 0.1, 0.2, 0.3],
	    'subsample': [.1, .2, .3, .4, 0.5],
	    'colsample_bytree': [.4, .5],
	    'objective': ['binary:logistic'],
	    'n_estimators': np.arange(20, 500, 10),
	}

        clf = xgb.XGBClassifier()
        grid_search = GridSearchCV(clf, param_grid=param_grid, cv=args.n_cv)
	grid_search.fit(train_df[predictors], train_y_df['Survived'])

        print('========xgb grid search results============')
	report(random_search.grid_scores_)

    else:
        params = {'max_depth': 9, 'learning_rate': 0.1, 'colsample_bytree': 0.5, 'n_estimators': 75, 'subsample': .5, 'gamma': 0.3, 'objective':'binary:logistic', 'eval_metric': 'auc'} #0.845, cv=3 
        bst = xgb.train(params, train_dmatrix)
        predictions = pd.Series(bst.predict(test_dmatrix))
        predictions_proba = predictions.copy()
        predictions[predictions >= .5] = 1 
        predictions[predictions < .5] = 0 
        predictions = [int(x) for x in predictions.tolist()] #Otherwise will be provided float

        submission_test = pd.DataFrame({
	    #'PassengerId': test_id_df,
	    'Survived': predictions, 
	    })
        submission_test.to_csv(args.test_predict_file,  index=False)

        submission_proba_test = pd.DataFrame({
	    #'PassengerId': test_id_df,
	    'Survived': predictions_proba,
	    })
        submission_proba_test.to_csv(args.test_soft_predict_file, index=False)

#==================Cross Validation========================================
        kf = KFold(train_df.shape[0], n_folds=args.n_cv, random_state=1)
       
        predictions = []
        for train, test in kf:
            bst = xgb.train(params, train_dmatrix.slice(train)) 
            cv_predictions = bst.predict(train_dmatrix.slice(test))
	    predictions.append(cv_predictions)

        predictions = np.concatenate(predictions, axis=0)
	predictions_proba = predictions.copy()
        predictions[predictions >= .5] = 1 
        predictions[predictions < .5] = 0 
	predictions = [int(x) for x in predictions]

        submission_val = pd.DataFrame({
	    #'PassengerId': train_id_df,
	    'Survived': predictions,
	    })
        submission_val.to_csv(args.val_predict_file, index=False)

	submission_proba_val = pd.DataFrame({
            #'PassengerId': train_id_df,
	    'Survived': predictions_proba,
	    })
        submission_proba_val.to_csv(args.val_soft_predict_file, index=False)
