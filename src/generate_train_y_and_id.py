#/usr/bin/python
import pandas as pd
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--train-raw-file', action='store', dest='train_raw_file', help='train raw data')
    parser.add_argument('--test-raw-file', action='store', dest='test_raw_file', help='test raw data')
    parser.add_argument('--train-y-file', action='store', dest='train_y_file', help='train y data')
    parser.add_argument('--train-id-file', action='store', dest='train_id_file', help='train id data')
    parser.add_argument('--test-id-file', action='store', dest='test_id_file', help='test id data')
    args = parser.parse_args()

    train_raw_data = pd.read_csv(args.train_raw_file)
    test_raw_data = pd.read_csv(args.test_raw_file)
    train_y = pd.DataFrame({
          #'PassengerId': train_raw_data['PassengerId'],
	  'Survived': train_raw_data['Survived'],
	  })
    train_y.to_csv(args.train_y_file, index=False)

    train_id = pd.DataFrame({
          'PassengerId': train_raw_data['PassengerId'],
	  })
    train_id.to_csv(args.train_id_file, index=False)
    
    test_id = pd.DataFrame({
          'PassengerId': test_raw_data['PassengerId'],
	  })
    test_id.to_csv(args.test_id_file, index=False)
