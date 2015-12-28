#/usr/bin/python
import pandas as pd

raw_data_train = pd.read_csv('./raw_data/train.csv')
train_y = pd.DataFrame({
          'PassengerId': raw_data_train['PassengerId'],
	  'Survived': raw_data_train['Survived'],
	})
train_y.to_csv('./raw_data/train.y', index=False)
