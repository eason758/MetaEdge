import numpy as np
import pandas as pd
from numpy import random

def SplitDataset(data, p, test_set_type, SAR):
    if test_set_type == 1:
        print('Verify on random sample')
        data = shuffle(data, random_state= 0)
    else:
        print('Verify on time series')

    print('shape of data:', data.shape)
    TO_SAR = data[data[SAR] != 0]
    Non_SAR = data[data[SAR] == 0]
    print('total number of TO_SAR:', TO_SAR.shape[0])
    print('*'*32)

    train_TO_SAR = TO_SAR[:int(TO_SAR.shape[0] * p)]
    train_Non_SAR = Non_SAR[:int(Non_SAR.shape[0] * p)]

    #train = train_Non_SAR.append(train_TO_SAR)
    train = pd.concat([train_Non_SAR, train_TO_SAR], axis= 0, ignore_index= True)
    train = train.sort_values(by= 'run_date')

    print(f"time interval in train set:{train.iloc[0]['run_date']}~{train.iloc[train.shape[0]-1]['run_date']}")
    print('TO_SAR in train set:', train_TO_SAR.shape[0])
    print('split dataset to train, test ....')
    print('*'*32)

    test_TO_SAR = TO_SAR[int(TO_SAR.shape[0] * p):]
    test_Non_SAR = Non_SAR[int(Non_SAR.shape[0] * (p)):]
    #test = test_Non_SAR.append(test_TO_SAR)
    test = pd.concat([test_Non_SAR, test_TO_SAR], axis= 0, ignore_index= True)
    diff = set(train) - set(test)

    if len(diff) != 0:
        raise ValueError('train set and test set are overlapped!')

    test = test.sort_values(by= 'run_date')

    print(f"time interval in test set:{test.iloc[0]['run_date']}~{test.iloc[test.shape[0]-1]['run_date']}")
    print('TO_SAR in test set:', test_TO_SAR.shape[0])
    print('TO_SAR/Total in test set', test_TO_SAR.shape[0] / (test_Non_SAR.shape[0] + test_TO_SAR.shape[0]))
    print('*'*32)

    train = train.drop('run_date', axis= 1)
    test = test.drop('run_date', axis= 1)
    print('train data shape:', train.shape)
    print('test data shape:', test.shape)

    return train, test

def ReadFile(df, col):
    df = df.drop(['alert_id', 'primary_entity_level_code', 'primary_entity_number', 'scenario_name' \
                    , 'Cust_No', 'Cust_No', 'Acct_No'], axis= 1)
    ratio = 0.6
    test_set_type = 0
    train, test = SplitDataset(df, ratio, test_set_type, col)
    train = train.reset_index(drop= True)
    test = test.reset_index(drop= True)
    
    return train, test

    
PATH = 'D:/Temp/NCTU/crystal/第二階段/new_alert/日數分類_營業日/'
FILE = 'TWN_A11_01_day15_with_aggregated_txn_scenario_txn_type_key_營業日.csv'
df = pd.read_csv(PATH + FILE)

df['TO_SAR'] = df['TO_SAR'].replace(np.nan, 0)
df['TO_SAR'] = df['TO_SAR'].replace('F', 1)
df['TO_SAR'] = df['TO_SAR'].replace('NF', 1)

SAR_type = 'TO_SAR'
train_set, test_set = ReadFile(df, SAR_type)

train_set = train_set[['Credit_Amt', 'number_of_Credit', 'Debit_Amt', 'number_of_Debit']]
test_set = test_set[['Credit_Amt', 'number_of_Credit', 'Debit_Amt', 'number_of_Debit']]

z = np.random.normal(loc= 0, scale= 1000, size= train_set.shape[0]).reshape(-1, 1)
a = np.random.normal(loc= 0, scale= 5, size= train_set.shape[0]).reshape(-1, 1)
train_noise = np.hstack((z, a))
train_noise = np.hstack((train_noise, train_noise))
train_set = train_set + train_noise

z = np.random.normal(loc= 0, scale= 1000, size= test_set.shape[0]).reshape(-1, 1)
a = np.random.normal(loc= 0, scale= 5, size= test_set.shape[0]).reshape(-1, 1)
test_noise = np.hstack((z, a))
test_noise = np.hstack((test_noise, test_noise))

test_set = test_set + test_noise
print(f'train_set:{train_set.shape}, test_set:{test_set.shape}')

train_set.to_csv('noise_train.csv', index= False)
test_set.to_csv('noise_test.csv', index= False)