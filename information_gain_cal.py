# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:54:27 2022

@author: NCTUUser2
"""
from sre_constants import IN
import numpy as np
import pandas as pd
from math import log2
from random import shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm
'''
function base
'''
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

def Entropy(data, SAR):
    total_cnt = len(data)
    if total_cnt == 0:
        return 0
    else:
        target_0 = len(data[data[SAR] == 0])
        target_1 = len(data[data[SAR] != 0])

        if target_0 == 0 or target_1 == 0:
            entropy = 0
        else:
            entropy = -((target_0 / total_cnt) * log2((target_0) / total_cnt) \
                        + (target_1 / total_cnt) * log2((target_1 / total_cnt)))
    
    return entropy

def ConditionEntropy(new_sam_0, new_sam_1):
    total_cnt = len(new_sam_0) + len(new_sam_1)
    condentropy0 = Entropy(new_sam_0, SAR_type) * (len(new_sam_0) / total_cnt)
    condentropy1 = Entropy(new_sam_1, SAR_type) * (len(new_sam_1) / total_cnt)

    return condentropy0 + condentropy1

def InformationGain(x_thre, y_thre, x_name, y_name, data):
    idx1 = set(data[data[x_name] >= x_thre].index)
    idx2 = set(data[data[y_name] >= y_thre].index)
    idx = list(idx1 & idx2)
    
    new_sam_1 = data.iloc[idx]
    new_sam_0 = data.drop(index= new_sam_1.index)

    if len(set(new_sam_1.index) & set(new_sam_0.index)) != 0:
        raise ValueError('new sam 0 and new sam 1 are overlapped!')
    
    condition_entropy = ConditionEntropy(new_sam_0, new_sam_1)
    IG = -(father_entropy - condition_entropy)
    return len(new_sam_0), len(new_sam_1), IG

def RecallFilterRate(x_thre, y_thre, x_name, y_name, data):
    idx1 = set(data[data[x_name] >= x_thre].index)
    idx2 = set(data[data[y_name] >= x_thre].index)
    idx = list(idx1 & idx2)
    
    new_sam_1 = data.iloc[idx]
    new_sam_0 = data.drop(index= new_sam_1.index)

    if len(set(new_sam_1.index) & set(new_sam_0.index)) != 0:
        raise ValueError('new sam 0 and new sam 1 are overlapped!')

    TP = new_sam_1[new_sam_1[SAR_type] != 0].shape[0]
    TN = new_sam_0[new_sam_0[SAR_type] == 0].shape[0]
    FP = new_sam_1[new_sam_1[SAR_type] == 0].shape[0]
    FN = new_sam_0[new_sam_0[SAR_type] != 0].shape[0]

    recall = TP / (TP + FN)
    filter_rate = (TN + FN) / (TP + TN + FP + FN)

    return recall, filter_rate

'''
main function
'''
PATH = 'D:/Temp/NCTU/crystal/第二階段/new_alert/日數分類_營業日/'
FILE = 'TWN_A11_01_day15_with_aggregated_txn_scenario_txn_type_key_營業日.csv'
df = pd.read_csv(PATH + FILE)
df.columns

df['TO_SAR'] = df['TO_SAR'].replace(np.nan, 0)
df['TO_SAR'] = df['TO_SAR'].replace('F', 1)
df['TO_SAR'] = df['TO_SAR'].replace('NF', 1)

SAR_type = 'TO_SAR'
train_set, test_set = ReadFile(df, SAR_type)

SAR = train_set[train_set.TO_SAR!= 0]
Non_SAR = train_set[train_set.TO_SAR== 0]

father_entropy = Entropy(train_set, SAR_type)
x_name = 'Credit_Amt'
y_name = 'number_of_Credit'

#x, y = np.meshgrid(train_set[x_name], train_set[y_name])
# x = train_set[x_name].to_numpy()
# y = train_set[y_name].to_numpy()

x = np.unique(train_set['Credit_Amt'])
x = np.linspace(x[0], x[-1], 1000)
y = np.unique(train_set['number_of_Credit'])
y = np.linspace(y[0], y[-1], 1000)
x, y = np.meshgrid(x, y)

results = pd.DataFrame(columns= ['Credit_Amt', 'number_of_Credit', 'new sam 0', 'new sam 1','information gain', 'recall', 'filter rate'])

for i in tqdm(range(x.shape[0])):
    for j in range(x.shape[1]):
    #for j in range(len(y)):
        new_sam_0, new_sam_1, information_gain = InformationGain(x[i][j], y[i][j], x_name, y_name, train_set)
        recall, filter_rate = RecallFilterRate(x[i][j], y[i][j], x_name, y_name, train_set)

        tmp = pd.DataFrame({'Credit_Amt': [x[i][j]], 'number_of_Credit': [y[i][j]], \
                'information gain': information_gain, 'new sam 0': new_sam_0, 'new sam 1': new_sam_1, 'recall': recall, 'filter rate': filter_rate})

        results = pd.concat([results, tmp], axis= 0, ignore_index= True)

print(results.shape)

results.to_csv('credit_information_gain.csv', index= False)