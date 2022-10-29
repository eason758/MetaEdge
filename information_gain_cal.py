# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:54:27 2022

@author: NCTUUser2
"""
import numpy as np
import pandas as pd
from math import log2
from random import shuffle
import matplotlib.pyplot as plt

def ReadFile(df, col):
    df = df.drop(['alert_id','primary_entity_level_code','primary_entity_number', 'scenario_name','Cust_No','Acct_No'], axis = 1)
    ratio = 0.6
    test_set_type = 0
    train, test = SplitDataset(df, ratio, test_set_type, col)
    train = train.reset_index(drop = True)
    test = test.reset_index(drop = True)
    
    return train, test

def SplitDataset(data, p, test_set_type, SAR):
    if test_set_type == 1:
        print('Verify on random sample')
        data = shuffle(data, random_state=0)
    else:
        print('Verify on time series')
        
    print('shape of data:', data.shape)
    #data = data.reset_index(drop = True)
    TO_SAR = data[data[SAR]!=0]
    Non_SAR = data[data[SAR]==0]
    print('total number of TO_SAR:', TO_SAR.shape[0])
    print('*'*32)
    train_TO_SAR = TO_SAR[:int(TO_SAR.shape[0] * p)]
    train_Non_SAR = Non_SAR[:int(Non_SAR.shape[0] * p)]
    
    train = train_Non_SAR.append(train_TO_SAR)
    train = train.sort_values(by=['run_date'])
   
    print('time interval in train set:{}~{}'.format(train.iloc[0]['run_date'], train.iloc[train.shape[0]-1]['run_date']))
    print('TO_SAR in train set:', train_TO_SAR.shape[0])
    print('TO_SAR/Total in train set', train_TO_SAR.shape[0] / (train_Non_SAR.shape[0] + train_TO_SAR.shape[0]))
    print('split dataset to train, test...')
    print('*'*32)
    test_TO_SAR = TO_SAR[int(TO_SAR.shape[0] * p):]
    test_Non_SAR = Non_SAR[int(Non_SAR.shape[0]*(p)):]
    test = test_Non_SAR.append(test_TO_SAR)
    diff = set(train) - set(test)
    if len(diff)!=0:
        raise ValueError
    test = test.sort_values(by=['run_date'])
    print('time interval in test set:{}~{}'.format(test.iloc[0]['run_date'], test.iloc[test.shape[0]-1]['run_date']))
    print('TO_SAR in test set:', test_TO_SAR.shape[0])
    print('TO_SAR/Total in test set', test_TO_SAR.shape[0] / (test_Non_SAR.shape[0]+ test_TO_SAR.shape[0]))
    print('*'*32)
    
    train = train.drop(['run_date'], axis = 1)
    test = test.drop(['run_date'], axis = 1)
    print('train data shape: ', train.shape)
    print('test data shape: ', test.shape)
    return train, test


PATH = str('D:/Temp/NCTU/crystal/第二階段/new_alert/日數分類_營業日/')
FILE = str('TWN_A11_01_day15_with_aggregated_txn_scenario_txn_type_key_營業日.csv')

df = pd.read_csv(PATH + FILE)

df['TO_SAR'] = df['TO_SAR'].replace(np.nan, 0)
df['TO_SAR'] = df['TO_SAR'].replace('F', 1)
df['TO_SAR'] = df['TO_SAR'].replace('NF', 1)

SAR_type = 'TO_SAR'
train_set, test_set = ReadFile(df, SAR_type)

x_name = 'Credit_Amt'
y_name = 'number_of_Credit'

x_array = train_set[x_name].to_numpy()
y_array = train_set[y_name].to_numpy()

def informationGain(x_thre, y_thre, x_name, y_name, data, results):
    idx1 = set(data[data[x_name] >= x_thre].index)
    idx2 = set(data[data[y_name] >= y_thre].index)
    
    idx = list(idx1 & idx2)
    new_sam_1 = data.iloc[idx]
    new_sam_0 = data.drop(index= new_sam_1.index)
    
    if len(set(new_sam_1.index) & set(new_sam_0.index)) != 0:
        raise ValueError('new sam 0 nad new sam 1 are overlapped!')
        
    IG = father_entropy - conditionalEntropy(new_sam_0, new_sam_1)
    
    results = results.append({'credit amt': x_thre, 'number of credit': y_thre, \
                              'new_sam_1': len(new_sam_1), 'new_sam_0': len(new_sam_0), \
                                  'information gain': {IG}}, ignore_index= True)
    return results
        
def conditionalEntropy(new_sam_0, new_sam_1):
    total = len(new_sam_1) + len(new_sam_0)
    return Entropy(new_sam_0, SAR_type) * (len(new_sam_0) / total) \
            + Entropy(new_sam_1, SAR_type) * (len(new_sam_1) / total)

def Entropy(data, feature):
    total_cnt = len(data)
    if total_cnt == 0:
        return 0
    else:
        target0_cnt = len(data[data[feature]== 1])
        target1_cnt = len(data[data[feature]!= 1])
        
        if (target0_cnt == 0) | (target1_cnt == 0):
            entropy = 0
        else:
            entropy = -((target0_cnt / total_cnt) * log2(target0_cnt / total_cnt) + (target1_cnt / total_cnt) * log2(target1_cnt / total_cnt))
        
        return entropy

def SplitMedian(data, feature):
    split_number = np.unique(train_set[feature])
    split_number = split_number[~np.isnan(split_number)]
    split_number.sort(axis= 0)
    median = np.empty(len(split_number) - 1)
    for i in range(1, len(split_number)):
        median[i-1] = (split_number[i] + split_number[i-1]) / 2
    print(f'len of median: {len(median)}')
    
    return median
            
father_entropy = Entropy(train_set, SAR_type)
results = pd.DataFrame(columns= ['credit amt', 'number of credit', 'new_sam_1', 'new_sam_0', 'information gain'])

for i in range(len(x_array)):
    results = informationGain(x_array[i], y_array[i], x_name, y_name, train_set, results)
    
results.to_csv('information_gain.csv', index= False)

results = pd.read_csv(r'D:\Temp\NCTU\mark\information_gain.csv')
'''
plot the 3D picture to show the distribution of information gain between 
Credit_Amt and number of Credit. 
'''
fig = plt.figure(figsize= (10, 10))
ax = plt.axes(projection='3d')
#ax.plot3D(results.credit_amt, results.number_of_credit, results.information_gain)
ax.scatter(results.credit_amt, results.number_of_credit, results.information_gain, cmap= 'viridis')
ax.set_xlabel('Credit Amt')
ax.set_ylabel('number of Credit')
ax.set_zlabel('information gain')
plt.show()

plt.figure()
plt.scatter(results.credit_amt, results.number_of_credit)
plt.xlabel('Credit Amt')
plt.ylabel('number of Credit')
plt.savefig('credit_amt_and_number_of_credit.png')
plt.show()