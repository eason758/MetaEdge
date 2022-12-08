# %%
import os
from math import ceil
import numpy as np
import pandas as pd
from scipy.special import entr
import matplotlib.pyplot as plt
from math import log2, log
from scipy.spatial import cKDTree
# import warnings
# warnings.filterwarnings('ignore')
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# %%
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


# file
def ReadFile(df, col):
    df = df.drop(['alert_id','primary_entity_level_code','primary_entity_number', 'scenario_name','Cust_No','Acct_No'], axis = 1)
    ratio = 0.6
    test_set_type = 0
    train, test = SplitDataset(df, ratio, test_set_type, col)
    train = train.reset_index(drop = True)
    test = test.reset_index(drop = True)
    
    return train, test


def getData(df, ratio, oversample, test_set_type, col):
    
    train, test= SplitDataset(df, ratio, test_set_type, col)
         
    print('TO_SAR=1:To_SAR=0 = {}:{}'.format(list(train['TO_SAR'].values).count(1), list(train['TO_SAR'].values).count(0) ))
    print('*'*32)
    
    if oversample == 0:
        print('Not doing oversampling...')
    elif oversample == 1:
        print('Oversampling by duplicate TO SAR=1 data...')
        resample_ratio = train[train['TO_SAR']==0].shape[0] / train[train['TO_SAR']==1].shape[0]
        #print('resample ratio:', resample_ratio)
        train = train.append([train[train['TO_SAR']==1]]*(int(resample_ratio)-1))
        print('TO SAR=1:To SAR=0 ={}:{} in train set'.format(list(train['TO_SAR'].values).count(1), list(train['TO_SAR'].values).count(0) ))
   

    print('fradulent in train/test:'.format(list(train['TO_SAR'].values).count(1),
                                            list(test['TO_SAR'].values).count(1)))
   
    train_data = train.drop(['TO_SAR'], axis = 1).to_numpy()
    train_label = train['TO_SAR'].to_numpy()
    test_data = test.drop(['TO_SAR'], axis = 1).to_numpy()
    test_label = test['TO_SAR'].to_numpy()
   
    return train_data, train_label, test_data, test_label, test

# %%
def getResult(y_true, y_pred):
    results = pd.DataFrame(columns = ['SAR', 'Non SAR','newSAM=1_SAR=1(TP)','newSAM=1_SAR=0(FP)',\
                                      'newSAM=0_SAR=1(FN)', 'newSAM=0_SAR=0(TN)', 'recall','filter rate'])
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_pred)): 
        if y_true[i]==y_pred[i]==1:
            TP += 1
        if y_pred[i]==1 and y_true[i]==0: 
            FP += 1
        if y_true[i]==y_pred[i]==0:
            TN += 1
        if y_pred[i]==0 and y_true[i]==1:
            FN += 1

    HRS = (TN+FN)/(TN+FN+TP+FP)
   
    results = results.append({'SAR':list(y_true).count(1), \
                              'Non SAR':list(y_true).count(0),\
                              'newSAM=1_SAR=1(TP)':TP, 'newSAM=1_SAR=0(FP)':FP, 'newSAM=0_SAR=1(FN)':FN,\
                              'newSAM=0_SAR=0(TN)':TN, 'recall': str(round(TP/(TP+FN), 4)),\
                              'filter rate':str((round(HRS, 4)))}, ignore_index = True)
    display(results)


def train(threshold_test, train_data, train_label, test_data, test_label):
    print('Training session starts...')
    model = XGBClassifier(use_label_encoder = False, eval_metric = 'logloss', n_jobs = -1).fit(train_data, train_label)
    if threshold_test:
        recalls = []
        HRSs = []
        for th in np.arange(1e-10, 1e-9, 1e-10):
            print('threshold = ', th)
            predicted_proba = model.predict_proba(test_data)
            y_pred = (predicted_proba[:,1]>=th)
            getResult(test_label,y_pred)
        
    else:
        predicted_proba = model.predict_proba(test_data)
        y_pred = (predicted_proba[:,1]>=0.5)
        getResult(test_label, y_pred)

# %%
def DrawFeatureDistribution(data, feature, day, label):
    data[feature].value_counts() # 存款金額
    #print(data[feature].value_counts())
    y, x = data[feature].value_counts().values, data[feature].value_counts().index
    plt.figure()
    plt.title("distribution of {} of SAR={} in past {} days".format(feature, label, day))
    plt.xlabel("{} in past {} days".format(feature, day))
    plt.ylabel("number of data")
    plt.scatter(x, y)
    plt.show()

# %%
# Hyper parameter settings
RANK = 10
SLICE = 10
beta = 2 # give recall weight=2

# %%
# 閥值x>=且閥值y>=
def Search2D(xRange, yRange, x, y, defaultX, defaultY, SAR, Non_SAR, train_set, LOGIC, SAR_type):
    # x, y: feature
    numberOfSAR = [[0] * SLICE for i in range(SLICE)] #sar個數
    recallRate = [[0] * SLICE for i in range(SLICE)]
    filterRate = [[0] * SLICE for i in range(SLICE)]
    harmonicRecallFilter = [[0] * SLICE for i in range(SLICE)] # apply f1-score simulated formula
    for i in range(len(xRange)):
        for j in range(len(yRange)):
            thre1 = xRange[i]
            thre2 = yRange[j] 
            if thre1 < defaultX or thre2 < defaultY:
                continue
            idx1 = set(train_set[train_set[y] >= thre2].index)
            idx2 = set(train_set[train_set[x] >= thre1].index)

            idx = list(idx1 & idx2) # 閥值x>= "且" 閥值y>=
            new_SAM_1 = train_set.iloc[idx]
            new_SAM_0 = train_set.drop(index = new_SAM_1.index)
    
            if len(set(new_SAM_0.index) & set(new_SAM_1.index)) != 0:
                raise ValueError("new SAM 0 and new SAM 1 overlapped!")

            TP = new_SAM_1[new_SAM_1[SAR_type] != 0].shape[0]
            TN = new_SAM_0[new_SAM_0[SAR_type] == 0].shape[0]
            FP = new_SAM_1[new_SAM_1[SAR_type] == 0].shape[0]
            FN = new_SAM_0[new_SAM_0[SAR_type] != 0].shape[0]
        
            numberOfSAR[i][j] = TP
            recallRate[i][j] = TP / (TP + FN)
            filterRate[i][j] = (TN + FN) / (TP + FP + TN + FN)
            harmonicRecallFilter[i][j] = ((1 + beta * beta) * recallRate[i][j] * filterRate[i][j])\
            / (recallRate[i][j] + beta * beta * filterRate[i][j])
            #print(thre1, thre2, 'TP:',TP,  'FP:',FP,  'FN:',FN,  'TN:',TN,  'recall:',recallRate[i][j] , 'filter:', filterRate[i][j], harmonicRecallFilter[i][j])
            
    if LOGIC == "AND":
        mat = recallRate
    elif LOGIC == "OR":
        mat = harmonicRecallFilter
    
    max_pos = [] 
    maxVal = np.amax(mat) # find the biggest values
    #print('max val', maxVal)
    countMax = 0
    countZero = 0
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == maxVal:
                countMax += 1
            if mat[i][j] == 0:
                countZero += 1
                
    print('max val: ', maxVal, 'number of max val: ', countMax)
    print(f'Rank: {RANK}, countMax: {countMax}, {SLICE}- countZero: {SLICE**2 -countZero}')
    for i in range(min(max(RANK, countMax), SLICE**2 - countZero)):
        idx = np.array(mat).argmax() # find the index where the biggest values is
        max_pos.append(idx)
        print('{}th, recall: {}, val1: {}, val2: {}'.format(idx, recallRate[int(idx / SLICE)][idx % SLICE], \
                                                            xRange[int(idx / SLICE)],yRange[idx % SLICE]))
        mat[int(idx / SLICE)][idx % SLICE]=-1
        
    print('*'*32)
    candidate_rule = []
    for i in range(min(RANK, SLICE**2 - countZero)):
        idx = max_pos[-(i + 1)]
        val0 = recallRate[int(idx / SLICE)][idx % SLICE]
        val1 = xRange[int(idx / SLICE)]
        val2 = yRange[idx % SLICE]
        candidate_rule.append([val1, val2])
   
    """results = pd.DataFrame(columns = [x,y,'SAR', 'Non SAR','newSAM=1_SAR=1(TP)','newSAM=1_SAR=0(FP)',\
                                  'newSAM=0_SAR=1(FN)', 'newSAM=0_SAR=0(TN)', 'recall','filter rate'])
    
    for thre1, thre2 in candidate_rule:
        idx = list(set(train_set[train_set[x] >= thre1].index) & set(train_set[train_set[y] >= thre2].index))
        new_SAM_1_rule1 = train_set.iloc[idx]
        new_SAM_1 = new_SAM_1_rule1
        new_SAM_1 = new_SAM_1.drop_duplicates()
        new_SAM_0 = train_set.drop(index = new_SAM_1.index)

        if (new_SAM_0.shape[0] + new_SAM_1.shape[0]) != train_set.shape[0]:
            raise ValueError("new SAM 0 and new SAM 1 overlapped!")
            
        TP = new_SAM_1[new_SAM_1[SAR_type] != 0].shape[0]
        TN = new_SAM_0[new_SAM_0[SAR_type] == 0].shape[0]
        FP = new_SAM_1[new_SAM_1[SAR_type] == 0].shape[0]
        FN = new_SAM_0[new_SAM_0[SAR_type] != 0].shape[0]
        total = train_set.shape[0]
        results = results.append({x:thre1, y:thre2,'SAR':train_set[train_set[SAR_type] != 0].shape[0], \
                                  'Non SAR':train_set[train_set[SAR_type] == 0].shape[0],\
                                  'newSAM=1_SAR=1(TP)':TP, 'newSAM=1_SAR=0(FP)':FP, 'newSAM=0_SAR=1(FN)':FN,\
                                  'newSAM=0_SAR=0(TN)':TN, 'recall': str(round(TP/(TP + FN), 4)),\
                                  'filter rate':str((round((TN + FN) / total, 4)))}, ignore_index = True)
    display(results)"""
    
    return candidate_rule


def Search1D(xRange, x, defaultX, SAR, Non_SAR, train_set, LOGIC, SAR_type):
    numberOfSAR = [0] * (2 * SLICE - 1) #sar個數
    recallRate = [0] * (2 * SLICE - 1) 
    filterRate= [0] *(2 * SLICE - 1)  
    harmonicRecallFilter = [0] * (2 * SLICE - 1)  # apply f1-score simulated formula
    
    for i in range(len(xRange)):
        thre1 = xRange[i]
        
        if thre1 < defaultX:
            continue
            
        idx1 = set(train_set[train_set[x] >= thre1].index)
        idx = list(idx1) # 閥值x>= 

        new_SAM_1_rule1 = train_set.iloc[idx]
        new_SAM_1 = new_SAM_1_rule1
        new_SAM_1 = new_SAM_1.drop_duplicates()
        new_SAM_0 = train_set.drop(index = new_SAM_1.index)

        if (new_SAM_0.shape[0] + new_SAM_1.shape[0]) != train_set.shape[0]:
            raise ValueError("new SAM 0 and new SAM 1 overlapped!")

        TP = new_SAM_1[new_SAM_1[SAR_type] != 0].shape[0]
        TN = new_SAM_0[new_SAM_0[SAR_type] == 0].shape[0]
        FP = new_SAM_1[new_SAM_1[SAR_type] == 0].shape[0]
        FN = new_SAM_0[new_SAM_0[SAR_type] != 0].shape[0]
    
        numberOfSAR[i] = TP
        recallRate[i] = TP / (TP + FN)
        filterRate[i] = (TN + FN)/(TP + FP + TN + FN)
        harmonicRecallFilter[i] = float((1 + beta * beta) * recallRate[i] * filterRate[i]\
                                        / (recallRate[i] + beta * beta * filterRate[i]))
        
        print(recallRate[i])
        
    if LOGIC == "AND":
        mat = recallRate
    elif LOGIC == "OR":
        mat = harmonicRecallFilter

        
    print('recall', mat)
    max_pos = [] 
    maxVal = max(mat)
    countMax = mat.count(maxVal)
    countZero = mat.count(0)
    
    print('max val: ', maxVal, 'number of max val: ', countMax)
    print('number of zero:', countZero)
    for i in range(min(max(RANK, countMax), 19-countZero)):
        idx = np.array(mat).argmax()
        max_pos.append(idx)
        print('{}th, recall: {} , val: {}'.format(idx, recallRate[idx % (2 * SLICE - 1)], xRange[idx % (2 * SLICE - 1)]))
        mat[idx % (2 * SLICE - 1)] = -1
        
    candidate_rule = []
    
    for i in range(min(RANK, 19 - countZero)):
        idx = max_pos[-(i + 1)]
        x = xRange[idx % (2 * SLICE - 1)]
        candidate_rule.append(x)
        
    
    results = pd.DataFrame(columns = [x, 'SAR', 'Non SAR','newSAM=1_SAR=1(TP)','newSAM=1_SAR=0(FP)',\
                                  'newSAM=0_SAR=1(FN)', 'newSAM=0_SAR=0(TN)', 'recall','filter rate'])
    
    """for thre1 in candidate_rule:
        print(thre1)
        display(train_set[train_set[x] >= float(thre1)])
        idx = train_set[train_set[x] >= thre1].index
        new_SAM_1 = train_set.iloc[idx]
        new_SAM_0 = train_set.drop(index = new_SAM_1.index)

        if (new_SAM_0.shape[0] + new_SAM_1.shape[0]) != train_set.shape[0]:
            raise ValueError("new SAM 0 and new SAM 1 overlapped!")
            
        TP = new_SAM_1[new_SAM_1[SAR_type] != 0].shape[0]
        TN = new_SAM_0[new_SAM_0[SAR_type] == 0].shape[0]
        FP = new_SAM_1[new_SAM_1[SAR_type] == 0].shape[0]
        FN = new_SAM_0[new_SAM_0[SAR_type] != 0].shape[0]
        total = train_set.shape[0]
        results = results.append({x:thre1, 'SAR':train_set[train_set[SAR_type] != 0].shape[0], \
                                  'Non SAR':train_set[train_set[SAR_type] == 0].shape[0],\
                                  'newSAM=1_SAR=1(TP)':TP, 'newSAM=1_SAR=0(FP)':FP, 'newSAM=0_SAR=1(FN)':FN,\
                                  'newSAM=0_SAR=0(TN)':TN, 'recall': str(round(TP/(TP + FN), 4)),\
                                  'filter rate':str((round((TN + FN) / total, 4)))}, ignore_index = True)
    
    display(results)"""

    return candidate_rule

# %%
def Entropy(data, feature):
    total_cnt = len(data)
    if total_cnt == 0:
        return 0
    else:
        target_cnt = np.array([len(data[data[feature] == 0]), len(data[data[feature] != 0])])
        pk = target_cnt / total_cnt
        vec = entr(pk)
        S = np.sum(vec, axis= 0)
        S /= np.log(2) # 換底公式
        return S

def SplitMedian(data, feature):
    split_number = np.unique(train_set[feature])
    split_number = split_number[~np.isnan(split_number)]
    split_number.sort(axis= 0)
    median = np.empty(len(split_number) - 1)
    for i in range(1, len(split_number)):
        median[i-1] = (split_number[i] + split_number[i-1]) / 2
    print(f'len of median: {len(median)}')
    
    return median

# %%
# Hyper parameter settings
RANK = 10
SLICE = 10
beta = 2 # give recall weight=2

# %%
def Search2Rule(xRange, yRange, x, y, defaultX, defaultY, SAR, Non_SAR, train_set, LOGIC, SAR_type):
    numberOfSAR = [[0] * SLICE for i in range(SLICE)] # SAR個數
    recallRate = [[0] * SLICE for i in range(SLICE)]
    filterRate = [[0] * SLICE for i in range(SLICE)]
    harmonicRecallFilter = [[0] * SLICE for i in range(SLICE)]
    information_gain = [[0] * SLICE for i in range(SLICE)]
    total_cnt = train_set.shape[0]
    father_entropy = Entropy(train_set, 'TO_SAR')
    
#     results = pd.DataFrame(columns = [x, y, 'information gain','SAR', 'Non SAR','newSAM=1_SAR=1(TP)','newSAM=1_SAR=0(FP)',\
#                                   'newSAM=0_SAR=1(FN)', 'newSAM=0_SAR=0(TN)', 'recall','filter rate'])
    
    for i in range(len(xRange)):
        for j in range(len(yRange)):
            thre1 = xRange[i]
            thre2 = yRange[j]
            if thre1 < defaultX or thre2 < defaultY:
                continue
            idx1 = set(train_set[train_set[x] >= thre1].index)
            idx2 = set(train_set[train_set[y] >= thre2].index)
        
            idx = list(idx1 & idx2) # 閥值x>= "且" 閥值y>=
            new_SAM_1 = train_set.iloc[idx]
            new_SAM_0 = train_set.drop(index= new_SAM_1.index)
        
            if len(set(new_SAM_0.index) & set(new_SAM_1.index)) != 0:
                raise ValueError('new SAM 0 and new SAM 1 overlapped!')
            
            if len(new_SAM_0) == 0 or len(new_SAM_1) == 0:
                continue
        
            TP = new_SAM_1[new_SAM_1[SAR_type] != 0].shape[0]
            TN = new_SAM_0[new_SAM_0[SAR_type] == 0].shape[0]
            FP = new_SAM_1[new_SAM_1[SAR_type] == 0].shape[0]
            FN = new_SAM_0[new_SAM_0[SAR_type] != 0].shape[0]
        
            numberOfSAR[i][j] = TP
            recallRate[i][j] = TP / (TP + FN)
            filterRate[i][j] = (TN + FN) / (TP + FP + TN + FN)
            harmonicRecallFilter[i][j] = ((1 + beta * beta) * recallRate[i][j] * filterRate[i][j]) \
                        / (recallRate[i][j] + beta * beta * filterRate[i][j])
            #print(f'new_SAM_0: {len(new_SAM_0)}, new_SAM_1: {len(new_SAM_1)}, val1: {thre1}, val2: {thre2}')
            condition_entropy_0 = Entropy(new_SAM_0, 'TO_SAR') * (len(new_SAM_0) / total_cnt)
            condition_entropy_1 = Entropy(new_SAM_1, 'TO_SAR') * (len(new_SAM_1) / total_cnt)
            information_gain[i][j] = father_entropy - (condition_entropy_0 + condition_entropy_1)
            
            
#             results = results.append({x:thre1, y: thre2, 'information gain': information_gain[i][j], 'SAR':SAR.shape[0], \
#                             'Non SAR':Non_SAR.shape[0],\
#                             'newSAM=1_SAR=1(TP)':TP, 'newSAM=1_SAR=0(FP)':FP, 'newSAM=0_SAR=1(FN)':FN,\
#                             'newSAM=0_SAR=0(TN)':TN, 'recall': recallRate[i][j],\
#                             'filter rate':filterRate[i][j]}, ignore_index = True)
               
    if LOGIC == 'AND':
        mat = recallRate
        #mat = information_gain
    elif LOGIC == 'OR':
        mat = harmonicRecallFilter
        #mat = information_gain
        
    max_pos = []
    maxVal = np.amax(mat)
    countMax = 0
    countZero = 0
    
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == maxVal:
                countMax += 1
            if mat[i][j] == 0:
                countZero += 1
    print('max val: ', maxVal, 'number of max val: ', countMax)
    for i in range(min(max(RANK, countMax), SLICE - countZero)):
        idx = np.array(mat).argmax()
        max_pos.append(idx)
        print('{}th, information gain: {}, recall: {}, val1: {}, val2: {}'.format(idx \
                , information_gain[int(idx / SLICE)][idx % SLICE], recallRate[int(idx / SLICE)][idx % SLICE] \
                , xRange[int(idx / SLICE)], yRange[int(idx % SLICE)]))
        mat[int(idx / SLICE)][idx % SLICE] = -1
    
    print('*' * 32)
    candidate_rule = []
    for i in range(min(RANK, SLICE - countZero)):
        idx = max_pos[-(i + 1)]
        val_IG = information_gain[int(idx / SLICE)][idx % SLICE]
        val_recall = recallRate[int(idx / SLICE)][idx % SLICE]
        val1 = xRange[int(idx / SLICE)]
        val2 = yRange[idx % SLICE]
        candidate_rule.append([val_IG, val_recall, val1, val2])
    
    return candidate_rule

# %%
# newton method funcion for k dimension
def informationGain(point, feature_name, data):
    N = len(feature_name)
    idx = []
    for i, name in enumerate(feature_name):
        idx.append(set(data[data[name] >= float(point[i])].index))
    
    intersect_idx = idx[0]

    for i in range(1, N):
        intersect_idx = intersect_idx & idx[i]
    intersect_idx = list(intersect_idx)

    new_sam_1 = data.iloc[intersect_idx]
    new_sam_0 = data.drop(index= new_sam_1.index)
    #print(len(new_sam_1), len(new_sam_0))
    if len(set(new_sam_1.index) & set(new_sam_0.index)) != 0:
        raise ValueError('new sam 0 and new sam 1 are overlapped')
        
    return -(father_entropy - conditionalEntropy(new_sam_0, new_sam_1))

def conditionalEntropy(new_sam_0, new_sam_1):
    total = len(new_sam_1) + len(new_sam_0)
    return Entropy(new_sam_0, SAR_type) * (len(new_sam_0) / total) \
            + Entropy(new_sam_1, SAR_type) * (len(new_sam_1) / total)


def gradientIG(point, feature_name, data):
    K = len(feature_name)
    D = 1
    
    while True:
        _, auxiliary_idx  = kdtree.query(point.flatten(), k= D)
        if D == 1:
            auxiliary = data.loc[auxiliary_idx, feature_name].to_numpy(dtype= 'float64').reshape(K, -1)
        else:
            auxiliary_idx = auxiliary_idx[-1]
            auxiliary = data.loc[auxiliary_idx, feature_name].to_numpy(dtype= 'float64').reshape(K, -1)
        
        if np.all(auxiliary != point):
            break
        D += 10
    
    delta = auxiliary - point
    tmp = []
    for i in range(K):
        tmp.append(point.copy())

    for i in range(K):
        for j in range(K):
            if i == j:
                tmp[i][j] = auxiliary[i]
    
    z = informationGain(point, feature_name, data)
    z_delta = []

    for i in range(K):
        z_delta.append(informationGain(tmp[i], feature_name, data))
    
    grad = (np.asarray(z_delta).reshape((K, -1)) - z) / delta

    return grad

# %%
def Recall(point, feature_name, data):
    N = len(feature_name)
    idx = []
    for i, name in enumerate(feature_name):
        idx.append(set(data[data[name] >= float(point[i])].index))
    
    intersect_idx = idx[0]

    for i in range(1, N):
        intersect_idx = intersect_idx & idx[i]
    intersect_idx = list(intersect_idx)

    new_sam_1 = data.iloc[intersect_idx]
    new_sam_0 = data.drop(index= new_sam_1.index)

    if len(set(new_sam_1.index) & set(new_sam_0.index)) != 0:
        raise ValueError('new sam 1 and new sam 0 are overlapped')
    
    TP = new_sam_1[new_sam_1[SAR_type] != 0].shape[0]
    TN = new_sam_0[new_sam_0[SAR_type] == 0].shape[0]
    FP = new_sam_1[new_sam_1[SAR_type] == 0].shape[0]
    FN = new_sam_0[new_sam_0[SAR_type] != 0].shape[0]
    
    return -(TP / (TP + FN))

def RecallFilterRate(point, feature_name, data):
    N = len(feature_name)
    idx = []
    for i, name in enumerate(feature_name):
        idx.append(set(data[data[name] >= float(point[i])].index))
    
    intersect_idx = idx[0]

    for i in range(1, N):
        intersect_idx = intersect_idx & idx[i]
    intersect_idx = list(intersect_idx)

    new_sam_1 = data.iloc[intersect_idx]
    new_sam_0 = data.drop(index= new_sam_1.index)

    if len(set(new_sam_1.index) & set(new_sam_0.index)) != 0:
        raise ValueError('new sam 1 and new sam 0 are overlapped')
    
    TP = new_sam_1[new_sam_1[SAR_type] != 0].shape[0]
    TN = new_sam_0[new_sam_0[SAR_type] == 0].shape[0]
    FP = new_sam_1[new_sam_1[SAR_type] == 0].shape[0]
    FN = new_sam_0[new_sam_0[SAR_type] != 0].shape[0]
    
    recall = TP / (TP + FN)
    filter_rate = (TN + FN) / (TP + TN + FP + FN)

    return recall, filter_rate


def gradientPenalty(point, upperbound=np.zeros((2,1))):
    if np.any(point < 0): # any 是只要有一個是 True 就 output True，但 all 是要全部都為 True 才 output True
        raise ValueError('point are negative!')

    if np.linalg.norm(upperbound) == 0:
        return 1 / point
    else:
        return 1 / (upperbound - point)


def gradientRecall(point, feature_name, data):
    N = len(feature_name)
    K = 1
    
    while True:
        _, auxiliary_idx  = kdtree.query(point.flatten(), k= K)
        if K == 1:
            auxiliary = data.loc[auxiliary_idx, feature_name].to_numpy().reshape(N, -1)
        else:
            auxiliary_idx = auxiliary_idx[-1]
            auxiliary = data.loc[auxiliary_idx, feature_name].to_numpy().reshape(K, -1)
        
        if auxiliary.all() != point.all():
            break
        K += 1    

    delta = auxiliary - point
    tmp = []
    for i in range(N):
        tmp.append(point)

    for i in range(N):
        for j in range(N):
            if i == j:
                tmp[i][j] = auxiliary[i]
    
    z = Recall(point, feature_name, data)
    z_delta = []

    for i in range(N):
        z_delta.append(Recall(tmp[i], feature_name, data))
    grad = (z_delta - z) / delta

    return grad

# %%
def sigmoid(x):
    return 1.0 / (1 + np.exp(-(sigmoid_b + sigmoid_w * x)))

# %%
def sigmoidDiff(x):
    tmp = sigmoid(x)
    return tmp * (1 - tmp)

# %%
def plot_results(results):
    x = list(results.loc[results['iteration number'] % 50 == 0].index)
    x.append(len(results) - 1)

    plt.figure()
    plt.plot(x, results.loc[x, 'information gain'], 'o--')
    plt.plot(x, results.loc[x, 'recall'], 'o--')
    plt.plot(x, results.loc[x, 'filter rate'], 'o--')
    plt.xlabel('iteration number')

    plt.legend(['information gain', 'recall', 'filter rate'])
    plt.show()

# %%
def plot_seperator(results):
    x = list(results.loc[results['iteration number'] % 50 == 0].index)
    x.append(len(results) - 1)

    plot_name = ['information gain', 'recall', 'filter rate', 'loss']
    color = ['r', 'b', 'g', 'c']
    plt.figure()
    for i, name in enumerate(plot_name):
        plt.subplot(2,2,i+1)
        plt.plot(x, results.loc[x, name], 'o--', color= color[i])
        plt.title(name)
    plt.tight_layout()
    plt.show()


