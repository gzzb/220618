#from statistics import linear_regression
from numpy import log10
import numpy
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
import pickle
from importlib import reload
#import GA_ctrl
from copy import copy, deepcopy


log = True
power = 1
groupby = 'Institution'
#groupby = 'Documation'
log_dict = {'Institution':True, 'Documation':False}
#log = log_dict[groupby]
pct = 1

df = pd.read_csv('dragon.csv')
y = df['BIO']
try:
    df['Documation'] = df['PMID'].astype(str)+df['Patent Number'].astype(str)
except:
    pass
Institution_set = set(df[groupby])
if log:
    y = log10(1000000000/y)
df['y'] = y

def get_ccc(y_train, y_test, pred_test):
  SSRes = numpy.sum(abs(y_test-numpy.mean(y_test))*abs(pred_test-numpy.mean(pred_test)))*2
  SStot = numpy.sum((y_test-numpy.mean(y_test))**2) + numpy.sum((pred_test-numpy.mean(pred_test))**2) + len(pred_test) * (numpy.mean(y_test)-numpy.mean(pred_test))**2
  r2 = SSRes/SStot
  return r2

def q2f2(y_test, pred_test):
  SSRes = numpy.sum(y_test-pred_test)**2
  SStot = numpy.sum((y_test-numpy.mean(y_test))**2)
  r2 = 1 - (SSRes/SStot)
  return r2


def get_AD(Xy_cor, X_ext):
	X_pre = deepcopy(X_ext)
	#X_pre = X_pre.drop(['LogBIO'], axis=1)
	X_cor = Xy_cor.drop(['y'], axis=1)
	n,p = X_pre.shape
	XT = X_cor.T
	XTX = np.dot(XT, X_cor)
	XTX_1 = np.linalg.pinv(XTX)
	hat = []
	for index, xi in X_pre.iterrows():
		hi = np.dot(xi.T, XTX_1)
		hi = np.dot(hi, xi)
		hat.append(hi)
	X_pre['hat'] = hat
	X_AD = X_pre[X_pre['hat']<=3*p/n]
	AD_pct = len(X_AD) / n
	return AD_pct


df_performance = pd.DataFrame()

for Institution in Institution_set:

    df_Institution = df[df[groupby]==Institution]
    try:
        X = df_Institution.drop(['Institution','BIO', 'PMID', 'Patent Number', 'Documation'],axis=1)
    except:
        X = df_Institution.drop(['Institution','BIO'],axis=1)
    X1 = X.dropna(axis=0)
    X2 = X.dropna(axis=1)
    shape1 = X1.shape[0] * X1.shape[1]
    shape2 = X2.shape[0] * X2.shape[1]
    if shape1 > shape2:
        Xy_CID = X1
    else:
        Xy_CID = X2
    Xy = Xy_CID.drop('ID', axis=1)
    Xy = Xy_CID.groupby('ID').agg('mean')
    sample_size = copy(len(Xy))
    if sample_size < 50:
        continue

    for i in range(1, 23):
        if i < 10:
            fn = 'Dragon.B0'+str(i)
        else:
            fn = 'Dragon.B'+str(i)
        try:
            X = pd.read_csv(fn,'\\s+')
            X = X.dropna(axis=1)
            X_cols = set(X.columns)
            Xy_cols = set(Xy.columns)
            cols = X_cols & Xy_cols
            if len(cols) == 0:
                continue
            X = Xy[list(cols)]
            if (not log) and power<0.5:
                X = 1000000000 / (10**X)
            y = copy(Xy['y'])
            pls = LinearRegression()
            pls.fit(X,y)
        except:
           continue
        r = pls.score(X,y)
        Xy['IDX'] = Xy.index
        #'''
        Xy['pred'] = pls.predict(X)
        Xy['err'] = abs(Xy['pred']-y)
        Xy = Xy.sort_values('err')
        Xy['err'] = Xy['err'].rank(method='max')
        Xy['err'] = Xy['err'] % 5
        df_test = Xy[Xy['err']==3]
        df_train = Xy[Xy['err']!=3]
        df_train.index = df_train['IDX']
        df_test.index = df_test['IDX']
        df_train = df_train.drop(['IDX','pred','err'],axis=1)
        df_test = df_test.drop(['IDX','pred','err'],axis=1)
        '''
        df_train.to_csv('training_1.csv')
        df_test.to_csv('test_1.csv')
        X_test = df_test.drop('y',axis=1)
        X_train = df_train.drop('y',axis=1)
        y_train = df_train['y']
        y_test = df_test['y']
        pls.fit(X_train,y_train)
        y_pred = pls.predict(X_test)
        reload(GA_ctrl)
        train_file = "training_2.csv"
        training = pd.read_csv(train_file,index_col=0)
        test_file = "test_2.csv"
        test = pd.read_csv(test_file,index_col=0)
        X_test = test.drop['y']
        = get_ccc(y_train,y_test,y_pred)
        '''
        #AD = get_AD(df_train, X_test)
        #n = min(df_train.shape[0]-1,df_train.shape[1]-1)
        #n = min(X.shape[0],X.shape[1])
        '''
        pls = LinearRegression()
        y = df_train['y']
        X = df_train.drop('y',axis=1)
        #try:
        pls.fit(X,y)
        y = df_test['y']
        X = df_test.drop('y',axis=1)
        pred = pls.predict(X).reshape(-1,)
        '''
        
        new = pd.DataFrame({'Institution':[Institution],'descriptor':[df_test.columns],'train':[df_train],'test':[df_test],'r':[r],'n':[sample_size],'fn':[fn]})
        df_performance = df_performance.append(new)

#avg = df_performance['r'].mean()
#df_performance = df_performance[df_performance['r'] > avg]
#df_performance['AD'] = abs(df_performance['AD']-avg)
df_performance['score'] = df_performance['n']**2*df_performance['n'] * df_performance['r']
#avg = df_performance['score'].mean()
#df_performance['score'] = abs(df_performance['score']-avg)
df_performance = df_performance.sort_values('score',ascending=False)
n = int(len(df_performance) / 8)
df_performance = df_performance.head(n)

lr = LinearRegression()
def forward_selected(training, test):
    #remaining = set(X.columns)
    #print(remaining)
    selected = []
    y_training = training['y']
    y_test = test['y']
    remaining = list(training.columns)
    remaining.remove('y')
    #remaining.remove('IDX')
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        df_score = pd.DataFrame()
        for candidate in remaining:
            feature_cols = selected + [candidate]
            X_training = training[feature_cols]
            X_test = test[feature_cols]
            lr.fit(X_training,y_training)
            y_pred = lr.predict(X_test)
            ccc = get_ccc(y_training,y_test,y_pred)
            score = abs(ccc-0.85)
            new = pd.DataFrame({'score':[score],'X':[candidate]})
            df_score = df_score.append(new)
        df_score = df_score.sort_values('score')
        best_new_score = list(df_score['score'])[0]
        if best_new_score > current_score:
            current_score = list(df_score['score'])[0]
        df_tmp = df_score.head(1)
        best_candidate = list(df_tmp['X'])[0]
        remaining.remove(best_candidate)
        selected.append(best_candidate)
        p = len(feature_cols)
        if p >= len(X):
            return selected
    return selected


def BE(training, test):
    #remaining = set(X.columns)
    #print(remaining)
    y_training = training['y']
    y_test = test['y']
    remaining = list(training.columns)
    remaining.remove('y')
    #remaining.remove('IDX')
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        df_score = pd.DataFrame()
        for candidate in remaining:
            feature_cols = deepcopy(remaining)
            feature_cols.remove(candidate)
            X_training = training[feature_cols]
            X_test = test[feature_cols]
            lr.fit(X_training,y_training)
            y_pred = lr.predict(X_test)
            ccc = get_ccc(y_training,y_test,y_pred)
            score = abs(ccc-0.85)
            new = pd.DataFrame({'score':[score],'X':[candidate]})
            df_score = df_score.append(new)
        df_score = df_score.sort_values('score')
        if list(df_score['score'])[0] > list(df_score['score'])[1]:
            return remaining
        if abs(list(df_score['score'])[1]-list(df_score['score'])[0])<0.01:
            return remaining
        df_tmp = df_score.head(1)
        best_candidate = list(df_tmp['X'])[0]
        remaining.remove(best_candidate)
        selected = deepcopy(remaining)
        p = len(feature_cols)
        if p >= len(X):
            return selected
    return selected

i = -1
while True:
    try:
        df_train = list(df_performance['train'])[i]
        df_test = list(df_performance['test'])[i]
        #cols = forward_selected(df_train, df_test)
        #cols = BE(df_train, df_test)
        cols.add('y')
        #df_train[cols].to_csv('training_1.csv')
        #df_test[cols].to_csv('test_1.csv')
        df_train.to_csv('training_1.csv')
        df_test.to_csv('test_1.csv')
        df_test = pd.read_csv('test_1.csv',index_col=0)
        cols = list(df_test.columns)
        val_cols = cols + ['ID']
        ls = list(df_train.index)+list(df_test.index)
        df_val = df[val_cols]
        df_val = df_val.groupby('ID').agg('mean')
        df_val = df_val[~df_val.index.isin(ls)]
        df_val.to_csv('val.csv')
        break
    except:
        i -= 1