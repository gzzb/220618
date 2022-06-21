import statsmodels.formula.api as smf
from sklearn import linear_model
#from sklearn.cross_validation import cross_val_predict
from sklearn.model_selection import cross_val_predict
import pandas as pd
import numpy as np
import numpy
from math import atan,pi
from scipy import stats
from sklearn.svm import SVR
import random
import copy
from copy import deepcopy
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")


def hat_matrix(X1):#, X2): #Hat Matrix
    hat_mat =  numpy.dot(numpy.dot(X1, numpy.linalg.inv(numpy.dot(X1.T, X1))), X1.T)
    return hat_mat


def pearson(y_true, y_pred):
    err = y_true - y_pred
    SE = err * err
    PRESS = np.sum(SE)
    y_avg = np.mean(y_true)
    n = len(y_true)
    y_mean = [y_avg for i in range(n)]
    err = y_true - y_mean
    SE = err * err
    TSS = np.sum(SE)
    r2 = 1 - PRESS/TSS
    return r2


def training_test(training, test, response, clf):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """

    from scipy import stats

    y = training.y
    y_ran = copy.copy(y)
    y_ran = y_ran.reset_index()
    y_ran = y_ran['y']
    #y_ran = y_ran.drop('IDX',axis=1)
    #print(y_ran)
    random.shuffle(y_ran)
    X = training.drop([response], axis=1)
    remaining = set(X.columns)
    y_train = y
    X_train = X
    y_test = test.y
    X_test = test.drop([response], axis=1)
    #remaining.remove(response)
    n = len(X)
    p = 0
    for i in remaining:
        p = p + 1
    print("length:",p)
    aa = {'length':[p/p]}
    lr1 = clf
    lr1.fit(X, y)
    lr_ran = clf
    lr_ran.fit(X, y_ran)
    
    lr0 = linear_model.LinearRegression(fit_intercept=False)
    lr0.fit(X, y)
    pred = lr1.predict(X)
    ss = numpy.sum(y-pred)**2
    mae = numpy.sum(abs(y-pred))
    pred_ran = lr_ran.predict(X)
    #import scipy.stats as stats
    #r, p = q2f2(y, pred)
    q = q2f2(y, pred)
    #q2 = q
    #q2, p = stats.pearsonr(y, pred)
    q2 = lr1.score(X,y)
    r0 = lr0.score(X,y)
    delta_r = abs(q2-r0)
    rm2 = q2*((1-delta_r)**0.5)

    k1, b1, r_value, p_value, std_err = stats.linregress(y, pred)
    k1 = atan(k1)
    k1 = pi/4 - abs(k1-pi/4)
    b1 = abs(b1)
    k2, b2, r_value, p_value, std_err = stats.linregress(pred, y)
    k2 = pi/4 - abs(k2-pi/4)
    b2 = abs(b2)
    k = max(k1, k2)
    k = abs(k)
    b = max(1/b1, 1/b2)

    r2_ran, p = stats.pearsonr(y_ran, pred_ran)
    pred_test = lr1.predict(X_test)
    print("r2 (tr):", q2)
    aa['r2 (tr)'] = [q2]
    print("r0 (tr):", r0)
    aa['r0 (tr)'] = [r0]
    print("rm2(tr):", rm2)
    aa['rm2(test)'] = [rm2]
    print("PRESS (tr):", ss)
    aa['PRESS'] = [1/ss]
    print("MAE (tr):", mae)
    aa['MAE'] = [1/mae]

    #print("BIC (tr):", b_ic)
    #print("AIC_m1.5 (tr):", aic_m15)
    rou = stats.spearmanr(y, pred)
    #print("rou (tr):", rou.correlation)
    #print("slope (training):", slope)
    #print("intercept (training):", intercept)
    q2_ran = qm2(training, 'y', clf)
    aa['q2(RAN)'] = [abs(q2_ran)]
    y_LOO, q2_loo = q2_LOO(training, clf)
    aa['q2(LOO)'] = [q2_loo]
    AE = abs(y-y_LOO)
    train = pd.DataFrame({'y':y, 'y_pre':y_LOO, 'y_cal':pred, 'AE':AE})
    train.to_csv('Train.csv')
    q2_VEN = q2_ven(clf)
    aa['q2(VEN)'] = [q2_VEN]
    print('delta_r:',delta_r)
    aa['delta_r'] = [1/delta_r]
    training_ran = copy.copy(training)
    training_ran['y'] = y_ran
    pred_loo_ran, q2_loo_ran = q2_LOO(training_ran, clf)
    q2_loo_ran, p = stats.pearsonr(y_ran, pred_loo_ran)
    r2_ran = r2_ran*r2_ran
    q2_loo_ran = q2_loo_ran*q2_loo_ran
    print('r2(y_ran):', r2_ran)
    print('q2(y_ran):', q2_loo_ran)
    aa['r2(y_ran)'] = [1/r2_ran]
    aa['q2(y_ran)'] = [1/q2_loo_ran]

    print("----------------------------------")

    AD = get_AD(training, test, response)
    print("slope (test):", k)
    print("intercept (test):", b)
    #print("AD(test):", AD)
    aa['slope (test)'] = [k]
    aa['intercept (test)'] = [b]
    aa['AD(test)'] = [AD]
    print('AD:',AD)
   
    #print(aa)
    df = pd.DataFrame.from_dict(aa, 'index')
    df.to_csv('ctrl.csv')
            

def q2_LOO(data, clf):
    n = len(data)
    y = training['y']
    X = data.drop(['y'], axis=1)
    #lr = clf
    y_pred = cross_val_predict(clf, X, y, cv = n)
    """lr = clf(fit_intercept=False)
    predicted0 = cross_val_predict(lr, X, y, cv = n)"""
    #cv = pd.DataFrame({'y':y, 'y1':predicted})
    """cv = pd.DataFrame({'y':y, 'y1':predicted, 'y0':predicted0})
    cv = pd.DataFrame({'y':y, 'y0':predicted0})"""
    #r2 = smf.ols('y~y1', cv).fit().rsquared_adj
    """r02 = smf.ols('y~y0',cv).fit().rsquared_adj"""
    """import math
    rm2 = r2 * math.sqrt(abs(1-abs(r2-r02)))"""
    r, p = stats.pearsonr(y, y_pred)
    q = q2f2(y, y_pred)
    q2 = q
    print("q2(LOO):", q2)
    return y_pred, q2

def SS(y_train, y_test, pred_test):
  PRESS = numpy.sum(y_test-pred_test)**2
  #SSRes = SSRes/len(y_test)
  TSS = numpy.sum((y_train-numpy.mean(y_train))**2)
  #SStot = SStot/len(y_train)
  #r2 = 1 - (SSRes/SStot)
  return PRESS, TSS

def q2f3(y_train, y_test, pred_test):
  SSRes = numpy.sum(y_test-pred_test)**2
  SSRes = SSRes/len(y_test)
  SStot = numpy.sum((y_train-numpy.mean(y_train))**2)
  SStot = SStot/len(y_train)
  r2 = 1 - (SSRes/SStot)
  return r2

def q2f2(y_test, pred_test):
  SSRes = numpy.sum(y_test-pred_test)**2
  SStot = numpy.sum((y_test-numpy.mean(y_test))**2)
  r2 = 1 - (SSRes/SStot)
  return r2

def q2f1(y_train, y_test, pred_test):
  SSRes = numpy.sum(y_test-pred_test)**2
  SStot = numpy.sum((y_test-numpy.mean(y_train))**2)
  r2 = 1 - (SSRes/SStot)
  return r2

def ccc(y_train, y_test, pred_test):
  SSRes = numpy.sum(abs(y_test-numpy.mean(y_test))*abs(pred_test-numpy.mean(pred_test)))*2
  SStot = numpy.sum((y_test-numpy.mean(y_test))**2) + numpy.sum((pred_test-numpy.mean(pred_test))**2) + len(pred_test) * (numpy.mean(y_test)-numpy.mean(pred_test))**2
  r2 = SSRes/SStot
  return r2

def qm2(data, response, clf):
    n = len(data)
    y = data.y
    X = data.drop([response], axis=1)
    lr = clf
    predicted = cross_val_predict(lr, X, y, cv = 4)
    #lr = clf(fit_intercept=False)
    #predicted0 = cross_val_predict(lr, X, y, cv = n)
    #cv = pd.DataFrame({'y':y, 'y1':predicted, 'y0':predicted0})
    #cv = pd.DataFrame({'y':y, 'y1':predicted})
    r, p = stats.pearsonr(y, predicted)
    q = q2f2(y, predicted)
    q2 = q
    print("q2(RAN):", q2)
    return q2
    #r02 = smf.ols('y~y0',cv).fit().rsquared
                
    #import math
    #rm2 = r2 * math.sqrt(abs(1-abs(r2-r02)))
                        
    #return rm2

def get_AD(Xy_cor, X_ext, response):
    #X_pre = deepcopy(X_ext)
    if X_ext.shape[1] < Xy_cor.shape[1]:
        X_cor = Xy_cor.drop(['y'], axis=1)
    else:
        X_cor = deepcopy(Xy_cor)
    if X_ext.shape[1] > Xy_cor.shape[1]:
        X_pre = X_ext.drop(['y'], axis=1)
    else:
        X_pre = deepcopy(X_ext)
    #X_cor = Xy_cor.drop(['y'], axis=1)
    n,p = X_cor.shape
    XT = X_cor.T
    XTX = np.dot(XT, X_cor)
    XTX_1 = np.linalg.pinv(XTX)
    hat = []
    for index, xi in X_pre.iterrows():
        try:
            hi = np.dot(xi.T, XTX_1)
        except:
            print(X_cor.columns)
            print(X_ext.columns)
        hi = np.dot(hi, xi)
        hat.append(hi)
    X_pre['hat'] = hat
    X_pre['hat'] = X_pre[X_pre['hat'] < (2*p/n)]
    return len(X_pre)/n




def q2_ven(clf):
    cor = pd.read_csv("training_2.csv")
    try:
      cor = cor.drop('IDX', axis=1)
    except:
      pass

    result = pd.DataFrame(columns=['y_exp', 'y_pre'])
    
    group = 0
    va = cor[cor.index%4==group]
    tr = cor[cor.index%4!=group]
    y = tr.y
    X = tr.drop(["y"], axis=1)
    lr = clf
    lr.fit(X, y)
    y = va.y
    X = va.drop(["y"], axis=1)
    y_pre = lr.predict(X)
    cv = pd.DataFrame({'y_exp':y, 'y_pre':y_pre})
    result=result.append(cv)
    
    group = 1
    va = cor[cor.index%4==group]
    tr = cor[cor.index%4!=group]
    y = tr.y
    X = tr.drop(["y"], axis=1)
    lr = clf
    lr.fit(X, y)
    y = va.y
    X = va.drop(["y"], axis=1)
    y_pre = lr.predict(X)
    cv = pd.DataFrame({'y_exp':y, 'y_pre':y_pre})
    result=result.append(cv)
    
    group = 2
    va = cor[cor.index%4==group]
    tr = cor[cor.index%4!=group]
    y = tr.y
    X = tr.drop(["y"], axis=1)
    lr = clf
    lr.fit(X, y)
    y = va.y
    X = va.drop(["y"], axis=1)
    y_pre = lr.predict(X)
    cv = pd.DataFrame({'y_exp':y, 'y_pre':y_pre})
    result=result.append(cv)
    
    group = 3
    va = cor[cor.index%4==group]
    tr = cor[cor.index%4!=group]
    y = tr.y
    X = tr.drop(["y"], axis=1)
    lr = clf
    lr.fit(X, y)
    y = va.y
    X = va.drop(["y"], axis=1)
    y_pre = lr.predict(X)
    cv = pd.DataFrame({'y_exp':y, 'y_pre':y_pre})
    result=result.append(cv)

    y_true = result.y_exp
    y_pred = result.y_pre
    r, p = stats.pearsonr(y_true, y_pred)
    q= q2f2(y_true, y_pred)
    q2 = q
    #r2 = smf.ols('y_exp~y_pre', result).fit().rsquared
    print("q2(VEN):", q2)
    return q2


train_file = "training_2.csv"
training = pd.read_csv(train_file,index_col=0)
test_file = "test_2.csv"
test = pd.read_csv(test_file,index_col=0)

clf = linear_model.LinearRegression()
#clf = SVR(gamma='scale')
training_test(training, test, 'y', clf)