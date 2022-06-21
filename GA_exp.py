import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from scipy.stats import pearsonr,spearmanr
import pickle
from copy import deepcopy
import Test_exp
from importlib import reload
import random
import warnings
warnings.filterwarnings("ignore")
import itertools

'''
df_data = pd.read_csv('training_1.csv',index_col=0)
df_test = pd.read_csv('test_1.5.csv',index_col=0)
df_test = df_test.drop('y',axis=1)

clf = LinearRegression()
def q2sd(cor, pre, clf):

    val = pd.DataFrame(columns=['ture', 'pred'])
    import warnings
    warnings.filterwarnings("ignore")
    try:
        X_pre = pre.drop(["y"], axis=1)
    except:
        X_pre = deepcopy(pre)

    group = 0
    va = cor[cor.index%4==group]
    tr = cor[cor.index%4!=group]
    y_true = tr.y
    X = tr.drop(["y"], axis=1)
    clf.fit(X, y_true)
    y_true = va.y
    X = va.drop(["y"], axis=1)
    y_pred = clf.predict(X)
    cv = pd.DataFrame({'true':y_true, 'pred':y_pred})
    val = val.append(cv)
    y0 = clf.predict(X_pre)
    
    group = 1
    va = cor[cor.index%4==group]
    tr = cor[cor.index%4!=group]
    y_true = tr.y
    X = tr.drop(["y"], axis=1)
    clf.fit(X, y_true)
    y_true = va.y
    X = va.drop(["y"], axis=1)
    y_pred = clf.predict(X)
    cv = pd.DataFrame({'true':y_true, 'pred':y_pred})
    val = val.append(cv)
    y1 = clf.predict(X_pre)

    group = 2
    va = cor[cor.index%4==group]
    tr = cor[cor.index%4!=group]
    y_true = tr.y
    X = tr.drop(["y"], axis=1)
    clf.fit(X, y_true)
    y_true = va.y
    X = va.drop(["y"], axis=1)
    y_pred = clf.predict(X)
    cv = pd.DataFrame({'true':y_true, 'pred':y_pred})
    val = val.append(cv)
    y2 = clf.predict(X_pre)

    group = 3
    va = cor[cor.index%4==group]
    tr = cor[cor.index%4!=group]
    y_true = tr.y
    X = tr.drop(["y"], axis=1)
    clf.fit(X, y_true)
    y_true = va.y
    X = va.drop(["y"], axis=1)
    y_pred = clf.predict(X)
    cv = pd.DataFrame({'true':y_true, 'pred':y_pred})
    val = val.append(cv)
    y3 = clf.predict(X_pre)

    u = (y0+y1+y2+y3) / 4
    y1 = (y1-u)**2
    y2 = (y2-u)**2
    y3 = (y3-u)**2
    y0 = (y0-u)**2
    u = (y0+y1+y2+y3) / 4
    u = np.mean(u)

    return u

u_ctrl = q2sd(df_data,df_test,clf)
'''

df_val = pd.read_csv('val.csv')
df_val = df_val.dropna()
y_val = df_val['y']
df_data = pd.read_csv('training_1.csv')
sample_size = len(df_data)
df_test = pd.read_csv('test_1.csv')
y_train = df_data['y']
field = list(df_data.columns)
field.remove('y')
field.remove('IDX')

df_ctrl = pd.read_csv('ctrl.csv',index_col=0)
df_变异 = pd.DataFrame()
df_变异['ctrl'] = df_ctrl['0']

## 遗传算法过程
### 初代种群生成
def ga_generate_ori():
    new_field = []
    df_field = pd.DataFrame()
    for i in field:
        prob = np.random.uniform(0, 1)
        row = pd.DataFrame({'col':[i],'p':[prob]})
        df_field = df_field.append(row)
    df_field = df_field.sort_values('p')
    df_field = df_field.tail(int(sample_size/3))
    field_list = list(df_field['col'])
    ## 选取原始特征过程
    for i in field:
        if i in field_list:
            new_field.append(True)
        else:
             new_field.append(False)
    new_field = np.array(new_field)
    return new_field

lr = LinearRegression()
## 种群交配过程
def ga_cross_next_group(ori_group, dict_score = 'ori', change_prob = 0.2):
    new_dict = ori_group.copy()
    if dict_score == 'ori':
        score = 1.0 / len(ori_group)
        dict_score = {k:score for k in ori_group.keys()}
    g, p = np.array([[k, v] for k, v in dict_score.items()]).T
    flag = max(ori_group.keys())

    ## 按照种群分数进行选择交配
    ## 选择交配种群
    try:
        cross_group = np.random.choice(g, size = 5, p = p, replace= False)
    except:
        #p[:] = [x - p.min() for x in p] 
        cross_group = np.random.choice(g, size = 5, replace= False)
    for (fa,mo) in itertools.combinations(cross_group, 2):
        flag += 1
        fa_code, mo_code = ori_group[fa], ori_group[mo]
        ## 随机选择切分点
        cut_point = np.random.randint(1, len(fa_code)-1)
        ## 切分基因
        fa_code0, fa_code1 = fa_code[:cut_point], fa_code[cut_point:]
        mo_code0, mo_code1 = mo_code[:cut_point], mo_code[cut_point:]
        # print(fa_code0, mo_code1)
        ## 基因交换
        new1 = np.hstack([fa_code0, mo_code1])
        ## 变异过程
        prob = np.random.uniform(0, 1)
        #print(len(fa_code))
        if prob < change_prob:
            ## 随机挑一个基因点
            change_point = np.random.randint(0, len(fa_code))
            ## 改变该点的值
            X1 = evalue_cols[new1]
            X_training_1 = df_data[X1]
            training_15 = deepcopy(X_training_1)
            training_15['y'] = y_train
            training_15.to_csv('training_15.csv',index=False)
            test_15 = df_test[X1]
            test_15.to_csv('test_15.csv',index=False)
            reload(Test_exp)
            df_new1 = pd.read_csv('exp.csv',index_col=0)
            df_变异['new1'] = df_new1['0']
            df_变异['sig'] = abs(df_变异['new1'])/df_变异['new1']
            df_变异['pct'] = df_变异['new1'] / df_变异['ctrl']
            df_变异['pct'] = df_变异['pct'] * df_变异['sig']

            new2 = deepcopy(new1)
            new2[change_point] = not new2[change_point]
            X1 = evalue_cols[new2]
            X_training_1 = df_data[X1]
            training_15 = deepcopy(X_training_1)
            training_15['y'] = y_train
            training_15.to_csv('training_15.csv',index=False)
            test_15 = df_test[X1]
            test_15.to_csv('test_15.csv',index=False)
            reload(Test_exp)
            df_new2 = pd.read_csv('exp.csv',index_col=0)
            df_变异['new2'] = df_new2['0']
            df_by = df_变异.sort_values('pct')
            df_by = df_by.dropna()
            score_2 = list(df_by['new2'])[0]
            score_1 = list(df_by['new1'])[0]
            if score_2 > score_1:
                new1 = deepcopy(new2)
            
        new_dict[flag] = new1
    return new_dict


def q2f2(y_test, pred_test):
  SSRes = np.sum(y_test-pred_test)**2
  SStot = np.sum((y_test-np.mean(y_test))**2)
  r2 = 1 - (SSRes/SStot)
  return r2


def get_q2f3(y_train, y_test, pred_test):
  SSRes = np.sum(y_test-pred_test)**2
  SSRes = SSRes/len(y_test)
  SStot = np.sum((y_train-np.mean(y_train))**2)
  SStot = SStot/len(y_train)
  r2 = 1 - (SSRes/SStot)
  return r2


def q2ven(cor, test, fold):

    clf = LinearRegression()
    cor['y'] = y_train
    df_val = pd.DataFrame()
    for group in range(fold):
        va = cor[cor.index%fold==group]
        tr = cor[cor.index%fold!=group]
        y_tr = tr.y
        X = tr.drop(["y"], axis=1)
        clf.fit(X, y_tr)
        y_te = va.y
        X = va.drop(["y"], axis=1)
        pre_te = clf.predict(X).reshape(-1,)
        val = pd.DataFrame()
        val['true'] = y_te
        val['pred'] = pre_te
        df_val = df_val.append(val)
    return q2f2(df_val['true'] ,df_val['pred'])


lr = LinearRegression()


def get_noise(train_new, test_new, train_old, test_old):

    y_true = train_new['y']
    X_cor = train_new.drop(["y"], axis=1)
    X_ext = test_new.drop(["y"], axis=1)
    lr.fit(X_cor, y_true)
    y_cal = lr.predict(X_cor)
    try:
        y_pre = lr.predict(X_ext)
    except:
        print(train_new.head())
        print(test_new.head())

    cor_tmp = train_old.drop(["y"], axis=1)
    cor_ext = test_old.drop(["y"], axis=1)
    lr.fit(cor_tmp, y_true)
    cal_tmp = lr.predict(cor_tmp)
    pre_tmp = lr.predict(cor_ext)
    q2_cor = pearsonr(y_cal, cal_tmp)[0]
    q2_ext = pearsonr(y_pre, pre_tmp)[0]
    
    return q2_cor / q2_ext
    

def forward_selected(train_raw, test_raw, output=False):
    y_train, y_test = df_data['y'], df_test['y']
    X = train_raw.append(test_raw)
    i = int(X.shape[1]/2)
    df = pd.DataFrame({'q':[0]})
    try:
        mat = np.mat(train_raw)
        rank = np.linalg.matrix_rank(mat)
        alg = PCA(rank)
        mdl = alg.fit(X)
        train_new = mdl.transform(train_raw)
        test_new = mdl.transform(test_raw)
        train_new = pd.DataFrame(train_new)
        test_new = pd.DataFrame(test_new)
        train_new['y'] = y_train
        test_new['y'] = y_test
        score_old = list(df['q'])[-1]
        q = q2ven(train_new, test_new, 10)
        X_training = deepcopy(train_new)
        X_training = X_training.drop('y',axis=1)
        lr = LinearRegression()
        lr.fit(X_training,y_train)
        X_test = deepcopy(test_new)
        X_test = X_test.drop('y',axis=1)
        y_pred = lr.predict(X_test)
        X_test['y'] = y_pred
        X_test = X_test.sort_values('y')
        X_test = X_test.drop('y',axis=1)
        n = min(10, len(X_test))
        X_test = X_test.head(n)
        AD = get_AD(train_new, test_new)
        '''
        if len(df) > 1:
            train_old = list(df['train'])[-1]
            test_old = list(df['test'])[-1]
            noise = get_noise(train_new, test_new, train_old, test_old)
            new_score = new_score / noise
        '''
        if new_score > score_old:
            new_row = pd.DataFrame({'q':[new_score],'train':[train_new],'test':[test_new]})
            df = df.append(new_row)
        else:
            if output:
                return train_new, test_new
            else:
                return score_old
    except:
        if output:
            return train_new, test_new
        else:
            return score_old


def BE(train_raw, test_raw, output=False):
    y_train, y_test = df_data['y'], df_test['y']
    X = train_raw.append(test_raw)
    i = int(X.shape[1])
    df = pd.DataFrame({'q':[0]})
    try:
        while True:
            i -= 1
            #try:
            alg = PCA(i)
            mdl = alg.fit(X)
            train_new = mdl.transform(train_raw)
            test_new = mdl.transform(test_raw)
            train_new = pd.DataFrame(train_new)
            test_new = pd.DataFrame(test_new)
            train_new['y'] = y_train
            test_new['y'] = y_test
            score_old = list(df['q'])[-1]
            new_score = q2ven(train_new, test_new, 10)
            '''
            if len(df) > 1:
                train_old = list(df['train'])[-1]
                test_old = list(df['test'])[-1]
                noise = get_noise(train_new, test_new, train_old, test_old)
                new_score = new_score / noise
            '''
            if new_score > score_old:
                new_row = pd.DataFrame({'q':[new_score],'train':[train_new],'test':[test_new]})
                df = df.append(new_row)
            else:
                if output:
                    return train_new, test_new
                else:
                    return score_old
    except:
        if output:
            return train_new, test_new
        else:
            return score_old


def get_AD(Xy_cor, X_ext):
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


def get_ccc(y_test, pred_test):
  SSRes = np.sum(abs(y_test-np.mean(y_test))*abs(pred_test-np.mean(pred_test)))*2
  SStot = np.sum((y_test-np.mean(y_test))**2) + np.sum((pred_test-np.mean(pred_test))**2) + len(pred_test) * (np.mean(y_test)-np.mean(pred_test))**2
  r2 = SSRes/SStot
  return r2

def qm2(data, response):
    n = len(data)
    y = data.y
    X = data.drop([response], axis=1)
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


## 对每一个个体评分
def ga_get_score(alg, df_data, xy_cols):
    xy_train, xy_test = df_data[xy_cols], df_test[xy_cols]
    Xy_val = df_val[xy_cols]
    q = q2ven(xy_train, xy_test, 10)
    return alg, q

## 种群个体能力评价
def ga_evalue_group(group, evalue_df, evalue_col):
    score_dict = {}
    for g, code in group.items():
        cols = evalue_col[code]
        _, score = ga_get_score(alg=PLSRegression(len(cols)), df_data = evalue_df ,xy_cols = cols)
        score_dict[g] = score
    return score_dict

## 丢弃弱者
def ga_kill_group(ori_group, dict_score):
    ## 二代目
    sub_group = ga_cross_next_group(ori_group, dict_score= dict_score)
    ## 评价
    score_dict = ga_evalue_group(sub_group, df_data, evalue_cols)
    score_se = pd.Series(score_dict)
    score_se = score_se.sort_values(ascending= False)[:10] / (score_se.sort_values(ascending= False)[:10].sum())
    score_dict = dict(score_se)
    liv_group = {i:sub_group[i] for i in score_dict.keys()}
    print('开启贤者模式')
    return liv_group, score_dict


## 初始化过程
evalue_cols = np.array(field)

## 随机产生初代子类
group_num = 10
ori_group = {i:ga_generate_ori() for i in range(group_num)}


## 产生第一代杂交类
for i in range(6):
    if i == 0:
        sub, sco = ga_kill_group(ori_group, 'ori')
    else:
        sub, sco = ga_kill_group(sub, sco)


best_code = pd.Series(sco).sort_values()[-1:].index[0]
best_field = list(evalue_cols[sub[best_code]])

final_field = ['IDX','y'] + best_field
train = df_data[final_field]
train.to_csv('training_2.csv',index=False)
df_data = pd.read_csv('test_1.csv')
test = df_data[final_field]
test.to_csv('test_2.csv',index=False)
'''
train, test = forward_selected(train, test, True)
train.to_csv('training_2.csv',index=False)
test.to_csv('test_2.csv',index=False)
'''