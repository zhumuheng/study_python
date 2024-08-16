## 导入模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # 画图
import seaborn as sns # 可视化
from scipy import stats # Q_Q图

import os

import warnings
warnings.filterwarnings('ignore') # 忽略警告

from sklearn.linear_model import Ridge # 岭回归
from sklearn.metrics import mean_squared_error # 计算MSE
from  sklearn import preprocessing # 归一化

## 文件及路径
train_data_file = "D:/study_python/工业蒸汽量预测/.venv/input/zhengqi_train.txt"
test_data_file = "D:/study_python/工业蒸汽量预测/.venv/input/zhengqi_test.txt"

## 读取文件
train_data = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
test_data = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')

drop_colums = ['V5','V9','V11','V17','V22','V28']
## 合并训练集数据和测试集数据,pandas做归一化
train_x = train_data.drop(['target'], axis=1)
data_all = pd.concat([train_x,test_data])

data_all.drop(drop_colums,axis=1,inplace=True)
print(data_all.head())

## 使用pandas对合并后的数据每一列进行归一化
cols_numeric = list(data_all.columns)

def scale_minmax(col):
    '''定义归一化计算公式'''
    return (col-col.min())/(col.max()-col.min())
#
# data_all[cols_numeric] = data_all[cols_numeric].apply(scale_minmax,axis=0)
# print(data_all[cols_numeric].describe())
## 分开归一化
train_data_process = train_data[cols_numeric]
train_data_process = train_data_process[cols_numeric].apply(scale_minmax,axis=0)

test_data_process = test_data[cols_numeric]
test_data_process = test_data[cols_numeric].apply(scale_minmax,axis=0)

# ## 使用sklearn对合并后的数据每一列进行归一化，好处就是可以全部使用训练集中的最大值最小值出来测试集和验证集
# features_columns = list(train_x)
# min_max_scaler = preprocessing.MinMaxScaler()
# min_max_scaler = min_max_scaler.fit(train_x(features_columns))

## 对特征变量进行box-cox变换，并画图
cols_numeric_left = cols_numeric[0:13]
cols_numeric_right = cols_numeric[13:]
train_data_process = pd.concat([train_data_process,train_data['target']],axis=1)

fcols = 6
frows = len(cols_numeric_left)
plt.figure(figsize=(4*fcols,4*frows))

i=0
for var in cols_numeric_left:
    dat = train_data_process[[var, 'target']].dropna()
    i += 1
    plt.subplot(frows, fcols, i)
    sns.distplot(dat[var],fit=stats.norm)
    plt.title(var + 'original')
    plt.xlabel('')
    plt.ylabel('')
    i += 1
    plt.subplot(frows, fcols, i)
    _ = stats.probplot(dat[var], plot=plt)
    plt.title('skew'+ '{:4f}'.format(stats.skew(dat[var])))
    plt.xlabel('')
    plt.ylabel('')
    i += 1
    plt.subplot(frows, fcols, i)
    plt.plot(dat[var], dat['target'], '.', alpha=0.5)
    plt.title('corr'+'{:.2f}'.format(np.corrcoef(dat[var], dat['target'])[0][1]))
    i += 1
    plt.subplot(frows, fcols, i)
    trans_var, lambda_var = stats.boxcox(dat[var].dropna() + 1)
    trans_var = scale_minmax(trans_var)
    sns.distplot(trans_var, fit=stats.norm)
    plt.title(var + 'transformed')
    plt.xlabel('')
    i += 1
    plt.subplot(frows, fcols, i)
    _ = stats.probplot(trans_var, plot=plt)
    plt.title('skew' + '{:4f}'.format(stats.skew(trans_var)))
    plt.xlabel('')
    plt.ylabel('')
    i += 1
    plt.subplot(frows, fcols, i)
    plt.plot(trans_var, dat['target'], '.', alpha=0.5)
    plt.title('corr' + '{:.2f}'.format(np.corrcoef(trans_var, dat['target'])[0][1]))
plt.savefig('box-cox-qq.png')