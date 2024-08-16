import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import config

import warnings
warnings.filterwarnings('ignore') # 忽略警告

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


train_data = pd.read_csv(config.train_data_file, sep='\t', encoding='utf-8')
test_data = pd.read_csv(config.test_data_file, sep='\t', encoding='utf-8')

## 获取异常数据函数
def find_outliers(model, x, y, sigma=3):
    # predict y values using model
    try:
        y_pred = pd.Series(model.predict(x), index=y.index) # 创建一维的数据结构Series
    # if predicting fails ,try fit the model first
    except:
        model.fit(x,y)
        y_pred = pd.Series(model.predict(x), index=y.index) # 创建一维的数据结构Series
    # calculate residuals between the model prediction and true y values
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()

    # calculate z statistic,define outlier to be where |Z|>sigma
    z= (resid - mean_resid)/std_resid
    outliers = z[abs(z)>sigma].index

    # prind and polt the results
    print('R2=',model.score(x, y))
    print('mse=',mean_squared_error(y, y_pred))
    print('-'*10)
    print('mean of residuals:', mean_resid)
    print('std of residuals:', std_resid)
    print('-' * 10)
    print(len(outliers), 'outliers:')
    print(outliers.tolist())

    plt.figure(figsize=(15,5))
    ax_131 = plt.subplot(1,3,1)
    plt.plot(y, y_pred,'.')
    plt.plot(y.loc[outliers],y_pred.loc[outliers],'ro')
    plt.legend(['accept','outliers'])
    plt.xlabel('y')
    plt.ylabel('y_pred')

    ax_132 = plt.subplot(1, 3, 2)
    plt.plot(y, y_pred, '.')
    plt.plot(y.loc[outliers], y.loc[outliers]-y_pred.loc[outliers], 'ro')
    plt.legend(['accept', 'outliers'])
    plt.xlabel('y')
    plt.ylabel('y-y_pred')

    ax_133 = plt.subplot(1, 3, 3)
    z.plot.hist(bins=50, ax=ax_133)
    z.loc[outliers].plot.hist(color='r',bins=50, ax=ax_133)
    plt.legend(['accept', 'outliers'])
    plt.xlabel('z')

    plt.savefig('outliers1.png')
    return outliers

x_train = train_data.iloc[:,0:-1]
y_train = train_data.iloc[:,-1]
outliers = find_outliers(Ridge(), x_train, y_train)