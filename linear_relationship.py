## 导入模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

import warnings
warnings.filterwarnings('ignore') # 忽略警告

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

## 文件及路径
train_data_file = "D:/study_python/工业蒸汽量预测/.venv/input/zhengqi_train.txt"
test_data_file = "D:/study_python/工业蒸汽量预测/.venv/input/zhengqi_test.txt"

## 读取文件
train_data = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
test_data = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')

fcols = 6
frows = len(test_data.columns)
plt.figure(figsize=(5*fcols,4*frows))

i=0
for col in test_data.columns:
    i += 1
    ax = plt.subplot(frows,fcols,i)
    sns.regplot(x=col,y='target',data=train_data,ax=ax,
                scatter_kws={'marker':'.','s':3,'alpha':0.3},
                line_kws={'color':'k'})
    plt.xlabel(col)
    plt.ylabel('target')

    i += 1
    ax = plt.subplot(frows,fcols,i)
    sns.distplot(train_data[col].dropna())
    plt.xlabel(col)
plt.savefig('Linear relationship diagram.png')