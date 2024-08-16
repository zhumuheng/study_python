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

## 全部变量直方图和Q_Q图 观察正态分布情况
train_cols = 6
train_rows = len(train_data.columns)
plt.figure(figsize=(4*train_cols,4*train_rows))

i = 0
for col in train_data.columns:
    i+=1
    ax = plt.subplot(train_rows,train_cols,i)
    sns.distplot(train_data[col],fit=stats.norm)

    i+=1
    ax = plt.subplot(train_rows, train_cols, i)
    res = stats.probplot(train_data[col], plot=plt)
plt.tight_layout()
plt.savefig('Q_Q2.png')