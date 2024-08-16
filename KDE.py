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


## KDE分布图（核密度估计），观察训练集和测试集的中的特征分布情况，发现两个数据集中分布不一致的情况的特征变量
dist_cols = 6
dist_rows = len(test_data.columns)
plt.figure(figsize=(4*dist_cols,4*dist_rows))
i = 1
for col in test_data.columns:
    ax = plt.subplot(dist_rows, dist_cols, i)
    ax = sns.kdeplot(train_data[col],color='Red', shade=True)
    ax = sns.kdeplot(test_data[col], color='Blue', shade=True)
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    ax = ax.legend(['train','test'])
    i += 1
plt.savefig('KDEplot.png')

