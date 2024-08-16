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

## 查看特征变量的相关性
## 计算相关性系数
pd.set_option('display.max_columns',10)
pd.set_option('display.max_rows',10)
data_train1 = train_data.drop(['V5','V9','V11','V17','V22','V28'], axis=1)

train_corr = data_train1.corr()
train_corr.to_excel('correlation.xlsx') # 导出相关系数表

ax = plt.figure(figsize=(20,16))
ax = sns.heatmap(train_corr,vmax=.8, square=True, annot=True) #
'''
vmax=.8：设置颜色比例的最大值为 0.8，这样可以更好地对比不同的相关性值;
square=True：使每个单元格呈正方形，annot=True：在每个单元格中显示数值;
cmap='coolwarm'：使用 coolwarm 颜色图，您也可以选择其他颜色图（如 viridis, plasma 等）;
'''
plt.savefig('Correlation_heatmap.png')
# plt.show()

## 筛选出相关系数大于0.5的特征变量

threshold = 0.5
corrmat = train_data.corr()
# top_corr_features = corrmat[abs(corrmat['target']) > threshold]
# top_corr_features = top_corr_features['target'].index
# print(top_corr_features)
# print()
top_corr_features = corrmat.index[abs(corrmat['target']) > threshold]
plt.figure(figsize=(10,10))
g = sns.heatmap(train_data[top_corr_features].corr(), annot=True)
plt.savefig('top_Correlation_heatmap.png')

## 移除相关性低的变量

corr_matrix = data_train1.corr().abs()
drop_col = corr_matrix[corr_matrix['target']<threshold].index
data_all.drop(drop_col, axis=1, inplace=True)


