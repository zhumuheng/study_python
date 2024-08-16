##  导入模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # 画图
import seaborn as sns  # 可视化
from scipy import stats  # Q_Q图

import os

import warnings

warnings.filterwarnings('ignore')  # 忽略警告

from sklearn.linear_model import Ridge  # 岭回归
from sklearn.metrics import mean_squared_error  # 计算MSE
from sklearn import preprocessing  # 归一化
from statsmodels.stats.outliers_influence import variance_inflation_factor  # 多重共线性方差膨胀因子
from sklearn.decomposition import PCA  # 主成分分析法
from sklearn.model_selection import train_test_split  # 切分数据
from sklearn.metrics import mean_squared_error  # 评价指标
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.neighbors import KNeighborsRegressor  # K近邻居回归模型
from sklearn.ensemble import RandomForestRegressor  # 随机森林
from sklearn.tree import DecisionTreeRegressor  # 决策树
from sklearn.svm import SVR  # 支持向量回归
import lightgbm as lgb  # lightgbm模型
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

## 文件及路径
train_data_file = "D:/study_python/工业蒸汽量预测/.venv/input/zhengqi_train.txt"
test_data_file = "D:/study_python/工业蒸汽量预测/.venv/input/zhengqi_test.txt"

##  读取文件
train_data = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
test_data = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')

##  查看文件数据
# train_data.info() # 查看基本信息
# test_data.info() # 查看基本信息
# print(train_data.describe()) # 查看统计信息
# print(test_data.describe()) # 查看统计信息
# print(train_data.head()) # 查看字段信息
# print(test_data.head()) # 查看字段信息

##  可视化数据分布
##  训练集中V0的箱线图
# fig = plt.figure(figsize=(4,6)) # 指定绘图对象的宽度和高度
# sns.boxplot(train_data['V0'], orient='h', width=0.5) # V0的箱线图，orient='v'按Y轴，orient='h' 按X轴，width=0.5指定宽度
# plt.show() # 打印图片

## 绘制V0-V37 箱线图
# columns = train_data.columns.tolist()[:39] # 将表头转化成列表
# fig = plt.figure(figsize=(40,10), dpi=75) # 指定绘图对象的宽度和高度和分辨率
# for i in range(38):
#     plt.subplot(7,8,i+1) # 7行8列子图
#     sns.boxplot(train_data[columns[i]], orient='v', width=0.5) # 箱式图子图
#     plt.ylabel(columns[i])
# plt.show()

## 获取异常数据并画图
## 获取异常数据函数
# def find_outliers(model, x, y, sigma=3):
#     # predict y values using model
#     try:
#         y_pred = pd.Series(model.predict(x), index=y.index) # 创建一维的数据结构Series
#     # if predicting fails ,try fit the model first
#     except:
#         model.fit(x,y)
#         y_pred = pd.Series(model.predict(x), index=y.index) # 创建一维的数据结构Series
#     # calculate residuals between the model prediction and true y values
#     resid = y - y_pred
#     mean_resid = resid.mean()
#     std_resid = resid.std()
#
#     # calculate z statistic,define outlier to be where |Z|>sigma
#     z= (resid - mean_resid)/std_resid
#     outliers = z[abs(z)>sigma].index
#
#     # print and polt the results
#     print('R2=',model.score(x, y))
#     print('mse=',mean_squared_error(y, y_pred))
#     print('-'*10)
#     print('mean of residuals:', mean_resid)
#     print('std of residuals:', std_resid)
#     print('-' * 10)
#     print(len(outliers), 'outliers:')
#     print(outliers.tolist())
#
#     plt.figure(figsize=(15,5))
#     ax_131 = plt.subplot(1,3,1)
#     plt.plot(y, y_pred,'.')
#     plt.plot(y.loc[outliers],y_pred.loc[outliers],'ro')
#     plt.legend(['accept','outliers'])
#     plt.xlabel('y')
#     plt.ylabel('y_pred')
#
#     ax_132 = plt.subplot(1, 3, 2)
#     plt.plot(y, y_pred, '.')
#     plt.plot(y.loc[outliers], y.loc[outliers]-y_pred.loc[outliers], 'ro')
#     plt.legend(['accept', 'outliers'])
#     plt.xlabel('y')
#     plt.ylabel('y-y_pred')
#
#     ax_133 = plt.subplot(1, 3, 3)
#     z.plot.hist(bins=50, ax=ax_133)
#     z.loc[outliers].plot.hist(color='r',bins=50, ax=ax_133)
#     plt.legend(['accept', 'outliers'])
#     plt.xlabel('z')
#
#     plt.savefig('outliers1.png')
#     return outliers
#
# x_train = train_data.iloc[:,0:-1]
# y_train = train_data.iloc[:,-1]
# outliers = find_outliers(Ridge(), x_train, y_train)

## 直方图和Q_Q图 观察正态分布情况
## V0直方图和Q_Q图 观察正态分布情况
# plt.figure(figsize=(10,5))
#
# ax = plt.subplot(1,2,1)
# sns.distplot(train_data['V0'],fit=stats.norm)
# ax = plt.subplot(1,2,2)
# res = stats.probplot(train_data['V0'], plot=plt)
# plt.show()

## 全部变量直方图和Q_Q图 观察正态分布情况
# train_cols = 6
# train_rows = len(train_data.columns)
# plt.figure(figsize=(4*train_cols,4*train_rows))
#
# i = 0
# for col in train_data.columns:
#     i+=1
#     ax = plt.subplot(train_rows,train_cols,i)
#     sns.distplot(train_data[col],fit=stats.norm)
#
#     i+=1
#     ax = plt.subplot(train_rows,train_cols,i)
#     res = stats.probplot(train_data[col], plot=plt)
# plt.tight_layout()
# plt.savefig('Q_Q2.png')

## KDE分布图（核密度估计），观察训练集和测试集的中的特征分布情况，发现两个数据集中分布不一致的情况的特征变量
## 单个变量KDE分布图
# plt.figure(figsize=(8,4),dpi=150)
# ax = sns.kdeplot(train_data['V0'],color='Red',shade=True)
# ax = sns.kdeplot(test_data['V0'],color='Blue',shade=True)
# ax.set_xlabel('V0')
# ax.set_ylabel('Frequency')
# ax = ax.legend('train','test')
# plt.show()

## 全部变量KDE分布图
# dist_cols = 6
# dist_rows = len(test_data.columns)
# plt.figure(figsize=(4*dist_cols,4*dist_rows))
# i = 1
# for col in test_data.columns:
#     ax = plt.subplot(dist_rows, dist_cols, i)
#     ax = sns.kdeplot(train_data[col],color='Red', shade=True)
#     ax = sns.kdeplot(test_data[col], color='Blue', shade=True)
#     ax.set_xlabel(col)
#     ax.set_ylabel('Frequency')
#     ax = ax.legend(['train','test'])
#     i+=1
# plt.savefig('KDEplot.png')

## V0 和 target
# fcols = 2
# frows = 1
# plt.figure(figsize=(8,4),dpi=150)
#
# ax = plt.subplot(1,2,1)
# sns.regplot(x='V0', y='target', data=train_data, ax=ax,
#             scatter_kws={'marker':'.','s':3,'alpha':0.3},
#             line_kws={'color':'k'} )
# plt.xlabel('V0')
# plt.ylabel('target')
#
# ax = plt.subplot(1,2,2)
# sns.distplot(train_data['V0'].dropna())
# plt.xlabel('V0')
# plt.show()

## 全部特征变量和target的线性回归关系图

# fcols = 6
# frows = len(test_data.columns)
# plt.figure(figsize=(5*fcols,4*frows))
#
# i=0
# for col in test_data.columns:
#     i += 1
#     ax = plt.subplot(frows,fcols,i)
#     sns.regplot(x=col,y='target',data=train_data,ax=ax,
#                 scatter_kws={'marker':'.','s':3,'alpha':0.3},
#                 line_kws={'color':'k'})
#     plt.xlabel(col)
#     plt.ylabel('target')
#
#     i += 1
#     ax = plt.subplot(frows,fcols,i)
#     sns.distplot(train_data[col].dropna())
#     plt.xlabel(col)
# plt.savefig('Linear_relationship.png')

## 查看特征变量的相关性
## 计算相关性系数
# pd.set_option('display.max_columns',10)
# pd.set_option('display.max_rows',10)
# data_train1 = train_data.drop(['V5','V9','V11','V17','V22','V28'], axis=1)
#
# train_corr = data_train1.corr()
# train_corr.to_excel('correlation.xlsx') # 导出相关系数表
#
# ax = plt.figure(figsize=(20,16))
# ax = sns.heatmap(train_corr,vmax=.8, square=True, annot=True) #
# '''
# vmax=.8：设置颜色比例的最大值为 0.8，这样可以更好地对比不同的相关性值;
# square=True：使每个单元格呈正方形，annot=True：在每个单元格中显示数值;
# cmap='coolwarm'：使用 coolwarm 颜色图，您也可以选择其他颜色图（如 viridis, plasma 等）;
# '''
# plt.savefig('Correlation_heatmap.png')
# # plt.show()

## 筛选出相关系数大于0.5的特征变量

# threshold = 0.5
# corrmat = train_data.corr()
# # top_corr_features = corrmat[abs(corrmat['target']) > threshold]
# # top_corr_features = top_corr_features['target'].index
# # print(top_corr_features)
# # print()
# top_corr_features = corrmat.index[abs(corrmat['target']) > threshold]
# plt.figure(figsize=(10,10))
# g = sns.heatmap(train_data[top_corr_features].corr(), annot=True)
# plt.savefig('top_Correlation_heatmap.png')
#
# ## 移除相关性低的变量
#
# corr_matrix = data_train1.corr().abs()
# drop_col = corr_matrix[corr_matrix['target']<threshold].index
# data_all.drop(drop_col, axis=1, inplace=True)

## Box-Cox 变换 线性回归基于正态分布，因此在数据分析统计之前需要将数据进行转换使其符合正态分布

# drop_columns = ['V5','V9','V11','V17','V22','V28']
# ## 合并训练集数据和测试集数据,pandas做归一化
# train_x = train_data.drop(['target'], axis=1)
# data_all = pd.concat([train_x,test_data])
#
# data_all.drop(drop_columns,axis=1,inplace=True)
# print(data_all.head())
#
# ## 使用pandas对合并后的数据每一列进行归一化
# cols_numeric = list(data_all.columns)
#
# def scale_minmax(col):
#     '''定义归一化计算公式'''
#     return (col-col.min())/(col.max()-col.min())
# #
# # data_all[cols_numeric] = data_all[cols_numeric].apply(scale_minmax,axis=0)
# # print(data_all[cols_numeric].describe())
# ## 分开归一化
# train_data_process = train_data[cols_numeric]
# train_data_process = train_data_process[cols_numeric].apply(scale_minmax,axis=0)
#
# test_data_process = test_data[cols_numeric]
# test_data_process = test_data[cols_numeric].apply(scale_minmax,axis=0)
#
# # ## 使用sklearn对合并后的数据每一列进行归一化，好处就是可以全部使用训练集中的最大值最小值出来测试集和验证集
# # features_columns = list(train_x)
# # min_max_scaler = preprocessing.MinMaxScaler()
# # min_max_scaler = min_max_scaler.fit(train_x(features_columns))
#
# ## 对特征变量进行box-cox变换，并画图
# cols_numeric_left = cols_numeric[0:13]
# cols_numeric_right = cols_numeric[13:]
# train_data_process = pd.concat([train_data_process,train_data['target']],axis=1)
#
# fcols = 6
# frows = len(cols_numeric_left)
# plt.figure(figsize=(4*fcols,4*frows))
#
# i=0
# for var in cols_numeric_left:
#     dat = train_data_process[[var, 'target']].dropna()
#     i += 1
#     plt.subplot(frows, fcols, i)
#     sns.distplot(dat[var],fit=stats.norm)
#     plt.title(var + 'original')
#     plt.xlabel('')
#     plt.ylabel('')
#     i += 1
#     plt.subplot(frows, fcols, i)
#     _ = stats.probplot(dat[var], plot=plt)
#     plt.title('skew'+ '{:4f}'.format(stats.skew(dat[var])))
#     plt.xlabel('')
#     plt.ylabel('')
#     i += 1
#     plt.subplot(frows, fcols, i)
#     plt.plot(dat[var], dat['target'], '.', alpha=0.5)
#     plt.title('corr'+'{:.2f}'.format(np.corrcoef(dat[var], dat['target'])[0][1]))
#     i += 1
#     plt.subplot(frows, fcols, i)
#     trans_var, lambda_var = stats.boxcox(dat[var].dropna() + 1)
#     trans_var = scale_minmax(trans_var)
#     sns.distplot(trans_var, fit=stats.norm)
#     plt.title(var + 'transformed')
#     plt.xlabel('')
#     i += 1
#     plt.subplot(frows, fcols, i)
#     _ = stats.probplot(trans_var, plot=plt)
#     plt.title('skew' + '{:4f}'.format(stats.skew(trans_var)))
#     plt.xlabel('')
#     plt.ylabel('')
#     i += 1
#     plt.subplot(frows, fcols, i)
#     plt.plot(trans_var, dat['target'], '.', alpha=0.5)
#     plt.title('corr' + '{:.2f}'.format(np.corrcoef(trans_var, dat['target'])[0][1]))
# plt.savefig('box-cox-qq.png')

# ## 特征选择
# ## 数据预处理
# ## 标准化
# from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import load_iris
# iris = load_iris()
#
# ## 标准化后，返回值为标准化后的数据
# StandardScaler().fit_transform(iris.data)
#
# ## 归一化-区间缩放法
# from sklearn.preprocessing import MinMaxScaler
# ## 区间缩放，返回值为缩放到[0,1]区间的数据
# MinMaxScaler().fit_transform(iris.data)
#
# ## 归一化 可以把数据映射到[0,1]或者[a,b]
# from sklearn.preprocessing import Normalizer
# Normalizer().fit_transform(iris.data)
#
# from sklearn.preprocessing import MinMaxScaler
# ## 区间缩放，自定义为缩放到[-1,1]区间的数据
# a, b = -1, 1
# print(MinMaxScaler(feature_range=(a,b)).fit_transform(iris.data))

## 箱线图
# plt.figure(figsize=(18, 10))  # 定义画布大小
# plt.boxplot(x=train_data.values, labels=train_data.columns)  # 定义X和labels
# plt.hlines([-7.5, 7.5], 0, 40, colors='r')  # 定义上线和下线，及线条最小值和最大值（起点终点），及上线和下线颜色
# plt.savefig('boxplot3.png')

## 箱线图结果,V9中存在异常值需删除
train_data = train_data[train_data['V9'] > -7.5]
test_data = test_data[test_data['V9'] > -7.5]
print('train_data\n', train_data.describe())
print('test_data\n', test_data.describe())

## 数据归一化
features_columns = [col for col in train_data.columns if col not in ['target']]
print(features_columns)
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler = min_max_scaler.fit(train_data[features_columns])

train_data_scaler = min_max_scaler.transform(train_data[features_columns])
test_data_scaler = min_max_scaler.transform(test_data[features_columns])

train_data_scaler = pd.DataFrame(train_data_scaler)
train_data_scaler.columns = features_columns

test_data_scaler = pd.DataFrame(test_data_scaler)
test_data_scaler.columns = features_columns

train_data_scaler['target'] = train_data['target']

print('train_data_scaler\n', train_data_scaler.describe())
print('test_data_scaler\n', test_data_scaler.describe())

## KDE分布
## KDE分布图（核密度估计），观察训练集和测试集的中的特征分布情况，发现两个数据集中分布不一致的情况的特征变量
# dist_cols = 6
# dist_rows = len(test_data.columns)
# plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))
# i = 1
# for col in test_data.columns:
#     ax = plt.subplot(dist_rows, dist_cols, i)
#     ax = sns.kdeplot(train_data[col], color='Red', shade=True)
#     ax = sns.kdeplot(test_data[col], color='Blue', shade=True)
#     ax.set_xlabel(col)
#     ax.set_ylabel('Frequency')
#     ax = ax.legend(['train', 'test'])
#     i += 1
# plt.savefig('KDEplot.png')

##  KDE结果V5，v9，v11，v17，v22，v28，在训练集和测试集中分布差异较大，会影响模型的泛化能力，因此删除
train_data_scaler = train_data_scaler.drop(labels=['V5', 'V9', 'V11', 'V17', 'V22', 'V28'], axis=1)
test_data_scaler = test_data_scaler.drop(labels=['V5', 'V9', 'V11', 'V17', 'V22', 'V28'], axis=1)

## 计算相关性特征，热力图可视化
# plt.figure(figsize=(20, 16))
column = train_data_scaler.columns.tolist()
mcorr = train_data_scaler[column].corr(method='spearman')  # 计算 train_data_scaler 数据框中各列之间的 Spearman 相关系数矩阵
# mask = np.zeros_like(mcorr, dtype=np.bool)  # 创建一个与 mcorr 形状相同的布尔型矩阵，初始化为全零。这个矩阵将用作掩码，以隐藏热力图的上三角部分
# mask[np.triu_indices_from(mask)] = True  # 将 mask 矩阵的上三角部分设置为 True，这样在绘制热力图时，这些位置的值将被隐藏
# cmap = sns.diverging_palette(220, 10, as_cmap=True)  # 创建一个发散色调调色板（从蓝色到红色），用于在热力图中区分正负相关性值
# g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')
# '''
# mcorr 是要可视化的相关性矩阵。
# mask 用于隐藏上三角部分。
# cmap 指定颜色映射。
# square=True 使每个单元格呈方形。
# annot=True 在每个单元格中显示数值。
# fmt='0.2f' 指定显示的数值格式为小数点后两位。
# '''
# plt.savefig('corr1.png')

##  特征降维，进行特征相关性初筛，计算相关系数并筛选大于0.1的特征变量
mcorr = mcorr.abs()
numerical_corr = mcorr[mcorr['target'] > 0.1]['target']
print(numerical_corr.sort_values(ascending=False))  # ascending=False 表示要对数据进行降序排序（从高到低）
# print(numerical_corr.index)

##  多重共线性分析
new_numerica = ['V0', 'V2', 'V3', 'V4', 'V6', 'V10', 'V13',
                'V15', 'V16', 'V18', 'V19', 'V20', 'V24',
                'V30', 'V31', 'V37']  # 删除与已有特征相关性大于0.75的
X = np.matrix(train_data_scaler[new_numerica])  # 将 train_data_scaler 数据框中列名为 new_numerica 的特征子集转换为 numpy 矩阵对象，并将其赋值给 X
print('X\n', X)
VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]  # 计算矩阵 X 中第 i 个特征的方差膨胀因子 (VIF)
print(VIF_list)

##  多重共线性分析结果均大于10，需要使用 PCA 降维-主成分分析法
## 保持90%的信息
pca = PCA(n_components=0.9)
new_train_pca_90 = pd.DataFrame(pca.fit_transform(train_data_scaler.iloc[:, 0:-1]))
new_test_pac_90 = pd.DataFrame(pca.fit_transform(test_data_scaler))
new_train_pca_90['target'] = train_data_scaler['target']
print(new_train_pca_90.describe())
print(train_data_scaler.describe())

##  PCA处理后保留16个主成分
pca = PCA(n_components=16)
new_train_pca_16 = pca.fit_transform(train_data_scaler.iloc[:, 0:-1])
new_train_pca_16 = pd.DataFrame(new_train_pca_16)
new_test_pac_16 = pd.DataFrame(pca.fit_transform(test_data_scaler))
new_train_pca_16['target'] = train_data_scaler['target']

##  切分数据集
new_train_pca_16 = new_train_pca_16.fillna(0)
train = new_train_pca_16[new_test_pac_16.columns]
target = new_train_pca_16['target']

##  切分数据，训练数据为80%，验证数据为20%，简单交叉验证
train_data, test_data, train_target, test_target = train_test_split(train, target, test_size=0.2, random_state=0)
# ## 线性回归
# clf = LinearRegression()
# clf.fit(train_data, train_target)
# test_pred = clf.predict(test_data)
# score = mean_squared_error(test_target, test_pred)
# print('LinearRegression: ', score)
#
# ## K近邻回归模型
# clf = KNeighborsRegressor(n_neighbors=3)  # 最近的3个
# clf.fit(train_data,train_target)
# test_pred = clf.predict(test_data)
# score = mean_squared_error(test_target, test_pred)
# print('KNeighborsRegressor: ', score)
#
# ##  决策树回归模型
#
# clf = DecisionTreeRegressor()
# clf.fit(train_data,train_target)
# test_pred = clf.predict(test_data)
# score = mean_squared_error(test_target, test_pred)
# print('DecisionTreeRegressor: ', score)
#
# ##  随机森林回归
#
# clf = RandomForestRegressor(n_estimators=200)  # 200棵树
# clf.fit(train_data,train_target)
# test_pred = clf.predict(test_data)
# score = mean_squared_error(test_target, test_pred)
# print('RandomForestRegressor: ', score)
#
# ##  LGB回归模型
# clf = lgb.LGBMRegressor(
#     learning_rate=0.01,
#     max_depth=-1,
#     n_estimators=5000,
#     boosting_type='gbdt',
#     random_state=2019,
#     objective='regression'
# )
#
# clf.fit(X=train_data, y=train_target, eval_metric='MSE')
# score = mean_squared_error(test_target, test_pred)
# print('LGBMRegressor: ', score)

##  K折交叉验证

from sklearn.model_selection import KFold
from sklearn.linear_model import SGDRegressor  # 随机梯度下降

# kf = KFold(n_splits=5)  # 5折交叉验证
# for k, (train_index, test_index) in enumerate(kf.split(train)):
#     train_data, test_data, train_target, test_target =
#     train.values[train_index], train.values[test_index], target[train_index], target[test_index]
#     clf = SGDRegressor(max_iter=1000, tol=1e-3)
#     clf.fit(train_data, train_target)
#     score_train = mean_squared_error(train_target, clf.predict(train_data))
#     score_test = mean_squared_error(test_target, clf.predict(test_data))
#     print(k, '折', 'SGDRegressor train mse:', score_train)
#     print(k, '折', 'SGDRegressor test mse:', score_test)

from sklearn.model_selection import GridSearchCV  # 网格搜索
from sklearn.model_selection import RandomizedSearchCV  # 随机参数优化
from sklearn.ensemble import RandomForestRegressor  # 随机森林

# randomForestRegressor = RandomForestRegressor()
# parameters = {'n_estimators': [50, 100, 200], 'max_depth': [1, 2, 3]}
# clf = GridSearchCV(randomForestRegressor, parameters, cv=5)  # 5折交叉验证
# clf.fit(train_data, train_target)
#
# score_test = mean_squared_error(test_target, clf.predict(test_data))
# print('RandomForestRegressor GridSearchCV test mse:', score_test)
#
# parma_result = clf.cv_results_
#
# parma_result_keys = parma_result.keys()
# for key in parma_result_keys:
#     print(str(key)+':'+str(parma_result.get(key)))
# print('best_estimators_:' + str(clf.best_estimator_))
# print('best_params_:' + str(clf.best_params_))
# print('best_score:' + str(clf.best_score_))

##  随机参数优化
# clf = RandomizedSearchCV(randomForestRegressor, parameters, cv=5)
# clf.fit(train_data, train_target)
#
# score_test = mean_squared_error(test_target, clf.predict(test_data))
# print('RandomForestRegressor RandomizedSearchCV test mse:', score_test)
#
# parma_result = clf.cv_results_
#
# parma_result_keys = parma_result.keys()
# for key in parma_result_keys:
#     print(str(key) + ':' + str(parma_result.get(key)))
# print('best_estimators_:' + str(clf.best_estimator_))
# print('best_params_:' + str(clf.best_params_))
# print('best_score:' + str(clf.best_score_))

##  学习曲线
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

# plt.figure(figsize=(18, 10), dpi=150)
#
# print('学习曲线')
#
#
# def plot_learning_curve(estimator,
#                         title,
#                         x,
#                         y,
#                         ylim=None,
#                         cv=None,
#                         n_jobs=1,
#                         train_sizes=np.linspace(.1, 1.0, 5)):
#     """
#     定义学习曲线函数
#     """
#
#     # plt.figure()
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel('Training examples')
#     plt.ylabel('Score')
#     train_sizes, train_scores, test_scores = learning_curve(estimator, x, y, cv=cv, n_jobs=n_jobs,
#                                                           train_sizes=train_sizes)
#     print('train_sizes:', train_sizes)
#     print('train_scores:', train_scores)
#     print('test_scores:', test_scores)
#
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()  # 启用网格线，以便更清晰地观察图表上的数据点
#
#     print('train_scores_mean:', train_scores_mean)
#     print(type(train_scores_mean))
#     print('train_scores_std:', train_scores_std)
#     print('test_scores_mean:', test_scores_mean)
#     print('test_scores_std:', test_scores_std)
#
#     plt.fill_between(train_sizes,
#                      train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std,
#                      alpha=0.1,
#                      color='r')
#
#     print('train_scores_mean - train_scores_std', train_scores_mean - train_scores_std)
#     print('train_scores_mean + train_scores_std:', train_scores_mean + train_scores_std)
#
#     plt.fill_between(train_sizes,
#                      test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std,
#                      alpha=0.1,
#                      color='g')
#
#     print('test_scores_mean - test_scores_std', test_scores_mean - test_scores_std)
#     print('test_scores_mean + test_scores_std:', test_scores_mean + test_scores_std)
#
#     plt.plot(train_sizes,
#              train_scores_mean,
#              'o-',
#              color='r',
#              label='Training score')
#     plt.plot(train_sizes,
#              test_scores_mean,
#              'o-',
#              color='g',
#              label='Cross-validatioon score')
#     plt.legend(loc='best')
#     # plt.show()
#     return plt
#
#
# x = new_train_pca_16[new_test_pac_16.columns].values
# y = new_train_pca_16['target'].values
# print(x)
# print('*'*20)
# print(y)
#
# title = 'linerRegressor'
# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
# estimator = SGDRegressor()
# plot_learning_curve(estimator, title, x, y, ylim=(0.5, 1.01), cv=cv, n_jobs=-1)
# plt.show()

##  验证曲线

from sklearn.model_selection import validation_curve

# x = new_train_pca_16[new_test_pac_16.columns].values
# print('x:', x)
# y = new_train_pca_16['target'].values
# print('y:', y)
# param_range = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
# # param_range = np.logspace(-6, -1, 5)
# train_scores, test_scores = validation_curve(SGDRegressor(max_iter=1000,
#                                                           tol=1e-3,
#                                                           penalty='l1'),  # max_iter=1000: 最大迭代次数，模型在训练时最多运行 1000 次迭代,tol=1e-3: 收敛的容忍度，当模型的优化收敛速度低于这个阈值时，迭代将停止,penalty='L1': 使用 L1 正则化（Lasso）
#                                              x,
#                                              y,
#                                              param_name='alpha',
#                                              param_range=param_range,
#                                              cv=10,
#                                              scoring='r2',
#                                              n_jobs=1)
#
#
# print('train_scores:', train_scores)
# print('test_scores:', test_scores)
#
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
#
# print('train_scores_mean:', train_scores_mean)
# print(type(train_scores_mean))
# print('train_scores_std:', train_scores_std)
# print('test_scores_mean:', test_scores_mean)
# print('test_scores_std:', test_scores_std)
#
# plt.title('validation_curve with SGDRegressor')
# plt.xlabel('alpha')
# plt.ylabel('score')
# plt.ylim(0.0, 1.1)
# plt.semilogx(param_range, train_scores_mean, label='Training score', color='r')
# plt.fill_between(param_range,
#                  train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std,
#                  alpha=0.2,
#                  color='r')
# print('train_scores_mean - train_scores_std:', train_scores_mean - train_scores_std)
# print('train_scores_mean + train_scores_std:', train_scores_mean + train_scores_std)
#
# plt.semilogx(param_range,
#                  test_scores_mean,
#                  label='Cross-validatioon score',
#                  color='g')
#
# print('test_scores_mean:', test_scores_mean)
#
# plt.fill_between(param_range,
#                  test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std,
#                  alpha=0.2,
#                  color='g')
#
# print('test_scores_mean - test_scores_std:', test_scores_mean - test_scores_std)
# print('test_scores_mean - test_scores_std:', test_scores_mean - test_scores_std)
# plt.legend(loc='best')
# plt.show()

##  模型融合


