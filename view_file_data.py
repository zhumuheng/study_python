import config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


## 读取文件
train_data = pd.read_csv(config.train_data_file, sep='\t', encoding='utf-8')
test_data = pd.read_csv(config.test_data_file, sep='\t', encoding='utf-8')

# 查看文件数据
train_data.info() # 查看基本信息
test_data.info() # 查看基本信息
print(train_data.describe()) # 查看统计信息
print(test_data.describe()) # 查看统计信息
print(train_data.head()) # 查看字段信息
print(test_data.head()) # 查看字段信息

# 绘制V0-V37 箱线图
colums = train_data.columns.tolist()[:39] # 将表头转化成列表
fig = plt.figure(figsize=(40,10), dpi=75) # 指定绘图对象的宽度和高度和分辨率
for i in range(38):
    plt.subplot(7,6,i+1) # 7行8列子图
    sns.boxplot(train_data[colums[i]], orient='v', width=0.5) # 箱式图子图
    plt.ylabel(colums[i])
plt.savefig('boxplot.png')