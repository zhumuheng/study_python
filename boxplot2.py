import config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


## 读取文件
train_data = pd.read_csv(config.train_data_file, sep='\t', encoding='utf-8')
test_data = pd.read_csv(config.test_data_file, sep='\t', encoding='utf-8')

plt.figure(figsize=(18,10))
plt.boxplot(x=train_data.values,tick_labels=train_data.columns)
plt.hlines([-7.5,7.5],0,40,colors='r')
plt.show()

