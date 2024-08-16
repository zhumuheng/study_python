# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.linear_model import SGDRegressor
#
#
# # train_scores_mean = np.array([0.62981592, 0.64235553, 0.64841535, 0.65026326, 0.65233059])
# # print(train_scores_mean)
# # train_sizes = np.array([ 230,  750, 1269, 1788, 2308])
# # plt.plot(train_sizes,
# #              train_scores_mean,
# #              'o-',
# #              color='r',
# #              label='Training score')
# # plt.show()
#
# param_range = np.logspace(-6, -1, 5)
# print(param_range)
#
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.datasets import load_digits
# from sklearn.svm import SVC
# from sklearn. model_selection import validation_curve
#
# digits = load_digits()
# X, y = digits.data, digits.target
#
# param_range = np.logspace(-6, -1, 5)
# train_scores, test_scores = validation_curve(
#     SVC(), X, y, param_name="gamma", param_range=param_range,
#     cv=10, scoring="accuracy", n_jobs=1)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
#
# plt.title("Validation Curve with SVM")
# plt.xlabel("$\gamma$")
# plt.ylabel("Score")
# plt.ylim(0.0, 1.1)
# plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
# plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.2, color="r")
# plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
#              color="g")
# plt.fill_between(param_range, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.2, color="g")
# plt.legend(loc="best")
# plt.show()

# from xgboost import XGBRegressor
#
# # 创建一个 XGBRegressor 实例
# model = XGBRegressor(
#     max_depth=2,
#     n_estimators=200,
#     random_state=42
# )
#
# # 获取所有参数
# params = model.get_params()
#
# # 打印参数
# for param, value in params.items():
#     print(f"{param}: {value}")

import re

code_str = """
class MyClass:
    pass

class MyOtherClass:
    pass
"""

class_names = re.findall(r'class (\w+):', code_str)
print(class_names)  # 输出: ['MyClass', 'MyOtherClass']

