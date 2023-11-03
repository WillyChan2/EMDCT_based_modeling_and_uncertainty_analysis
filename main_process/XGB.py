from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import torch
import math


def prepara_data(data):
    data_position = data[['X', 'Y', 'Z']]
    data_label = data[['Strata']]
    data_label = data_label.to_numpy().reshape(-1, 1).astype(int)
    return data_position, data_label


# A method mainly used to evaluate the performance
def Model_Estimating(x_test, y_test, path1, path2, path3, path5):
    # 在训练集和测试集上分布利用训练好的模型进行预测
    print("-------------------------------------------------")
    print('estimating_model...')
    start = time.process_time()
    test_predict = clf.predict(x_test)
    test_predict_proba = clf.predict_proba(x_test)
    print('The accuracy of the XGboost is:', metrics.accuracy_score(y_test, test_predict))
    print('The recall of the XGboost is:', metrics.recall_score(y_test, test_predict, average='weighted'))
    print('The recall of the XGboost for every single class is:',
          metrics.recall_score(y_test, test_predict, average=None))
    print('The precision of the XGboost is:', metrics.precision_score(y_test, test_predict, average='weighted'))
    print('The precision of the XGboost for every single class is:',
          metrics.precision_score(y_test, test_predict, average=None))
    print('The F1 of the XGboost is:', metrics.f1_score(y_test, test_predict, average='weighted'))
    print('The f1_score of the XGboost for every single class is:',
          metrics.f1_score(y_test, test_predict, average=None))
    proba1 = np.column_stack((y_test, test_predict, x_test, test_predict_proba))  # 把点的坐标和地层值组合起来，形成数组 array
    tot = pd.DataFrame(proba1)
    tot.columns = ['label_true', 'label_predict', 'x', 'y', 'z', 'predict_Strata1',
                   'predict_Strata2', 'predict_Strata3', 'predict_Strata4', 'predict_Strata5', 'predict_Strata6',
                   'predict_Strata7']
    np.savetxt(path1, proba1, delimiter=',', fmt='%s')  # 输出完整内容

    confusion_matrix_result = metrics.confusion_matrix(y_test, test_predict)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.heatmap(confusion_matrix_result, xticklabels=['1 ', '2', '3', '4', '5', '6', '7'],
                yticklabels=['1 ', '2', '3', '4', '5', '6', '7'], fmt=".0f", annot=True, annot_kws={"size": 8},
                linewidths=0.5, cmap='Blues', cbar_kws={"shrink": .8}, square=True)  # 画热力图
    ax.set_title('confusion matrix of XGBoost')  # 标题
    ax.set_xlabel('predict labels')  # x轴
    ax.set_ylabel('true labels')  # y轴
    params = {'axes.labelsize': 12,
              'font.size': 12,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'mathtext.fontset': 'stix',
              'font.family': 'sans-serif',
              'font.sans-serif': 'Times New Roman'}
    plt.rcParams.update(params)
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.savefig(path2)
    print('The confusion matrix result:\n', confusion_matrix_result)
    samples1 = np.column_stack((x_test, y_test, test_predict))
    np.savetxt(path5, samples1, delimiter=',')
    end = time.process_time()
    runTime = (end - start)
    print('Finished estimating, and it costs:', runTime, 's')
    print("============================")

    print('uncertainty analysis...')
    start = time.process_time()
    test_predict_proba = proba1[:, 5:12]
    test_predict_proba = torch.tensor(test_predict_proba, dtype=torch.float32).cuda()
    log_data_label = torch.log(test_predict_proba) * (-1)
    info_entropy = log_data_label.mul(test_predict_proba)
    info_entropy = np.nan_to_num(np.array(info_entropy.cpu()))
    cell_info_entropy = np.sum(info_entropy, axis=1)
    one_info_entropy = cell_info_entropy / math.log(OUTPUT_UNITS)

    samples = np.column_stack((np.array(proba1), one_info_entropy))
    np.savetxt(path3, samples, delimiter=',', fmt='%.3f')
    end = time.process_time()
    runTime = (end - start)
    print('Finish analysing, and it costs:', runTime, 's')


OUTPUT_UNITS = 7
data_train = pd.read_csv('../data/dataset_example/points_train_Boreholes149.csv')
data_general_test = pd.read_csv('../data/dataset_example/general_testing_operation60+61.csv')
# the path to export
path1 = "../data/result/test_predict_proba.csv"  # Detailed probability
path2 = "../data/result/test_confusion matrix.jpg"  # Confusion matrix
path3 = "../data/result/test_cloud_points_entropy.csv"  # Test set borehole information entropy results storage
path5 = "../data/result/test_points_results.txt"  # Test set formation result

x_train, y_train = prepara_data(data_train)
x_general_test, y_general_test = prepara_data(data_general_test)

clf = XGBClassifier(learning_rate=0.1,
                    n_estimators=450,
                    max_depth=5,
                    min_child_weight=6,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)
clf.fit(x_train, y_train)
Model_Estimating(x_general_test, y_general_test, general_test_path1, general_test_path2, general_test_path3,
                 general_test_path5)
