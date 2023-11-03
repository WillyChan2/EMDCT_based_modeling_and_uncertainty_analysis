import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import time
import numpy
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def DataAnalysis(data):
    data = data[['Strata', 'X', 'Y', 'Z']]
    data.info()
    data = data.fillna(-1)
    for col in data.columns:
        if col != 'Strata':
            sns.boxplot(x='Strata', y=col, saturation=0.5, palette='pastel', data=data)
            plt.title(col)


# A method mainly used to evaluate the performance
def estimate_model(x, y, path1, path2, path3, path5, model):
    print("-------------------------------------------------")
    print('estimating_model...')
    start = time.process_time()
    test_predict = model.predict(x)
    test_predict_proba = model.predict_proba(x)
    print('The test predict Probability of each class:\n', test_predict_proba)

    proba1 = np.column_stack((y, test_predict, x, test_predict_proba))  # 把点的坐标和地层值组合起来，形成数组 array
    tot = pd.DataFrame(proba1)
    tot.columns = ['label_true', 'label_predict', 'x', 'y', 'z', 'predict_Strata1',
                   'predict_Strata2', 'predict_Strata3', 'predict_Strata4', 'predict_Strata5', 'predict_Strata6', 'predict_Strata7']
    np.savetxt(path1, proba1, delimiter=',', fmt='%s')

    samples1 = np.column_stack((x, test_predict))
    np.savetxt(path5, samples1, delimiter=',')

    print('The accuracy of the test_DeepForest is:', metrics.accuracy_score(y, test_predict))
    print('The recall of the test_DeepForest is:', metrics.recall_score(y, test_predict, average='weighted'))
    print('The recall of the test_DeepForest for every single class is:',
          metrics.recall_score(y, test_predict, average=None))
    print('The precision of the test_DeepForest is:', metrics.precision_score(y, test_predict, average='weighted'))
    print('The precision of the test_DeepForest for every single class is:',
          metrics.precision_score(y, test_predict, average=None))

    print('The F1 of the test_DeepForest is:', metrics.f1_score(y, test_predict, average='weighted'))
    print('The f1_score of the test_DeepForest for every single class is:',
          metrics.f1_score(y, test_predict, average=None))

    confusion_matrix_result = metrics.confusion_matrix(y, test_predict)  # 错了，已修改
    print('The confusion matrix result:\n', confusion_matrix_result)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.heatmap(confusion_matrix_result, xticklabels=['1 ', '2', '3', '4', '5','6' ,'7'],
                yticklabels=['1 ', '2', '3', '4', '5', '6', '7'], fmt=".0f", annot=True, annot_kws={"size": 8},
                linewidths=0.5, cmap='Blues', cbar_kws={"shrink": .8}, square=True)  # 画热力图
    ax.set_title('confusion matrix of RF')  # 标题
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
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    plt.savefig(path2)
    x = np.array(x)
    row = 0
    for i in y:
        if np.argmax(test_predict[row]) == 0:
            test_predict_proba = np.delete(test_predict_proba, row, axis=0)  # 剔除无效信息，label=0的
            x = np.delete(x, row, axis=0)  # 剔除无效 信息，label=-1的对应位置
        else:
            row = row + 1
    test_predict_proba = torch.tensor(test_predict_proba, dtype=torch.float32).cuda()  # 标准化后 转为张量
    log_data_label = torch.log(test_predict_proba) * (-1)
    info_entropy = log_data_label.mul(test_predict_proba)
    info_entropy = np.nan_to_num(numpy.array(info_entropy.cpu()))
    cell_info_entropy = np.sum(info_entropy, axis=1)  # 每一个点的信息熵，信息熵越大则越不确定. inf*0=nan,排除这一部分，规定0ln0=0
    samples1 = np.column_stack((x, cell_info_entropy))  # 把点的坐标和信息熵组合起来，形成数组 array
    np.savetxt(path3, samples1, delimiter=',')  # TODO
    end = time.process_time()
    runTime = (end - start)
    print('Finished estimating, and it costs:', runTime, 's')


OUTPUT_UNITS = 7
data_train = pd.read_csv('../data/dataset_example/points_train_Boreholes149.csv')
data_general_test = pd.read_csv('../data/dataset_example/general_testing_operation60+61.csv')
# the path to export
path1 = "../data/result/test_predict_proba.csv"  # Detailed probability
path2 = "../data/result/test_confusion matrix.jpg"  # Confusion matrix
path3 = "../data/result/test_cloud_points_entropy.csv"  # Test set borehole information entropy results storage
path5 = "../data/result/test_points_results.txt"  # Test set formation result
DataAnalysis(data_general_test)
x_train, x_0, y_train, y_0 = train_test_split(data_train[['X', 'Y', 'Z']], data_train[['Strata']], test_size=0.000001,
                                              random_state=2020)
x_general_test, x_0, y_general_test, y_0 = train_test_split(data_general_test[['X', 'Y', 'Z']],
                                                            data_general_test[['Strata']], test_size=0.000001,
                                                            random_state=2020)
print('====================================================')
print('training...')
start = time.process_time()
rfc = RandomForestClassifier(random_state=5, n_estimators=100, oob_score=True)
rfc.fit(x_train, y_train.values.ravel())
print(rfc)
end = time.process_time()
runTime = (end - start)
print('Finished Training, and it costs:', runTime, 's')
print('=====================================================')
estimate_model(x_general_test, y_general_test, general_test_path1, general_test_path2, general_test_path3, general_test_path5, rfc)