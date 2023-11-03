import pandas as pd
import numpy as np
import torch
import math
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


def prepara_data(data):
    data_position = data[['X', 'Y', 'Z']]
    data_position.astype(np.float64)
    data_label = data[['Strata']]
    data_label = data_label.to_numpy().reshape(-1).astype(int)
    return data_position, data_label


OUTPUT_UNITS = 7
data_train = pd.read_csv('../data/dataset_example/points_train_Boreholes149.csv')
data_general_test = pd.read_csv('../data/dataset_example/general_testing_operation60+61.csv')
# the path to export
path1 = "../data/result/test_predict_proba.csv"  # Detailed probability
path2 = "../data/result/test_confusion matrix.jpg"  # Confusion matrix
path3 = "../data/result/test_cloud_points_entropy.csv"  # Test set borehole information entropy results storage
path5 = "../data/result/test_points_results.txt"  # Test set formation result

# 2.SVM- basic data processing
x_train, y_train = prepara_data(data_train)
x_test, y_test = prepara_data(data_general_test)

# 3. Feature engineering: standardization
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4. SVM classifier model training
start = time.process_time()
print('training...............')
svm_model = SVC(kernel="rbf", gamma=0.33333, decision_function_shape='ovo', C=1, probability=True, max_iter=4000, random_state=25)
svm_model.fit(x_train, y_train)

# 5. Predictive test data sets
print('testing...............')
y_predict = svm_model.predict(x_test)

# 6. Print forecast results and model scores
print("Predicted labels: ", y_predict)
accuracy = svm_model.score(x_test, y_test)
print("SVM_Accuracy score: ", accuracy)
test_predict_proba = svm_model.predict_proba(x_test)
print('The test predict Probability of each class:\n', test_predict_proba)
x_test = transfer.inverse_transform(x_test)
proba1 = np.column_stack((y_test, y_predict, x_test, test_predict_proba))
tot = pd.DataFrame(proba1)
tot.columns = ['label_true', 'label_predict', 'x', 'y', 'z', 'predict_Strata1',
               'predict_Strata2', 'predict_Strata3', 'predict_Strata4', 'predict_Strata5', 'predict_Strata6',
               'predict_Strata7']
np.savetxt(path1, proba1, delimiter=',', fmt='%.3f')
samples1 = np.column_stack((x_test, y_test, y_predict))
np.savetxt(path5, samples1, delimiter=',', fmt='%.3f')

# 7. Confusion Matrix
conf_mat = confusion_matrix(y_test, y_predict)
fig = plt.figure()
ax = fig.add_subplot(111)
sns.heatmap(conf_mat, xticklabels=['1 ', '2', '3', '4', '5', '6', '7'],
            yticklabels=['1 ', '2', '3', '4', '5', '6', '7'], fmt=".0f", annot=True, annot_kws={"size": 8},
            linewidths=0.5, cmap='Blues', cbar_kws={"shrink": .8}, square=True)
ax.set_title('confusion matrix of SVM')
ax.set_xlabel('predict labels')
ax.set_ylabel('true labels')
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

# Uncertainty analysis
print("============================")
print('uncertainty analysis...')
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
