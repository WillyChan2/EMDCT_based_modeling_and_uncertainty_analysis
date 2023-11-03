import time
import math
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import Delaunay
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from deepforest import CascadeForestClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
import catboost as cb
from sklearn import svm
from sklearn.preprocessing import StandardScaler


# A 3-point function that computes the outer circle for filtering the training set
def circumcircle(p1, p2, p3):
    x = p1[0] + p1[1] * 1j
    y = p2[0] + p2[1] * 1j
    z = p3[0] + p3[1] * 1j
    w = (z - x) / (y - x)
    res = (x - y) * (w - abs(w) ** 2) / 2j / w.imag - x
    X = -res.real
    Y = -res.imag
    rad = abs(res + x) * 2
    return X, Y, rad


# merge the accuracy
def merge(proba, acc, id):
    global overall_result
    overall_acc.pop()
    overall_acc.append(acc)
    overall_name.append(id)
    overall_result = np.concatenate((overall_result, proba), axis=0)


# A method mainly used to evaluate the performance of an algorithm on sub-blocks
def estimate_model(x, y, path1, path2, path5, model, id_model, b=True):
    print("-------------------------------------------------")
    print('estimating_model...')
    start = time.process_time()
    # The trained model is used to make prediction on the training set and the test set respectively
    test_predict = model.predict(x)
    # Generate prediction probabilities:
    # where the first column represents the probability of predicting class 1,
    # the second column represents the probability of predicting class 2,
    # and the third column represents the probability of predicting class 3.
    test_predict_proba = model.predict_proba(x)
    accuracy = metrics.accuracy_score(y, test_predict)
    print('The accuracy of the test is:', accuracy)
    if overall_acc[-1] != 0:  # 0 means this subblock has not been predicted before
        if accuracy <= overall_acc[-1]:  # Select the highest accuracy
            b = False
            print('b = False, not good')
    if b:
        x = transfer.inverse_transform(x)  # inverse-standardization
        # 1)probability
        print('The test predict Probability of each class:\n', test_predict_proba)
        proba1 = np.column_stack((y, test_predict, x, test_predict_proba))
        tot = pd.DataFrame(proba1)
        tot.columns = ['label_true', 'label_predict', 'x', 'y', 'z', 'predict_Strata1',
                       'predict_Strata2', 'predict_Strata3', 'predict_Strata4', 'predict_Strata5', 'predict_Strata6',
                       'predict_Strata7']
        np.savetxt(path1, proba1, delimiter=',', fmt='%s')  # 输出完整内容
        # 2)Visualization of scatter points
        merge(proba1, accuracy, id_model)
        samples1 = np.column_stack((x, test_predict))  # Combine the coordinates of the points with the values of the strata
        np.savetxt(path5, samples1, delimiter=',')
        # Evaluate model effect
        print('The accuracy of the test_all is:', metrics.accuracy_score(y, test_predict))
        print('The recall of the test_all is:', metrics.recall_score(y, test_predict, average='weighted'))
        print('The recall of the test_all for every single class is:',
              metrics.recall_score(y, test_predict, average=None))
        print('The precision of the test_all is:',
              metrics.precision_score(y, test_predict, average='weighted'))
        print('The precision of the test_all for every single class is:',
              metrics.precision_score(y, test_predict, average=None))
        print('The F1 of the test_all is:', metrics.f1_score(y, test_predict, average='weighted'))
        print('The f1_score of the test_all for every single class is:',
              metrics.f1_score(y, test_predict, average=None))
        # 3)Confusion matrix
        confusion_matrix_result = metrics.confusion_matrix(y, test_predict)
        print('The confusion matrix result:\n', confusion_matrix_result)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sns.heatmap(confusion_matrix_result, xticklabels=['1 ', '2', '3', '4', '5', '6', '7'],
                    yticklabels=['1 ', '2', '3', '4', '5', '6', '7'], fmt=".0f", annot=True, annot_kws={"size": 8},
                    linewidths=0.5, cmap='Blues', cbar_kws={"shrink": .8}, square=True)
        ax.set_title('confusion matrix')
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
    end = time.process_time()
    runTime = (end - start)
    print('Finish estimating, and it costs:', runTime, 's')


# Inserts a character at the specified position in the string
def str_insert(str_origin, pos, str_add):
    str_list = list(str_origin)
    str_list.insert(pos, str_add)
    str_out = ''.join(str_list)
    return str_out


# information entropy
def info_entopy(a, b):
    print("=====================================================")
    print('uncertainty analysis...')
    start = time.process_time()
    test_predict_proba = a[:, 5:12]
    test_predict_proba = torch.tensor(test_predict_proba, dtype=torch.float32).cuda()
    log_data_label = torch.log(test_predict_proba) * (-1)
    info_entropy = log_data_label.mul(test_predict_proba)
    info_entropy = np.nan_to_num(np.array(info_entropy.cpu()))
    cell_info_entropy = np.sum(info_entropy, axis=1)  # This is the information entropy at each point, definite 0ln0=0
    one_info_entropy = cell_info_entropy / math.log(OUTPUT_UNITS)  # standardization
    one_info_entropy, a = Duplicate_check(one_info_entropy, a)
    samples = np.column_stack((np.array(a), one_info_entropy))  # Combine the coordinates with the information entropy
    np.savetxt(test_path3, samples, delimiter=',', fmt='%.3f')
    np.savetxt(test_path4, b, delimiter=',', fmt='%.3f')
    end = time.process_time()
    runTime = (end - start)
    print('Finish analysing, and it costs:', runTime, 's')


# Duplicate checking
def Duplicate_check(entropy, a):
    i_positions = []
    for item1 in a:  # String combination
        i_position = str(item1[2]) + ',' + str(item1[3]) + ',' + str(item1[4])
        i_positions.append(i_position)
    for item2 in i_positions:
        if i_positions.count(item2) > 1:
            print(item2)
            dup_idx = [i for i, j in enumerate(i_positions) if j == item2]  #dup_idx is the corresponding number of repeated coordinates
            min_entropy = 1
            for id1 in reversed(dup_idx):
                if entropy[id1] < min_entropy:
                    min_entropy = entropy[id1]
                    min_id = id1
            dup_idx.remove(min_id)  # Retain the information entropy minimum, the most certain result
            for id2 in reversed(dup_idx):   # Duplicate coordinate points are deleted
                entropy = np.delete(entropy, id2, axis=0)
                a = np.delete(a, id2, axis=0)
                del i_positions[id2]
        dic = {}.fromkeys(i_positions)
        if len(dic) == len(i_positions):
            print('列表里的元素互不重复！可以跳出循环')
            break
    print('for完了', entropy.size)
    return entropy, a


# Mian
# Input the test and training sets
data_train = '../data/dataset_example/points_train_Boreholes149.csv'  # training set
data_general_test = '../data/dataset_example/general_testing_operation60+61.csv'  # test set
boreholes_position = '../data/dataset_example/delaunay2.txt'  # Triangulating the location
boreholes_position0 = '../data/dataset_example/delaunay5.txt'  # The location of the corner of the study area
# the path to export
general_test_path1 = "../data/EMDCT_result/proba/test_predict_proba.csv"  # Detailed probability
general_test_path2 = "../data/EMDCT_result/confusion_matrix/test_confusion matrix.jpg"  # Confusion matrix
general_test_path5 = "../data/EMDCT_result/points_results/points_results.txt"  # Test set formation result
test_path3 = "../data/EMDCT_result/test_entropy.csv"  # Test set borehole information entropy results storage
test_path4 = "../data/EMDCT_result/test_accuracy.csv"  # The accuracy of each subblock
test_path6 = "../data/EMDCT_result/delaunay_num.csv"  # The number of sampling points per subblock
test_path7 = "../data/EMDCT_result/delaunay_name.csv"  # The algorithm used for each subblock

STATE = 2
transfer = StandardScaler()
OUTPUT_UNITS = 7  # Number of Stratigraphic classification
overall_result = np.array([[1, 1, 40201.010, 	69471.318, 	508.8607595, 0.98, 0, 0, 0.01, 0.01, 0, 0],
                           [1, 1, 40201.010, 	69471.318, 	508.8607595, 0.98, 0, 0, 0.01, 0.01, 0, 0]])
overall_acc = []
overall_num = []
overall_name = []
train = pd.read_csv(data_train).values
test = pd.read_csv(data_general_test).values
positions = pd.read_csv(boreholes_position, header=None)
positions0 = pd.read_csv(boreholes_position0, header=None)
pos = pd.concat([positions, positions0], axis=0).values
tri = Delaunay(pos)  # The Delaunay triangle
plt.triplot(pos[:, 0], pos[:, 1], tri.simplices.copy())
plt.plot(pos[:, 0], pos[:, 1], 'o')
print(train.shape[0])
print(test.shape[0])

#  Loop through the triangular grid
for idx in range(tri.simplices.shape[0]):
    print("===========================================================================")
    print('sub-delaunay:({}/ {})'.format(idx, tri.simplices.shape[0]))
    pts = np.array(pos)  # Get the three vertices of the triangle
    triangle = pts[tri.simplices[idx]]  # The index of the points that make up the simplex.
    x_c, y_c, radius = circumcircle(triangle[0], triangle[1], triangle[2])
    train0 = [[40201.010, 	69471.318, 	508.8607595, 	1],
              [40201.010, 	69471.318, 	417.721519, 	2],
              [40201.010, 	69471.318, 	316.4556962, 	3],
              [40201.010, 	69471.318, 	215.1898734,    4],
              [40201.010, 	69471.318, 	103.7974684,    5],
              [40201.010, 	69471.318, 	-58.2278481, 	6],
              [40201.010, 	69471.318, 	-169.6202532, 	7]]
    test0 = [[40201.010, 	69471.318, 	508.8607595, 	1],
              [40201.010, 	69471.318, 	417.721519, 	2],
              [40201.010, 	69471.318, 	316.4556962, 	3],
              [40201.010, 	69471.318, 	215.1898734,    4],
              [40201.010, 	69471.318, 	103.7974684,    5],
              [40201.010, 	69471.318, 	-58.2278481, 	6],
              [40201.010, 	69471.318, 	-169.6202532, 	7]]
    for i in range(train.shape[0]):
        distance = (train[i, 0] - x_c) ** 2 + (train[i, 1] - y_c) ** 2
        if distance <= 1.5 * radius ** 2:  # An external circle is the training set
            train0.append([train[i, 0], train[i, 1], train[i, 2], train[i, 3]])
    for j in range(test.shape[0]):
        c_idx = int(tri.find_simplex([(test[j, 0], test[j, 1])]))
        if c_idx == idx:  # If the point currently traversed is inside the triangle, it is added to the set
            test0.append([test[j, 0], test[j, 1], test[j, 2], test[j, 3]])
    train1 = np.array(train0)
    test1 = np.array(test0)
    x_train = train1[:, :3]
    y_train = train1[:, 3]
    x_general_test = test1[:, :3]
    y_general_test = test1[:, 3]
    print('number of sub-train set: {}'.format(train1.shape[0]))
    print('number of sub-test set: {}'.format(test1.shape[0]))
    # standardization
    x_train = transfer.fit_transform(x_train)
    x_general_test = transfer.transform(x_general_test)
    overall_acc.append(0)
    overall_num.append(test1.shape[0])

    # Train model by model until you pick the highest one
    for i in range(7):
        if i == 0:  # SVM
            print('--------------------------------------------------')
            print('training SVM...')
            start = time.process_time()
            estimator = svm.SVC(kernel="rbf", gamma=0.33333, decision_function_shape='ovo', C=1, probability=True,
                                max_iter=4000, random_state=STATE)
            estimator.fit(x_train, y_train.ravel())
            print(estimator)
            end = time.process_time()
            runTime = (end - start)
            print('Finish Training, and it costs:', runTime, 's')
            index = str(idx)
            general_test_path1_d = str_insert(general_test_path1, -4, index)
            general_test_path2_d = str_insert(general_test_path2, -4, index)
            general_test_path5_d = str_insert(general_test_path5, -4, index)
            estimate_model(x_general_test, y_general_test,
                           general_test_path1_d,
                           general_test_path2_d,
                           general_test_path5_d, estimator, i)
        elif i == 1:  # catboost
            print('--------------------------------------------------')
            print('training CB...')
            start = time.process_time()
            estimator = cb.CatBoostClassifier(eval_metric='AUC', verbose=False, random_state=STATE, iterations=1000,
                                               learning_rate=0.03, depth=6, l2_leaf_reg=3)
            estimator.fit(x_train, y_train)
            print(estimator)
            end = time.process_time()
            runTime = (end - start)
            print('Finish Training, and it costs:', runTime, 's')
            index = str(idx)
            general_test_path1_d = str_insert(general_test_path1, -4, index)
            general_test_path2_d = str_insert(general_test_path2, -4, index)
            general_test_path5_d = str_insert(general_test_path5, -4, index)
            estimate_model(x_general_test, y_general_test,
                           general_test_path1_d,
                           general_test_path2_d,
                           general_test_path5_d, estimator, i)
        elif i == 2:  # RF
            print('--------------------------------------------------')
            print('training RF...')
            start = time.process_time()
            estimator = RandomForestClassifier(random_state=5, n_estimators=100, oob_score=True,
                                               criterion="gini", min_samples_split=2, min_samples_leaf=1)
            estimator.fit(x_train, y_train)
            print(estimator)
            end = time.process_time()
            runTime = (end - start)
            print('Finish Training, and it costs:', runTime, 's')
            index = str(idx)
            general_test_path1_d = str_insert(general_test_path1, -4, index)
            general_test_path2_d = str_insert(general_test_path2, -4, index)
            general_test_path5_d = str_insert(general_test_path5, -4, index)
            estimate_model(x_general_test, y_general_test,
                           general_test_path1_d,
                           general_test_path2_d,
                           general_test_path5_d, estimator, i)
        elif i == 3:  # decision tree
            print('--------------------------------------------------')
            print('training DT...')
            start = time.process_time()
            estimator = DecisionTreeClassifier(criterion="gini",  min_samples_split=2, min_samples_leaf=1,)
            estimator.fit(x_train, y_train)
            print(estimator)
            end = time.process_time()
            runTime = (end - start)
            print('Finish Training, and it costs:', runTime, 's')
            index = str(idx)
            general_test_path1_d = str_insert(general_test_path1, -4, index)
            general_test_path2_d = str_insert(general_test_path2, -4, index)
            general_test_path5_d = str_insert(general_test_path5, -4, index)
            estimate_model(x_general_test, y_general_test,
                           general_test_path1_d,
                           general_test_path2_d,
                           general_test_path5_d, estimator, i)
        elif i == 4:  # deep forest
            print('--------------------------------------------------')
            print('training DF...')
            start = time.process_time()
            estimator = CascadeForestClassifier(random_state=15, n_estimators=8, n_trees=150, max_layers=6,
                                                n_tolerant_rounds=2)
            estimator.fit(x_train, y_train)
            print(estimator)
            end = time.process_time()
            runTime = (end - start)
            print('Finish Training, and it costs:', runTime, 's')
            index = str(idx)
            general_test_path1_d = str_insert(general_test_path1, -4, index)
            general_test_path2_d = str_insert(general_test_path2, -4, index)
            general_test_path5_d = str_insert(general_test_path5, -4, index)
            estimate_model(x_general_test, y_general_test,
                           general_test_path1_d,
                           general_test_path2_d,
                           general_test_path5_d, estimator, i)
        elif i == 5:  # XGB
            print('--------------------------------------------------')
            print('training...')
            start = time.process_time()
            estimator = XGBClassifier(learning_rate=0.1, n_estimators=450, max_depth=5, min_child_weight=6,
                                gamma=0, subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
                                nthread=4, scale_pos_weight=1, seed=27)
            estimator.fit(x_train, y_train)
            print(estimator)
            end = time.process_time()
            runTime = (end - start)
            print('Finish Training, and it costs:', runTime, 's')
            index = str(idx)
            general_test_path1_d = str_insert(general_test_path1, -4, index)
            general_test_path2_d = str_insert(general_test_path2, -4, index)
            general_test_path5_d = str_insert(general_test_path5, -4, index)
            estimate_model(x_general_test, y_general_test,
                           general_test_path1_d,
                           general_test_path2_d,
                           general_test_path5_d, estimator, i)
        else:  # KNN
            print('--------------------------------------------------')
            print('training KNN...')
            start = time.process_time()
            estimator = KNeighborsClassifier(leaf_size=30, n_neighbors=5)
            param_dict = {"n_neighbors": [1, 2, 3, 4, 5]}
            estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
            estimator.fit(x_train, y_train)
            print(estimator)
            end = time.process_time()
            runTime = (end - start)
            print('Finish Training, and it costs:', runTime, 's')
            index = str(idx)
            general_test_path1_d = str_insert(general_test_path1, -4, index)
            general_test_path2_d = str_insert(general_test_path2, -4, index)
            general_test_path5_d = str_insert(general_test_path5, -4, index)
            estimate_model(x_general_test, y_general_test,
                           general_test_path1_d,
                           general_test_path2_d,
                           general_test_path5_d, estimator, i)

    print('A sub-delaunay({}/ {}) is finished, and the best accuracy is: {}'.format(idx, tri.simplices.shape[0], overall_acc[-1]))

info_entopy(overall_result, overall_acc)
np.savetxt(test_path6, overall_num, delimiter=',', fmt='%.0f')
np.savetxt(test_path7, overall_name, delimiter=',', fmt='%.0f')


