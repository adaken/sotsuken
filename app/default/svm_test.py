# coding: utf-8

if __name__ == '__main__':
    # -*- coding: utf-8 -*-
    from sklearn.svm import LinearSVC
    import numpy as np
    import scipy as sp

    # 学習データ
    data_training_tmp = np.loadtxt(r'C:\Users\locked\desktop\CodeIQ_auth.txt', delimiter=' ')
    data_training = [[x[0], x[1]] for x in data_training_tmp]
    label_training = [int(x[2]) for x in data_training_tmp]

    # 試験データ
    data_test = np.loadtxt(r'C:\Users\locked\desktop\CodeIQ_mycoins.txt', delimiter=' ')

    # 学習
    estimator = LinearSVC(C=1.0)
    estimator.fit(data_training, label_training)

    # 予測するよー
    label_prediction = estimator.predict(data_test)

    print(label_prediction)