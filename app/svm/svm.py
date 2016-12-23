# coding: utf-8
from libsvm.python.svm import *
from libsvm.python.svmutil import *

def svm_test1():
    label, data = [1, -1], [[0, 1, -1], [0, 0, 1]]
    prob = svm_problem(label, data)
    param = svm_parameter('-s 0 -t 2')
    m = svm_train(prob, param)
    test_label, test_data = [1, 1, -1, -1], [[]]

def main():
    # データを読み込む
    def read_data():
        pass
    # スケーリング

    #a

if __name__ == '__main__':
    prob = svm_problem([-1,1,1,-1], [[0,0], [0,1],[1,0],[1,1]])    # 教師データ (XOR)
    param = svm_parameter('-s 0 -t 2 -c 1')    # パラメータ (C-SVC, RBF カーネル, C=1)
    machine = svm_train(prob, param)    # 学習

    p_labels, p_acc, p_vals = svm_predict([-1,-1,1,1,1],
                                           [[0.2,0.3],[0.8,0.7],[0.3,0.7],[0.8,0.2],[0.51,0.49]],
                                           machine)    # 識別対象
    print(p_labels)    # 識別結果を表示

    prob = svm_problem([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1]*200,
                       [[1], [2], [3], [4], [5], [6], [11], [8], [9], [10]]*200)    # 教師データ (XOR)
    param = svm_parameter('-s 0 -t 2 -c 1')    # パラメータ (C-SVC, RBF カーネル, C=1)
    machine = svm_train(prob, param)    # 学習

    p_labels, p_acc, p_vals = svm_predict([-1, -1, 1, 1, -1],
                                           [[1], [3], [2], [4], [7]],
                                           machine)    # 識別対象
    print(p_labels)    # 識別結果を表示