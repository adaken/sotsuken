# coding: utf-8
from libsvm.python.svm import *
from libsvm.python.svmutil import *

def get_input():


if __name__ == '__main__':
    prob = svm_problem([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1]*200,
                       [[1], [2], [3], [4], [5], [6], [11], [8], [9], [10]]*200)    # 教師データ (XOR)
    param = svm_parameter('-s 0 -t 2 -c 1')    # パラメータ (C-SVC, RBF カーネル, C=1)
    machine = svm_train(prob, param)    # 学習

    p_labels, p_acc, p_vals = svm_predict([-1, -1, 1, 1, -1],
                                           [[1], [3], [2], [4], [7]],
                                           machine)    # 識別対象
    print(p_labels)    # 識別結果を表示