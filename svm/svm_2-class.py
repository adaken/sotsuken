# coding: utf-8

from libsvm.python.svm import *
from libsvm.python.svmutil import *
import numpy as np
from util.excelwrapper import ExcelWrapper
from util.fft import fft
from util.util import make_input_from_xlsx


if __name__ == '__main__':
    """
    教師データ生成
    """
    xls = r"E:\work\data\new_run.xlsx"
    input_vec = make_input_from_xlsx(filename=xls, sheetname='Sheet4', col='F', read_range=(2, None),
                                     sampling='rand', sample_cnt=50, overlap=0,
                                     fft_N=128, normalizing='01', label=1, log=False)
    #print >> file(r'D:\home\desk\log.txt', 'w'), input_vec
    #label = [1]*len(input_vec)
    labels = [vec[0] for vec in input_vec]
    vecs = [list(vec[1]) for vec in input_vec]


    xls = r"E:\work\data\walk.xlsx"
    walk_vec = make_input_from_xlsx(filename=xls, sheetname='Sheet4', col='F', read_range=(2, None),
                                     sampling='rand', sample_cnt=50, overlap=0,
                                     fft_N=128, normalizing='01', label=2, log=False)
    walk_labels = [vec[0] for vec in walk_vec]
    walk_vecs = [list(vec[1]) for vec in walk_vec]

    xls = r"E:\work\data\jump_128p_84data_fixed.xlsx"
    jamp_vec = make_input_from_xlsx(filename=xls, sheetname='Sheet', col='A', read_range=(2, None),
                                     sampling='std', sample_cnt=50, overlap=0,
                                     fft_N=128, normalizing='01', label=3, log=False)
    jamp_labels = [vec[0] for vec in jamp_vec]
    jamp_vecs = [list(vec[1]) for vec in jamp_vec]

    labels = labels + walk_labels + jamp_labels #labelsの結合
    vecs = vecs + walk_vecs + jamp_vecs #vecsの結合

    print "input finish"
    print labels, vecs

    """
    テストデータ生成
    """
    xls = r"E:\work\data\new_run.xlsx"
    run_vec = make_input_from_xlsx(filename=xls, sheetname='Sheet4', col='F', read_range=(2, None),
                                     sampling='rand', sample_cnt=10, overlap=0,
                                     fft_N=128, normalizing='01', label=1, log=False)
    run_labels = [vec[0] for vec in run_vec]
    run_vecs = [list(vec[1]) for vec in run_vec]


    xls = r"E:\work\data\walk.xlsx"
    test_walk_vec = make_input_from_xlsx(filename=xls, sheetname='Sheet4', col='F', read_range=(2, None),
                                     sampling='rand', sample_cnt=10, overlap=0,
                                     fft_N=128, normalizing='01', label=2, log=False)
    test_walk_labels = [vec[0] for vec in test_walk_vec]
    test_walk_vecs = [list(vec[1]) for vec in test_walk_vec]

    xls = r"E:\work\data\jump_128p_84data_fixed.xlsx"
    test_vec = make_input_from_xlsx(filename=xls, sheetname='Sheet', col='A', read_range=(2, None),
                                     sampling='std', sample_cnt=10, overlap=0,
                                     fft_N=128, normalizing='01', label=3, log=False)
    test_labels = [vec[0] for vec in test_vec]
    test_vecs = [list(vec[1]) for vec in test_vec]

    test_labels = test_labels + test_walk_labels + run_labels #labelsの結合
    test_vecs = test_vecs + test_walk_vecs + run_vecs #vecsの結合

    """
    学習の実行
    """
    prob = svm_problem(labels, vecs)    # 教師データ (XOR)
    param = svm_parameter('-s 0 -t 2 -c 1')    # パラメータ (C-SVC, RBF カーネル, C=1)
    machine = svm_train(prob, param)    # 学習

    svm_save_model('libsvm.model', machine)

    p_labels, p_acc, p_vals = svm_predict(test_labels,test_vecs,machine)    # テストデータ
    print(p_labels)    # 識別結果を表示
