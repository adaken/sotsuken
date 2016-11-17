# coding: utf-8

from libsvm.python.svm import *
from libsvm.python.svmutil import *
import numpy as np
from util.excelwrapper import ExcelWrapper
from util.fft import fft
from util.util import make_input_from_xlsx
import random
from collections import namedtuple

if __name__ == '__main__':
    """
    教師データ生成
    """
    Xl = namedtuple('Xl', 'filename, sheet, letter, label, sampling, overlap')
    xls =  (
         Xl(r'E:\work\data\new_run.xlsx', 'Sheet4', 'F', 1, 'rand', 0),
         Xl(r'E:\work\data\walk.xlsx', 'Sheet4', 'F', 2, 'rand', 0),
         Xl(r'E:\work\data\jump_128p_84data_fixed.xlsx', 'Sheet', 'A', 3, 'std', 0),
         Xl(r'E:\work\data\skip.xlsx', 'Sheet4', 'F', 4, 'rand', 0)
        )
    input_data = []
    for xl in xls:
        input_vec = make_input_from_xlsx(filename=xl.filename, sheetname=xl.sheet,
                                               col=xl.letter, read_range=(2, None), overlap=xl.overlap,
                                               sampling=xl.sampling, sample_cnt=50, fft_N=128,
                                               normalizing='01', label=xl.label, log=False)
        input_data += input_vec

    random.shuffle(input_data)

    labels = [vec[0] for vec in input_data]
    vecs = [list(vec[1]) for vec in input_data]

    print "input finish"
    print labels, vecs

    """
    テストデータ生成
    """
    test_data = []
    for xl in xls:
        test_vec = make_input_from_xlsx(filename=xl.filename, sheetname=xl.sheet,
                                               col=xl.letter, read_range=(2, None), overlap=xl.overlap,
                                               sampling=xl.sampling, sample_cnt=10, fft_N=128,
                                               normalizing='01', label=xl.label, log=False)
        test_data += test_vec

    random.shuffle(test_data)

    test_labels = [vec[0] for vec in test_data]
    test_vecs = [list(vec[1]) for vec in test_data]

    """
    学習の実行
    """
    prob = svm_problem(labels, vecs)    # 教師データ (XOR)
    param = svm_parameter('-s 0 -t 2 -c 1')    # パラメータ (C-SVC, RBF カーネル, C=1)
    machine = svm_train(prob, param)    # 学習

    svm_save_model('libsvm.model', machine)

    p_labels, p_acc, p_vals = svm_predict(test_labels,test_vecs,machine)    # テストデータ

    p_labels = [int(i) for i in p_labels]
    print p_labels    # 識別結果を表示

    print test_labels

