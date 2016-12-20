#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import datasets
from sklearn import cross_validation
from sklearn import svm
from util.excelwrapper import ExcelWrapper
from util.fft import fft
from util.util import make_input_from_xlsx
import random
from collections import namedtuple
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_digits
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

if __name__ == '__main__':
    """
    データ生成
    """
    Xl = namedtuple('Xl', 'filename, sheet, letter, label, sampling, overlap')
    xls =  (
         Xl(r'E:\work\data\run_1122_data.xlsx', 'Sheet1', 'F', 1, 'std', 0),
         Xl(r'E:\work\data\walk_1122_data.xlsx', 'Sheet1', 'F', 2, 'std', 0),
         Xl(r'E:\work\data\jump_128p_174data_fixed.xlsx', 'Sheet', 'A', 3, 'std', 0),
         Xl(r'E:\work\data\skip.xlsx', 'Sheet4', 'F', 4, 'rand', 0)
        )
    input_data = []
    for xl in xls:
        input_vec = make_input_from_xlsx(filename=xl.filename, sheetname=xl.sheet,
                                               col=xl.letter, read_range=(2, None), overlap=xl.overlap,
                                               sampling=xl.sampling, sample_cnt=120, fft_N=128,
                                               normalizing='01', label=xl.label, log=False)
        input_data += input_vec

    random.shuffle(input_data)

    labels = [vec[0] for vec in input_data]
    vecs = [list(vec[1]) for vec in input_data]

    """
    教師データの学習分類
    """
    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_validation.cross_val_score(clf, vecs, labels, cv=5)

    print scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
