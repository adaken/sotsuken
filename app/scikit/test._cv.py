#!/usr/bin/env python
# -*- coding: utf-8 -*-


from sklearn import svm
from app.util.inputmaker import make_input
import random
from collections import namedtuple
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import numpy as np
from app import R, T, L
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
if __name__ == '__main__':
    """
    教師データ生成
    """
    Xl = namedtuple('Xl', 'filename, sheet, letter, label, sampling, overlap')
    xls =  (
         Xl(R(r'data\raw\run_1122_data.xlsx'), ['Sheet1'], 'F', 'run', 'std', 0),
         Xl(R(r'data\raw\walk_1122_data.xlsx'), ['Sheet1'], 'F', 'walk', 'std', 0),
         Xl(R(r'data\raw\jump_128p_174data_fixed.xlsx'), ['Sheet'], 'A', 'jump', 'std', 0),
        )
    input_vecs = []
    input_labels = []
    for xl in xls:
        input_vec, labels = make_input(xlsx=xl.filename, sheetnames=xl.sheet,col=xl.letter,
                                                min_row=2,fft_N=128, sample_cnt=100,
                                                label=xl.label,sampling=xl.sampling,
                                                overlap=xl.overlap,normalizing='01', log=False)
        map(input_vecs.append, input_vec)
        input_labels += labels

    from app.util.inputmaker import random_input_iter
    input_vecs1, input_labels1 = [], []
    for i, j in random_input_iter(input_vecs, input_labels):
        input_vecs1.append(i)
        input_labels1.append(j)

    """
    教師データの学習分類
    """
    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_validation.cross_val_score(clf, input_vecs1, input_labels1, cv=5)

    print scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print scores.mean()
