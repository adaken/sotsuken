#!/usr/bin/env python
# -*- coding: utf-8 -*-


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
    Xl = namedtuple('Xl', 'filename, label')
    xls =  (
         Xl(R(r'data\acc\pass_acc_128p_131data.xlsx'), 'pass',),
         Xl(R(r'data\acc\placekick_acc_128p_101data.xlsx'), 'pk'),
         Xl(R(r'data\acc\run_acc_128p_132data.xlsx'), 'run'),
         Xl(R(r'data\acc\tackle_acc_128p_111data.xlsx'), 'tackle')
        )
    tr_vecs = []
    tr_labels = []
    for xl in xls:
        tr_vec, tr_label = make_input(xlsx=xl.filename, sheetnames=None,col=None,
                                                min_row=2,fft_N=128, sample_cnt=100,
                                                label=xl.label,normalizing='01', log=False)
        map(tr_vecs.append, tr_vec)
        tr_labels += tr_label

    from app.util.inputmaker import random_input_iter
    tr_vecs_rand, tr_labels_rand = [], []
    for i, j in random_input_iter(tr_vecs, tr_labels):
        tr_vecs_rand.append(i)
        tr_labels_rand.append(j)

    """
    教師データの学習分類
    """
    clf = SVC(C=1000, kernel='rbf', gamma = 0.001)
    scores = cross_validation.cross_val_score(clf, tr_vecs_rand, tr_labels_rand, cv=5)

    print scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print scores.mean()
