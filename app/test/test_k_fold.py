# coding: utf-8

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
from sklearn.cross_validation import StratifiedKFold

def kfold(labels, features, k=5):
    skf = StratifiedKFold(labels, n_folds=k, shuffle=False)

    i = 0
    for tr, ts in skf: # k回ループ
        ret_tr, ret_ts = [], []
        for i in tr:
            ret_tr.append(features[i])

        for j in ts:
            ret_ts.append(features[j])
        yield ret_tr, ret_ts

if __name__ == '__main__':
    """データ生成"""
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

    splited = list(kfold(input_labels, input_vecs, k=5))
    splited = map(list, zip(*splited))
    tr, ts = splited
    #print "tr_len:", len(tr[0])
    #print "tr_value:", tr[0]
    #print "ts_len:", len(ts[0])
    #print "ts_value:", ts[0]
    
    L.mkdir('kfold')
    print >> file(L('kfold/tr.txt'), 'w'), tr
    print >> file(L('kfold/ts.txt'), 'w'), ts
    for i, t in enumerate(tr):
        print "tr_len", len(t)
    else:
        print "合計ループ回数:", i+1

    print "\n"

    for i, t in enumerate(ts):
        print "ts_len", len(t)
    else:
        print "合計ループ回数:", i+1

