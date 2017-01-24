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
from sklearn.cross_validation import StratifiedKFold
from app.util import ConfusionMatrix

def kfold(labels, features, k=5):
    skf = StratifiedKFold(labels, n_folds=k, shuffle=False)

    for tr, ts in skf: # k回ループ
        tr_labels, ts_labels = [], []
        ret_tr, ret_ts = [], []
        for i in tr:
            tr_labels.append(labels[i])
            ret_tr.append(features[i])

        for j in ts:
            ts_labels.append(labels[j])
            ret_ts.append(features[j])
        yield tr_labels, ret_tr, ts_labels, ret_ts

def train(tr,tr_labels,ts, ts_labels, confm):
    clf = SVC(C=100, kernel='linear')    # パラメータ (C-SVC, RBF カーネル, C=1000)
    #clf = OneVsRestClassifier(est)  #多クラス分類器One-against-restによる識別
    clf.fit(tr, tr_labels)
    ts_pred = clf.predict(ts)
    l = confusion_matrix(ts_labels, ts_pred)

    confm(l)
    print confm.precision
    print confm.recall
    print confm.fmeasure


if __name__ == '__main__':
    """
    データ生成
    """
    Xl = namedtuple('Xl', 'filename, label')
    xls =  (
         Xl(R(r'data\acc\pass_acc_128p_131data.xlsx'), 'pass',),
         Xl(R(r'data\acc\placekick_acc_128p_101data.xlsx'), 'pkick'),
         Xl(R(r'data\acc\run_acc_128p_132data.xlsx'), 'run'),
         Xl(R(r'data\acc\tackle_acc_128p_111data.xlsx'), 'tackle'),
         Xl(R(r'data/raw/invectest/walk.xlsx'), 'walk')
        )
    input_vecs = []
    input_labels = []
    for xl in xls:
        input_vec, labels = make_input(xlsx=xl.filename, sheetnames=None,col=None,
                                                min_row=2,fft_N=128, sample_cnt=100,
                                                label=xl.label,normalizing='01', log=False)
        #input_vecs.append(input_vec)
        #input_labels.append(labels)
        map(input_vecs.append, input_vec)
        input_labels += labels

    k = 5 # 分割数
    class_n = len(xls) # クラス数：自動取得


    """
    # 分割数に応じたlabelsの分割(labels,分割数,シャッフル)
    skf = StratifiedKFold(input_labels, n_folds=k, shuffle=False)

    train_vecs = [[]]*k
    train_labels = [[]]*k
    test_vecs = [[]]*k
    test_labels = [[]]*k

    i = 0
    # skfよりtr(train),ts(test)listの取り出し
    for tr, ts in skf:

        # vecs[tr],labels[tr]をtrain用のlistにまとめる
        for j in tr:
            train_vecs[i].append(input_vecs[j])
            train_labels[i].append(input_labels[j])
            if len(train_vecs[i]) == 240 :
                i += 1
                break

        # vecs[ts],labels[ts]をtest用のlistにまとめる
        for j in ts:
            test_vecs[i].append(input_vecs[j])
            test_labels[i].append(input_labels[j])
    """

    # 分割数に応じたlabels,vecsの分割(labels,vecs,分割数)
    it = kfold(input_labels, input_vecs, k=k)

    # ConfusionMatrixの表示
    confm = ConfusionMatrix(class_n)

    # 分割検定の実施
    for tr_labels, tr, ts_labels, ts in it:
        train(tr, tr_labels, ts,ts_labels, confm)

    confm = ConfusionMatrix(class_n)(confm.sum_matrix / float(k))
    print "適合率", confm.precision.mean()
    print "再現率", confm.recall.mean()
    print "F値   ", confm.fmeasure.mean()