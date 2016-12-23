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

if __name__ == '__main__':
    """
    データ生成
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
        #input_vecs.append(input_vec)
        #input_labels.append(labels)
        map(input_vecs.append, input_vec)
        input_labels += labels

    #print len(input_labels)
    #print len(input_vecs)

    #from sklearn import datasets
    #iris = datasets.load_iris()
    #data = iris.data[0:100][:,::2]
    #print iris.target[0:100]

    #label = np.r_[np.repeat(0,20), np.repeat(1,10)]

    k = 5 # 分割数

    # 分割数に応じたlabelsの分割(labels,分割数,シャッフル)
    skf = StratifiedKFold(input_labels, n_folds=k, shuffle=False)

    train_vecs = [[]]*k
    train_labels = [[]]*k
    test_vecs = [[]]*k
    test_labels = [[]]*k

    i = 0
    # skfよりtr(train),ts(test)listの取り出し
    for tr, ts in skf:
        #print("%s %s" % (tr, ts))
        #print tr
        #print ts

        # vecs[tr],labels[tr]をtrain用のlistにまとめる
        for j in tr:
            print j
            #for i in xrange(k):
            train_vecs[i].append(input_vecs[j])
            train_labels[i].append(input_labels[j])
            print len(train_vecs[i])
            print len(train_labels[i])
            if len(train_vecs[i]) == 240 :
                i = 1
        
        # vecs[ts],labels[ts]をtest用のlistにまとめる
        for j in ts:
            #print j
            #for i in xrange(k):
            test_vecs[i].append(input_vecs[j])
            test_labels[i].append(input_labels[j])
        #break
    print len(train_vecs[1])
    print len(test_vecs[1])
    
    print len(train_vecs[0])
    print len(test_vecs[0])

    print len(train_labels[0])
    print len(test_labels[0])

    """
    教師データの学習分類
    """
    est = svm.SVC(C=1, kernel='rbf', gamma=0.01)    # パラメータ (C-SVC, RBF カーネル, C=1)
    clf = OneVsRestClassifier(est)  #他クラス分類器One-against-restによる識別
    clf.fit(train_vecs[0], train_labels[0])
    test_pred = clf.predict(test_vecs[0])

    """
    clf2 = SVC(C=1, kernel='rbf', gamma=0.01)    # パラメータ (C-SVC, RBF カーネル, C=1)
    clf2.fit(input_vecs1, input_labels1)
    test_pred2 = clf2.predict(test_vecs1)  #他クラス分類器One-versus-oneによる識別


    学習モデルのローカル保存

    joblib.dump(clf, R('misc\model\clf.pkl'))
    joblib.dump(clf2, R('misc\model\clf2.pkl'))
    """

    #confusion matrix（ラベルの分類表。分類性能が高いほど対角線に値が集まる）
    print confusion_matrix(test_labels[0], test_pred)
    #print confusion_matrix(test_labels1, test_pred2)

    #分類結果 適合率 再現率 F値の表示
    print classification_report(test_labels[0], test_pred)
    #print classification_report(test_labels1, test_pred2)

    #正答率 分類ラベル/正解ラベル
    print accuracy_score(test_labels[0], test_pred)
    #print accuracy_score(test_labels1, test_pred2)

    print test_labels[0]       #分類前ラベル
    print list(test_pred)   #One-against-restによる識別ラベル
    #print list(test_pred2)  #One-versus-oneによる識別ラベル

    """
    target_names = ['class 0', 'class 1', 'class 2']
    print(classification_report(test_labels, test_pred, target_names=target_names))
    """