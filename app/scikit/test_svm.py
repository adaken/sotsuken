#!/usr/bin/env python
# -*- coding: utf-8 -*-


from app.util.inputmaker import make_input
from collections import namedtuple
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import numpy as np
from app import R, T, L

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
    input_vecs = []
    input_labels = []
    for xl in xls:
        input_vec, labels = make_input(xlsx=xl.filename, sheetnames=None,col=None,
                                                min_row=2,fft_N=128, sample_cnt=80,
                                                label=xl.label,normalizing='01', log=False)
        map(input_vecs.append, input_vec)
        input_labels += labels

    from app.util.inputmaker import random_input_iter
    input_vecs1, input_labels1 = [], []
    for i, j in random_input_iter(input_vecs, input_labels):
        input_vecs1.append(i)
        input_labels1.append(j)

    """
    tmp = np.c_[input_vec, labels]
    random.shuffle(tmp)
    input_vec = tmp[:, :-1]
    labels  = tmp[:, -1]
    #labels = [vec[0] for vec in input_data]
    #vecs = [list(vec[1]) for vec in input_data]
    """
    """
    テストデータ生成
    """
    test_vecs = []
    test_labels = []
    for xl in xls:
        test_vec, test_label = make_input(xlsx=xl.filename, sheetnames=None,col=None,
                                                min_row=128*80+1,fft_N=128, sample_cnt=20,
                                                label=xl.label,normalizing='01', log=False)
        map(test_vecs.append, test_vec)
        test_labels += test_label

    test_vecs1, test_labels1 = [], []
    for i, j in random_input_iter(test_vecs, test_labels):
        test_vecs1.append(i)
        test_labels1.append(j)

    print "input_vec_len    :", len(input_vecs)
    #print "input_vec_shape  :", input_vecs.shape
    print "labels_len       :", len(input_labels)
    print "test_vec_len     :", len(test_vecs)
    #print "test_vec_shape   :", test_vecs.shape
    print "test_labels      :", len(test_labels)

    """
    tmpt = np.c_[test_vec, test_labels]
    random.shuffle(tmpt)
    test_vec = tmpt[:, :-1]
    test_labels  = tmpt[:, -1]
    #test_labels = [vec[0] for vec in test_data]
    #test_vecs = [list(vec[1]) for vec in test_data]
    """
    """
    教師データの学習分類
    """
    # test_gridsearchを参照
    est = SVC(C=1000, kernel='rbf', gamma = 0.001)    # パラメータ (C-SVC, RBF カーネル, C=1000)
    clf = OneVsRestClassifier(est)  #他クラス分類器One-against-restによる識別
    clf.fit(input_vecs1, input_labels1)
    test_pred = clf.predict(test_vecs1)

    clf2 = SVC(C=1000, kernel='rbf', gamma = 0.001)    # パラメータ (C-SVC, RBF カーネル, C=1000)
    clf2.fit(input_vecs1, input_labels1)
    test_pred2 = clf2.predict(test_vecs1)  #他クラス分類器One-versus-oneによる識別

    """
    学習モデルのローカル保存
    """
    #joblib.dump(clf, R('misc\model\clf.pkl'))
    #joblib.dump(clf2, R('misc\model\clf2.pkl'))

    #confusion matrix（ラベルの分類表。分類性能が高いほど対角線に値が集まる）
    print confusion_matrix(test_labels1, test_pred)
    print confusion_matrix(test_labels1, test_pred2)

    #分類結果 適合率 再現率 F値の表示
    print classification_report(test_labels1, test_pred)
    print classification_report(test_labels1, test_pred2)

    #正答率 分類ラベル/正解ラベル
    print accuracy_score(test_labels1, test_pred)
    print accuracy_score(test_labels1, test_pred2)

    print test_labels1       #分類前ラベル
    print list(test_pred)   #One-against-restによる識別ラベル
    print list(test_pred2)  #One-versus-oneによる識別ラベル

    """
    target_names = ['class 0', 'class 1', 'class 2']
    print(classification_report(test_labels, test_pred, target_names=target_names))
    """