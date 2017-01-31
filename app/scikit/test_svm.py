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
from app.util.normalize import scale_zero_one

if __name__ == '__main__':
    """
    教師データ生成
    """
    Xl = namedtuple('Xl', 'filename, label')
    xls =  (
         Xl(R(r'data\acc\pass_acc_128p_131data.xlsx'), 'pass',),
         Xl(R(r'data\acc\placekick_acc_128p_101data.xlsx'), 'pkick'),
         Xl(R(r'data\acc\run_acc_128p_132data.xlsx'), 'run'),
         Xl(R(r'data\acc\tackle_acc_128p_111data.xlsx'), 'tackle'),
         Xl(R(r'data/raw/invectest/walk.xlsx'), 'walk')
        )
    N = 64
    tr_vecs = []
    tr_labels = []
    for xl in xls:
        print "read", xl.label
        tr_vec, tr_label = make_input(xlsx=xl.filename, sheetnames=None,col=None,
                                      min_row=2,fft_N=N, sample_cnt=80,
                                      label=xl.label,normalizing=None, log=False, read_N=N)
        map(tr_vecs.append, tr_vec)
        tr_labels += tr_label

    tr_vecs = scale_zero_one(np.array(tr_vecs))

    from app.util.inputmaker import random_input_iter
    tr_vecs_rand, tr_labels_rand = [], []
    for i, j in random_input_iter(tr_vecs, tr_labels):
        tr_vecs_rand.append(i)
        tr_labels_rand.append(j)

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
    ts_vecs = []
    ts_labels = []
    for xl in xls:
        ts_vec, ts_label = make_input(xlsx=xl.filename, sheetnames=None,col=None,
                                                min_row=128*80+1,fft_N=N, sample_cnt=20,
                                                label=xl.label,normalizing=None, log=False,read_N=N)
        map(ts_vecs.append, ts_vec)
        ts_labels += ts_label

    ts_vecs = scale_zero_one(np.array(ts_vecs))

    ts_vecs_rand, ts_labels_rand = [], []
    for i, j in random_input_iter(ts_vecs, ts_labels):
        ts_vecs_rand.append(i)
        ts_labels_rand.append(j)

    print "tr_vecs_len     :", len(tr_vecs)
    #print "tr_vecs_shape  :", tr_vecs.shape
    print "tr_labels_len   :", len(tr_labels)
    print "ts_vecs_len     :", len(ts_vecs)
    #print "ts_vecs_shape  :", ts_vecs.shape
    print "ts_labels       :", len(ts_labels)

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
    est = SVC(C=1000, kernel='linear',gamma=0.001)    # パラメータ (C-SVC, RBF カーネル, C=1000)
    clf = OneVsRestClassifier(est)  #多クラス分類器One-against-restによる識別
    clf.fit(tr_vecs_rand, tr_labels_rand)
    pred = clf.predict(ts_vecs_rand)

    clf2 = SVC(C=1000, kernel='linear',gamma=0.001)    # パラメータ (C-SVC, RBF カーネル, C=1000)
    clf2.fit(tr_vecs_rand, tr_labels_rand)
    pred2 = clf2.predict(ts_vecs_rand)  #多クラス分類器One-versus-oneによる識別

    """
    学習モデルのローカル保存
    """
    joblib.dump(clf, R('misc/model/rbf_1k_1k-_A_{}p.pkl'.format(N)))
    joblib.dump(clf2, R('misc/model/rbf_1k_1k-_VS_{}p.pkl'.format(N)))

    #One-against-oneの結果
    #confusion matrix（ラベルの分類表。分類性能が高いほど対角線に値が集まる）
    print confusion_matrix(ts_labels_rand, pred)
    #分類結果 適合率 再現率 F値の表示
    print classification_report(ts_labels_rand, pred)
    #正答率 分類ラベル/正解ラベル
    print accuracy_score(ts_labels_rand, pred)
    print   #改行

    #One-versus-oneの結果
    print confusion_matrix(ts_labels_rand, pred2)
    print classification_report(ts_labels_rand, pred2)
    print accuracy_score(ts_labels_rand, pred2)

    print "分類前:", ["{:^6}".format(p) for p in ts_labels_rand] #分類前ラベル
    print "OAR   :", ["{:^6}".format(p) for p in pred]   #One-against-restによる識別ラベル
    print "OVO   :", ["{:^6}".format(p) for p in pred2]  #One-versus-oneによる識別ラベル

    """
    target_names = ['class 0', 'class 1', 'class 2']
    print(classification_report(ts_labels, pred, target_names=target_names))
    """