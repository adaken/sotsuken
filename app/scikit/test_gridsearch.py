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
from sklearn.grid_search import GridSearchCV
import numpy as np
from app import R, T, L

if __name__ == '__main__':
    # 調整パラメータ定義
    tuned_parameters = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},
    {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [0.001, 0.0001]},
    {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}
    ]

    """
    教師データ生成
    """
    Xl = namedtuple('Xl', 'filename, label')
    xls =  (
         Xl(R(r'data\acc\pass_acc_128p_131data.xlsx'), 'pass',),
         Xl(R(r'data\acc\placekick_acc_128p_101data.xlsx'), 'pk'),
         Xl(R(r'data\acc\run_acc_128p_132data.xlsx'), 'run'),
         Xl(R(r'data\acc\tackle_acc_128p_111data.xlsx'), 'tackle'),
         Xl(R(r'data/raw/invectest/walk.xlsx'), 'walk')
        )
    tr_vecs = []
    tr_labels = []
    for xl in xls:
        tr_vec, tr_label = make_input(xlsx=xl.filename, sheetnames=None,col=None,
                                                min_row=2,fft_N=128, sample_cnt=80,
                                                label=xl.label,normalizing='01', log=False)
        map(tr_vecs.append, tr_vec)
        tr_labels += tr_label

    from app.util.inputmaker import random_input_iter
    tr_vecs_rand, tr_labels_rand = [], []
    for i, j in random_input_iter(tr_vecs, tr_labels):
        tr_vecs_rand.append(i)
        tr_labels_rand.append(j)

    """
    テストデータ生成
    """
    ts_vecs = []
    ts_labels = []
    for xl in xls:
        ts_vec, ts_label = make_input(xlsx=xl.filename, sheetnames=None,col=None,
                                                min_row=128*80+1,fft_N=128, sample_cnt=20,
                                                label=xl.label,normalizing='01', log=False)
        map(ts_vecs.append, ts_vec)
        ts_labels += ts_label

    ts_vecs_rand, ts_labels_rand = [], []
    for i, j in random_input_iter(ts_vecs, ts_labels):
        ts_vecs_rand.append(i)
        ts_labels_rand.append(j)


    score = 'f1'
    clf = GridSearchCV(
        SVC(), # 識別器
        tuned_parameters, # 最適化したいパラメータセット
        cv=5, # 交差検定の回数
        scoring='%s_weighted' % score ) # モデルの評価関数の指定

    clf.fit(tr_vecs_rand, tr_labels_rand)
    print("# Tuning hyper-parameters for %s" % score)
    print
    print("Best parameters set found on development set: %s" % clf.best_params_)
    print

    # それぞれのパラメータでの試行結果の表示
    print("Grid scores on development set:")
    print
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
    print

    # テストデータセットでの分類精度を表示
    print("The scores are computed on the full evaluation set.")
    print
    labels_true, labels_pred = ts_labels_rand, clf.predict(ts_vecs_rand)
    print(classification_report(labels_true, labels_pred))

    """
    実行結果
    0.972 (+/-0.031) for {'kernel': 'linear', 'C': 1000}
    0.962 (+/-0.055) for {'kernel': 'rbf', 'C': 1000, 'gamma': 0.001}
    グリッドサーチにより、パラメータを調整
    カーネルはlinearよりrbfの方が低いが、svmの実行結果では高いスコアを出したため、こちらを採用。


    """