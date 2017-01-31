#!/usr/bin/env python
# -*- coding: utf-8 -*-
from app.util.inputmaker import make_input
import random
from collections import namedtuple
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from app import R, T, L


if __name__ == '__main__':
    """
    テストデータ生成
    """
    Xl = namedtuple('Xl', 'filename, label')
    xls =  (
         Xl(R(r'data\acc\pass_acc_128p_131data.xlsx'), 'pass',),
         Xl(R(r'data\acc\placekick_acc_128p_101data.xlsx'), 'pk'),
         Xl(R(r'data\acc\run_acc_128p_132data.xlsx'), 'run'),
         Xl(R(r'data\acc\tackle_acc_128p_111data.xlsx'), 'tackle')
        )
    ts_vecs = []
    ts_labels = []
    for xl in xls:
        ts_vec, ts_label = make_input(xlsx=xl.filename, sheetnames=None,col=None,
                                                min_row=128*80+1,fft_N=128, sample_cnt=20,
                                                label=xl.label,normalizing='01', log=False)
        map(ts_vecs.append, ts_vec)
        ts_labels += ts_label

    ts_vecs_rand, ts_labels_rand = [], []
    from app.util.inputmaker import random_input_iter
    for i, j in random_input_iter(ts_vecs, ts_labels):
        ts_vecs_rand.append(i)
        ts_labels_rand.append(j)


    clf = joblib.load(R('misc\model\clf.pkl'))
    pred = clf.predict(ts_vecs_rand)  #多クラス分類器One-versus-oneによる識別

    #confusion matrix（ラベルの分類表。分類性能が高いほど対角線に値が集まる）
    print confusion_matrix(ts_labels_rand, pred)

    #分類結果 適合率 再現率 F値の表示
    print classification_report(ts_labels_rand, pred)

    #正答率 分類ラベル/正解ラベル
    print accuracy_score(ts_labels_rand, pred)

    print ts_labels_rand       #分類前ラベル
    print list(pred)   #One-against-restによる識別ラベル