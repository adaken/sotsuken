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
    Xl = namedtuple('Xl', 'filename, sheet, letter, label, sampling, overlap')
    xls =  (
         Xl(R(r'data\raw\run_1122_data.xlsx'), ['Sheet1'], 'F', 'run', 'std', 0),
         Xl(R(r'data\raw\walk_1122_data.xlsx'), ['Sheet1'], 'F', 'walk', 'std', 0),
         Xl(R(r'data\raw\jump_128p_174data_fixed.xlsx'), ['Sheet'], 'A', 'jump', 'std', 0),
        #Xl(R(r'data\skip.xlsx'), 'Sheet4', 'F', 4, 'rand', 0)
        )
    test_vecs = []
    test_labels = []
    for xl in xls:
        test_vec, test_label = make_input(xlsx=xl.filename, sheetnames=xl.sheet,col=xl.letter,
                                                min_row=12802,fft_N=128, sample_cnt=21,
                                                label=xl.label,sampling=xl.sampling,
                                                overlap=xl.overlap,normalizing='01', log=False)
        map(test_vecs.append, test_vec)
        test_labels += test_label

    from app.util.inputmaker import random_input_iter
    test_vecs1, test_labels1 = [], []
    for i, j in random_input_iter(test_vecs, test_labels):
        test_vecs1.append(i)
        test_labels1.append(j)

    clf = joblib.load(R('misc\model\clf.pkl'))
    test_pred = clf.predict(test_vecs1)  #他クラス分類器One-versus-oneによる識別

    #confusion matrix（ラベルの分類表。分類性能が高いほど対角線に値が集まる）
    print confusion_matrix(test_labels1, test_pred)

    #分類結果 適合率 再現率 F値の表示
    print classification_report(test_labels1, test_pred)

    #正答率 分類ラベル/正解ラベル
    print accuracy_score(test_labels1, test_pred)

    print test_labels1       #分類前ラベル
    print list(test_pred)   #One-against-restによる識別ラベル