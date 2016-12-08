#!/usr/bin/env python
# -*- coding: utf-8 -*-


from sklearn import svm
from util.util import make_input_from_xlsx
import random
from collections import namedtuple
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.externals import joblib

if __name__ == '__main__':
    """
    教師データ生成
    """
    Xl = namedtuple('Xl', 'filename, sheet, letter, label, sampling, overlap')
    xls =  (
         Xl(r'E:\work\data\run_1122_data.xlsx', 'Sheet1', 'F', 'run', 'std', 0),
         Xl(r'E:\work\data\walk_1122_data.xlsx', 'Sheet1', 'F', 'walk', 'std', 0),
         #Xl(r'E:\work\data\jump_128p_174data_fixed.xlsx', 'Sheet', 'A', 'jump', 'std', 0),
         Xl(r'E:\work\data\acc_stop_1206.xlsx', 'Sheet4', 'F', 'stop', 'rand', 0)
        )
    input_data = []
    for xl in xls:
        input_vec = make_input_from_xlsx(filename=xl.filename, sheetname=xl.sheet,
                                               col=xl.letter, read_range=(2, None), overlap=xl.overlap,
                                               sampling=xl.sampling, sample_cnt=100, fft_N=128,
                                               normalizing='01', label=xl.label, log=False)
        input_data += input_vec

    random.shuffle(input_data)

    labels = [vec[0] for vec in input_data]
    vecs = [list(vec[1]) for vec in input_data]

    """
    テストデータ生成
    """
    test_data = []
    for xl in xls:
        test_vec = make_input_from_xlsx(filename=xl.filename, sheetname=xl.sheet,
                                               col=xl.letter, read_range=(12802, None), overlap=xl.overlap,
                                               sampling=xl.sampling, sample_cnt=20, fft_N=128,
                                               normalizing='01', label=xl.label, log=False)
        test_data += test_vec

    random.shuffle(test_data)

    test_labels = [vec[0] for vec in test_data]
    test_vecs = [list(vec[1]) for vec in test_data]

    """
    教師データの学習分類
    """
    est = svm.SVC(C=1, kernel='rbf', gamma=0.01)    # パラメータ (C-SVC, RBF カーネル, C=1)
    clf = OneVsRestClassifier(est)  #他クラス分類器One-against-restによる識別
    clf.fit(vecs, labels)
    test_pred = clf.predict(test_vecs)

    clf2 = SVC(C=1, kernel='rbf', gamma=0.01)    # パラメータ (C-SVC, RBF カーネル, C=1)
    clf2.fit(vecs, labels)
    test_pred2 = clf2.predict(test_vecs)  #他クラス分類器One-versus-oneによる識別

    """
    学習モデルのローカル保存
    """
    joblib.dump(clf, 'E:\clf.pkl')
    joblib.dump(clf2, 'E:\clf2.pkl')

    #confusion matrix（ラベルの分類表。分類性能が高いほど対角線に値が集まる）
    print confusion_matrix(test_labels, test_pred)
    print confusion_matrix(test_labels, test_pred2)

    #分類結果 適合率 再現率 F値の表示
    print classification_report(test_labels, test_pred)
    print classification_report(test_labels, test_pred2)

    #正答率 分類ラベル/正解ラベル
    print accuracy_score(test_labels, test_pred)
    print accuracy_score(test_labels, test_pred2)

    print test_labels       #分類前ラベル
    print list(test_pred)   #One-against-restによる識別ラベル
    print list(test_pred2)  #One-versus-oneによる識別ラベル

    """
    target_names = ['class 0', 'class 1', 'class 2']
    print(classification_report(test_labels, test_pred, target_names=target_names))
    """