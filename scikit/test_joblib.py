#!/usr/bin/env python
# -*- coding: utf-8 -*-
from util.util import make_input_from_xlsx
import random
from collections import namedtuple
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

if __name__ == '__main__':
    """
    テストデータ生成
    """
    Xl = namedtuple('Xl', 'filename, sheet, letter, label, sampling, overlap')
    xls =  (
         Xl(r'E:\work\data\run_1122_data.xlsx', 'Sheet1', 'F', 1, 'std', 0),
         Xl(r'E:\work\data\walk_1122_data.xlsx', 'Sheet1', 'F', 2, 'std', 0),
         Xl(r'E:\work\data\jump_128p_174data_fixed.xlsx', 'Sheet', 'A', 3, 'std', 0),
        #Xl(r'E:\work\data\skip.xlsx', 'Sheet4', 'F', 4, 'rand', 0)
        )
    input_data = []
    test_data = []
    for xl in xls:
        test_vec = make_input_from_xlsx(filename=xl.filename, sheetname=xl.sheet,
                                               col=xl.letter, read_range=(12802, None), overlap=xl.overlap,
                                               sampling=xl.sampling, sample_cnt=21, fft_N=128,
                                               normalizing='01', label=xl.label, log=False)
        test_data += test_vec

    random.shuffle(test_data)

    test_labels = [vec[0] for vec in test_data]
    test_vecs = [list(vec[1]) for vec in test_data]

    clf = joblib.load('E:\clf.pkl')
    test_pred = clf.predict(test_vecs)  #他クラス分類器One-versus-oneによる識別
    print confusion_matrix(test_labels, test_pred)