# coding: utf-8

import json
import numpy as np

def save_input_to_json(labels, features, savename):
    """入力ベクトルをjsonで保存

    :param labels : iterable
    :param features : iterable
    :param savename : str

    """

    assert len(labels) == len(features), "labelsとfeaturesのサイズが違うよ"

    #list_ = [{"label":l, "features":f} for l, f in zip(labels, features)]
    list_ = [[l, f] for l, f in zip(labels, features)]
    with open(savename, 'w') as fp:
        s = json.dumps(list_, indent=None, sort_keys=False)
        #fp.write(s.replace('},', '},\n')) # '},'で改行させてファイルに書き込む
        fp.write(s.replace('],', '],\n'))

def get_input_from_json(filename):
    """jsonから入力ベクトルを読み込む

    :param filename : str
        jsonのパス

    :return (labels, features) : list of strs, list of lists
        ラベルのlistと特徴ベクトルのlist

    """

    with open(filename) as fp:
        l, f = [], []
        for i, j in json.load(fp):
            l.append(i)
            f.append(j)
        return l, f

def save_features_to_json(features, savename):
    """入力ベクトルをラベルなしで保存"""
    
    if isinstance(features, np.ndarray):
        features = features.tolist()
    with open(savename, 'w') as fp:
        s = json.dumps(features)
        fp.write(s.replace('],', '],\n'))

if __name__ == '__main__':
    from app import R
    from app.util import make_input

    xls = [('placekick', R('data/acc/placekick_128p_52data.xlsx'), 52),
           ('run', R('data/acc/run_acc_128p_132data.xlsx'), 132),
           ('tackle', R('data/acc/tackle_acc_128p_92data.xlsx'), 92)]
    for act, xl, cnt in xls:
        vec = make_input(xl, cnt, log=True)
        print "shape:", vec.shape
        save_features_to_json(vec, R('data/acc/fft/{}_acc_128p_{}data.json'.format(act, cnt)))
