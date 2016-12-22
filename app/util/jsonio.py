# coding: utf-8

import json

def tojson(labels, features, savename):
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

def fromjson(filename):
    """jsonから入力ベクトルを読み込む

    :param filename : str

    """

    with open(filename) as fp:
        l, f = [], []
        for i, j in json.load(fp):
            l.append(i)
            f.append(j)
        return l, f


if __name__ == '__main__':
    from app import L
    tojson([1, 2, 3], [[52, 62, 9, 8], [67, 62, 6, 42], [624, 35, 525, 5215]], L('test.json'))
    labels, features = fromjson(L('test.json'))
    print labels
    print features