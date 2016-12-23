# coding: utf-8

import json

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

if __name__ == '__main__':
    from app import R
    from . import make_input, ExcelWrapper

    def test():
        from app import L
        save_input_to_json([1, 2, 3], [[5.2, 6.2, 9., 8.], [67., 6.2, 6., 42.], [6.24, 3.5, 52.5, 5215.]], L('test.json'))
        labels, features = get_input_from_json(L('test.json'))
        print labels
        print features

    def conv(xlsx, label):
        ws = ExcelWrapper(xlsx).get_sheet(sheetname)
