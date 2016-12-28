# coding: utf-8

import os
import json
import numpy as np
from app.util import ExcelWrapper
from datetime import datetime

def _support_datetime_default(o):
    """datetimeオブジェクトをjsonで扱う場合のフック関数"""

    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(repr(o) + " is not JSON serializable")

def _rename_to_json(old):
    """.jsonにリネーム"""

    name, _ = os.path.splitext(old)
    return name + '.json'

def save_inputs_as_json(labels, features, savename):
    """ラベルと特徴ベクトルをjsonで保存

    :param labels : iterable
    :param features : iterable
    :param savename : str
    """

    list_ = [[l, f] for l, f in zip(labels, features)]
    with open(savename, 'w') as fp:
        s = json.dumps(list_, indent=None, sort_keys=False)
        fp.write(s.replace('],', '],\n'))

def save_features_as_json(features, savename):
    """特徴ベクトルのみ保存(ラベルなし)

    :param features : list fo list
    :param savename : str
    """

    with open(savename, 'w') as fp:
        s = json.dumps(features, indent=None)
        fp.write(s.replace('],', '],\n'))

def save_xlsx_as_json(xlsx, sheet, cols, row_range, savename, header=None):
    """xlsxの指定した複数の列をjsonで保存

    :param header list of str or None, default: None
        ヘッダのリスト
        Noneの場合は各列の1行目をヘッダとして扱う
    """

    ws = ExcelWrapper(xlsx)[sheet]
    if header is None:
        header = [ws.pickup_cell((c, row_range[0])) for c in cols]
        row_range = row_range[0]+1, row_range[1]

    dict_ = {}
    for c, h in zip(cols, header):
        list_ = ws.get_col(c, row_range=row_range)
        data = list_[1:]
        dict_[h] = data

    with open(savename, 'w') as fp:
        s = json.dumps(dict_, fp, indent=None,
                       default=_support_datetime_default)
        fp.write(s.replace(']', '],\n'))
    print "saved as json: {}".format(savename)

def save_gpsdata_as_json(xlsx, sheet, newname=None, cols=('A', 'J', 'K'),
                         row_range=(8, None)):
    """GPSデータのxlsxをjsonに変換して、/res/data/gps/に保存

    :param xlsx : str
        Excelファイルのパス

    :param sheet : str
        データのあるシート名
    """

    if newname is None:
        newname = _rename_to_json(os.path.basename(xlsx))
    savename = R('data/gps/{}'.format(newname))
    if os.path.exists(savename):
        raise ValueError(u"filename already exist, set the argument 'name' " \
                         u"to avoid overlap: {}".format(newname))
    save_xlsx_as_json(xlsx, sheet, cols, row_range, savename, header=None)

def save_acceleration_as_json(xlsx, sheet, newname=None, cols=('A', 'F'),
                              row_range=(2, None)):
    """加速度データのxlsxをjsonに変換して、/res/data/acc/に保存

    :param xlsx : str
        Excelファイルのパス

    :param sheet : str
        データのあるシート名
    """

    if newname is None:
        newname = _rename_to_json(os.path.basename(xlsx))
    savename = R('data/acc/{}'.format(newname))
    if os.path.exists(savename):
        raise ValueError(u"filename already exist, set the argument 'name' " \
                         u"to avoid overlap: {}".format(newname))
    save_xlsx_as_json(xlsx, sheet, cols, row_range, savename, header=None)

if __name__ == '__main__':
    from app import R, L
    from app.util import make_input

    def f1():
        xls = [#('placekick', R('data/acc/placekick_128p_52data.xlsx'), 52),
               #('run', R('data/acc/run_acc_128p_132data.xlsx'), 132),
               #('tackle', R('data/acc/tackle_acc_128p_92data.xlsx'), 92),
               ('dropkick',
                R('data/acc/dropkick_acc_128p_16data_20161215.xlsx'), 16)]
        for act, xl, cnt in xls:
            vec = make_input(xl, cnt, log=True)
            print "shape:", vec.shape
            save_features_as_json(vec, R('data/acc/fft/{}_acc_128p_{}data.json'
                                         .format(act, cnt)))

    def main():
        for xl in R('data/gps/players').ls(True)[1]:
            save_gpsdata_as_json(xl, 'Sheet1')

