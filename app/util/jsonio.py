# coding: utf-8

import os
import json
from app.util import ExcelWrapper
from datetime import datetime
from time import mktime
import re


"""内部関数"""

def _support_datetime_default(o):
    """datetimeオブジェクトをjsonで扱う場合のフック関数"""

    if isinstance(o, datetime):
        #return o.isoformat()
        return list(o.timetuple())
    raise TypeError(repr(o) + " is not JSON serializable")

def _rename_to_json(old):
    """.jsonにリネーム"""

    name, _ = os.path.splitext(old)
    return name + '.json'

def _get_datetime(t):
    """timetupleからdatetime"""

    return datetime.fromtimestamp(mktime(t))

def _format_str(jsonstr):
    """jsonstringを整形"""

    #return re.sub(r'], ', r'],\n', jsonstr)
    reg = '"\w+": '
    gs = re.findall(reg, jsonstr)
    for s in gs:
        jsonstr = re.sub(s, '\n\t' + s, jsonstr)
    return re.sub('}', '\n}',jsonstr)


def _check_overwrite(savename, overwrite, newname):
    """上書きをチェック"""

    if os.path.exists(savename):
        if overwrite:
            print "overewrite a file:", newname
            return
        raise ValueError(u"filename already exist, set the argument 'newname' "\
                         u"to avoid overlap: {}".format(newname))

def _get_xl_prop(json_):
    """プロパティの辞書を取得"""

    d = {}
    d['min_row'] = json_['min_row']
    d['max_row'] = json_['max_row']
    d['length'] = json_['length']
    return d



"""json読み込み用関数"""

def iter_gps_json(gps_json, prop=False):
    """GPSデータの各列のgeneratorを返す

    :return tuple of generators
        (Time, Latitude, Longitude)
        'prop'がTrueの場合は後ろにプロパティの辞書を追加
    """

    with open(gps_json, 'r') as fp:
        j = json.load(fp)
        ret = [(_get_datetime(t) for t in j['Time']),
               (l for l in j['Latitude']),
               (l for l in j['Longitude'])]
        if prop:
            ret.append(_get_xl_prop(j))
        return tuple(ret)

def iter_acc_json(acc_json, prop=False):
    """加速度データの各列のgeneratorを返す

    :return tuple of generators
        (Time, Magnitude Vector)
        'prop'がTrueの場合は後ろにプロパティの辞書を追加
    """

    with open(acc_json, 'r') as fp:
        j = json.load(fp)
        ret = [(_get_datetime(t) for t in j['Time']),
               (l for l in j['Magnitude Vector'])]
        if prop:
            ret.append(_get_xl_prop(j))
        return tuple(ret)



"""json保存用関数"""

def save_inputs_as_json(labels, features, savename):
    """ラベルと特徴ベクトルをjsonで保存

    :param labels : iterable
    :param features : iterable
    :param savename : str
    """

    list_ = [[l, f] for l, f in zip(labels, features)]
    with open(savename, 'w') as fp:
        s = json.dumps(list_, indent=None, sort_keys=False)
        fp.write(_format_str(s))

def save_features_as_json(features, savename):
    """特徴ベクトルのみ保存(ラベルなし)

    :param features : list fo list
    :param savename : str
    """

    with open(savename, 'w') as fp:
        s = json.dumps(features, indent=None)
        fp.write(_format_str(s))

def save_xlsx_as_json(xlsx, sheet, cols, row_range, savename, header=None,
                      prop=False):
    """xlsxの指定した複数の列をjsonで保存

    :param header : list of str or None, default: None
        ヘッダのリスト
        Noneの場合は各列の1行目をヘッダとして扱う

    :param prop : bool, default: False
        詳細を含むかどうか
        Trueの場合は以下のキーでアクセス
        'min_row': 読み取った最初の行番号(ヘッダ行を含まない)
        'max_row': 読み取った最後の行番号
        'length' : 列の長さ(ヘッダ行を含まない)
    """

    ws = ExcelWrapper(xlsx)[sheet]
    if header is None:
        header = [ws.pickup_cell((c, row_range[0])) for c in cols]
        row_range = row_range[0]+1, row_range[1]

    d = {}
    for c, h in zip(cols, header):
        list_ = ws.get_col(c, row_range=row_range)
        d[h] = list_

    if prop:
        d['min_row'] = row_range[0]
        d['max_row'] = ws.ws.max_row if row_range[1] is None else row_range[1]
        d['length'] = len(list_)

    with open(savename, 'w') as fp:
        s = json.dumps(d, fp, indent=None,
                       default=_support_datetime_default)
        fp.write(_format_str(s))
    print "saved as json: {}".format(savename)

def save_gps_as_json(xlsx, sheet, newname=None, cols=('A', 'J', 'K'),
                         row_range=(8, None), overwrite=False):
    """GPSデータのxlsxをjsonに変換して、/res/data/gps/に保存

    :param xlsx : str
        GPSデータのExcelファイルのパス

    :param sheet : str
        データのあるシート名

    :param newname : str, default: None
        ファイル名を変更する場合は指定
        パスは必要なし
        Noneの場合はもとのファイル名のまま拡張子のみを変更する

    :param overwrite, bool, default: False
        上書きを許可する
    """

    if newname is None:
        newname = _rename_to_json(os.path.basename(xlsx))
    savename = R('data/gps/{}'.format(newname))
    _check_overwrite(savename, overwrite, newname)
    save_xlsx_as_json(xlsx, sheet, cols, row_range, savename, header=None,
                      prop=True)

def save_acc_as_json(xlsx, sheet, newname=None, cols=('A', 'F'),
                              row_range=(1, None), overwrite=False):
    """加速度データのxlsxをjsonに変換して、/res/data/acc/に保存

    :param xlsx : str
        加速度データのExcelファイルのパス

    :param sheet : str
        データのあるシート名

    :param newname : str, default: None
        ファイル名を変更する場合は指定
        パスは必要なし
        Noneの場合はもとのファイル名のまま拡張子のみを変更する

    :param overwrite : bool, default: False
        上書きを許可する
    """

    if newname is None:
        newname = _rename_to_json(os.path.basename(xlsx))
    savename = R('data/acc/{}'.format(newname))
    _check_overwrite(savename, overwrite, newname)
    save_xlsx_as_json(xlsx, sheet, cols, row_range, savename, header=None,
                      prop=True)

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
            save_gps_as_json(xl, 'Sheet1')

    def main2():
        xl = R('data/raw/pass&runpass/pass&runpass1_1222.xlsx')
        save_acc_as_json(xl, 'Sheet2')

    def main3():
        gps = R('data/raw/gps_random_1206.xlsx')
        acc = R('data/raw/acc_random_1206.xlsx')
        save_gps_as_json(gps, 'Sheet1', overwrite=True)
        save_acc_as_json(acc, 'Sheet4', overwrite=True)

    def iter_test():
        times, lats, lons = iter_gps_json(R('data/gps/gps_random_1206.json'))
        print times.next()
        print type(times.next())

    #main()
    #iter_test()
    #main2()
    main3()