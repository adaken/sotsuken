# coding: utf-8

import os
import json
from app.util import ExcelWrapper
from datetime import datetime
from time import mktime
import re
from itertools import izip_longest


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

    gs = re.findall('"[\w ]+": ', jsonstr)
    for s in set(gs):
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

def iter_inputs_json(inputs_json, trans=False):
    """入力ベクトルのgeneratorを返す

    :param trans : bool
        True : ([labels], [features])
        False: [(label, features)...]

    :return generator of tuple(str, list)
    """

    with open(inputs_json, 'r') as fp:
        j = json.load(fp)
        if trans:
            return (d['label'] for d in j), (d['features'] for d in j)
        return ((d['label'], d['features']) for d in j)

def iter_gps_json(gps_json, prop=False):
    """GPSデータの各列のgeneratorを返す

    :return tuple of generators
        (Time_iter, Latitude_iter, Longitude_iter)
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
        (Time_iter, Magnitude Vector_iter)
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

    :param labels : iterable of str or int
    :param features : iterable of list
    :param savename : str
    """

    obj = []
    for l, f in izip_longest(labels, features):
        if l is None or f is None:
            raise ValueError(u"labels and features must be the same lengths")
        obj.append({'label': l, 'features': f})

    with open(savename, 'w') as fp:
        s = json.dumps(obj, indent=None, sort_keys=None)
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
        # 入力ベクトルのjsonを作成
        xls = [
            ('placekick', R('data/acc/placekick_acc_128p_101data.xlsx'), 101),
            ('run', R('data/acc/run_acc_128p_132data.xlsx'), 132),
            ('tackle', R('data/acc/tackle_acc_128p_111data.xlsx'), 111),
            ('pass', R('data/acc/pass_acc_128p_131data.xlsx'), 131)
            ]

        for act, xl, cnt in xls:
            vecs, labels = make_input(xl, cnt, label=act, log=True)
            print "shape:", vecs.shape
            save_inputs_as_json(labels, vecs.tolist(),
                                R('data/fft/{}_acc_128p_{}data.json'
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

    def iter_test2():
        inp = iter_inputs_json(R('data/fft/run_acc_128p_132data.json'))
        print inp.next()
        print inp.next()

    f1()
    #main()
    #iter_test()
    #iter_test2()
    #main2()
    #main3()