# coding: utf-8

import openpyxl as px
from excelwrapper import ExcelWrapper
from util import timecounter
from app import R, L, T
import numpy as np
import warnings

def clip_xlsx(xlsx, sheetname, savename, col='F', row_range=(1, None), N=128,
           threshold=3.5, limit=100., interval=1000):
    """突発的な加速度を切り出してExcelに保存

    リストを先頭から見ていき、threshold以上の加速度を発見した場合に
    そのインデックスから前後N/2を切り出す

    :param threshold : int or float
        閾値

    :param interval : int, default: 1000
        予測される動作の間隔(行)

    """

    Hz = 100
    ws = ExcelWrapper(xlsx).get_sheet(sheetname)
    col_v = ws.get_col(col, row_range, iter_cell=False, log=True)
    ret = [['Magnitude Vector']]
    half = N / 2
    #interval = int(Hz / 1000. * interval)

    old = 0
    i = 0
    while i < len(col_v):
        if threshold < col_v[i] < limit:
            vec = col_v[i-(half):i+(half)]
            ret += [[elem] for elem in vec]
            print "{}.\tpoint: {},\t{}:{},\tvalue: {},\tinterval: {}".format(len(ret)/N, i+2, i-half, i+half, col_v[i], i-old)
            old = i
            i += half + 1 + interval
        else:
            i += 1

    print "検出した数:", len(ret) / N
    wb = px.Workbook()
    new_ws = wb.active
    for row in ret:
        new_ws.append(row)
    wb.save(savename)
    print "finish"

def clip_xlsx2(acc, savename, still_min=0, still_max=1.15, still_len=300,
               act_len=3, sample_n=128, error_margin=0.02):
    """
    :param acc : ndarray
    :param savename : str

    :param still_min : float
        静止状態の最小値

    :param still_max : float
        静止状態の最大値

    :param still_len : int, default: 250
        静止期間だと認識するのに必要な行数

    :param act_len : int, default: 3
        動作期間だと認識するのに必要な行数

    :param sample_n : int, default: 128
        切り出される加速度リスト1つの行数
    """

    def is_still(a):
        return (still_min < a) * (a < still_max)

    def is_probably_still(a):
        b = is_still(a)
        p = b.sum() / float(b.size) # Trueの確率
        return 1. - p < error_margin

    def find_stillness(idx):
        j = idx + still_len # 最小限の静止期間の最後の添字
        a = acc[idx:j]
        if not is_probably_still(a): # 動作を含む
            return None
        for i, v in enumerate(acc[j:], j): # 活動状態にぶつかるまでループ
            if not is_still(v):
                return i # 活動期間の最初の添字

    def find_activity(idx):
        j = idx + act_len # 最小限の活動期間の最後の添字
        a = acc[idx:j]
        b = is_still(a)
        if np.any(b): # 静止を含む
            return None
        for i, v in enumerate(acc[j:], j): # 静止状態にぶつかるまでループ
            if is_still(v):
                return i # 静止期間最初の添字

    def clip_activity(start, stop, n=sample_n/2):
        b = np.zeros(acc.shape, dtype=bool)
        b[start:stop] = True # 活動期間のマスク
        idxs = np.where(b)[0] # マスクされた添字
        i = idxs[acc[b].argmax()] # 最大値の添字
        return i, acc[i-n:i+n]

    def save(data):
        wb = px.Workbook()
        ws = wb.active
        for v in data: ws.append(v)
        wb.save(savename)

    if isinstance(acc, list):
        acc = np.array(acc)
        print "convert to ndarray"

    i, oldc, cnt = 0, 0, 0
    data = [['Magnitude Vector']]
    print "#  \t" + "act-start\t" + "act-end\t" + "interval"
    while i < len(acc) - still_len:
        j = find_stillness(i)
        if j is None:
            i += 1
            continue

        k = find_activity(j)
        if k is None:
            i += 1
            continue

        c, v = clip_activity(j, k)
        if not len(v) == sample_n:
            warnings.warn(u"less data sample: {}/{}".format(len(v), sample_n))
            i = k
            continue
        cnt += 1
        data += [[e] for e in v]
        print "{:<3d}\t{:>9d}\t{:>7d}\t{:>8d}".format(cnt, j, k-1, c-oldc)
        oldc = c
        i = k
    else:
        print "detected:", cnt

    save(data)
    print "Finish\n"

if __name__ == '__main__':
    def func1():
        xlsx, sheetname = R('data/raw/placekick/place_1222_fix.xlsx'), 'Sheet2'
        savename = T('clip/placekick.xlsx', mkdir=True)
        clip_xlsx(xlsx, sheetname, savename, 'F', (2, None), threshold=4.9, interval=2000)

    def func2():
        res = R('data/raw/pass')
        savename = T('clip').mkdir('pass').p
        for ab, fn in zip(res.ls(True)[1], res.filenames):
            s =  savename + '\\' + fn
            clip_xlsx(ab, 'Sheet2', s, 'F', (2, None), 128, 1.6, 200)

    def func3():
        r = R('data/raw/20170106/tackle/fix')
        savename = T.mkdir('clip').mkdir('tackle')
        for a, f in zip(r.ls(absp=True)[1], r.filenames):
            acc = ExcelWrapper(a).get_sheet('Sheet2').get_col('F', (2, None), log=True)
            s = savename.p + '\\' + f
            print a
            clip_xlsx2(acc, s, still_max=2.25, act_len=2)

    def func4():
        r = R('data/raw/tackle/1215_tackle1_fix.xlsx')
        savename = T('clip/test.xlsx')
        acc = ExcelWrapper(r).get_sheet('Sheet2').get_col('F', (2, None), log=True)
        clip_xlsx2(acc, savename, still_max=3)

    func3()
