# coding: utf-8

import openpyxl as px
from excelwrapper import ExcelWrapper
from util import timecounter
from app import R, L, T
import numpy as np

def clip_xlsx(xlsx, sheetname, savename, col='F', row_range=(1, None), N=128,
           threshold=3.5, interval=1000):
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
        if col_v[i] > threshold:
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

def clip_xlsx2(acc, savename, still_min=0, still_max=1.25, still_len=300, act_len=10,
               sample_n=128):
    """
    :param acc : ndarray
    :param savename : str
    
    :param still_min : float
        静止状態の最小値
        
    :param still_max : float
        静止状態の最大値
        
    :param still_len : int, default: 300
        静止期間だと認識するのに必要な行数
        
    :param act_len : int, default: 10
        動作期間だと認識するのに必要な行数
    
    :param sample_n : int, default: 128
        切り出される加速度リスト1つの行数        
    """
    
    def is_still(a):
        return (still_min < a) * (a < still_max)
    
    def find_stillness(idx):
        j = idx + still_len # 最小限の静止期間の最後の添字
        a = acc[idx:j]
        print "a_len:", len(a)
        b = is_still(a)
        print "is_still?:", b
        if not np.all(b): # 動作を含む
            return None
        for i, v in enumerate(a[j:], j): # 活動状態にぶつかるまでループ
            if not is_still(v): return i # 活動期間の最初の添字
    
    def find_activity(idx):
        j = idx + act_len # 最小限の活動期間の最後の添字
        a = acc[idx:j]
        b = is_still(a)
        if np.any(b): # 静止を含む
            return None
        for i, v in enumerate(acc[j:], j): # 静止状態にぶつかるまでループ
            if is_still(v): return i # 静止期間最初の添字
         
    def clip_activity(start, stop, n=sample_n/2):
        a = acc[start:stop] # 活動期間の配列
        i = a.argmax() # 最大値の添字
        return a[i-n:i+n]
    
    def save(data):
        wb = px.Workbook
        ws = wb.active
        for v in data: ws.append(v)
        wb.save(savename)
         
    i = 0
    data = [['Magnitude Vector']]
    while i < len(acc):
        j = find_stillness(i)
        if j is None:
            i += 1
            continue
        k = find_activity(j)
        if k is None:
            i += 1
            continue
        v = clip_activity(j, k)
        print "v_len", len(v)
        data += [[e] for e in v]
        i = k
    save(data)
    print "Finish"
        
    
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
        r = R('data/raw/pass/')

    func3()
