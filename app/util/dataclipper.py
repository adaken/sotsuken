# coding: utf-8

import openpyxl as px
from excelwrapper import ExcelWrapper
from util import timecounter
from app import R, L, T

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

def clip_xlsx2(xlsx, sheetname, savename, threshold):
    pass

if __name__ == '__main__':
    def func1():
        xlsx, sheetname = R('data/raw/placekick/place_1222_fix.xlsx'), 'Sheet2'
        savename = T('clip/placekick.xlsx', mkdir=True)
        clip_xlsx(xlsx, sheetname, savename, 'F', (2, None), threshold=4.9, interval=2000)

    def func2():
        res = R('data/raw/pass')
        savename = T('clip').mkdir('pass')
        for ab, fn in zip(res.ls(True)[1], res.filenames):
            s =  savename + '\\' + fn
            clip_xlsx(ab, 'Sheet2', s, 'F', (2, None), 128, 1.6, 200)

    def func3():
        r = R('data/raw/20170106/pass/fix')
        savename = T.mkdir('clip').mkdir('pass')
        for a, f in zip(r.ls(absp=True)[1], r.filenames):
            s = savename.p + '\\' + f
            print a
            print f
            clip_xlsx(a, 'Sheet2', s, 'F', (2, None), 128, 1.25, 2.0, 300)

    func3()
