# coding: utf-8

import openpyxl as px
from excelwrapper import ExcelWrapper
from timecounter import timecounter
from app import R, L, T

@timecounter
def clip_xlsx(xlsx, sheetname, savename, col='F', row_range=(1, None), N=128,
           threshold=3.5, interval=1000):
    """突発的な加速度を切り出してExcelに保存

    リストを先頭から見ていき、threshold以上の加速度を発見した場合に
    そのインデックスから前後N/2を切り出す

    :param threshold : int or float
        閾値

    :param interval : int, default: 1000
        予測される動作の間隔(ms)

    """

    Hz = 100
    ws = ExcelWrapper(xlsx).get_sheet(sheetname)
    col_v = ws.get_col(col, row_range, iter_cell=False, log=True)
    ret = [['Magnitude Vector']]
    half = N / 2
    interval = int(Hz / 1000. * interval)

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

if __name__ == '__main__':
    xlsx, sheetname = R('data/raw/placekick/place_1222_fix.xlsx'), 'Sheet2'
    savename = T('clip/placekick.xlsx', mkdir=True)
    clip_xlsx(xlsx, sheetname, savename, 'F', (2, None), threshold=4.9, interval=2000)