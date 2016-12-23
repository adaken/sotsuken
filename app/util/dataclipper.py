# coding: utf-8

import openpyxl as px
from excelwrapper import ExcelWrapper
from timecounter import timecounter
from app import R, L, T

@timecounter
def clip_xlsx(xlsx, sheetname, savename, col='F', row_range=(1, None), N=128,
           threshold=3.5, interval=0):
    """突発的な加速度を切り出してExcelに保存

    リストを先頭から見ていき、threshold以上の加速度を発見した場合に
    そのインデックスから前後N/2を切り出す

    :param interval : int, default: 0
        予測される動作の間隔(ms)

    """

    ws = ExcelWrapper(xlsx).get_sheet(sheetname)
    col_v = ws.get_col(col, row_range, iter_cell=False, log=True)
    ret = [['Magnitude Vector']]
    half = N / 2

    i = 0
    while i < len(col_v):
        if col_v[i] > threshold:
            vec = col_v[i-(half):i+(half)]
            ret += [[elem] for elem in vec]
            print "{}. {} : {}".format(len(ret) / N, i-half, i+half)
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
    xlsx, sheetname = R('data/raw/tackle/tackle_1222_fix.xlsx'), 'Sheet2'
    savename = L('clip/tackle.xlsx', mkdir=True)
    clip_xlsx(xlsx, sheetname, savename, 'F', (2, None), threshold=2.2, interval=350)