# coding: utf-8

import openpyxl as px
import re
from itertools import chain

class ExcelWrapper(object):

    def __init__(self, filename, sheetname):
        self.filename = filename
        self.wb = px.load_workbook(filename=filename, read_only=False)
        self.ws = self.wb[sheetname]

    def _read_value(self, col_range, row_range, mode, log):
        c1, c2 = col_range
        r1, r2 = row_range
        if r2 is None: r2 = self.ws.max_row
        if log: print "{}{}:{}{}を読み込み中です...".format(c1, r1, c2, r2)
        cells =  self.ws['{}{}:{}{}'.format(c1, r1, c2, r2)]
        rows = [[v.value for v in t] for t in (cell for cell in cells)] # 2Dリスト
        if mode == 'c': return list(map(list, zip(*rows))) # 列のリスト
        if mode == 'r': return rows # 行のリスト

    def select_column(self,
                      column_letter,
                      row_range=(1, None),
                      mode='c',
                      log = False):
        """
        Parameters
        ----------
        column_leter : str or tuple
            読み込む列のレター
            ('A', 'C')のように指定した場合は、'A', 'C'列を読み込む
            ('A:C', 'E', 'G')のように指定した場合は、'A', 'B', 'C', 'E', 'G'列を読み込む

        row_range : tuple of ints, default: (1, None), optional
            読み込む行の範囲
            終了行にNoneを指定すると最後の行までが範囲になる

        mode : str, default: 'c', optional
            column_letterにタプルを指定した場合の振る舞い
            'c': 指定した列を切り取り、列として結合
            'r': 指定した列を切り取り、行として結合

        log : bool, default: False, optional
            ログを出力するかどうか

        Returns
        -------
        column : list

        columns: list of lists
        """

        assert isinstance(column_letter, (str, tuple))
        assert mode in ('c', 'r'), "modeの値が不正です: '{}'".format(mode)

        orders = []
        if isinstance(column_letter, str):
            orders.append(column_letter)
        else:
            for i in column_letter: orders.append(i)

        cols = []
        for o in orders:
            if re.match(r'[A-Z]+$', o):
                cols.append(self._read_value(o*2, row_range, 'c', log)[0])
            elif re.match(r'[A-Z]+:[A-Z]+$', o):
                c1, c2 = o.split(':')
                if c1 == c2:
                    cols.append(self._read_value((c1, c2), row_range, 'c', log)[0])
                else:
                    for i in self._read_value((c1, c2), row_range, 'c', log): cols.append(i)
            else: raise AssertionError("column_letterの値が不正です: '{}'".format(o))

        if len(cols) == 1:
            if mode == 'c': return cols[0]
            if mode == 'r': return list(map(list, zip(*cols)))
        else:
            if mode == 'c': return cols
            if mode == 'r': return list(map(list, zip(*cols)))

if __name__ == '__main__':
    filepath = r'E:\read_test.xlsx'
    excel = ExcelWrapper(filepath, 'Sheet1')
    data = excel.select_column(('D:A'), (1, 5), log=True, mode='c')
    print "size:", len(data)
    print >> file('E:\log.txt', 'w'), data
    print "finish"