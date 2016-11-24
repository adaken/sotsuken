# coding: utf-8

import openpyxl as px

class ExcelWrapper(object):

    def __init__(self, filename, sheetname):
        self.filename = filename
        self.wb = px.load_workbook(filename=filename, read_only=False)
        self.ws = self.wb[sheetname]

    def _read(self, col_range, row_range, log=False):
        c1, c2 = col_range
        r1, r2 = row_range
        if r2 is None: r2 = self.ws.max_row
        if log: print "{}{}:{}{}を読み込み中です...".format(c1, r1, c2, r2)
        return self.ws['{}{}:{}{}'.format(c1, r1, c2, r2)]

    def cut_rect(self, col_range, row_range=(1, None), mode='v', log=False):
        """
        四角形に切り取る

        Parameters
        ----------
        col_range: tuple of str
            読み込む列の範囲

        row_range: tuple of ints, default: (1, None), optional
            読み込む行の範囲
            最低2行以上指定する
            終了行にNoneを指定すると最後の行までが範囲になる

        mode: str, default: 'v', optional
            'v': 列のリストを返す
            'h': 行のリストを返す

        log: bool, default: False, optional
            ログを表示するかどうか

        Return
        ------
        list of lists
        """

        assert len(row_range) == len(col_range) == 2
        assert row_range[1] - row_range[0] > 1
        assert mode in ('v', 'h')
        rect = self._read(col_range, row_range, log)
        rows = [[v.value for v in t] for t in (row for row in rect)]
        if mode == 'v':
            return list(map(list, zip(*rows)))
        elif mode == 'h':
            return rows

    def select_column(self,
                      column_letter,
                      row_range=(1, None),
                      mode='v',
                      log = False):
        """
        Parameters
        ----------
        column_leter : str or tuple
            読み込む列のレター
            ('A', 'C')のように指定した場合は、'A', 'C'列を読み込む

        row_range : tuple of ints, default:(1, None) optional
            読み込む行の範囲
            終了行にNoneを指定すると最後の行までが範囲になる

        mode : str, default: 'v', optional
            column_letterにタプルを指定した場合の振る舞い
            'v': 指定した列を切り取り、列として結合
            'h': 指定した列を切り取り、行として結合

        log : bool, default: False, optional
            ログを出力するかどうか

        Returns
        -------
        column : list of lists
        """

        assert isinstance(column_letter, (str, tuple))

        if isinstance(column_letter, str) or (isinstance(column_letter, tuple) and len(column_letter) == 1):
            col = self._read((column_letter,)*2, row_range, log)
            return [t[0].value for t in (row for row in col)]
        else:
            cols = []
            for letter in column_letter:
                col = self._read((letter,)*2, row_range, log)
                col = [t[0].value for t in (cell for cell in col)]
                cols.append(col)
            if mode == 'v':
                return cols
            elif mode == 'h':
                return list(map(list, zip(*cols)))

if __name__ == '__main__':
    filepath = r'E:\work\data\run.xlsx'
    letter = ('D', 'F', 'B')
    begin = 2
    end = 100
    sheet = "Sheet4"
    excel = ExcelWrapper(filepath, sheet)
    data = excel.cut_rect(col_range=('B', 'D'), row_range=(2, 4), mode='h', log=True)
    print "size:", len(data)
    print >> file('E:\log.txt', 'w'), data
    print "finish"

