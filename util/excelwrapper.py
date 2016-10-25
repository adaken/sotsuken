# coding: utf-8

import openpyxl as px

class ExcelWrapper(object):
    """
    シート単位でオブジェクトとして扱う
    """

    def __init__(self, filename, sheetname):
        self.filename = filename
        self.wb = px.load_workbook(filename=filename, read_only=False)
        self.ws = self.wb[sheetname]

    def select_column(self,
                      column_letter,
                      begin_row = 1,
                      end_row = None,
                      log = False):
        """
        Parameters
        ----------
        column_leter : char
            読み込む列のレター

        begin_row : int, default: 1, optional
            読み込みを開始する行

        end_row : int or None, default: None, optional
            読み込みを終了する行
            Noneを指定すると最後の行まで読み込む

        log : bool, default: False, optional
            ログを出力するかどうか

        Returns
        -------
        column : list
        """

        if end_row is None:
            # 最後の行
            end_row = self.ws.max_row
        if log:
            print "read", self.filename
            print "reading column '%s%d:%s%d'..." % (column_letter, begin_row, column_letter, end_row)
        column = self.ws['%s%d:%s' % (column_letter, begin_row, end_row)]
        return [data[0].value for data in [cell for cell in column]]

    def write_to_csv(self, columns, csv_path):
        import csv

        # 行のリストに転置
        rows = self.__transpose(columns)

        # csvに書き出す
        with open(csv_path, 'wb') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)

    def __transpose(self, list2):
        """
        2次元リストを転置します
        """
        row_size = len(list2)
        col_size = len(list2[0])
        # 行と列のサイズを入れ替える
        nlist2 = [[None for i in xrange(row_size)] for j in xrange(col_size)]
        for i in range(row_size):
            for j in range(col_size):
                nlist2[j][i] = list2[i][j]
        return nlist2
