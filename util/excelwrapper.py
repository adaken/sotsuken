# coding: utf-8

import openpyxl as px

class ExcelWrapper(object):
    """
    シート単位でオブジェクトとして扱う
    """

    def __init__(self, filename, sheetname):
            self.wb = px.load_workbook(filename=filename, read_only=True)
            self.ws = self.wb[sheetname]

    def select_column(self,
                      col_letter,
                      begin_row = 1,
                      end_row = None):

        if end_row is None:
            # 最後の行
            end_row = self.ws.max_row

        print "reading column '%s'..." % col_letter
        column = self.ws['%s%d:%s' % (col_letter, begin_row, end_row - 1)]
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
