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
        column_leter : str or tuple
            読み込む列のレター
            tuple('A', 'C')のように指定した場合は、'A', 'B', 'C'列を読み込む

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
        column = None
        if isinstance(column_letter, str):
            if log: print "reading column '%s%d:%s%d'..." % (column_letter, begin_row, column_letter, end_row)
            column = self.ws['%s%d:%s' % (column_letter, begin_row, end_row)]
            return [data[0].value for data in (cell for cell in column)]
        elif isinstance(column_letter, tuple):
            if log: print "reading column '%s%d:%s%d'..." % (column_letter[0], begin_row, column_letter[1], end_row)
            column = self.ws['%s%d:%s%d' % (column_letter[0], begin_row, column_letter[1], end_row)]
            return [[v.value for v in data] for data in (cell for cell in column)]

if __name__ == '__main__':
    filepath = r'E:\work\data\run.xlsx'
    letter = ('D', 'F')
    begin = 2
    end = 100
    sheet = "Sheet4"
    excel = ExcelWrapper(filepath,sheet)
    data = excel.select_column(column_letter=letter, begin_row=begin, end_row=end, log=True)
    print "size:", len(data)
    print >> file('E:\log.txt', 'w'), data
    print "finish"

