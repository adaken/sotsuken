# coding: utf-8

def cutoff(xlsx, sheetname, min_row=2, N=128, col='F', steel_min=300, steel_max=500):
    import openpyxl as px
    from util.excelwrapper import ExcelWrapper

    ws = ExcelWrapper(xlsx).get_sheet(sheetname)
    column = ws.get_col(col, row_range=(min_row, None), iter_cell=False, log=True)
    steel_cnt = 0
    n_list = [['Magnitude Vector']]
    flag = False
    for i, v in enumerate(column):
        if 0.7 < v < 1.3:
            steel_cnt += 1
            flag = True
        else:
            steel_cnt = 0
            flag = False
        if not flag:
            if steel_min < steel_cnt < steel_max:
                b = i - steel_min
                vec = column[b:b+N]
                n_list += [[elem] for elem in vec]
                steel_cnt = 0
    print "検出した数:", len(n_list) / N
    wb = px.Workbook()
    new_ws = wb.active
    for row in n_list:
        new_ws.append(row)
    wb.save(r'..\data\cutoff_test.xlsx')
    print "finish"

if __name__ == '__main__':
    xlsx, sheetname = r'C:\Users\locked\Desktop\20161215_rugby\tackle\1215_tackle2.xlsx', 'Sheet2'
    cutoff(xlsx, sheetname, min_row=9570)