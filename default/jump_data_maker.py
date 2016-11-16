# coding: utf-8

if __name__ == '__main__':
    import openpyxl as px
    from util.excelwrapper import ExcelWrapper

    filename = r'E:\work\data\jump_original.xlsx'
    sheets = ('Sheet4', 'Sheet5', 'Sheet6')
    n_list = [['Magnitude Vector']]
    cnt = 0
    for s in sheets:
        old_cnt = cnt
        print s
        r = 2
        ws = ExcelWrapper(filename, s)
        col = ws.select_column(column_letter='F', begin_row=r, end_row=None, log=False)
        while True:
            if col[r] < 0.75:
                n_list += [[i] for i in col[r:r+128]]
                r += 350
                cnt += 1
                if r + 128 > len(col):
                    break
                continue
            r += 1
        print "ジャンプ回数:", cnt - old_cnt
    print "ジャンプ合計回数:", cnt
    wb = px.Workbook()
    ws = wb.active
    for row in n_list:
        ws.append(row)
    wb.save(r'E:\work\data\new_jump.xlsx')
    print "finish"