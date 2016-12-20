# coding: utf-8

if __name__ == '__main__':
    import openpyxl as px
    from util.excelwrapper import ExcelWrapper

    filename = r'E:\work\data\jump_original_1122_fixed.xlsx'
    sheets = ('Sheet4', 'Sheet5', 'Sheet6')
    n_list = [['Magnitude Vector']]
    cnt = 0
    for s in sheets:
        old_cnt = cnt
        print s
        r = 0
        ws = ExcelWrapper(filename, s)
        col = ws.select_column(column_letter='F', (2, None), end_row=None, log=False)
        print "lencol:", len(col)
        while True:
            print r
            if col[r] < 0.65:
                n_list += [[i] for i in col[r:r+128]]
                r += 360
                cnt += 1
                if r > len(col) - 1:
                    print "break!"
                    break
                continue
            r += 1
            if r > len(col) - 1:
                print "break!"
                break
        print "ジャンプ回数:", cnt - old_cnt
    print "ジャンプ合計回数:", cnt
    wb = px.Workbook()
    ws = wb.active
    for row in n_list:
        ws.append(row)
    wb.save(r'E:\work\data\jump_128p_Xdata_1122.xlsx')
    print "finish"