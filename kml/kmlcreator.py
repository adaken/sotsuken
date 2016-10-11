# coding: utf-8

import time as timeutil

if __name__ == '__main__':

    time_start = timeutil.time()

    # リソース
    xlsx_path = r"E:\work\bicycle_gps_hirano.xlsx"
    kml_path = r"E:\work\out_bicycle.kml"
    icon_res = r"E:\work\circle_blue.png"

    # Excelを読み込む
    from util.excelwrapper import ExcelWrapper
    ws = ExcelWrapper(filename=xlsx_path, sheetname='Sheet1')

    # 読み込みを開始する行
    begin_row = 9

    # 列のリストを取得する関数
    getcol = lambda l : ws.select_column(col_letter=l, begin_row=begin_row)

    # kmlに書き出す
    from kml.kmlwrapper import KmlWrapper
    KmlWrapper().createAnimeKml(save_path=kml_path, times=getcol('A'), longitudes=getcol('K'),
                              latitudes=getcol('J'), format_time=True, sampling_interval=15,
                              icon_res=icon_res, icon_scale=0.3)
    print "completed!"

    time_elapsed = timeutil.time() - time_start
    print "elapsed time: %fsec" % time_elapsed
