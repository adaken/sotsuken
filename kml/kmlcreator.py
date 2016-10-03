# coding: utf-8

import time as timeutil

if __name__ == '__main__':

    time_start = timeutil.time()

    # リソース
    xlsx_path = r"E:\work\sample_rowdata.xlsx"
    kml_path = r"E:\work\out.kml"
    icon_res = r"E:\work\circle_blue.png"

    # Excelの列を読み込む
    print "reading xlsx..."
    from util.excelwrapper import ExcelWrapper
    ws = ExcelWrapper(filename=xlsx_path, sheetname='Sheet1')
    # 読み込みを開始する行
    begin_row = 9
    # Time列読み込み
    times = ws.select_column(col_letter='A', begin_row=begin_row, datatype='S30')
    # Latitude列読み込み
    lats = ws.select_column(col_letter='J', begin_row=begin_row, datatype='float')
    # Longitude列読み込み
    lons = ws.select_column(col_letter='K', begin_row=begin_row, datatype='float')

    # kmlに書き出す
    print "write to kml..."
    from kml.kmlwrapper import KmlWrapper
    kmlwrapper = KmlWrapper()
    kmlwrapper.createAnimeKml(save_path=kml_path, times=times, longitudes=lons,
                              latitudes=lats, format_time=True, sampling_interval=15,
                              icon_res=icon_res, icon_scale=0.3)
    print "completed!"

    time_elapsed = timeutil.time() - time_start
    print "elapsed time: %fsec" % time_elapsed
