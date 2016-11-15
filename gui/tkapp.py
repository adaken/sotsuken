# coding: utf-8

import sys, os.path
import Tkinter as tk
import tkFileDialog as tkfd
import time as timeutil
import subprocess as sb
from pip import cmdoptions
from Tkconstants import BOTH


path_name = ""
fTyp_xlsx = [('Excelファイル', '*.xlsx')]
fTyp_kml = [('kmlファイル', '*.kml')]
fTyp_png = [('pngファイル', '*.png')]
iDir = 'E:/work'


def load_file():

    filename = tkfd.askopenfilename(filetypes=fTyp_xlsx, initialdir=iDir)
    print "filename:", filename
    if filename != "":
        path_name = os.path.dirname(filename)

        filename2 = replace_ext(filename, 'kml')

    #kml
    #リソース作成
    r = open(filename2, 'w')
    r.write("")
    r.close()
    print "filename:", filename
    print "filename2:", filename2

    time_start = timeutil.time()

    # リソース
    xlsx_path = filename
    kml_path = filename2
    icon_res = r"E:\work\circle_blue.png"

    # Excelを読み込む
    from util.excelwrapper import ExcelWrapper
    ws = ExcelWrapper(filename=xlsx_path, sheetname='Sheet1')

    # 読み込みを開始する行
    begin_row = 9

    # 列のリストを取得する関数
    getcol = lambda l : ws.select_column(column_letter=l, begin_row=begin_row)

    # kmlに書き出す
    from kml.kmlwrapper import KmlWrapper
    KmlWrapper().createAnimeKml(save_path=kml_path, times=getcol('A'), longitudes=getcol('K'),
                    latitudes=getcol('J'), format_time=True, sampling_interval=15,
                    icon_res=icon_res, icon_scale=0.3)
    print "completed!"
    time_elapsed = timeutil.time() - time_start
    print "elapsed time: %fsec" % time_elapsed

    #kml開く
    sb.Popen(["C:\Program Files (x86)\Google\Google Earth\client\googleearth.exe", filename2])


def replace_ext(filename, extension):
    assert filename.find('.') is not -1
    for i in range(len(filename), 0, -1):
        c = filename[i-1]
        if c == '.':
            return filename[:i] + extension

if __name__ == '__main__':

    root = tk.Tk()
    root.title("卒業研究")
    root.geometry("640x480")
    
    #ラベルフレーム
    f1 = tk.LabelFrame(root,
                       text ='KML',
                       relief = 'raised',
                       width = 600,
                       height = 300,
                       labelanchor = 'nw',
                       bg = '#4169e1')
    #ボタン
    button1 = tk.Button(root,
                        text = 'kml作成', 
                        font = ('times', 15), 
                        bg = '#4169e1',
                        fg = '#fffafa',
                        relief = 'raised',
                        command = load_file)
    button1.pack(in_ = f1)
    
    #フレームの配置
    f1.pack(padx = 5, pady = 5)

    root.mainloop()
