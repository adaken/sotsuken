# coding: utf-8

import sys, os.path
import Tkinter as tk
import tkFileDialog as tkfd
import time as timeutil
import subprocess as sb
from pip import cmdoptions
from Tkconstants import BOTH, LEFT
import random
import kml

def replace_ext(filename, extension):
    assert filename.find('.') is not -1
    for i in range(len(filename), 0, -1):
        c = filename[i-1]
        if c == '.':
            return filename[:i] + extension

def replace_exts(filenames, extension):
    list_ = []
    for f in filenames:
        list_.append(replace_ext(f, extension))
    return list_

def init():
        """初期化処理"""

        #メニュー
        menu1 = tk.Menu(root)
        menu1.add_command(label = 'Exit', command = root.quit)
        #ラベルフレーム
        f1 = tk.Frame(root, relief = 'ridge',
                      width = 640, height = 240,
                      borderwidth = 4,
                      padx=5, pady=5,
                      bg = '#006400')
        f2 = tk.Frame(root)

        #ボタン
        b1 = tk.Button(f1, text = 'ファイル選択', relief = 'raised',
                            font = ('times', 10),
                            bg = '#006400', fg = '#fffafa', borderwidth = 4,
                            command = select_files)
        b2 = tk.Button(f1, text = '変換', relief = 'raised',
                            font = ('times', 10),
                            bg = '#006400', fg = '#fffafa', borderwidth = 4,
                            command = open_kml)

        #ラベル
        l1 = tk.Label(f1, width = 50, font = ('times', 12), pady=2, textvariable=buffs[0])
        l2 = tk.Label(f1, width = 50, font = ('times', 12), pady=2, textvariable=buffs[1])
        l3 = tk.Label(f1, width = 50, font = ('times', 12), pady=2, textvariable=buffs[2])
        l4 = tk.Label(f1, width = 50, font = ('times', 12), pady=2, textvariable=buffs[3])
        l5 = tk.Label(f1, width = 50, font = ('times', 12), pady=2, textvariable=buffs[4])
        l6 = tk.Label(f1, width = 50, font = ('times', 20), pady=2,
                     text = 'kml作成', bg='#006400', fg='#fffafa')

        #フレームの配置
        #e1.pack(side = LEFT)
        l6.grid(row=0, column=0, pady=10)
        l1.grid(row=1, column=0)
        l2.grid(row=2, column=0)
        l3.grid(row=3, column=0)
        l4.grid(row=4, column=0)
        l5.grid(row=5, column=0)
        b1.grid(row=6, column=0, pady=10)
        b2.grid(row=7, column=0)

        f1.pack(padx = 5, pady = 5)
        f2.pack()

def select_files():
    """ファイルを選択"""

    fTyp_xlsx = [('Excelファイル', '*.xlsx')]
    iDir = 'E:/work'
    filenames = tkfd.askopenfilenames(filetypes=fTyp_xlsx, initialdir=iDir)
    print "filenames:", filenames
    return filenames

def select_files_call_back(filenames):


def open_kml(filenames, icons):

    filenames2 = replace_exts(filenames, 'kml')
    #kml
    #リソース作成
    for i, zip_ in enumerate(zip(filenames, filenames2)):
        xl, kml = zip_

        r = open(kml, 'w')
        r.write("")
        r.close()
        buffs[i].set(kml)

        # リソース
        icon_res = icons[random.randint(0, 2)]

        # Excelを読み込む
        from util.excelwrapper import ExcelWrapper
        ws = ExcelWrapper(filename=xl, sheetname='Sheet1')

        # 読み込みを開始する行
        begin_row = 9

        # 列のリストを取得する関数
        getcol = lambda l : ws.select_column(column_letter=l,
                                             begin_row=begin_row)

        # kmlに書き出す
        from kml.kmlwrapper import KmlWrapper
        KmlWrapper().createAnimeKml(save_path=kml, times=getcol('A'), longitudes=getcol('K'),
                        latitudes=getcol('J'), format_time=True,
                        sampling_interval=15,
                        icon_res=icon_res, icon_scale=0.3)
        print "completed!"


if __name__ == '__main__':

    root = tk.Tk()
    root.title("卒業研究")
    root.geometry("800x300")

    buffs = [tk.StringVar() for i in xrange(5)]

    

    #sb.Popen(["C:\Program Files (x86)\Google\Google Earth\client\googleearth.exe", kml])

# アプリケーション開始
init()
root.mainloop()
