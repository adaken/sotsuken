# coding: utf-8

import sys, os.path
import Tkinter as tk
import tkFileDialog as tkfd
import time as timeutil

def replace_ext(filename, extension):
    assert filename.find('.') is not -1
    for i in range(len(filename), 0, -1):
        c = filename[i-1]
        if c == '.':
            return filename[:i] + extension

if __name__ == '__main__':

    root = tk.Tk()
    root.title("Tk App")
    root.geometry("700x900")



    path_name = ""
    fTyp_xlsx = [('Excelファイル', '*.xlsx')]
    fTyp_kml = [('kmlファイル', '*.kml')]
    fTyp_png = [('pngファイル', '*.png')]
    iDir = 'E:/work'
    dirname = tkfd.askdirectory(initialdir=iDir)
    # variable 用のオブジェクト
    action = tk.IntVar()
    action.set(0)
    level = tk.IntVar()
    level.set(1)
    d = tk.StringVar()
    d.set("")
    b = tk.StringVar()
    b.set("")
    f = tk.StringVar()
    f.set("")

    #ファイル選択
    def load_file():
        global path_name
        filename = tkfd.askopenfilename(filetypes=fTyp_xlsx, initialdir=iDir)
        print "filename:", filename
        if filename != "":
            path_name = os.path.dirname(filename)
        d.set(os.path.dirname(filename))
        b.set(os.path.basename(filename))
        f.set(filename)
        filename2 = replace_ext(filename, 'kml')

        #kml
        #リソース作成
        r = open(filename2, 'w')
        r.write("")
        r.close()
        print filename
        print filename2

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


    # メニューバーの生成
    menubar = tk.Menu(root)
    root.configure(menu = menubar)

    # メニューの設定
    games = tk.Menu(menubar, tearoff = False)
    levels = tk.Menu(menubar, tearoff = False)
    menubar.add_cascade(label="ファイル", underline = 0, menu=games)
    menubar.add_cascade(label="Level", underline = 0, menu=levels)

    # Games
    games.add_command(label = "Kml作成", under = 0, command = load_file)
    games.add_separator
    games.add_command(label = "exit", under = 0, command = sys.exit)

    # Labels
    levels.add_radiobutton(label = 'Level 1', variable = level, value = 1)
    levels.add_radiobutton(label = 'Level 2', variable = level, value = 2)
    levels.add_radiobutton(label = 'Level 3', variable = level, value = 3)

    # ラベル
    label1 = tk.Label(root, textvariable = d).pack()
    label2 = tk.Label(root, textvariable = b).pack()
    label3 = tk.Label(root, textvariable = f).pack()

    # スケールの値を格納する
    red = tk.IntVar()
    red.set(0)    #ファイル選択
    blue = tk.IntVar()
    blue.set(0)
    green = tk.IntVar()
    green.set(0)

    # ボタンの背景色を変更
    def change_color( n ):
        color = '#%02x%02x%02x' % (red.get(), green.get(), blue.get())
        button.configure(bg = color)

    # ボタン
    button = tk.Button(root, text = 'button', bg = '#000')
    button.pack(fill = 'both');

    # スケール
    s1 = tk.Scale(root, label = 'red', orient = 'h',
               from_ = 0, to = 255, variable = red,
               command = change_color)

    s2 = tk.Scale(root, label = 'blue', orient = 'h',
               from_ = 0, to = 255, variable = blue,
               command = change_color)

    s3 = tk.Scale(root, label = 'green', orient = 'h',
               from_ = 0, to = 255, variable = green,
               command = change_color)

    # ウィジェットの配置
    s1.pack(fill = 'both')
    s2.pack(fill = 'both')
    s3.pack(fill = 'both')



    root.mainloop()