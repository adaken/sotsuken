# coding: utf-8

import Tkinter as tk
import tkFileDialog as tkfd
import subprocess as sb

class Frame(tk.Frame):

    def __init__(self, root, master=None):
        tk.Frame.__init__(self, master)
        self.master.title('卒業研究')

        self.menu = tk.Menu(root)
        self.menu.add_command(label='Exit', command=root.quit)
        root.configure(menu=self.menu)

        self.filenames = None

        self.f1 = tk.Frame(root, relief = 'ridge',
                      width = 640, height = 240,
                      borderwidth = 4,
                      padx=5, pady=5,
                      bg = '#006400')

        self.select_button = tk.Button(self.f1, text = 'ファイル選択', relief = 'raised',
                            font = ('times', 10),
                            bg = '#006400', fg = '#fffafa', borderwidth = 4,
                            command = self.select_files)

        self.kml_button = tk.Button(self.f1, text = '変換', relief = 'raised',
                            font = ('times', 10),
                            bg = '#006400', fg = '#fffafa', borderwidth = 4,
                            command = self.open_kml)

        self.title_label = tk.Label(self.f1, width=50, font=('times', 20), pady=2,
                                     text='kml作成', bg='#006400', fg='#fffafa')

        # ラベルのバッファ
        self.filenames_buff = [tk.StringVar() for i in xrange(5)]

        self.labels = []
        map(self.labels.append,
            (tk.Label(self.f1, width=50, font=('times', 12), pady=2, relief='raised', textvariable=self.filenames_buff[i]) for i in xrange(5)))

        #フレームの配置

        # ラベルの配置
        self.title_label.grid(row=0, column=0, pady=10)
        for i, label in zip(xrange(1, len(self.labels)+1), self.labels):
            label.grid(row=i, column=0)

        # ボタンの配置
        self.select_button.grid(row=6, column=0, pady=5)
        self.kml_button.grid(row=7, column=0, pady=5)

        # フレームを配置
        self.f1.pack()

    def select_files(self):
        """ファイルを選択"""

        fTyp_xlsx = [('Excelファイル', '*.xlsx')]
        iDir = 'E:/work'
        filenames = tkfd.askopenfilenames(filetypes=fTyp_xlsx, initialdir=iDir)
        print "filenames:", filenames
        self.filenames =  filenames

        # ラベルに選択ファイル名をセット
        for i, filename in zip(xrange(len(self.filenames_buff)), filenames):
            self.filenames_buff[i].set(filename)


    def open_kml(self):
        filenames2 = self.replace_exts(self.filenames, 'kml')
        #kml
        #リソース作成
        for i, zip_ in enumerate(zip(self.filenames, filenames2)):
            xl, kml = zip_

            r = open(kml, 'w')
            r.write("")
            r.close()

            # リソース
            #icon_res = icons[random.randint(0, 2)]

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
                            sampling_interval=15, icon_scale=0.3)
            print "completed!"

            #GoogleEarthで表示
            sb.Popen(["C:\Program Files (x86)\Google\Google Earth\client\googleearth.exe", kml])

    def replace_ext(self, filename, extension):
        assert filename.find('.') is not -1
        for i in range(len(filename), 0, -1):
            c = filename[i-1]
            if c == '.':
                return filename[:i] + extension

    def replace_exts(self, filenames, extension):
        list_ = []
        for f in filenames:
            list_.append(self.replace_ext(f, extension))
        return list_

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("800x300")
    f = Frame(root)
    f.pack()
    f.mainloop()