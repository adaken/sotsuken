# coding: utf-8

import Tkinter as tk
import tkFileDialog as tkfd
from msilib.schema import SelfReg

class Frame(tk.Frame):

    def __init__(self, root, master=None):
        tk.Frame.__init__(self, master)
        self.master.title('TkApp')
        self.filenames = None
        self.filename = None
        self.create_menu()
        self.create_kml_widgets()

    def create_menu(self):
        self.menu = tk.Menu(root)
        root.configure(menu=self.menu)
        self.menu.add_command(label='kml作成', command=self.check_frame_kml)
        self.menu.add_command(label='svm', command=self.check_frame_svm)
        self.menu.add_command(label='Exit', command=root.quit)

    def check_frame_svm(self):
        if self.f1.winfo_exists() == 1:
            self.f1.destroy()
            self.create_svm_widgets()
        else:
            self.create_svm_widgets()
    def check_frame_kml(self):
        if self.f1.winfo_exists() == 1:
            self.f1.destroy()
            self.create_kml_widgets()
        else:
            self.create_kml_widgets()

    def create_kml_widgets(self):
        """ウィジェット"""

        #フレーム
        self.f1 = tk.LabelFrame(root, relief = 'ridge', width = 300, height = 300, text='kml作成', labelanchor=tk.N,
                           borderwidth = 4, padx=5, pady=5, bg = '#fffafa', font=('times', 20))

        #ボタン
        self.select_button = tk.Button(self.f1, text = 'ファイル選択', relief = 'raised',
                            font = ('times', 10), bg = '#fffafa', fg = '#000000', borderwidth = 4,
                            command = self.select_files)

        self.kml_button = tk.Button(self.f1, text = '変換', relief = 'raised',
                            font = ('times', 10), bg = '#fffafa', fg = '#000000', borderwidth = 4,
                            command = self.open_kml)
        #ラベル
        self.title_label = tk.Label(self.f1, width=50, font=('times', 20), pady=2,
                                    text='kml作成', bg='#fffafa', fg='#000000')

        #ラベルのバッファ
        self.filenames_buff = [tk.StringVar() for i in xrange(5)]

        self.labels = []
        map(self.labels.append,
            (tk.Label(self.f1, width=50, font=('times', 13), pady=2, relief='raised', textvariable=self.filenames_buff[i]) for i in xrange(5)))

        # ラベルの配置
        #self.title_label.grid(row=0, column=0, pady=10)
        for i, label in zip(xrange(1, len(self.labels)+1), self.labels):
            label.grid(row=i, column=0)

        # ボタンの配置
        self.select_button.grid(row=6, column=0, pady=5)
        self.kml_button.grid(row=7, column=0, pady=5)
        # フレームを配置
        self.f1.pack()
    def create_svm_widgets(self):
        #フレーム
        self.f1 = tk.LabelFrame(root, relief = 'ridge', text='scikit-learn', labelanchor=tk.N,
                           borderwidth = 4, padx=5, pady=5, bg = '#fffafa', font=('times', 20))

        #ボタン
        self.select_file1 = tk.Button(self.f1, text = 'ファイル選択', relief = 'raised',
                            font = ('times', 10), bg = '#fffafa', fg = '#000000', borderwidth = 4,
                            command = self.select_svm_file1, padx=5)
        self.select_file2 = tk.Button(self.f1, text = 'ファイル選択', relief = 'raised',
                            font = ('times', 10), bg = '#fffafa', fg = '#000000', borderwidth = 4,
                            command = self.select_svm_file2, padx=5)
        self.select_file3 = tk.Button(self.f1, text = 'ファイル選択', relief = 'raised',
                            font = ('times', 10), bg = '#fffafa', fg = '#000000', borderwidth = 4,
                            command = self.select_svm_file3, padx=5)

        #ラベルのバッファ
        self.filename1_buff=tk.StringVar()
        self.filename2_buff=tk.StringVar()
        self.filename3_buff=tk.StringVar()
        #ラベル
        self.title_label = tk.Label(self.f1, width=40, font=('times', 20), pady=2,
                                    text='Support Vector Machine', bg='#fffafa', fg='#000000')
        self.label_A=tk.Label(self.f1, text='A')
        self.label_B=tk.Label(self.f1, text='B')
        self.label_C=tk.Label(self.f1, text='C')

        self.filename1_label=tk.Label(self.f1, textvariable=self.filename1_buff, width=30)
        self.filename2_label=tk.Label(self.f1, textvariable=self.filename2_buff, width=30)
        self.filename3_label=tk.Label(self.f1, textvariable=self.filename3_buff, width=30)

        # ラベルの配置
        #self.title_label.grid(row=0, column=1)
        self.label_A.grid(row=1, column=0)
        self.label_B.grid(row=2, column=0)
        self.label_C.grid(row=3, column=0)
        self.filename1_label.grid(row=1, column=1)
        self.filename2_label.grid(row=2, column=1)
        self.filename3_label.grid(row=3, column=1)
        # ボタンの配置
        self.select_file1.grid(row=1, column=2)
        self.select_file2.grid(row=2, column=2)
        self.select_file3.grid(row=3, column=2)
        # フレームを配置
        self.f1.pack()


    def select_files(self): #kml
        """ファイルを選択"""

        fTyp_xlsx = [('Excelファイル', '*.xlsx')]
        iDir = r'E:/work'
        filenames = tkfd.askopenfilenames(filetypes=fTyp_xlsx, initialdir=iDir)
        print "filenames:", filenames
        self.filenames =  filenames

        # ラベルに選択ファイル名をセット
        for i, filename in zip(xrange(len(self.filenames_buff)), filenames):
            self.filenames_buff[i].set(filename)

    def open_kml(self):
        """Please wait..."""
        text1 = ["Please wait..."]*5
        for i, filename in zip(xrange(len(self.filenames_buff)), text1):
            self.filenames_buff[i].set(filename)
        self.after(500, self.make_kml)

    def make_kml(self):
        """kml作成"""
        from kml.kmlwrapper import KmlWrapper
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
            ws = ExcelWrapper(filename=xl).get_sheet('Sheet1')

            # 読み込みを開始する行
            begin_row = 9
            times, lats, lons = ws.iter_cols(('A', 'J', 'K'), row_range=(begin_row, None))
            import os
            print os.getcwd()
            from util.util import drow_circle
            icon = os.path.abspath(drow_circle((255, 0, 0), size=(16, 16), savepath=r'.\tmp\red.png'))
            KmlWrapper().createAnimeKml(kml, times, lons, lats, icon_scale=0.3, sampling_step=10, icon_res=icon)

            print "completed!"

            #GoogleEarthで表示
            #sb.Popen(["C:\Program Files (x86)\Google\Google Earth\client\googleearth.exe", kml])

            self.after(500, self.change_complete)

    def change_complete(self):
        text1 = ["Completed!"]*5
        for i, filename in zip(xrange(len(self.filenames_buff)), text1):
            self.filenames_buff[i].set(filename)

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

    def select_svm_file1(self): #svm
        fTyp_xlsx=[('Excelファイル', '*.xlsx')]
        iDir=r'E:work'
        filename1=tkfd.askopenfilename(filetypes=fTyp_xlsx, initialdir=iDir)
        #self.filename=filename1
        self.filename1_buff.set(filename1)
    def select_svm_file2(self):
        fTyp_xlsx=[('Excelファイル', '*.xlsx')]
        iDir=r'E:work'
        filename2=tkfd.askopenfilename(filetypes=fTyp_xlsx, initialdir=iDir)
        #self.filename=filename2
        self.filename2_buff.set(filename2)
    def select_svm_file3(self):
        fTyp_xlsx=[('Excelファイル', '*.xlsx')]
        iDir=r'E:work'
        filename3=tkfd.askopenfilename(filetypes=fTyp_xlsx, initialdir=iDir)
        #self.filename=filename3
        self.filename3_buff.set(filename3)



if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("800x400")
    f = Frame(root)
    f.pack()
    f.mainloop()
