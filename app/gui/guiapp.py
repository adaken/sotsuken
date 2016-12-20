# coding: utf-8

import Tkinter as tk
import tkFileDialog as tkfd
import ttk
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
        self.menu.add_command(label='som', command=self.check_frame_som)
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

    def check_frame_som(self):
        if self.f1.winfo_exists() == 1:
            self.f1.destroy()
            self.create_som_widgets()
        else:
            self.create_som_widgets()

    def create_kml_widgets(self):
        """ウィジェット"""

        #フレーム
        self.f1 = tk.LabelFrame(root, relief = 'ridge', width = 300, height = 300, text='kml作成', labelanchor=tk.N,
                           borderwidth = 4, padx=5, pady=5, bg = '#fffafa', font=('times', 30))

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
            (tk.Label(self.f1, width=40, font=('times', 13), pady=2, relief='ridge', textvariable=self.filenames_buff[i]) for i in xrange(5)))

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
                           borderwidth = 4, padx=5, pady=5, bg = '#fffafa', font=('times', 30), width=500, height=300)
        #ボタン
        self.select_fileA = tk.Button(self.f1, text = 'ファイル選択', relief = 'raised',
                            bg = '#fffafa', fg = '#000000', borderwidth = 4,
                            command = self.select_svm_fileA)
        self.select_fileB = tk.Button(self.f1, text = 'ファイル選択', relief = 'raised',
                            bg = '#fffafa', fg = '#000000', borderwidth = 4,
                            command = self.select_svm_fileB)
        self.select_fileC = tk.Button(self.f1, text = 'ファイル選択', relief = 'raised',
                            bg = '#fffafa', fg = '#000000', borderwidth = 4,
                            command = self.select_svm_fileC)

        #ラベルのバッファ
        self.filenameA_buff=tk.StringVar()
        self.filenameB_buff=tk.StringVar()
        self.filenameC_buff=tk.StringVar()
        #ラベル
        self.title_label = tk.Label(self.f1, width=40, font=('times', 20), pady=2,
                                    text='Support Vector Machine', bg='#fffafa', fg='#000000')
        self.label_A=tk.Label(self.f1, text='A')
        self.label_B=tk.Label(self.f1, text='B')
        self.label_C=tk.Label(self.f1, text='C')

        self.filenameA_label=tk.Label(self.f1, relief='ridge', textvariable=self.filenameA_buff, width=40)
        self.filenameB_label=tk.Label(self.f1, relief='ridge', textvariable=self.filenameB_buff, width=40)
        self.filenameC_label=tk.Label(self.f1, relief='ridge', textvariable=self.filenameC_buff, width=40)

        # ラベルの配置
        self.label_A.place(relx=0.05, rely=0.1)
        self.label_B.place(relx=0.05, rely=0.2)
        self.label_C.place(relx=0.05, rely=0.3)
        self.filenameA_label.place(relx=0.1, rely=0.1)
        self.filenameB_label.place(relx=0.1, rely=0.2)
        self.filenameC_label.place(relx=0.1, rely=0.3)
        self.select_fileA.place(relx=0.7, rely=0.09)
        self.select_fileB.place(relx=0.7, rely=0.19)
        self.select_fileC.place(relx=0.7, rely=0.29)
        # フレームを配置
        self.f1.pack()
    def create_som_widgets(self):
        #フレーム
        self.f1 = tk.LabelFrame(root, relief = 'ridge', text='SOM', labelanchor=tk.N,
                           borderwidth = 4, padx=5, pady=5, bg='#006400', fg='#fffafa', font=('times', 30), width=800, height=300)
        #ラベルフレーム
        self.frame_labelname=tk.LabelFrame(self.f1, relief='ridge', text='ラベル名', labelanchor=tk.NW,
                                           bg='#006400', fg='#fffafa')
        #var
        self.labelname1_buff=tk.StringVar()
        self.labelname2_buff=tk.StringVar()
        self.labelname3_buff=tk.StringVar()
        self.labelname4_buff=tk.StringVar()
        self.filename1_buff=tk.StringVar()
        self.filename2_buff=tk.StringVar()
        self.filename3_buff=tk.StringVar()
        self.filename4_buff=tk.StringVar()
        self.color1_buff=tk.StringVar()
        self.color2_buff=tk.StringVar()
        self.color3_buff=tk.StringVar()
        self.color4_buff=tk.StringVar()
        self.count_buff=tk.IntVar()
        self.point_buff=tk.IntVar()
        self.size_y_buff=tk.IntVar()
        self.size_x_buff=tk.IntVar()
        #ボタン
        self.select_file1 = tk.Button(self.f1, text = 'ファイル選択', relief = 'raised',
                            bg = '#fffafa', fg = '#000000', borderwidth = 4, command = self.select_som_file1)
        self.select_file2 = tk.Button(self.f1, text = 'ファイル選択', relief = 'raised',
                            bg = '#fffafa', fg = '#000000', borderwidth = 4, command = self.select_som_file2)
        self.select_file3 = tk.Button(self.f1, text = 'ファイル選択', relief = 'raised',
                            bg = '#fffafa', fg = '#000000', borderwidth = 4, command = self.select_som_file3)
        self.select_file4 = tk.Button(self.f1, text = 'ファイル選択', relief = 'raised',
                            bg = '#fffafa', fg = '#000000', borderwidth = 4, command = self.select_som_file4)
        #ラベル
        self.label_1=tk.Label(self.f1, bg='#006400', fg='#fffafa', text='ラベル名')
        self.label_2=tk.Label(self.f1, bg='#006400', fg='#fffafa', text='色')
        self.label_3=tk.Label(self.f1, bg='#006400', fg='#fffafa', text='ファイル名')
        self.label_4=tk.Label(self.f1, bg='#006400', fg='#fffafa', text='カウント')
        self.label_5=tk.Label(self.f1, bg='#006400', fg='#fffafa', text='ポイント')
        self.label_6=tk.Label(self.f1, bg='#006400', fg='#fffafa', text='サイズ')
        self.label_7=tk.Label(self.f1, bg='#006400', fg='#fffafa', text='縦')
        self.label_8=tk.Label(self.f1, bg='#006400', fg='#fffafa', text='横')
        self.filename1_label=tk.Label(self.f1, textvariable=self.filename1_buff, width=50, relief='ridge')
        self.filename2_label=tk.Label(self.f1, textvariable=self.filename2_buff, width=50, relief='ridge')
        self.filename3_label=tk.Label(self.f1, textvariable=self.filename3_buff, width=50, relief='ridge')
        self.filename4_label=tk.Label(self.f1, textvariable=self.filename4_buff, width=50, relief='ridge')

        #エントリー
        self.labelname1_entry=tk.Entry(self.frame_labelname, textvariable=self.labelname1_buff, width=15)
        self.labelname2_entry=tk.Entry(self.frame_labelname, textvariable=self.labelname2_buff, width=15)
        self.labelname3_entry=tk.Entry(self.frame_labelname, textvariable=self.labelname3_buff, width=15)
        self.labelname4_entry=tk.Entry(self.frame_labelname, textvariable=self.labelname4_buff, width=15)
        self.count_entry=tk.Entry(self.f1, textvariable=self.count_buff, width=5)
        self.point_entry=tk.Entry(self.f1, textvariable=self.point_buff, width=5)
        self.size_y_entry=tk.Entry(self.f1, textvariable=self.size_y_buff, width=5)
        self.size_x_entry=tk.Entry(self.f1, textvariable=self.size_x_buff, width=5)
        #コンボボックス
        self.color_box1=ttk.Combobox(self.f1, textvariable=self.color1_buff, state='readonly', width=10,
                                     value=('red', 'blue', 'green', 'yellow'))
        self.color_box2=ttk.Combobox(self.f1, textvariable=self.color2_buff, state='readonly', width=10,
                                     value=('red', 'blue', 'green', 'yellow'))
        self.color_box3=ttk.Combobox(self.f1, textvariable=self.color3_buff, state='readonly', width=10,
                                     value=('red', 'blue', 'green', 'yellow'))
        self.color_box4=ttk.Combobox(self.f1, textvariable=self.color4_buff, state='readonly', width=10,
                                     value=('red', 'blue', 'green', 'yellow'))
        #ウィジェット配置
        self.label_1.place(relx=0.05, rely=0.05)
        self.labelname1_entry.place(relx=0.05, rely=0.15)
        self.labelname2_entry.place(relx=0.05, rely=0.25)
        self.labelname3_entry.place(relx=0.05, rely=0.35)
        self.labelname4_entry.place(relx=0.05, rely=0.45)
        self.label_2.place(relx=0.2, rely=0.05)
        self.color_box1.place(relx=0.2, rely=0.15)
        self.color_box2.place(relx=0.2, rely=0.25)
        self.color_box3.place(relx=0.2, rely=0.35)
        self.color_box4.place(relx=0.2, rely=0.45)
        self.label_3.place(relx=0.35, rely=0.05)
        self.filename1_label.place(relx=0.35, rely=0.15)
        self.filename2_label.place(relx=0.35, rely=0.25)
        self.filename3_label.place(relx=0.35, rely=0.35)
        self.filename4_label.place(relx=0.35, rely=0.45)
        self.select_file1.place(relx=0.85, rely=0.15)
        self.select_file2.place(relx=0.85, rely=0.25)
        self.select_file3.place(relx=0.85, rely=0.35)
        self.select_file4.place(relx=0.85, rely=0.45)
        self.label_4.place(relx=0.15, rely=0.6)
        self.count_entry.place(relx=0.15, rely=0.7)
        self.label_5.place(relx=0.4, rely=0.6)
        self.point_entry.place(relx=0.4, rely=0.7)
        self.label_6.place(relx=0.7, rely=0.6)
        self.label_7.place(relx=0.67, rely=0.7)
        self.size_y_entry.place(relx=0.7, rely=0.7)
        self.label_8.place(relx=0.75, rely=0.7)
        self.size_x_entry.place(relx=0.78, rely=0.7)
        #フレーム配置
        self.f1.pack()
    def select_files(self): #kml関連
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




    def select_svm_fileA(self): #svm関連
        fTyp_xlsx=[('Excelファイル', '*.xlsx')]
        iDir=r'E:work'
        filenameA=tkfd.askopenfilename(filetypes=fTyp_xlsx, initialdir=iDir)
        #self.filename=filename1
        self.filenameA_buff.set(filenameA)
    def select_svm_fileB(self):
        fTyp_xlsx=[('Excelファイル', '*.xlsx')]
        iDir=r'E:work'
        filenameB=tkfd.askopenfilename(filetypes=fTyp_xlsx, initialdir=iDir)
        #self.filename=filename2
        self.filenameB_buff.set(filenameB)
    def select_svm_fileC(self):
        fTyp_xlsx=[('Excelファイル', '*.xlsx')]
        iDir=r'E:work'
        filenameC=tkfd.askopenfilename(filetypes=fTyp_xlsx, initialdir=iDir)
        #self.filename=filename3
        self.filenameC_buff.set(filenameC)

    def select_som_file1(self): #som関連
        fTyp_xlsx=[('Excelファイル', '*.xlsx')]
        iDir=r'E:work'
        filename1=tkfd.askopenfilename(filetypes=fTyp_xlsx, initialdir=iDir)
        #self.filename=filename3
        self.filename1_buff.set(filename1)
    def select_som_file2(self):
        fTyp_xlsx=[('Excelファイル', '*.xlsx')]
        iDir=r'E:work'
        filename2=tkfd.askopenfilename(filetypes=fTyp_xlsx, initialdir=iDir)
        #self.filename=filename3
        self.filename2_buff.set(filename2)
    def select_som_file3(self):
        fTyp_xlsx=[('Excelファイル', '*.xlsx')]
        iDir=r'E:work'
        filename3=tkfd.askopenfilename(filetypes=fTyp_xlsx, initialdir=iDir)
        #self.filename=filename3
        self.filename3_buff.set(filename3)
    def select_som_file4(self):
        fTyp_xlsx=[('Excelファイル', '*.xlsx')]
        iDir=r'E:work'
        filename4=tkfd.askopenfilename(filetypes=fTyp_xlsx, initialdir=iDir)
        #self.filename=filename3
        self.filename4_buff.set(filename4)

    def run_scikit_learn(self):
        from sklearn import svm
        from util.util import make_input_from_xlsx
        import random
        from collections import namedtuple
        from sklearn.metrics import confusion_matrix
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import classification_report
        from sklearn.externals import joblib

        """
        教師データ生成
        """
        Xl = namedtuple('Xl', 'filename, sheet, letter, label, sampling, overlap')
        xls =  (
             Xl(r'E:\work\data\walk_1122_data.xlsx', 'Sheet4', 'F', 'run', 'std', 0),
             Xl(r'E:\work\data\walk_1122_data.xlsx', 'Sheet4', 'F', 'walk', 'std', 0),
             #Xl(r'E:\work\data\jump_128p_174data_fixed.xlsx', 'Sheet', 'A', 'jump', 'std', 0),
             Xl(r'E:\work\data\acc_stop_1206.xlsx', 'Sheet4', 'F', 'stop', 'rand', 0)
            )
        input_data = []
        for xl in xls:
            input_vec = make_input_from_xlsx(filename=xl.filename, sheetname=xl.sheet,
                                                   col=xl.letter, read_range=(2, None), overlap=xl.overlap,
                                                   sampling=xl.sampling, sample_cnt=100, fft_N=128,
                                                   normalizing='01', label=xl.label, log=False)
            input_data += input_vec

        random.shuffle(input_data)

        labels = [vec[0] for vec in input_data]
        vecs = [list(vec[1]) for vec in input_data]

        """
        テストデータ生成
        """
        test_data = []
        for xl in xls:
            test_vec = make_input_from_xlsx(filename=xl.filename, sheetname=xl.sheet,
                                                   col=xl.letter, read_range=(12802, None), overlap=xl.overlap,
                                                   sampling=xl.sampling, sample_cnt=20, fft_N=128,
                                                   normalizing='01', label=xl.label, log=False)
            test_data += test_vec

        random.shuffle(test_data)

        test_labels = [vec[0] for vec in test_data]
        test_vecs = [list(vec[1]) for vec in test_data]

        """
        教師データの学習分類
        """
        est = svm.SVC(C=1, kernel='rbf', gamma=0.01)    # パラメータ (C-SVC, RBF カーネル, C=1)
        clf = OneVsRestClassifier(est)  #他クラス分類器One-against-restによる識別
        clf.fit(vecs, labels)
        test_pred = clf.predict(test_vecs)

        clf2 = SVC(C=1, kernel='rbf', gamma=0.01)    # パラメータ (C-SVC, RBF カーネル, C=1)
        clf2.fit(vecs, labels)
        test_pred2 = clf2.predict(test_vecs)  #他クラス分類器One-versus-oneによる識別

        """
        学習モデルのローカル保存
        """
        joblib.dump(clf, 'E:\clf.pkl')
        joblib.dump(clf2, 'E:\clf2.pkl')

        #confusion matrix（ラベルの分類表。分類性能が高いほど対角線に値が集まる）
        print confusion_matrix(test_labels, test_pred)
        print confusion_matrix(test_labels, test_pred2)

        #分類結果 適合率 再現率 F値の表示
        print classification_report(test_labels, test_pred)
        print classification_report(test_labels, test_pred2)

        #正答率 分類ラベル/正解ラベル
        print accuracy_score(test_labels, test_pred)
        print accuracy_score(test_labels, test_pred2)

        print test_labels       #分類前ラベル
        print list(test_pred)   #One-against-restによる識別ラベル
        print list(test_pred2)  #One-versus-oneによる識別ラベル

        """
        target_names = ['class 0', 'class 1', 'class 2']
        print(classification_report(test_labels, test_pred, target_names=target_names))
        """

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("800x400")
    f = Frame(root)
    f.pack()
    f.mainloop()
