# coding: utf-8

import Tkinter as tk
import tkFileDialog as tkfd
import subprocess as sb
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
        """kmlのウィジェット"""

        #フレーム
        self.f1 = tk.LabelFrame(root, relief = 'ridge', width = 300, height = 300, text='KML', labelanchor=tk.N,
                           borderwidth = 4, padx=5, pady=5, bg = '#fffafa', font=('times', 30))

        #ボタン
        self.select_button = tk.Button(self.f1, text = 'ファイル選択', relief = 'raised',
                            font = ('times', 10), bg = '#fffafa', fg = '#000000', borderwidth = 4,
                            command = self.select_files)

        self.kml_button = tk.Button(self.f1, text = '実行', relief = 'raised',
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
        """svmのウィジェット"""
        #フレーム
        self.f1 = tk.LabelFrame(root, relief = 'ridge', text='SVM', labelanchor=tk.N,
                           borderwidth = 4, padx=5, pady=5, font=('times', 30), width=800, height=300)
        #ラベルフレーム(lf)
        self.lf_labelname_skl = tk.LabelFrame(self.f1, text='ラベル名', labelanchor=tk.NW)
        self.lf_sheetname_skl = tk.LabelFrame(self.f1, text='Sheet名', labelanchor=tk.NW)
        self.lf_filename_skl = tk.LabelFrame(self.f1, text='ファイル名', labelanchor=tk.NW)
        self.f_button_skl = tk.Frame(self.f1)

        #ボタン
        self.select_fileA = tk.Button(self.f_button_skl, text = 'ファイル選択', relief = 'raised',
                            bg = '#fffafa', fg = '#000000', borderwidth = 4,
                            command = self.select_svm_fileA).pack()
        self.select_fileB = tk.Button(self.f_button_skl, text = 'ファイル選択', relief = 'raised',
                            bg = '#fffafa', fg = '#000000', borderwidth = 4,
                            command = self.select_svm_fileB).pack()
        self.select_fileC = tk.Button(self.f_button_skl, text = 'ファイル選択', relief = 'raised',
                            bg = '#fffafa', fg = '#000000', borderwidth = 4,
                            command = self.select_svm_fileC).pack()
        self.select_fileD = tk.Button(self.f_button_skl, text = 'ファイル選択', relief = 'raised',
                            bg = '#fffafa', fg = '#000000', borderwidth = 4,
                            command = self.select_svm_fileD).pack()
        self.run_skl_button=tk.Button(self.f1, text = '実行', relief = 'raised',
                            bg = '#fffafa', fg = '#000000', borderwidth = 4,
                            command = self.run_scikit_learn)

        #タイトルラベル
        self.title_label = tk.Label(self.f1, width=40, font=('times', 20), pady=2,
                                    text='Scikit-learn', bg='#fffafa', fg='#000000')

        #ラベル(ファイル名)
        self.filenameA_buff=tk.StringVar()
        self.filenameB_buff=tk.StringVar()
        self.filenameC_buff=tk.StringVar()
        self.filenameD_buff=tk.StringVar()
        self.filenameA_label=tk.Label(self.lf_filename_skl, relief='ridge', textvariable=self.filenameA_buff, width=60).pack()
        self.filenameB_label=tk.Label(self.lf_filename_skl, relief='ridge', textvariable=self.filenameB_buff, width=60).pack()
        self.filenameC_label=tk.Label(self.lf_filename_skl, relief='ridge', textvariable=self.filenameC_buff, width=60).pack()
        self.filenameD_label=tk.Label(self.lf_filename_skl, relief='ridge', textvariable=self.filenameD_buff, width=60).pack()

        #エントリー(ラベル名)
        self.e_ln1_skl_buff=tk.StringVar()
        self.e_ln2_skl_buff=tk.StringVar()
        self.e_ln3_skl_buff=tk.StringVar()
        self.e_ln4_skl_buff=tk.StringVar()
        self.e_ln1_skl=tk.Entry(self.lf_labelname_skl, textvariable=self.e_ln1_skl_buff, width=10).pack()
        self.e_ln2_skl=tk.Entry(self.lf_labelname_skl, textvariable=self.e_ln2_skl_buff, width=10).pack()
        self.e_ln3_skl=tk.Entry(self.lf_labelname_skl, textvariable=self.e_ln3_skl_buff, width=10).pack()
        self.e_ln4_skl=tk.Entry(self.lf_labelname_skl, textvariable=self.e_ln4_skl_buff, width=10).pack()
        #エントリー(ラベル名)初期内容
        self.e_ln1_skl_buff.set('pass')
        self.e_ln2_skl_buff.set('Pkick')
        self.e_ln3_skl_buff.set('run')
        self.e_ln4_skl_buff.set('tackle')

        #配置(ボタン)
        self.run_skl_button.place(relx=0.45, rely=0.6)
        #配置(フレーム)
        self.lf_labelname_skl.place(relx=0.15, rely=0.1)
        self.lf_filename_skl.place(relx=0.25, rely=0.1)
        self.f_button_skl.place(relx=0.8, rely=0.1)
        self.f1.pack()
    def create_som_widgets(self):
        """somのウィジェット"""
        #フレーム
        self.f1 = tk.LabelFrame(root, relief = 'ridge', text='SOM', labelanchor=tk.N,
                           borderwidth = 4, padx=5, pady=5, font=('times', 30), width=800, height=300)

        #ラベルフレーム
        self.lf_labelname_som = tk.LabelFrame(self.f1, text='ラベル名', labelanchor=tk.NW)
        self.lf_color_som=tk.LabelFrame(self.f1, text='色', labelanchor=tk.NW)
        self.lf_filename_som=tk.LabelFrame(self.f1, text='ファイル名', labelanchor=tk.NW)
        self.lf_button_som=tk.Frame(self.f1)
        self.lf_train_cnt=tk.LabelFrame(self.f1, text='学習回数', labelanchor=tk.NW)

        #ボタン(ファイル選択)
        self.select_file1 = tk.Button(self.lf_button_som, text = 'ファイル選択', relief = 'raised',
                            borderwidth = 4, command = self.select_som_file1).pack()
        self.select_file2 = tk.Button(self.lf_button_som, text = 'ファイル選択', relief = 'raised',
                            borderwidth = 4, command = self.select_som_file2).pack()
        self.select_file3 = tk.Button(self.lf_button_som, text = 'ファイル選択', relief = 'raised',
                            borderwidth = 4, command = self.select_som_file3).pack()
        self.select_file4 = tk.Button(self.lf_button_som, text = 'ファイル選択', relief = 'raised',
                            borderwidth = 4, command = self.select_som_file4).pack()
        #ボタン(実行)
        self.run_som_button = tk.Button(self.f1, text = '実行', relief = 'raised',
                            bg = '#fffafa', fg = '#000000', borderwidth = 4, command = self.run_som)
        #ラベル(ファイル名)
        self.filename1_buff=tk.StringVar()
        self.filename2_buff=tk.StringVar()
        self.filename3_buff=tk.StringVar()
        self.filename4_buff=tk.StringVar()
        self.filename1_label=tk.Label(self.lf_filename_som, textvariable=self.filename1_buff, width=60, relief='ridge').pack()
        self.filename2_label=tk.Label(self.lf_filename_som, textvariable=self.filename2_buff, width=60, relief='ridge').pack()
        self.filename3_label=tk.Label(self.lf_filename_som, textvariable=self.filename3_buff, width=60, relief='ridge').pack()
        self.filename4_label=tk.Label(self.lf_filename_som, textvariable=self.filename4_buff, width=60, relief='ridge').pack()

        #エントリー(ラベル名)
        self.labelname1_buff=tk.StringVar()
        self.labelname2_buff=tk.StringVar()
        self.labelname3_buff=tk.StringVar()
        self.labelname4_buff=tk.StringVar()
        self.labelname1_entry=tk.Entry(self.lf_labelname_som, textvariable=self.labelname1_buff, width=15).pack()
        self.labelname2_entry=tk.Entry(self.lf_labelname_som, textvariable=self.labelname2_buff, width=15).pack()
        self.labelname3_entry=tk.Entry(self.lf_labelname_som, textvariable=self.labelname3_buff, width=15).pack()
        self.labelname4_entry=tk.Entry(self.lf_labelname_som, textvariable=self.labelname4_buff, width=15).pack()
        #エントリー(ラベル名)初期値
        self.labelname1_buff.set('Pkick')
        self.labelname2_buff.set('Run')
        self.labelname3_buff.set('Tackle')
        self.labelname4_buff.set('Pass')
        #エントリー(その他)
        self.e_train_cnt_buff=tk.IntVar()
        self.e_train_cnt_buff.set('400')
        self.e_train_cnt=tk.Entry(self.lf_train_cnt, textvariable=self.e_train_cnt_buff, width=10).pack()

        #ウィジェット配置
        self.lf_labelname_som.place(relx=0.15, rely=0.1)
        self.lf_filename_som.place(relx=0.3, rely=0.1)
        self.lf_button_som.place(relx=0.85, rely=0.1)
        self.lf_train_cnt.place(relx=0.45, rely=0.6)
        self.run_som_button.place(relx=0.45, rely=0.8)
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
        from app.kml.kmlwrapper import KmlWrapper
        filenames2 = self.replace_exts(self.filenames, 'kml')
        #リソース作成
        for i, zip_ in enumerate(zip(self.filenames, filenames2)):
            xl, kml = zip_

            r = open(kml, 'w')
            r.write("")
            r.close()

            # リソース
            #icon_res = icons[random.randint(0, 2)]

            # Excelを読み込む
            from app.util.excelwrapper import ExcelWrapper
            ws = ExcelWrapper(filename=xl).get_sheet('Sheet1')

            # 読み込みを開始する行
            begin_row = 9
            times, lats, lons = ws.iter_cols(('A', 'J', 'K'), row_range=(begin_row, None))
            import os
            print os.getcwd()
            from app.util.iconmaker import drow_circle
            icon = os.path.abspath(drow_circle((255, 0, 0), size=(16, 16), savepath=r'.\tmp\red.png'))
            KmlWrapper().createAnimeKml(kml, times, lons, lats, icon_scale=0.3, sampling_step=10, icon_res=icon)

            print "completed!"

            #GoogleEarthで表示
            sb.Popen(["C:\Program Files (x86)\Google\Google Earth\client\googleearth.exe", kml])

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
        iDir=r'E:\Eclipse\pleiades\workspace\Sotsugyo_kenkyu\res\data'
        filenameA=tkfd.askopenfilename(filetypes=fTyp_xlsx, initialdir=iDir)
        #self.filename=filename1
        self.filenameA_buff.set(filenameA)
    def select_svm_fileB(self):
        fTyp_xlsx=[('Excelファイル', '*.xlsx')]
        iDir=r'E:\Eclipse\pleiades\workspace\Sotsugyo_kenkyu\res\data'
        filenameB=tkfd.askopenfilename(filetypes=fTyp_xlsx, initialdir=iDir)
        #self.filename=filename2
        self.filenameB_buff.set(filenameB)
    def select_svm_fileC(self):
        fTyp_xlsx=[('Excelファイル', '*.xlsx')]
        iDir=r'E:\Eclipse\pleiades\workspace\Sotsugyo_kenkyu\res\data'
        filenameC=tkfd.askopenfilename(filetypes=fTyp_xlsx, initialdir=iDir)
        #self.filename=filename3
        self.filenameC_buff.set(filenameC)
    def select_svm_fileD(self):
        fTyp_xlsx=[('Excelファイル', '*.xlsx')]
        iDir=r'E:\Eclipse\pleiades\workspace\Sotsugyo_kenkyu\res\data'
        filenameD=tkfd.askopenfilename(filetypes=fTyp_xlsx, initialdir=iDir)
        #self.filename=filename3
        self.filenameD_buff.set(filenameD)
    def run_scikit_learn(self):

        from app.util.inputmaker import make_input
        from collections import namedtuple
        from sklearn.metrics import confusion_matrix
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import classification_report
        from sklearn.externals import joblib
        import numpy as np
        from app import R, T, L

        #ファイル名取得
        file1=self.filenameA_buff.get()
        file2=self.filenameB_buff.get()
        file3=self.filenameC_buff.get()
        file4=self.filenameD_buff.get()
        label1=self.e_ln1_skl_buff.get()
        label2=self.e_ln2_skl_buff.get()
        label3=self.e_ln3_skl_buff.get()
        label4=self.e_ln4_skl_buff.get()

        """
        教師データ生成
        """
        Xl = namedtuple('Xl', 'filename, label')
        xls =  (
             Xl(R(file1), label1),
             Xl(R(file2), label2),
             Xl(R(file3), label3),
             Xl(R(file4), label4)
            )
        input_vecs = []
        input_labels = []
        for xl in xls:
            input_vec, labels = make_input(xlsx=xl.filename, sheetnames=None,col=None,
                                                    min_row=2,fft_N=128, sample_cnt=80,
                                                    label=xl.label,normalizing='01', log=False)
            map(input_vecs.append, input_vec)
            input_labels += labels

        from app.util.inputmaker import random_input_iter
        input_vecs1, input_labels1 = [], []
        for i, j in random_input_iter(input_vecs, input_labels):
            input_vecs1.append(i)
            input_labels1.append(j)

        """
        tmp = np.c_[input_vec, labels]
        random.shuffle(tmp)
        input_vec = tmp[:, :-1]
        labels  = tmp[:, -1]
        #labels = [vec[0] for vec in input_data]
        #vecs = [list(vec[1]) for vec in input_data]
        """
        """
        テストデータ生成
        """
        test_vecs = []
        test_labels = []
        for xl in xls:
            test_vec, test_label = make_input(xlsx=xl.filename, sheetnames=None,col=None,
                                                    min_row=128*80+1,fft_N=128, sample_cnt=20,
                                                    label=xl.label,normalizing='01', log=False)
            map(test_vecs.append, test_vec)
            test_labels += test_label

        test_vecs1, test_labels1 = [], []
        for i, j in random_input_iter(test_vecs, test_labels):
            test_vecs1.append(i)
            test_labels1.append(j)

        print "input_vec_len    :", len(input_vecs)
        #print "input_vec_shape  :", input_vecs.shape
        print "labels_len       :", len(input_labels)
        print "test_vec_len     :", len(test_vecs)
        #print "test_vec_shape   :", test_vecs.shape
        print "test_labels      :", len(test_labels)

        """
        tmpt = np.c_[test_vec, test_labels]
        random.shuffle(tmpt)
        test_vec = tmpt[:, :-1]
        test_labels  = tmpt[:, -1]
        #test_labels = [vec[0] for vec in test_data]
        #test_vecs = [list(vec[1]) for vec in test_data]
        """
        """
        教師データの学習分類
        """
        # test_gridsearchを参照
        est = SVC(C=1000, kernel='rbf', gamma = 0.001)    # パラメータ (C-SVC, RBF カーネル, C=1000)
        clf = OneVsRestClassifier(est)  #他クラス分類器One-against-restによる識別
        clf.fit(input_vecs1, input_labels1)
        test_pred = clf.predict(test_vecs1)

        clf2 = SVC(C=1000, kernel='rbf', gamma = 0.001)    # パラメータ (C-SVC, RBF カーネル, C=1000)
        clf2.fit(input_vecs1, input_labels1)
        test_pred2 = clf2.predict(test_vecs1)  #他クラス分類器One-versus-oneによる識別

        """
        学習モデルのローカル保存
        """
        #joblib.dump(clf, R('misc\model\clf.pkl'))
        #joblib.dump(clf2, R('misc\model\clf2.pkl'))

        #confusion matrix（ラベルの分類表。分類性能が高いほど対角線に値が集まる）
        print confusion_matrix(test_labels1, test_pred)
        print confusion_matrix(test_labels1, test_pred2)

        #分類結果 適合率 再現率 F値の表示
        print classification_report(test_labels1, test_pred)
        print classification_report(test_labels1, test_pred2)

        #正答率 分類ラベル/正解ラベル
        print accuracy_score(test_labels1, test_pred)
        print accuracy_score(test_labels1, test_pred2)

        print test_labels1       #分類前ラベル
        print list(test_pred)   #One-against-restによる識別ラベル
        print list(test_pred2)  #One-versus-oneによる識別ラベル

        """
        target_names = ['class 0', 'class 1', 'class 2']
        print(classification_report(test_labels, test_pred, target_names=target_names))
        """



    def select_som_file1(self): #som関連
        fTyp_xlsx=[('jsonファイル', '*.json')]
        iDir=r'E:\Eclipse\pleiades\workspace\Sotsugyo_kenkyu\res\data'
        filename1=tkfd.askopenfilename(filetypes=fTyp_xlsx, initialdir=iDir)
        #self.filename=filename3
        self.filename1_buff.set(filename1)
    def select_som_file2(self):
        fTyp_xlsx=[('jsonファイル', '*.json')]
        iDir=r'E:\Eclipse\pleiades\workspace\Sotsugyo_kenkyu\res\data'
        filename2=tkfd.askopenfilename(filetypes=fTyp_xlsx, initialdir=iDir)
        #self.filename=filename3
        self.filename2_buff.set(filename2)
    def select_som_file3(self):
        fTyp_xlsx=[('jsonファイル', '*.json')]
        iDir=r'E:\Eclipse\pleiades\workspace\Sotsugyo_kenkyu\res\data'
        filename3=tkfd.askopenfilename(filetypes=fTyp_xlsx, initialdir=iDir)
        #self.filename=filename3
        self.filename3_buff.set(filename3)
    def select_som_file4(self):
        fTyp_xlsx=[('jsonファイル', '*.json')]
        iDir=r'E:\Eclipse\pleiades\workspace\Sotsugyo_kenkyu\res\data'
        filename4=tkfd.askopenfilename(filetypes=fTyp_xlsx, initialdir=iDir)
        #self.filename=filename3
        self.filename4_buff.set(filename4)
    def run_som(self):
        from collections import namedtuple
        from app.util.inputmaker import make_input
        from modsom import SOM
        from app import R, T, L
        import matplotlib.pyplot as plt
        import json
        import numpy as np
        import matplotlib.cm as cm
        from itertools import chain, product, izip
        from app.util.jsonio import iter_inputs_json

        def som_json(jsons, labels, label_colors=None, train_cnt=50, mapsize=None):
            """jsonからsom

            :param jsons : list of str
                特徴ベクトルのみのjsonのパスのリスト

            :param labels : list
                マップに表示するそれぞれのjsonに対応するラベル

            :param label_colors : dict, None
                ラベルに対応するマップに表示する際の文字色
            """

            inputs = []
            for j, label in zip(jsons, labels):
                _, f_iter  = iter_inputs_json(j, True)
                inputs = chain(inputs, product(f_iter, [label]))

            if label_colors is None:
                len_ = len(labels)
                label_colors = {str(l): cm.autumn(float(i)/len_)
                                for i, l in enumerate(labels)}

            features, labels = zip(*inputs)
            features = np.array(features)
            labels = list(labels)

            if mapsize is None:
                som = SOM(features, labels, display='um')
            else:
                som = SOM(features, labels, shape=mapsize, display='um')

            map_, labels, coords = som.train(train_cnt)
            plt.imshow(map_, interpolation='nearest', cmap='gray')
            for label, coord in zip(labels, coords):
                x, y = coord
                plt.text(x, y, label, color=label_colors[label])
            plt.show()

            from app import R

            file1=self.filename1_buff.get()
            file2=self.filename2_buff.get()
            file3=self.filename3_buff.get()
            file4=self.filename4_buff.get()
            ln1=self.labelname1_buff.get()
            ln2=self.labelname2_buff.get()
            ln3=self.labelname3_buff.get()
            ln4=self.labelname4_buff.get()
            e_tc=self.e_train_cnt_buff.get()
            som_json([
                R(file1),
                R(file2),
                R(file3),
                R(file4)
                ],
                [ln1, ln2, ln3, ln4], train_cnt=e_tc)



if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("800x400")
    f = Frame(root)
    f.pack()
    f.mainloop()

