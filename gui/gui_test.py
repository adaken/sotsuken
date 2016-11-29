# coding: utf-8
if __name__ == '__main__':
    import sys
    import os.path
    import Tkinter as tk
    import tkFileDialog as tkfd

    root = tk.Tk()
    root.title("test")
    root.geometry("400x300")


# フレームの生成
f0 = Frame(root)
f1 = Frame(root)

# f0 にボタンを配置する
Button(f0, text = 'button 00').pack(side = LEFT)
Button(f0, text = 'button 01').pack(side = LEFT)
Button(f0, text = 'button 02').pack(side = LEFT)

# f1 にボタンを配置する
Button(root, text = 'button 10').pack(in_ = f1, fill = BOTH)
Button(root, text = 'button 11').pack(in_ = f1, fill = BOTH)
Button(root, text = 'button 12').pack(in_ = f1, fill = BOTH)

# フレームの配置
f0.pack()
f1.pack(fill = BOTH)
    #エントリー
    #editbox = tk.Entry(root, textvariable = buffer, width = 50)
    #editbox.pack()
    #editbox.focus_set()

    label = tk.Label()
    label.pack()

    fTyp = ['Excelファイル', '*.csv']

    iDir = 'c:/'

    dirname = tkfd.askdirectory(initialdir=iDir)

    #ファイル選択
    def load_file():
        global path_name
        filename = tkfd.askopenfilename(filetypes=fTyp, initialdir=iDir)
        label.set(os.path.dirname(filename))

    button1 = tk.Button(root, text = '参照', width = 10, command = load_file)
    button1.pack()


    def
    #button2 = tk.Button(root, text = '', width = 10, command = )
    #button2.pack

    #button.place() ボタン配置
    #数値を変更・入力できるエントリー
    #FFT,SOM,SVM

○entry widgetの場合

self.<entry>.get()

・・・引数なし。



○text widgetの場合

sel.<text>.get(<index_start>,<index_end>=None)

・・・どこからどこまで取得したいかを指定する。
    root.mainloop()