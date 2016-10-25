# coding: utf-8
if __name__ == '__main__':
    import sys
    import os.path
    import Tkinter as tk
    import tkFileDialog as tkfd

    root = tk.Tk()
    root.title("test")
    root.geometry("400x300")

    path_name = ""

    buffer = tk.StringVar()
    buffer.set("")

    def load_text():
        if buffer.get():
            value = eval(buffer.get())
            buffer.set(str(value))

    #エントリー
    #editbox = tk.Entry(root, textvariable = buffer, width = 50)
    #editbox.pack()
    #editbox.focus_set()

    label = tk.Label()
    label.pack()
    #ファイル選択
    def load_file():
        global path_name
        filename = tkfd.askopenfilename()
        label.set(os.path.dirname(filename))

    button1 = tk.Button(root, text = '参照', width = 10, command = load_file)
    button1.pack()
    #button2 = tk.Button(root, text = '', width = 10, command = )
    #button2.pack

    #button.place()

    root.mainloop()