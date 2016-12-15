# -*- coding: utf-8 -*-

import os
import Tkinter as Tk
import sys
import tkFileDialog as tkfd

class Application(Tk.Frame):
    def __init__(self, master=None):
        Tk.Frame.__init__(self, master)
        self.create_test_widgets()
        self.pack()
        
    def create_test_widgets(self):
        #var
        self.var_entry=Tk.StringVar()
        self.var_entry.trace('w', self.entry_changed)
        self.var_check=Tk.BooleanVar()
        
        #ウィジェット
        self.label = Tk.Label(self, text=u'入力ファイル')
        self.entry = Tk.Entry(self, textvariable=self.var_entry)
        self.button = Tk.Button(self, text=u'開く', command=self.button_pushed)
        self.check = Tk.Checkbutton(self, text=u'拡張子をtxtに限定',
                                    variable=self.var_check)
        self.text = Tk.Text(self)
        #ウィジェット配置
        self.label.grid(column=0, row=0)
        self.entry.grid(column=1, row=0)
        self.button.grid(column=2, row=0)
        self.check.grid(column=0, row=1)
        self.text.grid(column=0, columnspan=3, row=2)
        
    def button_pushed(self):
        self.var_entry.set(u'ボタンが押されました。')
        
    def entry_changed(self, *args):
        if os.path.exists(self.var_entry.get()):
            self.text.delete('1.0', Tk.END)
            self.text.insert('1,0', open(self.var_entry.get()).read())





    
root = Tk.Tk()
app = Application(master=root)
app.mainloop()