# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from svm import *

if __name__ == '__main__':

    # 加速度の配列をxlsxから読み込む
    from util.excelwrapper import ExcelWrapper
    ws = ExcelWrapper(filename=r'E:\work\fft_testdata.xlsx',
                       sheetname='Sheet1')
    acc = ws.select_column('A', 2)

    """
    # xの値を生成
    x = np.arange(-3.14, 3.14, 0.25)
    y = np.arange(-3.14, 3.14, 0.25)
    # 高さを計算
    sin1 = np.sin(x)
    sin2 = np.sin(5*y)
    print sin1
    print sin2

    acc = sin1 + sin2 # sin(x)の計算
    """
    # サンプリング周波数
    fs = 1024

    # サンプリング間隔(サンプリング周波数の逆数)
    ps = 1/float(fs)

    # X軸の長さ
    n = len(acc)

    # X軸 - 周波数
    fftfreq = np.fft.fftfreq(n, ps)
    print fftfreq

    # Y軸 - スペクトル
    fftdata = np.fft.fft(acc)[0:n/2]

    # スペクトルの絶対値
    fftmag = np.abs(fftdata)
    print fftmag

    # 図の作成
    fig, panel = plt.subplots(1, 1)
    print fig, panel
    panel.set_xlim(0, fftfreq[:len(fftmag)][-1])
    panel.plot(fftfreq[:len(fftmag)], fftmag)

    # 図をグラフとして保存
    #fig.savefig(r"E:\work\fft_graph2.png", dpi=200)

    # 図を表示
    plt.show()

