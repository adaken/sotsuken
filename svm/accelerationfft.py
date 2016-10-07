# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from svm import *

def apply_hunningwin(a, fs):
    """
    ハニング窓
    """
    hanning_window = np.hanning(fs)
    return a * hanning_window

def apply_hummingwin(a, fs):
    """
    ハミング窓
    """
    hamming_window = np.hamming(fs)
    return a * hamming_window

def apply_blackmanwin(a, fs):
    """
    ブラックマン窓
    """
    blackman_window = np.blackman(fs)
    return a * blackman_window

if __name__ == '__main__':

    # 加速度の配列をxlsxから読み込む
    from util.excelwrapper import ExcelWrapper
    ws = ExcelWrapper(filename=r'E:\work\fft_testdata.xlsx',
                       sheetname='Sheet1')
    acc = ws.select_column('D', 2)

    # リストを配列に変換
    acc = np.array(acc)
    print type(acc)

    """
    # xの値を生成
    x = np.arange(-3.14, 3.14, 0.25/40.764)
    y = np.arange(-3.14, 3.14, 0.25/40.764)

    # 高さを計算
    sin1 = np.sin(3*x)
    sin2 = np.sin(7*y)
    print sin1
    print sin2

    acc = sin1 + sin2
    """
    # DC成分を取り除く
    acc = acc - np.mean(acc)

    # サンプリング周波数
    fs = 1024

    # サンプリング間隔(サンプリング周波数の逆数)
    ps = 1/float(fs)

    # fsとの差を0で埋める
    #acc = np.hstack((acc, np.zeros(fs - acc.size)))
    print "acc:", acc
    print "size:", acc.size

    # 波形を描画（窓関数なし）
    plt.subplot(221)  # 2行2列のグラフの1番目の位置にプロット
    plt.plot(xrange(0, fs), acc)
    plt.axis([0, fs, -5.0, 5.0])
    plt.xlabel("time")
    plt.ylabel("amplitude")

    # ハニング窓を適用
    acc = apply_hunningwin(acc, fs)

    # ハミング窓を適用
    #acc = apply_hummingwin(acc, fs)

    # ブラックマン窓を適用
    #acc = apply_blackmanwin(acc, fs)

    # X軸の長さ
    n = acc.size

    # 波形を描画（窓関数あり）
    plt.subplot(222)  # 2行2列のグラフの2番目の位置にプロット
    plt.plot(xrange(0, fs), acc)
    plt.axis([0, fs, -5.0, 5.0])
    plt.xlabel("time")
    plt.ylabel("amplitude")

    # X軸 - 周波数
    fftfreq = np.fft.fftfreq(n, ps)
    print "X軸:", fftfreq

    # Y軸 - スペクトル
    fftdata = np.fft.fft(acc)[0:n/2]

    # スペクトルの絶対値
    fftmag = np.abs(fftdata)
    print "Y軸:", fftmag

    # スペクトルを描画
    plt.subplot(223)  # 2行2列のグラフの3番目の位置にプロット
    plt.plot(xrange(0, fftmag.size), fftmag)
    plt.axis([0, fftmag.size, 0, np.max(fftmag)])
    plt.xlabel("freq")
    plt.ylabel("spectrum")


    # 図をグラフとして保存
    #fig.savefig(r"E:\work\fft_graph.png", dpi=200)

    # 図を表示
    plt.show()

