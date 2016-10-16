# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
#import sklearn.svm as svm

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

def fft(sheet, col_letter, begin_row, end_row, fft_points):

    # 加速度のリストをxlsxから読み込む
    acc = ws.select_column(col_letter=col_letter, begin_row=begin_row,
                           end_row=end_row)

    # リストを配列に変換
    acc = np.array(acc)

    # DC成分を取り除く
    acc = acc - np.mean(acc)

    # サンプリング周波数(FFTのポイント数)
    fs = fft_points

    # サンプリング間隔(サンプリング周波数の逆数)
    ps = 1/float(fs)

    # fsとの差を0で埋める
    #acc = np.hstack((acc, np.zeros(fs - acc.size)))
    #print "acc:", acc
    print "acc_size:", acc.size

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
    #print "X軸:", fftfreq

    # FFT
    fftdata = np.fft.fft(acc)[0:n/2]

    # Y軸 - スペクトルの絶対値
    fftmag = np.abs(fftdata)
    #print "Y軸:", fftmag

    # スペクトルを描画
    plt.subplot(223)  # 2行2列のグラフの3番目の位置にプロット
    plt.plot(xrange(0, fftmag.size), fftmag)
    #plt.axis([0, fftmag.size, 0, np.max(fftmag)])
    plt.axis([0, fftmag.size, 0, 100])
    plt.xlabel("freq")
    plt.ylabel("spectrum")

    # 図をグラフとして保存
    #plt.savefig(r"E:\work\fig.png", dpi=200)

    # 図を表示
    #plt.show()

    return fftmag

if __name__ == '__main__':
    from util.excelwrapper import ExcelWrapper
    ws = ExcelWrapper(filename=r"E:\work\jump_run_acc_hirano.xlsx",
                      sheetname='Sheet5')

    col_letter = 'F'
    fft_points = 256
    begin_row = 7800
    end_row = begin_row + fft_points - 1

    for i in xrange(5):
        fft(sheet=ws, col_letter=col_letter, begin_row=begin_row,
             end_row=end_row, fft_points=fft_points)
        begin_row += fft_points
        end_row = begin_row + fft_points - 1
        
        # 図を保存
        plt.savefig(r"E:\work\fig\fig%03d.png" % i, dpi=200)

    """
    ここからSOM処理
    """

    