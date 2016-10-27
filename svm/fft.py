# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

def __apply_hunningwin(a, fs):
    """
    ハニング窓
    """
    hanning_window = np.hanning(fs)
    return a * hanning_window

def __apply_hummingwin(a, fs):
    """
    ハミング窓
    """
    hamming_window = np.hamming(fs)
    return a * hamming_window

def __apply_blackmanwin(a, fs):
    """
    ブラックマン窓
    """
    blackman_window = np.blackman(fs)
    return a * blackman_window

def fft(arr, fft_points, out_fig=False):
    """
    Parameters
    ----------
    arr : ndarray or list

    fft_points : int

    out_fig : bool, optional, default: False
        グラフを出力するかどうか

    Return
    ------
    fftmag : ndarray
        FFTされた信号の絶対値の配列

    fig : matplotlib.pyplot.Figure
        信号のグラフ
    """

    # リストなら配列に変換
    if isinstance(arr, list):
        arr = np.array(arr)

    # DC成分を除去
    #arr -= np.mean(arr)

    # サンプリング周波数(FFTのポイント数)
    fs = fft_points

    # fsとの差を0で埋める
    #arr = np.hstack((arr, np.zeros(fs - arr.size)))

    assert fs == arr.size, """
    size of arr and fft_points must be same value.
    fft_points:%d, arr_size:%d""" % (fft_points, arr.size)

    # ハニング窓を適用
    wind_arr = __apply_hunningwin(arr, fs)

    # ハミング窓を適用
    #wind_arr = apply_hummingwin(arr, fs)

    # ブラックマン窓を適用
    #wind_arr = apply_blackmanwin(arr, fs)

    # サンプリング間隔(サンプリング周波数の逆数)
    ps = 1/float(fs)

    # FFT
    fftdata = np.fft.fft(wind_arr)

    # 正規化
    fftdata /= fs / 2

    # X軸 - 周波数
    fftfreq = np.fft.fftfreq(fs, ps)[0:fs/2]

    # Y軸 - スペクトルの絶対値
    fftmag = np.abs(fftdata[0:fs/2])

    if out_fig:

        # グラフ
        x = np.arange(0, fs)
        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)

        # 元の波形
        ax1.plot(x, arr)
        ax1.set_title("original")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        #ax1.set_ylim(arr.min(), arr.max())

        # 窓関数を適用した波形
        ax2.plot(x, wind_arr)
        ax2.set_title("window_func")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        #ax2.set_ylim(wind_arr.min(), wind_arr.max())

        # FFTした波形
        ax3.plot(fftfreq, fftmag)
        ax3.set_title("fft")
        ax3.set_xlabel("freq[Hz]")
        ax3.set_ylabel("amp")
        #ax3.set_ylim(fftmag.min(), fftmag.max())

        fig.tight_layout()

        return fftmag, fig

    return fftmag

if __name__ == '__main__':

    # テスト用sin波形

    N = 256
    x = np.linspace(0, 1, N)
    f = 20
    y = 5 * np.sin(2*np.pi * x * 20) + 3 * np.sin(2*np.pi * x * 50) + 10 * np.sin(2*np.pi * x * 100)
    print x.size
    plt.plot(y)
    #plt.show()

    fftmag, fig = fft(arr=y, fft_points=N, out_fig=True)
    print fftmag
    fig.savefig(r"E:\work\fig\sin\sin.png")