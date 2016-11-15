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

def fft(arr, fft_points, window='haning', out_fig=False):
    """
    Parameters
    ----------
    arr : ndarray or list

    fft_points : int

    window : str
        窓関数を指定
        'hunning'
        'humming'
        'blackman'

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

    #arr -= np.mean(arr) # DC成分を除去

    fs = fft_points # サンプリング周波数(FFTのポイント数)

    #arr = np.hstack((arr, np.zeros(fs - arr.size))) # fsとの差を0で埋める

    assert fs == arr.size, """
    size of arr and fft_points must be same value.
    fft_points:%d, arr_size:%d""" % (fft_points, arr.size)

    wind_arr = None
    if window == 'hunning':
        wind_arr = __apply_hunningwin(arr, fs) # ハニング窓を適用
    elif window == 'humming':
        wind_arr = __apply_hummingwin(arr, fs) # ハミング窓を適用
    elif window == 'blackman':
        wind_arr = __apply_blackmanwin(arr, fs) # ブラックマン窓を適用
    else:
        raise ValueError("windowの値が不正です")


    ps = 1/float(fs) # サンプリング間隔(サンプリング周波数の逆数)
    fftdata = np.fft.fft(wind_arr) # FFT
    fftdata /= fs / 2 # 正規化
    fftfreq = np.fft.fftfreq(fs, ps)[0:fs/2] # X軸 - 周波数
    fftmag = np.abs(fftdata[0:fs/2]) # Y軸 - スペクトルの絶対値

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
        ax2.set_title("%s_window" % window)
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

    w = 'hunning'
    fftmag, fig = fft(arr=y, fft_points=N, out_fig=True, window=w)
    print fftmag
    fig.savefig(r"E:\work\fig\sin\sin_{}.png".format(w))