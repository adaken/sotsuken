# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
plt.hold(False)

def fftn(arrs, fft_N, wf='hanning', savepath=None):
    """高速フーリエ変換を行う

    2D配列の場合はfft_iter()より高速

    :param arrs : ndarray, list
        FFTする配列
        2Dndarray対応

    :param fft_N : int
        FFTポイント数

    :param wf : str, default: 'hanning'
        使用する窓関数
        'hanning', 'hamming', 'blackman'

    :param savepath : str, default: None
        指定したパスにグラフを保存する
        arrsが2Dの場合は最初の配列のみグラフを保存

    :return fftmags : ndarray
        FFTされた配列(絶対値)
        配列1つの長さはfft_N/2になる

    """

    assert isinstance(arrs, (list, np.ndarray))

    if isinstance(arrs, list):
        arrs = np.array(arrs)

    assert arrs.ndim in (1, 2)
    assert isinstance(fft_N, int)
    assert isinstance(wf, str)
    assert wf in ('hanning', 'hamming', 'blackman')

    arrs.astype(np.float32) # 配列の型を変換

    dim = arrs.ndim

    #arrs -= np.mean(arrs) # DC成分を除去

    # 窓関数を適用
    w_arrs = None
    if   wf == 'hanning':
        w_arrs = arrs * np.hanning(fft_N)
    elif wf == 'hamming':
        w_arrs = arrs * np.hamming(fft_N)
    elif wf == 'blackman':
        w_arrs = arrs * np.blackman(fft_N)

    sp = 1 / float(fft_N)                         # サンプリング間隔
    fftdata = np.fft.fft(w_arrs)                  # FFT
    fftdata /= fft_N / 2                          # 正規化
    fftfreq = np.fft.fftfreq(fft_N, sp)[:fft_N/2] # 周波数のリスト
    fftmags = None                                # スペクトルの絶対値
    if dim == 1:
        fftmags = np.abs(fftdata[:fft_N/2])
    else:
        fftmags = np.abs(fftdata[:, :fft_N/2])

    if savepath is not None:
        a, w, f = [None] * 3
        if dim == 1:
            a, w, f = arrs, w_arrs, fftmags
        else:
            a, w, f = arrs[0], w_arrs[0], fftmags[0]

        # グラフ
        x = np.arange(0, fft_N)
        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)

        # 元の波形
        ax1.plot(x, a)
        ax1.set_title("original")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        # 窓関数を適用した波形
        ax2.plot(x, w)
        ax2.set_title("%s_window" % wf)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")

        # FFTした波形
        ax3.plot(fftfreq, f)
        ax3.set_title("fft")
        ax3.set_xlabel("freq[Hz]")
        ax3.set_ylabel("amp")

        fig.tight_layout()
        fig.savefig(savepath)

    return fftmags

def fft_iter(arrs, fft_N, wf='hunning'):
    """FFTした配列を返すイテレータ

    :param arrs : iterable

    :param fft_N : int

    :return fft_iter : iterater

    """

    assert hasattr(arrs, '__iter__')
    assert isinstance(fft_N, int)

    for a in arrs:
        yield fftn(arrs=a, fft_N=fft_N, wf=wf)

if __name__ == '__main__':

    # テスト用sin波形

    N = 256
    x = np.linspace(0, 1, N)
    f = 20
    y = 5 * np.sin(2*np.pi * x * 20) + 3 * np.sin(2*np.pi * x * 50) + 10 * np.sin(2*np.pi * x * 100) + 1 * np.sin(2*np.pi * x * 40)

    arrs = np.array([y, y])
    print fftn(arrs, N, wf='hanning', savepath=r'E:\fft_result.png')

    """
    print x.size
    plt.plot(y)
    #plt.show()

    w = 'hunning'
    fftmag, fig = fft(arr=y, fft_points=N, out_fig=True, window=w)
    print fftmag
    fig.savefig(r"E:\work\fig\sin\sin_{}.png".format(w))
    """