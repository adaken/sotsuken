# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
plt.hold(False)

def fftn(arrs, fft_N, wf='hanning', savepath=None, fs=None, freq=False):
    """高速フーリエ変換を行う

    2D配列の場合はfft_iter()より高速

    :param arrs : ndarray or list of ndarrays
        FFTする配列
        2Dndarray対応

    :param fft_N : int
        FFTポイント数

    :param wf : str, default: 'hanning'
        使用する窓関数
        Noneは窓関数を使用しない
        'hanning', 'hamming', 'blackman', None

    :param savepath : str, default: None
        指定したパスにグラフを保存する
        arrsが2Dの場合は最初の配列のみグラフを保存

    :param fs : float, default: None
        サンプリング周波数
        Noneの場合は'arrs'の長さと同じ

    :param frec : bool, default: False
        Trueの場合は戻り値に周波数リストが加わる

    :return fftmags : ndarray
        FFTされた配列(絶対値)
        配列1つの長さはfft_N/2-1になる

    :return fftfreq : ndarray
        周波数スペクトルのX軸

    """

    assert isinstance(arrs, (list, np.ndarray))

    if isinstance(arrs, list):
        arrs = np.array(arrs)

    assert arrs.ndim in (1, 2)
    assert wf in ('hanning', 'hamming', 'blackman', None)

    arrs.astype(np.float64) # 配列の型を変換

    if fs is None:
        fs = float(arrs.shape[-1])
    fn = fs / 2    # ナイキスト周波数

    #arrs -= np.mean(arrs) # DC成分を除去

    # 窓関数を適用
    if   wf == 'hanning':
        w_arrs = arrs * np.hanning(arrs.shape[-1])
    elif wf == 'hamming':
        w_arrs = arrs * np.hamming(arrs.shape[-1])
    elif wf == 'blackman':
        w_arrs = arrs * np.blackman(arrs.shape[-1])
    else:
        w_arrs = arrs
        wf = 'non'

    print "sampling rate  : {}Hz".format(fs)
    print "nyquist freq   : {}Hz".format(fn)
    print "freq resolution: {}Hz".format(fs/float(fft_N))
    print "fft point  :", fft_N
    print "window func:", wf

    fftdata = np.fft.fft(w_arrs, fft_N) # FFT
    fftmags = np.abs(fftdata)  # パワースペクトル
    #fftmags /= fft_N          # 正規化
    fftfreq = fs / float(fft_N) * np.arange(fft_N) # 周波数
    fftfreq = fftfreq[1:fft_N/2]

    if arrs.ndim == 1:
        fftmags = fftmags[1:fft_N/2] # 0番目はDC成分なので除く
    else:
        fftmags = fftmags[:, 1:fft_N/2]

    if savepath is not None:
        if arrs.ndim == 1:
            a, w, m = arrs, w_arrs, fftmags
        else:
            a, w, m = arrs[0], w_arrs[0], fftmags[0]

        # グラフ
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        x = np.arange(0, arrs.shape[-1])
        lim = [None, arrs.shape[-1], None, None]

        # 元の波形
        ax1.plot(x, a)
        ax1.grid()
        ax1.set_title("Original")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Amplitude")
        ax1.axis(lim)

        # 窓関数を適用した波形
        ax2.plot(x, w)
        ax2.grid()
        ax2.set_title("{} window".format(wf.capitalize()))
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Amplitude")
        ax2.axis(lim)

        # FFTした波形
        ax3.plot(fftfreq, m)
        ax3.grid()
        ax3.set_title("FFT")
        ax3.set_xlabel("Freqency[Hz]")
        ax3.set_ylabel("Power")
        ax3.axis([None, fn, None, None])

        fig.tight_layout()
        fig.savefig(savepath)
    return (fftmags, fftfreq) if freq else fftmags

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
    from app import R, T, L

    def func1():
        x = 3 * np.sin(2*np.pi*20) * np.linspace(0, 1, 100)

    def fucn2():
        k = 1000
        fs = 44.1*k # サンプリング周波数
        ts = 1 / fs # サンプリング周期
        t = 110 # データ数
        N = 4096 # fftポイント
        amp = np.array([2, 3, 6, 10])[:, np.newaxis] # 振幅
        af = np.array([20*k, 10*k, 5*k, fs*k-15*k])[:, np.newaxis] * 2*np.pi # 角周波数
        # f(t) は sin
        ft = np.sum(amp * np.vectorize(np.sin)(af * np.arange(0, t*ts, ts)), axis=0)
        m = fftn(ft, N, savepath=L('fft_test_result_rmdc.png'), fs=fs)
    fucn2()
