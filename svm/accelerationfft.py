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

def fft(acc, fft_points):
    """
    Parameters
    -----------
    acc : array or list
    fft_points : int
    """
    # リストなら配列に変換
    if isinstance(acc, list):
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

    return fftmag

if __name__ == '__main__':

    """
    # Excelシート読み込み
    from util.excelwrapper import ExcelWrapper
    ws = ExcelWrapper(filename=r"E:\work\data\walk.xlsx",
                      sheetname='Sheet4')

    col_letter = 'F'
    fft_points = 256

    begin_row = 2
    end_row = begin_row + fft_points - 1

    for i in xrange(5):
        # 加速度のリストをxlsxから読み込む
        acc = ws.select_column(col_letter=col_letter,
                               begin_row=begin_row,
                               end_row=end_row)

        # リストを配列に変換
        acc = np.array(acc)

        # FFT
        fft(acc, fft_points)

        begin_row += fft_points
        end_row = begin_row + fft_points - 1

        # 図を保存
        plt.savefig(r"E:\work\fig\walk_fig%03d.png" % i, dpi=200)

    print "finish"
    """

    """
    ここからSOM処理
    """

    from util.excelwrapper import ExcelWrapper
    col_letter = 'F'
    fft_points = 256
    begin_row = 2
    end_row = begin_row + fft_points - 1

    # FFT結果のリスト
    ffts = []

    # walkのFFT
    ws_walk = ExcelWrapper(filename=r"E:\work\data\walk.xlsx",
                           sheetname='Sheet4')

    acc_walk = ws_walk.select_column(col_letter=col_letter,
                                begin_row=begin_row,
                                end_row=end_row)

    ffts.append(fft(acc_walk, fft_points))

    # runのFFT
    ws_run = ExcelWrapper(filename=r"E:\work\data\run.xlsx",
                          sheetname='Sheet4')

    acc_run = ws_run.select_column(col_letter=col_letter,
                                begin_row=begin_row,
                                end_row=end_row)

    ffts.append(fft(acc_run, fft_points))

    # skipのFFT
    ws_skip = ExcelWrapper(filename=r"E:\work\data\skip.xlsx",
                           sheetname='Sheet4')

    acc_skip = ws_skip.select_column(col_letter=col_letter,
                                begin_row=begin_row,
                                end_row=end_row)

    ffts.append(fft(acc_skip, fft_points))

    from sompy import SOM

    # 入力ベクトル
    #input_data = np.random.rand(3, 256)
    input_data = np.array(ffts, np.float32)
    print "input shape:", input_data.shape
    print "input_data_type:", input_data.dtype
    print input_data

    # データを正規化?

    # 出力するマップのサイズ
    output_shape = (40, 40)

    # SOMインスタンス
    som = SOM(output_shape, input_data)

    # SOMのパラメータを設定
    # neighborは近傍の比率:初期値0.25、learning_rateは学習率:初期値0.1
    som.set_parameter(neighbor=0.26, learning_rate=0.22)

    # 学習と出力マップの取得
    # 引数は学習ループの回数
    output_map = som.train(2000)
    print "output shape:", output_map.shape

    plt.imshow(output_map, interpolation='none')
    plt.show()