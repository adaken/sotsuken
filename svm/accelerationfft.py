# coding: utf-8

if __name__ == '__main__':

    """
    # Excelシート読み込み
    from util.excelwrapper import ExcelWrapper
    ws = ExcelWrapper(filename=r"E:\work\data\skip.xlsx",
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

        from fft import fft
        # FFT
        fftmag, fig = fft(arr=acc, fft_points=fft_points, out_fig=True)

        begin_row += fft_points
        end_row = begin_row + fft_points - 1

        # 図を保存
        fig.savefig(r"E:\work\fig\skip_fig%03d.png" % i)

    print "finish"
    """

    """
    ここからSOM処理
    """

    from util.excelwrapper import ExcelWrapper
    from fft import fft
    import numpy as np
    import matplotlib.pyplot as plt

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