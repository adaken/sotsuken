# coding: utf-8

from util.excelwrapper import ExcelWrapper
from fft import fft
import numpy as np
import matplotlib.pyplot as plt
from sompy import SOM

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
    # ここからSOM処理

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
    """

    def som_test():

        # xlsxの辞書
        xls = {'run':r'E:\work\data\run.xlsx',
               'walk':r'E:\work\data\walk.xlsx',
               'skip':r'E:\work\data\skip.xlsx'}
        sheet_name = 'Sheet4'

        fft_points = 256
        column_letter = 'F'
        begin_row = 2
        end_row = lambda begin : begin + fft_points - 1
        overlap = 128

        # ファイル1つにつき何回読み込むか
        read_count = 20

        # 入力ベクトルのサイズ
        input_row_size = read_count * len(xls)

        # 入力ベクトル
        input_vec = [None] * input_row_size

        # 空いている挿入インデックスを保持
        vacant_i = range(input_row_size)

        for xl in xls.values():

            # Excelシートを読み込む
            ws = ExcelWrapper(xl, sheet_name)

            # 読み込む行
            begin = begin_row
            end = end_row(begin)

            for i in xrange(read_count):

                # 列を読み込む
                acc = ws.select_column(column_letter, begin, end)

                # FFTしてランダムな位置に挿入
                r = int(np.random.rand() * len(vacant_i))
                input_vec[vacant_i[r]] = fft(acc, fft_points)
                del vacant_i[r]

                # 読み込む行を更新
                begin += fft_points - overlap
                end = end_row(begin)

                a = ["%03d" % (j+1) if v is None else " # " for j, v in enumerate(input_vec)]
                a = [a[j:j+read_count/2] for j in xrange(0, len(a), read_count/2)]
                for row in a:
                    print row

        input_vec = np.array(input_vec)
        print "input_vector_shape:", input_vec.shape

    som_test()
