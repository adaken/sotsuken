# coding: utf-8

from util.excelwrapper import ExcelWrapper
from fft import fft
import numpy as np
import matplotlib.pyplot as plt
from sompy import SOM

if __name__ == '__main__':

    def som_test():

        # xlsxの辞書
        xls = {'run':r'E:\work\data\run.xlsx',
               'walk':r'E:\work\data\walk.xlsx',
               'skip':r'E:\work\data\skip.xlsx'}
        sheet_name = 'Sheet4'

        # SOMで使う色
        som_colors = {'run':[255, 0, 0],    # 赤
                      'walk':[0, 0, 255],   # 青
                      'skip':[0, 255, 0]}   # 緑

        fft_points = 256
        column_letter = 'F'
        begin_row = 2
        end_row = lambda begin : begin + fft_points - 1
        overlap = 200

        # ファイル1つにつき何回読み込むか
        read_count = 40

        # 入力ベクトルのサイズ
        input_row_size = read_count * len(xls)

        # 入力ベクトル
        input_vector = [None] * input_row_size

        # 空いている挿入インデックスを保持
        vacant_i = range(input_row_size)

        for act, xl in xls.items():

            # Excelシートを読み込む
            ws = ExcelWrapper(xl, sheet_name)

            # 読み込む行
            begin = begin_row
            end = end_row(begin)

            for i in xrange(read_count):

                # 列を読み込む
                acc = ws.select_column(column_letter, begin, end)

                # ランダムな挿入インデックス
                r = int(np.random.rand() * len(vacant_i))

                # FFTしてデータに色情報を連結し、入力ベクトルに追加
                input_vector[vacant_i[r]] = np.r_[som_colors[act], fft(acc, fft_points)]
                del vacant_i[r]

                # 読み込む行を更新
                begin += fft_points - overlap
                end = end_row(begin)

                # 視覚的にテスト
                a = ["%03d" % (j+1) if v is None else " # " for j, v in enumerate(input_vector)]
                a = [a[j:j+read_count/2] for j in xrange(0, len(a), read_count/2)]
                for row in a:
                    print row

        # 配列に変換
        input_vector = np.array(input_vector, np.float32)
        print "input_vector_shape:", input_vector.shape
        print "input_data_type:", input_vector.dtype

        # データを正規化
        # (Xi - "Xの平均") / "Xの標準偏差" で平均0分散1にする
        #input_vector = (input_vector - np.mean(input_vector)) / np.std(input_vector)

        # 正規化方法2
        # (Xi - Xmin) / (Xmax - Xmin) で0<Xi<1にする
        vector_min = np.min(input_vector)
        vector_max = np.max(input_vector)
        input_vector = (input_vector - vector_min) / (vector_max - vector_min)

        # 出力するマップのサイズ
        output_shape = (40, 40)

        # SOMインスタンス
        som = SOM(output_shape, input_vector)

        # SOMのパラメータを設定
        # neighborは近傍の比率:初期値0.25、learning_rateは学習率:初期値0.1
        som.set_parameter(neighbor=0.26, learning_rate=0.22)

        # 学習と出力マップの取得
        # 引数は学習ループの回数
        output_map = som.train(4000)

        # 色のベクトルだけ取り出す
        output_map = output_map[:, :, 0:3]
        print "output_map_shape:", output_map.shape
        print output_map

        plt.imshow(output_map, interpolation='none')
        plt.show()

    som_test()
