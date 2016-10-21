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

        fft_points = 256
        column_letter = 'F'
        begin_row = 2
        end_row = lambda begin : begin + fft_points - 1

        # xlsx1つを読み込む回数
        read_count = 3

        # xlsx1つのサンプリング回数
        sample_count = 10

        # xlsxの重複サンプリングを許容する行数
        overlap = 0

        # 入力ベクトルのサイズ
        input_row_size = sample_count * read_count * len(xls)

        # 入力ベクトル
        input_vector = [None] * input_row_size

        # 空いている挿入インデックスを保持
        vacant_i = range(input_row_size)

        for act, xl in xls.items():

            # Excelシートを読み込む
            ws = ExcelWrapper(xl, sheet_name)

            for i in xrange(read_count):

                # 読み込む行
                begin = begin_row
                end = end_row(begin)

                for i2 in xrange(sample_count):

                    # 列を読み込む
                    acc = ws.select_column(column_letter, begin, end)

                    # ランダムな挿入インデックス
                    r = int(np.random.rand() * len(vacant_i))

                    # FFTして力ベクトルに追加
                    input_vector[vacant_i[r]] = fft(acc, fft_points)
                    del vacant_i[r]

                    # 読み込む行を更新
                    begin += fft_points - overlap
                    end = end_row(begin)

                    # 視覚的にテスト
                    if ((i2 + 1) % 5 is 0):
                        a = ["%03d" % (j+1) if v is None else " # " for j, v in enumerate(input_vector)]
                        a = [a[j:j+sample_count] for j in xrange(0, len(a), sample_count)]
                        for row in a: print row
                        print ""

        # 配列に変換
        input_vector = np.array(input_vector, np.float32)
        print "input_vector_shape:", input_vector.shape
        print "input_vector_data_type:", input_vector.dtype

        # 正規化方法1
        # (Xi - "Xの平均") / "Xの標準偏差" で平均0分散1にする
        normalize_ = lambda vec : (vec - np.mean(vec)) / np.std(vec)

        # 正規化方法2
        # (Xi - Xmin) / (Xmax - Xmin) で0<Xi<1にする
        normalize2_ = lambda vec : (vec - np.min(vec)) / (np.max(vec) - np.min(vec))

        # 正規化
        input_vector = normalize_(input_vector)
        print "normalized_input_vector_element:\n", input_vector

        # 出力するマップのサイズ
        output_shape = (40, 40)

        # SOMインスタンス
        som = SOM(output_shape, input_vector)

        # SOMのパラメータを設定
        # neighborは近傍の比率:初期値0.25、learning_rateは学習率:初期値0.1
        som.set_parameter(neighbor=0.26, learning_rate=0.22)

        # 学習と出力マップの取得
        # 引数は学習ループの回数
        output_map = som.train(3000)
        print "output_map_shape:", output_map.shape

        # 画像として表示するためにマップを変換
        color_dim = 3 # 4だとデータを切り捨てない
        map_x, map_y = output_shape
        color_map = np.empty(map_x * map_y * color_dim).reshape(map_x, map_y, color_dim)
        shapes = {'x':output_map.shape[0],
                  'y':output_map.shape[1],
                  'z':output_map.shape[2]}
        for x in np.arange(shapes['x']):
            for y in np.arange(shapes['y']):
                for z in np.arange(0, shapes['z'], shapes['z'] / color_dim):
                    color_map[x, y, color_dim * z / shapes['z']] = \
                    np.mean(output_map[x, y, z:z+shapes['z'] / color_dim])

        print "color_map_shape:", color_map.shape
        print "color_map_elements:\n", color_map

        plt.imshow(color_map[:, :, :], interpolation='none')
        plt.show()

    som_test()
