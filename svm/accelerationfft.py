# coding: utf-8

from util.excelwrapper import ExcelWrapper
from fft import fft
import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from sompy import SOM
import util.modsom as modsom

def normalize_standard(arr):
    """
    正規化方法1
    (Xi - "Xの平均") / "Xの標準偏差" で平均0分散1にする
    """
    return (arr - np.mean(arr)) / np.std(arr)

def normalize_scale(arr):
    """
    正規化方法2
    (Xi - Xmin) / (Xmax - Xmin) で0<Xi<1にする
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def main():

    # xlsxの辞書
    sheet_name = 'Sheet4'
    xls = {
        'run':(r'E:\work\data\run.xlsx', sheet_name),
        'walk':(r'E:\work\data\walk.xlsx', sheet_name),
        'skip':(r'E:\work\data\skip.xlsx', sheet_name)
    }
    colors = {
        'run':'red',
        'walk':'blue',
        'skip':'green'
    }
    fft_points = 256
    column_letter = 'F'
    begin_row = 2
    end_row = lambda begin : begin + fft_points - 1
    read_count = 1      # xlsx1つを読み込む回数
    sample_count = 10   # xlsx1つのサンプリング回数
    overlap = 0         # 重複サンプリングを許容する行数
    map_size = (50, 50) # 表示するマップの大きさ
    train_itr = 1000     # 学習ループの回数
    input_vector = []   # 入力ベクトル

    for act, v in xls.items():
        path, sheet = v
        ws = ExcelWrapper(path, sheet)
        for i in xrange(read_count):
            begin = begin_row # 読み込み開始位置
            end = end_row(begin) # 終了位置
            for j in xrange(sample_count):
                rows = ws.select_column(column_letter, begin, end, log=True)
                fftdata = fft(rows, fft_points, out_fig=False) # FFT
                fftdata = normalize_scale(fftdata) # 0～1に正規化
                input_vector.append((act, fftdata))
                begin += fft_points - overlap # 読み込む範囲を更新
                end = end_row(begin)

    som = modsom.SOM(map_size, input_vector, display='gray_scale')
    som.set_parameter(neighbor=0.25, learning_rate=0.3)
    map_, label_coord = som.train(train_itr)
    plt.imshow(map_, interpolation='nearest')
    for label, coord in label_coord:
        x, y = coord
        plt.text(x, y, label, color=colors[label])
    plt.show()

def som_gray_without_label():
    vec_size = 500
    vec_dim = 128
    data_type_count = 4
    map_size = (40, 40)
    train_itr = 30
    patterns = [np.random.random(vec_dim) for i in xrange(data_type_count)]
    input_vec = [patterns[np.random.randint(data_type_count)] for i in xrange(vec_size)]
    som = modsom.SOM(map_size, input_vec, display='gray_scale')
    map_ = som.train(train_itr)
    print "map_shape", map_.shape
    plt.imshow(map_, interpolation='nearest')
    plt.show()

def som_gray_with_label():
    vec_size = 500
    vec_dim = 128
    data_type_count = 8
    map_size = (40, 40)
    train_itr = 100
    # ラベル付き特徴ベクトルのリスト
    patterns =  [("pattern%d" % (i+1), np.random.randint(0, 2, vec_dim))
                 for i in xrange(data_type_count)]
    for i, v in enumerate(patterns): print "pattern:%d\n" % (i+1), v
    input_vec = [patterns[np.random.randint(data_type_count)] for i in xrange(vec_size)]
    som = modsom.SOM(map_size, input_vec, display='gray_scale')
    map_, label_coord = som.train(train_itr)
    print "map_shape:", map_.shape
    print "map:\n", map_
    print "label_coord:\n", label_coord
    for label, coord in label_coord:
        x, y = coord
        plt.text(x, y, label, ha='center', va='center', color='red')
    plt.imshow(map_, interpolation='nearest')
    plt.show()

def som_color_test():
    vec_size = 1000
    vec_dim = 128
    data_type_count = 5
    map_size = (40, 40)
    train_itr = 2
    vec_patterns = [[list(np.random.randint(0, 255, 3)), list(np.random.randint(0, 2, vec_dim))]
                    for i in xrange(data_type_count)]
    for i, v in enumerate(vec_patterns):
        print "pattern:%d\n" % (i+1), v[1]
        print "label:", v[0]
    input_vec = [vec_patterns[np.random.randint(data_type_count)] for i in xrange(vec_size)]
    som = modsom.SOM(map_size, input_vec)
    output_map = som.train(train_itr)
    print "output_map_shape", output_map.shape
    print "output_map:", output_map
    plt.imshow(output_map)
    plt.show()

def hirakawa_test():
    xls = "E:\work\data\hsan.xlsx"
    ws = ExcelWrapper(xls, 'Sheet4')
    N = 256
    begin_row = 2
    end_row = lambda begin : begin + N - 1

    def read_xls():
        begin = begin_row
        end = end_row(begin)
        for i in xrange(ws.ws.max_row / N):
            col = ws.select_column(column_letter='F', begin_row=begin, end_row=end,
                                   log=True)
            begin += N
            end = end_row(begin)
            yield col

    in_vector = []
    for rows in read_xls():
        in_vector.append(fft(arr=rows, fft_points=N))
        print "add to input_vector"
        print "input_vector_size", len(in_vector)
    print "learning map"
    map_ = modsom.SOM((40, 40), in_vector, display='gray_scale')
    plt.imshow(map_, interpolation='nearest')
    plt.show()

if __name__ == '__main__':

    main()
    #som_gray_with_label()
    #som_gray_without_label()
    #som_color_test()
    #hirakawa_test()