# coding: utf-8

from util.excelwrapper import ExcelWrapper
import numpy as np
import matplotlib.pylab as pylab
from sompy import SOM
import matplotlib.pyplot as plt
import util.modsom as modsom
from collections import namedtuple
from fft import fft

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
    Xls = namedtuple('Xls', 'label, path, sheet, rgb')
    xls = [Xls('r', r'E:\work\data\run.xlsx', 'Sheet4', [1, 0, 0]),
           Xls('w', r'E:\work\data\walk.xlsx', 'Sheet4', [0, 0, 1]),
           Xls('s', r'E:\work\data\skip.xlsx', 'Sheet4', [0, 1, 0])]
    font_colors = {'r':'red',
                   'w':'blue',
                   's':'green'}
    COLUMN_LETTER = 'F'
    FFT_POINTS = 128
    SAMPLE_CNT = 100    # xlsx1つのサンプリング回数
    MAP_SIZE = (40, 60) # 表示するマップの大きさ
    TRAIN_CNT = 150     # 学習ループの回数

    def sample_at_random(ws, c, N):
        r = np.random.randint(2, ws.ws.max_row - N)
        end_ = r + N - 1
        return  ws.select_column(column_letter=c, begin_row=r, end_row=end_, log=True)

    def ws_gen():
        for x in xls:
            yield x, ExcelWrapper(filename=x.path, sheetname=x.sheet)

    def sample_gen():
        for x, ws in ws_gen():
            for i in xrange(SAMPLE_CNT):
                yield x, sample_at_random(ws, COLUMN_LETTER, FFT_POINTS)

    def fft_gen():
        for x, sample_data in sample_gen():
            yield x, normalize_scale(fft(sample_data, FFT_POINTS))

    def input_vec_gen():
        for x, fftdata in fft_gen():
            yield x.label, x.rgb, fftdata

    input_vecs = [input_vec for input_vec in input_vec_gen()]

    som = modsom.SOM(MAP_SIZE, input_vecs, display='gray_scale')
    #som = modsom.SOM(MAP_SIZE, input_vecs)
    som.set_parameter(neighbor=0.2, learning_rate=0.3, input_length_ratio=0.25)
    map_, label_coord = som.train(TRAIN_CNT)
    plt.imshow(map_, interpolation='nearest')
    for label, coord in label_coord:
        x, y = coord
        #plt.text(x, y, label, color=colors[label])
        plt.text(x, y, label, color=font_colors[label])
    plt.savefig(r"E:\work\128pointFFT01scale150roopSOM.png")
    #plt.show()
    from libsvm.svm import *
    
    
def test1():
    Xls = namedtuple('Xls', 'label, path, sheet, rgb')
    xls = [Xls('r', r'E:\work\data\run.xlsx', 'Sheet4', [1, 0, 0]),
           Xls('w', r'E:\work\data\walk.xlsx', 'Sheet4', [0, 0, 1]),
           Xls('s', r'E:\work\data\skip.xlsx', 'Sheet4', [0, 1, 0])]
    font_colors = {'r':'red',
                   'w':'blue',
                   's':'green'}
    COLUMN_LETTER = 'F'
    SAMPLE_CNT = 100    # xlsx1つのサンプリング回数
    MAP_SIZE = (40, 60) # 表示するマップの大きさ
    TRAIN_CNT = 100     # 学習ループの回数

    def sample_at_random(ws, c, N):
        r = np.random.randint(2, ws.ws.max_row - N)
        end_ = r + N - 1
        return  ws.select_column(column_letter=c, begin_row=r, end_row=end_, log=False)

    def ws_gen():
        for x in xls:
            yield x, ExcelWrapper(filename=x.path, sheetname=x.sheet)

    def sample_gen(N):
        for x, ws in ws_gen():
            for i in xrange(SAMPLE_CNT):
                yield x, sample_at_random(ws, COLUMN_LETTER, N)

    def fft_gen(N, scale):
        for x, sample_data in sample_gen(N):
            if scale == "0-1":
                yield x, normalize_scale(fft(sample_data, N))
            elif scale == "std":
                yield x, normalize_standard(fft(sample_data, N))

    def input_vec_gen(N, scale):
        for x, fftdata in fft_gen(N, scale):
            yield x.label, fftdata

    fftps = (128,256)
    scales = ("0-1", "std")
    for i in xrange(2):
        print "%d回目" % (i+1)
        for N in fftps:
            print "%d_FFT_SOM" % N
            for sc in scales:
                print "scale: %s" % sc
                input_vecs = [input_vec for input_vec in input_vec_gen(N, sc)]
                som = modsom.SOM(MAP_SIZE, input_vecs, display='gray_scale')
                som.set_parameter(neighbor=0.2, learning_rate=0.3, input_length_ratio=0.25)
                map_, label_coord = som.train(TRAIN_CNT)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_title("%dPointFFT" % N)
                ax.imshow(map_, interpolation='nearest')
                for label, coord in label_coord:
                    x, y = coord
                    ax.text(x=x, y=y, s=label, color=font_colors[label])
                fig.savefig(r"E:\work\fig\fft_point_test\%d_fft%d_%s_100roop" % ((i+1), N, sc))

def som_gray_with_label():
    vec_size = 100
    vec_dim = 128
    data_type_count = 12
    map_size = (40, 60)
    train_itr = 100
    # ラベル付き特徴ベクトルのリストを生成
    patterns =  [("pattern%d" % (i+1), np.random.randint(0, 2, vec_dim))
                 for i in xrange(data_type_count)]
    for i, v in enumerate(patterns): print "pattern:%d\n" % (i+1), v
    input_vec = [patterns[np.random.randint(data_type_count)] for i in xrange(vec_size)]
    som = modsom.SOM(map_size, input_vec, display='gray_scale')
    som.set_parameter(neighbor=0.2, learning_rate=0.3, input_length_ratio=0.25)
    map_, label_coord = som.train(train_itr)
    print "map_shape:", map_.shape
    print "label_coord:\n", label_coord
    for label, coord in label_coord:
        x, y = coord
        plt.text(x, y, label, ha='center', va='center', withdash=True,  color='red')
    plt.imshow(map_, interpolation='nearest')
    plt.show()

def som_gray_without_label():
    vec_size = 100
    vec_dim = 128
    data_type_count = 8
    map_size = (40, 40)
    train_itr = 50
    patterns = [np.random.random(vec_dim) for i in xrange(data_type_count)]
    input_vec = [patterns[np.random.randint(data_type_count)] for i in xrange(vec_size)]
    som = modsom.SOM(map_size, input_vec, display='gray_scale')
    som.set_parameter(neighbor=0.2, learning_rate=0.3, input_length_ratio=0.25)
    map_ = som.train(train_itr)
    print "map_shape", map_.shape
    plt.imshow(map_, interpolation='nearest')
    plt.show()

def som_color_with_label():
    vec_size = 100
    vec_dim = 128
    data_type_count = 10
    map_size = (40, 60)
    train_itr = 70
    vec_patterns = [["pattern%d" % (i+1), # ラベル
                     np.random.randint(0, 255, 3), # 色
                     np.random.randint(0, 2, vec_dim)] # 入力ベクトル
                    for i in xrange(data_type_count)]
    for i, v in enumerate(vec_patterns):
        print "pattern:%d\n" % (i+1), v[2]
        print "label:", v[0]
        print "color:", v[1]
    input_vec = [vec_patterns[np.random.randint(data_type_count)] for i in xrange(vec_size)]
    som = modsom.SOM(map_size, input_vec)
    som.set_parameter(neighbor=0.2, learning_rate=0.3, input_length_ratio=0.25)
    map_, labels = som.train(train_itr)
    print "map_shape:", map_.shape
    print "map:\n", map_
    print "label_coord:\n", labels
    for l, c in labels:
        x, y = c
        plt.text(x, y, l)
    plt.imshow(map_, interpolation='nearest')
    plt.show()

def som_color_without_label():
    vec_size = 1000
    vec_dim = 10
    data_type_count = 10
    map_size = (50, 70)
    train_itr = 50
    vec_patterns = [[np.random.randint(0, 255, 3), np.random.randint(0, 2, vec_dim)]
                    for i in xrange(data_type_count)]
    for i, v in enumerate(vec_patterns):
        print "pattern:%d\n" % (i+1), v[1]
        print "label:", v[0]
    input_vec = [vec_patterns[np.random.randint(data_type_count)] for i in xrange(vec_size)]
    som = modsom.SOM(map_size, input_vec)
    som.set_parameter(neighbor=0.2, learning_rate=0.3, input_length_ratio=0.25)
    output_map = som.train(train_itr)
    print "map_shape", output_map.shape
    print "map\n", output_map
    plt.imshow(output_map, interpolation='nearest')
    plt.show()

def som_rgb_test():
    rgb = [
        ('red', [1, 0, 0]),
        ('green', [0, 1, 0]),
        ('blue', [0, 0, 1])
        ]
    in_size = 1000
    #in_vec = np.random.rand(in_size, 3)
    in_vec = [rgb[np.random.randint(3)] for i in xrange(in_size)]
    train_itr = 100
    som = modsom.SOM((40, 40), in_vec)
    som.set_parameter(neighbor=0.2, learning_rate=0.3, input_length_ratio=0.25)
    map_, label_coord = som.train(train_itr)
    print map_
    plt.imshow(map_, interpolation='nearest')
    for label, coord in label_coord:
        x, y = coord
        plt.text(x, y, label, color='white')
    plt.show()

def som_r_rgb_test():
    rgb = np.random.rand(2000, 3)
    som = modsom.SOM((60, 40), rgb)
    som.set_parameter(neighbor=0.2, learning_rate=0.3, input_length_ratio=0.25)
    map_ = som.train(100)
    plt.imshow(map_, interpolation='nearest')
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

    #test1()
    #main()
    #som_rgb_test()
    #som_r_rgb_test()
    som_gray_with_label()
    #som_gray_without_label()
    #som_color_with_label()
    #som_color_without_label()
    #hirakawa_test()