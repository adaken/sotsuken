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

def make_grayscaled_map(som_map):
    if isinstance(som_map, list):
        som_map = np.array(som_map)
    assert isinstance(som_map, np.ndarray), "list or ndarray only"

    # すべてのノードに対して近傍ノードとの平均ユークリッド距離を計算

    map_x, map_y, z = som_map.shape
    # 全ノードの位置
    x, y = np.meshgrid(np.arange(map_x), np.arange(map_y))
    nodes_idx = np.c_[x.ravel(), y.ravel()]
    # ノード[x,y]の近傍ノードの位置
    n_node_idx = lambda x, y : [[x-1, y+1], [x, y+1], [x+1, y+1], [x+1, y],
                                [x+1, y-1], [x, y-1], [x-1, y-1], [x-1, y]]
    # ユークリッド距離
    #eu_dist = lambda vec_p, vec_q : np.sqrt(np.sum((vec_p - vec_q) ** 2))
    # 近傍ノードとのユークリッド距離の平均値
    eu_dis_mean = lambda n_nodes, node : np.mean(np.linalg.norm(n_nodes - node))
    # 各ノードに対して計算
    max_shape = np.max(som_map.shape[0])
    gray_scaled_map = np.zeros(map_x * map_y * 4).reshape(map_x, map_y, 4)
    for i, j in nodes_idx:
        n_nodes = []
        for n_i, n_j in n_node_idx(i, j):
            if 0 <= n_i < max_shape and 0 <= n_j < max_shape:
                n_nodes.append(som_map[n_i, n_j])
        dis_mean = eu_dis_mean(n_nodes, som_map[i, j])
        gray_scaled_map[i, j, 3] = dis_mean
    normalized = normalize_scale((normalize_standard(gray_scaled_map)))
    return normalized

def main():

    # xlsxの辞書
    sheet_name = 'Sheet4'
    xls = {
        'run':(r'E:\work\data\run.xlsx', sheet_name),
        'walk':(r'E:\work\data\walk.xlsx', sheet_name),
        #'skip':(r'E:\work\data\skip.xlsx', sheet_name)
    }
    colors = {
        'run':[255, 0, 0],
        'skip':[0, 255, 0],
        'walk':[0, 0, 255]
    }
    fft_points = 256
    column_letter = 'F'
    begin_row = 2
    end_row = lambda begin : begin + fft_points - 1
    read_count = 100    # xlsx1つを読み込む回数
    sample_count = 5    # xlsx1つのサンプリング回数
    overlap = 0         # 重複サンプリングを許容する行数
    map_size = (40, 40) # 表示するマップの大きさ
    train_itr = 2       # 学習ループの回数
    input_vector = []   # 入力ベクトル

    for act, v in xls.items():
        path, sheet = v
        ws = ExcelWrapper(path, sheet)
        for i in xrange(read_count):
            begin = begin_row # 読み込み開始位置
            end = end_row(begin) # 終了位置
            for j in xrange(sample_count):
                rows = ws.select_column(column_letter, begin, end, log=True)
                fftdata = normalize_scale(fft(rows, fft_points))
                input_vector.append([colors[act], fftdata])
                begin += fft_points - overlap # 読み込む範囲を更新
                end = end_row(begin)

    som = modsom.SOM(map_size, input_vector)
    output_map = som.train(train_itr)
    plt.imshow(output_map)
    plt.show()

def som_gray_test():
    vec_size = 1000
    vec_dim = 128
    data_type_count = 5
    map_size = (40, 40)
    train_itr = 2
    patterns =  [np.random.randint(0, 2, vec_dim) for i in xrange(data_type_count)]
    for i, v in enumerate(patterns): print "pattern:%d\n" % (i+1), v
    input_vec = [patterns[np.random.randint(data_type_count)] for i in xrange(vec_size)]
    som = modsom.SOM(map_size, input_vec)
    som.set_parameter(neighbor=0.26, learning_rate=0.22)
    output_map = som.train(train_itr)
    output_map = make_grayscaled_map(output_map)
    print "output_map_shape", output_map.shape
    print "output_map:", output_map
    plt.imshow(output_map)
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
    som.set_parameter(neighbor=0.26, learning_rate=0.22)
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
    som_map = modsom.SOM()
    gray_map = make_grayscaled_map(som_map)
    plt.imshow(gray_map)
    plt.show()

if __name__ == '__main__':

    main()
    #som_gray_test()
    #som_color_test()
    #hirakawa_test()