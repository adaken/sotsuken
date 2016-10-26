# coding: utf-8

from util.excelwrapper import ExcelWrapper
from fft import fft
import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from sompy import SOM
import util.modsom as mos
from numpy.lib.function_base import meshgrid

def run_som(input_vector, map_size=(40, 40), neighbor=0.26,
            learning_rate=0.22, train_itr=2000):

    # 配列に変換
    input_vector = np.array(input_vector, np.float32)
    print "input_vector_shape:", input_vector.shape
    print "input_vector_data_type:", input_vector.dtype
    print "input_vector_elements:\n", input_vector

    # 正規化方法1
    # (Xi - "Xの平均") / "Xの標準偏差" で平均0分散1にする
    normalize_ = lambda vec : (vec - np.mean(vec)) / np.std(vec)

    # 正規化方法2
    # (Xi - Xmin) / (Xmax - Xmin) で0<Xi<1にする
    normalize2_ = lambda vec : (vec - np.min(vec)) / (np.max(vec) - np.min(vec))

    # 正規化
    input_vector = normalize2_(input_vector)
    print "normalized_input_vector_element:\n", input_vector

    # 出力するマップのサイズ
    output_shape = map_size

    # SOMインスタンス
    som = mos.SOM(output_shape, input_vector)

    # SOMのパラメータを設定
    # neighborは近傍の比率:初期値0.25、learning_rateは学習率:初期値0.1
    som.set_parameter(neighbor=neighbor, learning_rate=learning_rate)

    # 学習と出力マップの取得
    # 引数は学習ループの回数
    output_map = som.train(train_itr)
    print "output_map_shape:", output_map.shape
    return output_map

def make_img_map(som_map, color_dim=3):
    """
    画像として表示するためにSOMマップを変換する

    Parameter
    ---------
    som_map : ndarray
        3次元配列

    color_dim : int
        3 : RGB
        4 : RGBA

    Return
    ------
    color_map : ndarray
    """
    color_dim = color_dim # 4だとデータを切り捨てない
    map_x, map_y, map_z = som_map.shape
    color_map = np.empty(map_x * map_y * color_dim).reshape(map_x, map_y, color_dim)

    for x in np.arange(map_x):
        for y in np.arange(map_y):
            for z in np.arange(0, map_z, map_z / color_dim):
                color_map[x, y, color_dim * z / map_z] = \
                np.mean(som_map[x, y, z:z+map_z / color_dim])

    print "cutoff_data:", map_x, "*", map_y, "*", map_z % color_dim
    print "color_map_shape:", color_map.shape
    print "color_map_elements:\n", color_map
    return color_map

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
    # 近傍のノードとのユークリッド距離の平均値
    eu_dis_mean = lambda n_nodes, node : np.mean(np.linalg.norm(n_nodes - node))
    # 0<x<1にスケーリング
    normalize_ = lambda vec : (vec - np.mean(vec)) / np.std(vec)
    normalize2_ = lambda vec : (vec - np.min(vec)) / (np.max(vec) - np.min(vec))
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
    #normalized = normalize_(gray_scaled_map)
    normalized = normalize_(normalize2_(gray_scaled_map))

    return normalized
    #return gray_scaled_map

def insert_at_random(input_gen, output_list, log=False):
    vacant_i = range(len(output_list)) # 空いている挿入位置のリスト
    for i, vector in enumerate(input_gen):
        r = int(np.random.rand() * len(vacant_i))
        output_list[vacant_i[r]] = vector
        del vacant_i[r] # 挿入したインデックスをリストから削除

        # 視覚的にテスト
        if (log and (i + 1) % 10 is 0):
            a = ["%04d" % (j+1) if v is None else " # " for j, v in enumerate(output_list)]
            a = [a[j:j+15] for j in xrange(0, len(a), 15)]
            for row in a: print row
            print ""

def main():

    # xlsxの辞書
    sheet_name = 'Sheet4'
    xls = {
        #'run':(r'E:\work\data\run.xlsx', sheet_name),
        'walk':(r'E:\work\data\walk.xlsx', sheet_name),
        'skip':(r'E:\work\data\skip.xlsx', sheet_name)
    }

    fft_points = 256
    column_letter = 'F'
    begin_row = 2
    end_row = lambda begin : begin + fft_points - 1

    read_count = 100     # xlsx1つを読み込む回数
    sample_count = 10   # xlsx1つのサンプリング回数
    overlap = 0         # 重複サンプリングを許容する行数

    # 入力ベクトルのサイズ
    input_row_size = read_count * sample_count * len(xls)

    # 空の入力ベクトル
    input_vector = [None] * input_row_size

    def read_xlsx():
        ws_list = [ExcelWrapper(path, sheet) for path, sheet in xls.values()]
        for ws in ws_list:
            for i in xrange(read_count):

                # 読み込む範囲
                begin = begin_row
                end = end_row(begin)

                for j in xrange(sample_count):

                    # 読み込んでFFT
                    yield fft(ws.select_column(column_letter, begin, end, log=True),
                               fft_points)

                    # 読み込む範囲を更新
                    begin += fft_points - overlap
                    end = end_row(begin)

    # FFTしたデータをランダムな位置に挿入
    insert_at_random(read_xlsx(), input_vector, log=True)

    som_map = run_som(input_vector, train_itr=10000, map_size=(70, 70))
    gray_map = make_grayscaled_map(som_map)
    plt.imshow(gray_map)
    plt.show()

def som_test():
    vec_size = 1000
    vec_dim = 128
    data_type_count = 5
    map_size = (40, 40)
    patterns =  [np.random.randint(0, 2, vec_dim) for i in xrange(data_type_count)]
    for i, v in enumerate(patterns): print "pattern:%d\n" % (i+1), v
    vec_gen = (patterns[np.random.randint(data_type_count)] for i in xrange(vec_size))
    input_vec = [None] * vec_size
    insert_at_random(vec_gen, input_vec)
    som_map = run_som(input_vec, train_itr=2, map_size=map_size)
    gray_map = make_grayscaled_map(som_map)
    print "gray_map_shape", gray_map.shape
    print "gray_map:\n", gray_map
    plt.imshow(gray_map)
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
    som_map = run_som(in_vector, map_size=(40, 40), train_itr=10000)
    gray_map = make_grayscaled_map(som_map)
    plt.imshow(gray_map)
    plt.show()


def rgb_test():
    colors = {'red':[1, 0, 0],
              'green':[0, 1, 0],
              'blue':[0, 0, 1]}
    vec_size = 1000
    input_vec = [None] * vec_size
    def color_gen():
        for i in xrange(vec_size):
            r = int(np.random.rand() * 3)
            key = colors.keys()[r]
            yield colors[key]
    insert_at_random(color_gen(), input_vec)
    som_map = run_som(input_vec, train_itr=4)
    gm = make_grayscaled_map(som_map)
    plt.imshow(gm)
    plt.show()

if __name__ == '__main__':

    #main()
    #som_test()
    rgb_test()
    #hirakawa_test()