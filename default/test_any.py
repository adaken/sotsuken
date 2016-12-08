# coding: utf-8
from fileinput import close
from _struct import unpack
from util.util import timecounter

def csv_test():
    import csv
    with open('eggs.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
        spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
        snakes = [["AAAAAAAAAA", "ABABABA"], ["BBBBBBBBBB", "BABAB"]]
        spamwriter.writerows(snakes)

def array_test():
    a = [[0 for i in range(2)] for j in range(3)]
    a[0][0] = 2
    print a

def transpose_test():

    def print_array(a):
        for i in xrange(len(a)):
            for j in a[i]:
                print j,
            print

    def transpose(list2):
        row_size = len(list2)
        col_size = len(list2[0])
        nlist2 = [[None for i in xrange(row_size)] for j in xrange(col_size)]
        for i in xrange(row_size):
            for j in range(col_size):
                nlist2[j][i] = list2[i][j]
        return nlist2

    array = [[1, 2], [3, 4], [5, 6]]

    print_array(array)
    print
    print_array(transpose(array))

def strop_test():
    s = "http://www"
    print s[0:7]

def numpy_test():
    import numpy as np
    a = np.array([
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
    [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
    [[25, 26, 27, 28], [29, 30, 31, 32], [33, 34, 35, 36]]
    ])
    print a.shape
    print a
    print a - [1, 2, 3, 4]
    print a - [[1, 2, 3, 4]]

def rand_test():
    import numpy as np
    print np.random.rand(10, 3)

def som_test():
    from sompy import SOM
    import numpy as np
    import matplotlib.pyplot as plt

    # 入力ベクトル
    # 1000行3列
    input_data = np.random.rand(1000, 3)

    # 出力するマップのサイズ
    output_shape = (40, 40)

    # SOMインスタンス
    som = SOM(output_shape, input_data)

    # SOMのパラメータを設定
    # neighborは近傍の比率:初期値0.25、learning_rateは学習率:初期値0.1
    som.set_parameter(neighbor=0.26, learning_rate=0.22)

    # 学習と出力マップの取得
    # 引数は学習ループの回数
    output_map = som.train(10000)

    print "output_shape:", output_map.shape

    plt.imshow(output_map, interpolation='none')
    plt.show()

def som_test2():
    import numpy as np

    from sompy import SOM
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    N = 20
    colors = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    som = SOM((N, N), colors)
    som.set_parameter(neighbor=0.3)
    ims = []
    for i in range(1000):
        m = som.train(10)
        img = np.array(m.tolist(), dtype=np.uint8)
        im = plt.imshow(m.tolist(), interpolation='none', animated=True)
        ims.append([im])
    fig = plt.figure()
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
    plt.show()

def som_test3():
    import numpy as np
    import matplotlib.pyplot as plt

    from sompy import SOM
    colors = [ [1, 0, 0], [0, 1, 0], [0, 0, 1] ]
    input_data = np.array([colors[i] for i in (np.random.rand(1000) * 3).astype(np.int8)])
    print "input_shape:", input_data.shape

    # 出力するマップのサイズ
    output_shape = (40, 40)

    # SOMインスタンス
    som = SOM(output_shape, input_data)

    # SOMのパラメータを設定
    # neighborは近傍の比率:初期値0.25、learning_rateは学習率:初期値0.1
    som.set_parameter(neighbor=0.5, learning_rate=0.22)

    # 学習と出力マップの取得
    # 引数は学習ループの回数
    output_map = som.train(3000)

    print "output_shape:", output_map.shape
    print output_map

    plt.imshow(output_map, interpolation='none')
    plt.show()

def for_test():
    a = []
    for x in range(3):
        for y in [100, 200, 300]:
            a.append(x + y)
    print a

    print [x + y for x in range(3) for y in [100, 200, 300]]

    b = []
    for inner_list in [[1, 3], [5], [7, 9]]:
        for x in inner_list:
            b.append(x)
    print b

    print [x for inner_list in [[1, 3], [5], [7, 9]] for x in inner_list]

    print ["{}".format(i * j) for i in range(1, 10) for j in range(1, 10)]


if __name__ == "__main__":
    import numpy as np
    import random

    a = np.array([[1, 2, 3], [4, 5, 6]])
    print a
    
    b = np.array([7, 8])
    print b
    
    t = np.c_[a, b]
    print t
    
    print t[:, :-1]
    print t[:, -1:]
