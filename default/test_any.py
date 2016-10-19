# coding: utf-8

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
    output_map = som.train(2000)

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

if __name__ == "__main__":
    pass