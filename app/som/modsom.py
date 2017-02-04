# coding: utf-8

"""
The MIT License (MIT)

Copyright (c) 2016 Yota Ishikawa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import numbers
from numpy import random as rand
from app.util import scale_zero_one
from app.util import random_idx_gen

class SOM:
    """オンラインSOM"""

    conf_display = ['um', 'umr']

    def __init__(self, input_, labels=None, shape=(40, 50), display='um'):
        """コンストラクタ

        :param input_ : numpy.ndarray
            2次元配列の特徴ベクトル

        :param labels : list or numpy.ndarray of strs or ints, or None,
        default: None
            ラベルリスト
            None以外を指定した場合、train()の戻り値にラベルと座標のリストが
            追加される

        :param shape : tuple of ints, default: (40, 50)
            作成するマップのサイズ(行数、列数)

        :param display : str or None, default: 'um'
            作成されたマップに加える処理
            Noneの場合はなにもしない
            'um' : マップをユニタリ行列に変換
            'umr': ユニタリ行列の値を反転

        """

        """引数を検証"""
        SOM._validate_input_type(input_)
        conved_input = SOM._conversion_input(input_) # 型変換
        SOM._validate_input_dim(conved_input)
        SOM._validate_labels(labels, input_)
        SOM._validate_shape(shape)
        SOM._validate_display(display)

        """インスタンス変数"""
        self.input_layer = conved_input # 入力層
        self.labels = labels # ラベル
        self.input_num = conved_input.shape[0] # 特徴ベクトルの総数
        self.input_dim = conved_input.shape[1] # 特徴ベクトルの次元
        self.out_shape = shape # マップの大きさ
        self.display = display # マップの表示オプション
        self.output_layer = rand.standard_normal( # 出力層: 1次元配列
            size=(shape[0] * shape[1], self.input_dim))

        c, r = np.meshgrid(range(shape[1]), range(shape[0]))
        # 出力層の座標の配列: 2次元配列
        self.index_map = np.c_[r.ravel(), c.ravel()]

        """係数"""
        self._param_input_length_ratio = 0.25
        # 残りの学習回数
        self._life = self.input_num * self._param_input_length_ratio
        self._param_neighbor = 0.25
        self._param_learning_rate = 0.1

    def set_parameter(self, neighbor=None, learning_rate=None,
                      input_length_ratio=None):
        if neighbor:
            self._param_neighbor = neighbor
        if learning_rate:
            self._param_learning_rate = learning_rate
        if input_length_ratio:
            self._param_input_length_ratio = input_length_ratio
            self._life = self.input_num * self._param_input_length_ratio

    def set_default_parameter(self, neighbor=0.25, learning_rate=0.1,
                              input_length_ratio=0.25):
        if neighbor:
            self._param_neighbor = neighbor
        if learning_rate:
            self._param_learning_rate = learning_rate
        if input_length_ratio:
            self._param_input_length_ratio = input_length_ratio
            self._life = self.input_num * self._param_input_length_ratio

    def train(self, n):
        """SOMを開始させる

        :param n : int
            学習ループの回数
            入力データ数と'n'の差が大きいとオーバーフローする

        :return map_ : ndarray
            作成されたマップ(2次元配列)
            __init__()の引数displayの指定により変化
            処理は__init__()を参照

        :return labels : list of strs or ints
            coordsに対応するラベルのリスト
            __init__()の引数'label'がNoneの場合は返り値は'map_'のみ

        :return coords : list of tuple
            labelsに対応するマップ上でのラベルの相対座標
            __init__()の引数'label'がNoneの場合は返り値は'map_'のみ

        """

        loop_ = float(n * self.input_num)
        loop_p = 0

        print "input dimension :", self.input_dim
        print "number of inputs:", self.input_num
        print "number of loops :", int(loop_)
        print "learning: 0%",

        for i in xrange(n):
            for j in random_idx_gen(self.input_num):
                data = self.input_layer[j] # 入力ベクトル
                win_idx = self._get_winner_node(data) # BMUの座標
                self._update(win_idx, data, i) # 近傍を更新
                loop_p += 1
                if loop_p/loop_*100%10==0:
                    print "%d%%"%(loop_p/loop_*100),
        print ""
        return self._make_returns()

    def _get_labels(self):
        """出力層のノードとラベルを関連付ける"""

        labels = []
        coords = []
        for i, data in enumerate(self.input_layer):
            label = self.labels[i]
            win_idx = self._get_winner_node(data)
            y, x = win_idx # 出力画像に合わせるため相対座標に変換
            labels.append(label)
            coords.append((x, y))
        return labels, coords

    def _make_returns(self):
        """train()の戻り値を作成"""

        ret = []
        map_ = self.output_layer.reshape(
            (self.out_shape[0], self.out_shape[1], self.input_dim))

        if self.display is None:
            ret.append(map_)
        elif self.display == 'um':
            ret.append(SOM.to_umatrix(map_))
        elif self.display == 'umr':
            ret.append(SOM.to_umatrix(map_, reverse=True))

        if self.labels is not None:
            labels, coords = self._get_labels()
            ret.append(labels)
            ret.append(coords)

        return ret[0] if len(ret) == 1 else tuple(ret)

    def _get_winner_node(self, data):
        """勝者ノードを決定"""

        sub = self.output_layer - data
        dis = np.linalg.norm(sub, axis=1) # ユークリッド距離を計算
        bmu = np.argmin(dis)              # 最も近いノードのインデックス
        # 出力層上でのインデックスに変換
        return np.unravel_index(bmu, self.out_shape)

    def _update(self, bmu, data, i):
        """ノードを更新"""

        dis = np.linalg.norm(self.index_map - bmu, axis=1) # BMUとの距離
        L = self._learning_rate(i)                         # 学習率係数
        S = self._learning_radius(i, dis)                  # 学習半径
        self.output_layer += L * S[:, np.newaxis] * (data - self.output_layer)

    def _learning_rate(self, t):
        """学習率係数"""

        return self._param_learning_rate * np.exp(-t/self._life)

    def _learning_radius(self, t, d):
        """近傍関数、勝者ノードとの距離に従いガウス関数で減衰する"""

        s = self._neighbourhood(t)
        return np.exp(-d**2/(2*s**2))

    def _neighbourhood(self, t):
        """学習が進むに連れ減衰する係数"""

        initial = max(self.out_shape) * self._param_neighbor
        return initial * np.exp(-t/self._life)

    @classmethod
    def to_umatrix(cls, map_, range_=1, reverse=False):
        """マップをユニタリ行列に変換

        マップのそれぞれのノードについて、近傍とのユークリッド距離の平均を計算
        これによりノードごとに色での階調表現ができる
        ユニタリ行列にすることで視覚的にマップを評価できる

        :parma map_
            somされたマップ

        :parma range_ : int, default: 1
            近傍ノードの範囲

        :param reverse : bool, default: False
            ノードの値を反転

        """

        def eudis(i, j, map_,imap, range_, max_i, max_j):
            """マップの添字から近傍とのユークリッド距離の平均を計算"""

            # 近傍ノードの添字の範囲
            i_start, i_stop = max(0, i-range_), min(max_i, i+range_)
            j_start, j_stop = max(0, j-range_), min(max_j, j+range_)

            # 近傍ノードの添字
            n = imap[i_start:i_stop, j_start:j_stop]
            n = n[(n != [i, j]).any(axis=2)] # 自身の添字を除く
            n_i, n_j = n.T # 転置して2つのリストに

            # ユークリッド距離の平均
            return np.mean(np.linalg.norm(map_[n_i, n_j] - map_[i, j], axis=1))


        # ベクトル関数化
        # 0, 1番目の引数はベクトル、2, 3, 4, 5, 6番目は定数
        veudis = np.vectorize(eudis, excluded=(2, 3, 4, 5, 6))

        len_i, len_j, _ = map_.shape # 出力のサイズ
        i_, j_ = np.meshgrid(np.arange(len_i), np.arange(len_j), indexing='ij')
        imap = np.c_[i_.ravel(), j_.ravel()] # 添字の配列
        i, j = imap.T # それぞれの座標に分ける
        imap3d = imap.reshape(len_i, len_j, 2) # 3次元に変換
        um = veudis(i, j, map_, imap3d, range_+1, len_i, len_j)
        ret = scale_zero_one(um.reshape(len_i, len_j), None)

        return 1 - ret if not reverse else ret

    @classmethod
    def _conversion_input(cls, input_):
        """特徴ベクトルをndarray, np.float64に揃える"""

        if isinstance(input_, list):
            return np.array(input_, dtype=np.float64)
        else:
            if not input_.dtype == np.float64:
                return input_.astype(np.float64)
            else:
                return input_

    @classmethod
    def _validate_input_type(cls, input_):
        """引数input_の型をチェック"""

        if not isinstance(input_, (list, np.ndarray)):
            raise TypeError(u"'input_' must be a list or nupmy.ndarray, " \
                            u"not {}".format(input_.__class__.__name__))

        if not isinstance(input_, list): return

        if not all(isinstance(v, list) for v in input_):
            raise ValueError(u"invalid list: 'input_' must be a 2D")

        if not all(len(v) == len(input_[0]) for v in input_):
            raise ValueError(u"invalid length: all of inner lists in " \
                             u"'input_' must be same length")

        if not all(isinstance(v, numbers.Number)
                   for vec in input_ for v in vec):
            raise ValueError(u"invalid element: elements of 'input_' " \
                             u"must be numbers")

    @classmethod
    def _validate_input_dim(cls, input_):
        """引数input_の次元をチェック"""

        if isinstance(input_, np.ndarray):
            if not input_.ndim == 2:
                raise ValueError(u"'input_' must be a 2D list or " \
                                 u"2D nupmy.ndarray, not {}D"
                                 .format(input_.ndim))

    @classmethod
    def _validate_labels(cls, labels, input_):
        """引数labelsをチェック"""

        if labels is None: return

        if not isinstance(labels, (list, np.ndarray)):
            raise TypeError(u"'labels' must be a list or nupmy.ndarray, " \
                            u"not {}".format(labels.__class__.__name__))

        if not all(isinstance(l, (int, str, unicode)) for l in labels):
            raise ValueError(u"invalid element: elements of 'labels' " \
                             u"must be ints or strs")

        len_, len__ = len(input_), len(labels)
        if not len_ == len__:
            raise ValueError(u"invalid length: 'input_' and 'labels' " \
                             u"must be same length: {}:{}"
                             .format(len_, len__) )

    @classmethod
    def _validate_shape(cls, shape):
        """引数shapeをチェック"""

        #if shape is None: return

        if not isinstance(shape, (list, tuple)):
            raise TypeError(u"'shape' must be a list or tuple, " \
                            u"not {}". format(shape.__class__.__name__))

        if not len(shape) == 2:
            raise ValueError(u"'shape' must be a 2 length list or tuple")

        if not all(isinstance(v, int) for v in shape):
            raise ValueError(u"elements of 'shape' must be ints" )

    @classmethod
    def _validate_display(cls, display):
        """引数displayをチェック"""

        if display is None: return

        if not isinstance(display, (str, unicode)):
            raise TypeError(u"'display' must be a str, " \
                            u"not {}". format(display.__class__.__name__))

        if not display in cls.conf_display:
            raise ValueError(u"no such option: 'display' must be choosed " \
                             u"from {}".format(cls.conf_display))

if __name__ == '__main__':
    from app.util import timecounter
    import matplotlib.pyplot as plt

    @timecounter
    def f1(n, m):
        for _ in xrange(n):
            SOM.to_grayscale(m, range_=1)
        return SOM.to_grayscale(m)

    #rgb_ = np.random.rand(2500, 3)
    rgb_ = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0],
                     [0, 1, 1], [1, 0, 0], [1, 0, 1],
                     [1, 1, 0], [1, 1, 1]]*200)
    np.random.shuffle(rgb_)
    plt.imshow(rgb_.reshape(32, 50, 3), interpolation='nearest')
    plt.subplot(121).imshow(rgb_.reshape(32, 50, 3), interpolation='nearest')
    som = SOM(rgb_, None, (32, 50), None)
    #som = SOM(rgb_, None, (32, 50), 'um')
    map_ = som.train(200)
    #f1(n, map_)
    #map_ = SOM.to_grayscale(map_, reverse=True)
    #map_ = f1(20, map_)
    plt.subplot(122).imshow(map_, cmap=None, interpolation='nearest')
    #plt.tight_layout()
    plt.show()
    urgb = SOM.to_umatrix(rgb_.reshape(32, 50, 3))
    umap = SOM.to_umatrix(map_)
    plt.subplot(121).imshow(urgb, cmap='gray',interpolation='nearest')
    plt.subplot(122).imshow(umap, cmap='gray', interpolation='nearest')
    plt.show()