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
from numpy import random as rand

class SOM:
    def __init__(self, shape, input_data):
        assert isinstance(shape, (int, list, tuple))
        assert isinstance(input_data, (list, np.ndarray))
        if len(input_data[0]) == 2: # ラベルあり
            print "ラベルあり"
            self.with_label = True
            l, v = self._split_input(input_data)
            self.input_labels = np.array(l) if isinstance(l, list) else l
            self.input_layer = np.array(v, np.float32) if isinstance(v, list) else v
        else:                       # ラベルなし
            print "ラベルなし"
            self.with_label = False
            self.input_layer = np.array(input_data, dtype=np.float32) \
            if isinstance(input_data, list) else input_data
        input_shape = tuple(self.input_layer.shape)
        assert len(input_shape) == 2
        self.shape = tuple(shape) # 入力層のサイズ(X x Y)
        self.input_num = input_shape[0] # 入力ベクトルの総数
        self.input_dim = input_shape[1] # 入力ベクトルの次元
        # 出力層、平均0分散1のランダム値で初期化
        self.output_layer = rand.standard_normal((self.shape[0] * self.shape[1], self.input_dim))
        x, y = np.meshgrid(range(self.shape[0]), range(self.shape[1]))
        # 出力層のインデックスの配列[ [x1, y1], [x2, y2] ]
        self.index_map = np.hstack((y.flatten()[:, np.newaxis],
                                    x.flatten()[:, np.newaxis]))
        self._param_input_length_ratio = 0.25
        self._life = self.input_num * self._param_input_length_ratio
        self._param_neighbor = 0.25
        self._param_learning_rate = 0.1

    def _split_input(self, input_data):
        labels = [i[0] for i in input_data]
        vectors = [i[1] for i in input_data]
        return labels, vectors

    def set_parameter(self, neighbor=None, learning_rate=None, input_length_ratio=None):
        if neighbor:
            self._param_neighbor = neighbor
        if learning_rate:
            self._param_learning_rate = learning_rate
        if input_length_ratio:
            self._param_input_length_ratio = input_length_ratio
            self._life = self.input_num * self._param_input_length_ratio

    def set_default_parameter(self, neighbor=0.25, learning_rate=0.1, input_length_ratio=0.25):
        if neighbor:
            self._param_neighbor = neighbor
        if learning_rate:
            self._param_learning_rate = learning_rate
        if input_length_ratio:
            self._param_input_length_ratio = input_length_ratio
            self._life = self.input_num * self._param_input_length_ratio

    def _get_winner_node(self, data):
        """勝者ノードを決定"""
        sub = self.output_layer - data
        dis = np.linalg.norm(sub, axis=1) # ユークリッド距離を計算
        bmu = np.argmin(dis) # 最も距離が近いノードのインデックス
        return np.unravel_index(bmu, self.shape)

    def _update(self, bmu, data, i):
        """ノードを更新"""
        dis = np.linalg.norm(self.index_map - bmu, axis=1) # 勝者ノードとの距離
        L = self._learning_rate(i) # 学習率係数
        S = self._learning_radius(i, dis) # 学習半径
        self.output_layer += L * S[:, np.newaxis] * (data - self.output_layer)

    def _learning_rate(self, t):
        return self._param_learning_rate * np.exp(-t/self._life)

    def _learning_radius(self, t, d):
        """勝者ノードとの距離に従いガウス関数で減衰する係数"""
        s = self._neighbourhood(t)
        return np.exp(-d**2/(2*s**2))

    def _neighbourhood(self, t):
        initial = max(self.shape) * self._param_neighbor
        return initial * np.exp(-t/self._life)

    def _random_idx_gen(self, n):
        """要素が0からnまでの重複のないランダム値を返すジェネレータ"""
        vacant_idx = range(n)
        for i in xrange(n):
            r = rand.randint(0, len(vacant_idx))
            yield vacant_idx[r]
            del vacant_idx[r]

    def _label(self):
        """出力層のノードをラベリング"""
        x, y = self.shape
        rabeled_map = np.ones(x * y * 3).reshape(x, y, 3)
        for i in xrange(self.input_num):
            label = self.input_labels[i]
            data = self.input_layer[i]
            win_idx = self._get_winner_node(data)
            rabeled_map[win_idx] = label
        return rabeled_map

        """
        x, y = self.shape
        labeled_map = [[[0, 0, 0] for i in xrange(y)] for j in xrange(x)]
        for i in xrange(self.input_num):
            label = self.input_labels[i]
            data = self.input_layer[i]
            win_i, win_j = self._get_winner_node(data)
            labeled_map[win_i][win_j] = label
            self.output_layer[win_idx] = label
        return labeled_map
        """

    def train(self, n):
        print "ループ回数:", self.input_num * n
        for i in range(n):
            print "学習%d回目" % (i + 1)
            for j in self._random_idx_gen(self.input_num):
                data = self.input_layer[j] # 入力ベクトル
                win_idx = self._get_winner_node(data)
                self._update(win_idx, data, i)
        #return self.output_layer.reshape(self.shape + (self.input_dim,))
        if self.with_label:
            return self._label()
        else:
            return self.output_layer.reshape((self.shape[1], self.shape[0], self.input_dim))

if __name__ == '__main__':
    pass