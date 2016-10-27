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
    def __init__(self, shape, input_data, display=None):
        assert isinstance(shape, (int, list, tuple))
        assert isinstance(input_data, (list, np.ndarray))
        if self._has_label(input_data): # ラベルあり
            print "ラベルあり"
            self.with_label = True
            l, v = self._split_input(input_data)
            l_type = self._check_label_type(l)
            if l_type is 'str':
                self.input_label_type = l_type
                self.input_labels = l
            elif l_type is 'rgb':
                self.input_label_type = l_type
                self.input_labels = np.array(l) if isinstance(l, list) else l
            self.input_layer = np.array(v, np.float32) if isinstance(v, list) else v
        else:                           # ラベルなし
            print "ラベルなし"
            self.with_label = False
            self.input_layer = np.array(input_data, dtype=np.float32) \
            if isinstance(input_data, list) else input_data
        input_shape = tuple(self.input_layer.shape)
        assert len(input_shape) == 2
        self.shape = tuple(shape)       # 入力層のサイズ(X x Y)
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
        self.display = display

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
        bmu = np.argmin(dis)              # 最も距離が近いノードのインデックス
        return np.unravel_index(bmu, self.shape)

    def _update(self, bmu, data, i):
        """ノードを更新"""
        dis = np.linalg.norm(self.index_map - bmu, axis=1) # 勝者ノードとの距離
        L = self._learning_rate(i)                         # 学習率係数
        S = self._learning_radius(i, dis)                  # 学習半径
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

    def train(self, n):
        print "合計ループ回数:", n * self.input_num
        print "学習状況: 0%",
        for i in xrange(n):
            if (i+1)/float(n)*100%10==0: print "%d%%"%((i+1)/float(n)*100),
            for j in self._random_idx_gen(self.input_num):
                data = self.input_layer[j] # 入力ベクトル
                win_idx = self._get_winner_node(data)
                self._update(win_idx, data, i)
        print ""
        return self._return_map()

    def _return_map(self):
        map_ = self.output_layer.reshape((self.shape[1], self.shape[0], self.input_dim))
        if self.with_label:
            if self.display is None:
                return map_, self._get_str_label()
            elif self.display is 'gray_scale':
                return self._make_gray_scale_map(map_), self._get_str_label()
        else:
            if self.display is None:
                return map_
            elif self.display is 'gray_scale':
                return self._make_gray_scale_map(map_)

    def _has_label(self, input_data):
        if len(input_data[0]) == 2 \
        and isinstance(input_data[0][0], (str, list, np.ndarray)):
            return True
        return False

    def _split_input(self, input_data):
        labels = [i[0] for i in input_data]
        vectors = [i[1] for i in input_data]
        return labels, vectors

    def _check_label_type(self, labels):
        label = labels[0]
        assert isinstance(label, (list, np.ndarray, str))
        if isinstance(label, str):
            return 'str'
        elif isinstance(label, (list, np.ndarray)):
            assert len(label) == 3
            return 'rgb'

    def _random_idx_gen(self, n):
        """要素が0からnまでの重複のないランダム値を返すジェネレータ"""
        vacant_idx = range(n)
        for i in xrange(n):
            r = rand.randint(0, len(vacant_idx))
            yield vacant_idx[r]
            del vacant_idx[r]

    def _get_str_label(self):
        """
        出力層のノードとラベルを関連付ける

        Return
        ------
        label_cood : tuple(str, ndarray)
            input vector's label with the coord
        """
        label_coord = []
        for i, data in enumerate(self.input_layer):
            label = self.input_labels[i]
            win_idx = self._get_winner_node(data)
            label_coord.append((label, win_idx))
        return label_coord

    def _get_rgb_label(self):
        """出力層のノードをRGBでラベリング"""
        # 未完成です
        x, y = self.shape
        rgb_map = np.ones(x * y * 3).reshape(x, y, 3)
        for rgb, data in zip(self.input_labels, self.input_layer):
            win_idx = self._get_winner_node(data)
            rgb_map[win_idx] = rgb
        return rgb_map

    def _normalize_standard(self, arr):
        """
        正規化方法1
        (Xi - "Xの平均") / "Xの標準偏差" で平均0分散1にする
        """
        return (arr - np.mean(arr)) / np.std(arr)

    def _normalize_scale(self, arr):
        """
        正規化方法2
        (Xi - Xmin) / (Xmax - Xmin) で0<Xi<1にする
        """
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    def _make_gray_scale_map(self, map_):

        # すべてのノードに対して近傍ノードとの平均ユークリッド距離を計算

        map_x, map_y, z = map_.shape
        x, y = np.meshgrid(np.arange(map_x), np.arange(map_y)) # 全ノードのインデックス
        nodes_idx = np.c_[x.ravel(), y.ravel()]
        # ノード[x,y]の近傍ノードの位置
        n_node_idx = lambda x, y : [[x-1, y+1], [x, y+1], [x+1, y+1], [x+1, y],
                                    [x+1, y-1], [x, y-1], [x-1, y-1], [x-1, y]]
        # ユークリッド距離
        #eu_dist = lambda vec_p, vec_q : np.sqrt(np.sum((vec_p - vec_q) ** 2))
        # 近傍ノードとのユークリッド距離の平均値
        eu_dis_mean = lambda n_nodes, node : np.mean(np.linalg.norm(n_nodes - node))
        # 各ノードに対して計算
        gray_map = np.zeros(map_x * map_y * 4).reshape(map_x, map_y, 4)
        for i, j in nodes_idx:
            n_nodes = []
            for n_i, n_j in n_node_idx(i, j):
                if 0 <= n_i < map_x and 0 <= n_j < map_y:
                    n_nodes.append(map_[n_i, n_j])
            dis_mean = eu_dis_mean(n_nodes, map_[i, j])
            gray_map[i, j, 3] = dis_mean
        #normalized = normalize_scale(normalize_standard(gray_scaled_map))
        #normalized = normalize_standard(normalize_scale(gray_scaled_map))
        #normalized = normalize_standard(gray_scaled_map)
        normalized = self._normalize_scale(gray_map)
        return normalized

if __name__ == '__main__':
    pass