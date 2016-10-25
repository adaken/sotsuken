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
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class SOM:
    def __init__(self, shape, input_data):
        assert isinstance(shape, (int, list, tuple))
        assert isinstance(input_data, (list, np.ndarray))
        if isinstance(input_data, list):
            input_data = np.array(input_data, dtype=np.float32)
        input_shape = tuple(input_data.shape)
        assert 2 == len(input_shape)
        self.shape = tuple(shape) # 入力層のサイズ
        self.input_layer = input_data # 入力層
        self.input_num = input_shape[0] # 入力ベクトルの総数
        self.input_dim = input_shape[1] # 入力ベクトルの次元
        # 出力層、平均0分散1のランダム値で初期化
        self.output_layer = rand.standard_normal((self.shape[0]*self.shape[1], self.input_dim))
        x, y = np.meshgrid(range(self.shape[0]), range(self.shape[1]))
        # 出力層のインデックスの配列[ [x1, y1], [x2, y2] ]
        self.index_map = np.hstack((y.flatten()[:, np.newaxis],
                                    x.flatten()[:, np.newaxis]))
        self._param_input_length_ratio = 0.25
        self._life = self.input_num * self._param_input_length_ratio
        self._param_neighbor = 0.25
        self._param_learning_rate = 0.1

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
        sub = self.output_layer - data
        dis = np.linalg.norm(sub, axis=1)
        bmu = np.argmin(dis)
        return np.unravel_index(bmu, self.shape)

    def _update(self, bmu, data, i):
        dis = np.linalg.norm(self.index_map - bmu, axis=1)
        L = self._learning_rate(i)
        S = self._learning_radius(i, dis)
        self.output_layer += L * S[:, np.newaxis] * (data - self.output_layer)

    def _learning_rate(self, t):
        return self._param_learning_rate * np.exp(-t/self._life)

    def _learning_radius(self, t, d):
        s = self._neighbourhood(t)
        return np.exp(-d**2/(2*s**2))

    def _neighbourhood(self, t):
        initial = max(self.shape) * self._param_neighbor
        return initial*np.exp(-t/self._life)

    def train(self, n):
        for i in range(n):
            r = rand.randint(0, self.input_num)
            data = self.input_layer[r]
            win_idx = self._get_winner_node(data)
            self._update(win_idx, data, i)
        #return self.output_layer.reshape(self.shape + (self.input_dim,))
        return self.output_layer.reshape((self.shape[1], self.shape[0], self.input_dim))