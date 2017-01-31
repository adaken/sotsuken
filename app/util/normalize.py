# coding: utf-8

import numpy as np

def standardize(arr, axis=None):
    """標準化

    (Xi - "Xの平均") / "Xの標準偏差" で平均0分散1にする

    """

    assert isinstance(arr, np.ndarray)

    mean_ = np.mean(arr, axis)
    std_ = np.std(arr, axis)
    if axis is not None: # 次元を揃える
        s = arr.shape
        n_shape = s[:axis] + (1,) + s[axis+1:]
        mean_ = np.reshape(mean_, n_shape)
        std_ = np.reshape(std_, n_shape)
    return (arr - mean_) / std_

def scale_zero_one(arr, axis=None):
    """正規化

    (Xi - Xmin) / (Xmax - Xmin) で0<Xi<1にする

    """

    assert isinstance(arr, np.ndarray)

    max_ = np.max(arr, axis)
    min_ = np.min(arr, axis)
    if axis is not None: # 次元を揃える
        s = arr.shape
        n_shape = s[:axis] + (1,) + s[axis+1:]
        max_ = np.reshape(max_, n_shape)
        min_ = np.reshape(min_, n_shape)
    return (arr - min_) / (max_ - min_)

if __name__ == '__main__':
    a = np.array([[[1, 2, 3], [4, 5, 0], [5, 4, 6]],
                  [[4, 8, 6], [6, 5, 9], [1, 3, 2]],
                  [[9, 5, 2], [6, 6, 2], [9, 8, 9]],
                  [[5, 6, 4], [3, 4, 1], [1 ,7, 4]]], dtype=np.float64)
    a = np.random.rand(100, 3, 3)
    #print a
    #print "axis0\n{}".format(scale_zero_one(a, axis=0))
    #print "axis1\n{}".format(scale_zero_one(a, axis=1))
    print "axis2\n{}".format(scale_zero_one(a, axis=2))
    #print "axisNone:\n{}".format(scale_zero_one(a, axis=None))

    #print "axis0\n{}".format(standardize(a, axis=0))
