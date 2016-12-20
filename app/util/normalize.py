# coding: utf-8

import numpy as np

def standardize(arr):
    """
    正規化方法1
    (Xi - "Xの平均") / "Xの標準偏差" で平均0分散1にする
    """

    assert isinstance(arr,np.ndarray)
    assert arr.ndim in (1, 2)
    dim = arr.ndim
    if dim == 1:
        return (arr - np.mean(arr)) / np.std(arr)
    else:
        return np.array([standardize(a) for a in arr])

def scale_zero_one(arr):
    """
    正規化方法2
    (Xi - Xmin) / (Xmax - Xmin) で0<Xi<1にする
    """
    assert isinstance(arr,np.ndarray)
    assert arr.ndim in (1, 2)
    dim = arr.ndim
    if dim == 1:
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    else:
        return np.array([scale_zero_one(a) for a in arr])

if __name__ == '__main__':
    pass