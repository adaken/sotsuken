# coding: utf-8

import random
import time

"""基本的にbuiltin modules以外を参照しないutil関数郡

importの際の相互参照を防ぐため上記の以外書かないほうが吉

"""

def split_nlist(list_, n):
    """リストをn個のサブリストに分割

    n=2: [1, 2, 3, 4, 5, 6] -> [[1, 2], [3, 4], [5, 6]]

    あまりは切り捨てられる
    """

    return zip(*[iter(list_)]*n)

def get_iter_len(iterator):
    """iteratorの長さと元のiteratorを返す"""

    if hasattr(iterator, '__len__'):
        return len(iterator), iterator
    tmp = list(iterator)
    return len(tmp), iter(tmp)

def random_idx_gen(n):
    """要素が0からnまでの重複のないランダム値を返すジェネレータ"""

    vacant_idx = range(n)
    for _ in xrange(n):
        r = random.randint(0, len(vacant_idx) - 1)
        yield vacant_idx[r]
        del vacant_idx[r]

def timecounter(func):
    """関数の処理時間を計測して標準出力に出力するデコレータ"""

    def wrapper(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        elapsed = time.time() - start
        print "elapsed time for {}(): {}sec".format(func.__name__,
                                                    float(elapsed))
        return ret
    return wrapper

if __name__ == '__main__':
    pass