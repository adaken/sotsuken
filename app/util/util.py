# coding: utf-8

import random
import time
import collections
from itertools import islice

"""基本的にbuiltin modules以外を参照しないutil関数郡

importの際の相互参照を防ぐため上記の以外書かないほうが吉

"""

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
    for i in islice(xrange(10), 0, None, 2):
        print i