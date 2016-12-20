# coding: utf-8

import time

def timecounter(func):
    """
    関数の処理時間を計測して標準出力に出力するデコレータ
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        elapsed = time.time() - start
        print "elapsed time for {}(): {}sec".format(func.__name__, float(elapsed))
        return ret
    return wrapper

if __name__ == '__main__':
    pass