# coding: utf-8

import numpy as np
import itertools as it
from random import randint
from excelwrapper import ExcelWrapper
from normalize import standardize, scale_zero_one
from fft import fftn

def _sample_xlsx(xlsx, sample_cnt, sheetnames, col, min_row, fft_N, overlap, log):
    """Excelを順にサンプリング"""

    wb = ExcelWrapper(xlsx)
    input_vecs = []
    is_full = False
    vec_cnt = 0

    if sheetnames is None:
        sheetnames = wb.sheetnames

    def iter_alt(iter1, iter2):
        """交互にイテレート"""
        for i1, i2 in it.izip_longest(iter1, iter2):
            if i1 is not None:
                yield i1
            if i2 is not None:
                yield i2

    for sheetname in sheetnames:
        ws = wb.get_sheet(sheetname)

        if col is None:
            _, col = ws.find_letter_by_header('Magnitude Vector')

        vec_iter = ws.iter_part_col(col, fft_N, (min_row, None), log=log)

        if overlap:
            _vec_iter = ws.iter_part_col(col, fft_N, (min_row + fft_N - overlap, None), log=log)
            _iter = iter_alt(vec_iter, _vec_iter)
        else:
            _iter = vec_iter

        for vec in _iter:
            input_vecs.append(vec)
            vec_cnt += 1
            if vec_cnt == sample_cnt:
                is_full = True
                break
        if is_full:
            break
    else:
        if not vec_cnt == sample_cnt:
            raise AssertionError("指定したサンプル回数に対してデータが足りません: {}/{}"
                                 .format(vec_cnt, sample_cnt))
        else:
            raise RuntimeError

    return input_vecs

def _sample_xlsx_random(xlsx, sample_cnt, sheetnames, col, min_row, fft_N, overlap, log):
    """Excelをランダムサンプリング"""

    wb = ExcelWrapper(xlsx)
    ws = [wb.get_sheet(s) for s in sheetnames]
    n_ws = len(ws) - 1
    begin_limits = [s.ws.max_row - fft_N for s in ws] # 読み込み開始行の限界

    if sheetnames is None:
        sheetnames = wb.sheetnames

    if col is None:
        _, col = ws.find_letter_by_header('Magnitude Vector')

    input_vecs = []
    for r in (randint(0, n_ws) for i in xrange(sample_cnt)):
        begin = randint(min_row, begin_limits[r])
        vec = ws[r].get_col(col, (begin, begin + fft_N - 1), log=log)
        input_vecs.append(vec)

    return input_vecs


def make_input(xlsx, sample_cnt, sheetnames=None, col=None, min_row=2, fft_N=128, label=None,
               wf='hanning', normalizing='01', sampling='std', overlap=0, log=False):

    """Excelファイルから入力ベクトルを作成

    sheetnamesはリストで、サンプル回数に対して足りないデータはリストの次のシートから読み込む

    :param xlsx : str
        Excelファイルのパス

    :param sample_cnt : int
        欲しい入力ベクトルの数

    :param sheetnames : iterable of str or None
        読み込み可能なシート名のiterable
        Noneを指定ですべてのシート

    :param col : str, default: None
        読み込む列
        None指定で'Magnitude Vector'列を自動で検索

    :param min_row : int, default: 2
        読み込み開始行

    :param fft_N : int, default: 128
        FFTのポイント数、一度に読み込む行数

    :param label : str or int, default: None
        指定した場合は長さsample_cntのラベルのリストも返す

    :param wf : str, default: 'hanning'
        FFTで使う窓関数
        'hanning', 'hamming', 'blackman'

    :param normalizing : str, default: '01'
        入力ベクトルの正規化方法
        '01'  -> 各ベクトルの要素を0-1の間に丸める
        'std' -> 各ベクトルの要素を平均0、分散1にする

    :param sampling : str, default: 'std'
        サンプリング方法
        'rand'の場合、シートリストの全体からランダムにサンプリングを行う
        'std', 'rand'

    :param overlap : int, default: 0
        オーバーラップさせて読み込む行数
        samplingが'std'のときのみ使用される

    :param log : bool, default: False
        ログを出力するかどうか

    :return input_vectors : ndarray
    :return labels : list
        長さfft_N/2の2D配列
        またはtuple(input_vectors, labels)
    """

    assert normalizing in ('01', 'std')
    assert 0 <= overlap < fft_N

    args = (xlsx, sample_cnt, sheetnames, col, min_row, fft_N)
    kwargs = {'overlap': overlap, 'log': log}

    if sampling == 'std':
        input_vecs = _sample_xlsx(*args, **kwargs)
    elif sampling == 'rand':
        input_vecs = _sample_xlsx_random(*args, **kwargs)
    else:
        raise ValueError

    normalizer = scale_zero_one if normalizing == '01' else standardize
    input_vecs = np.array(input_vecs)
    input_vecs = normalizer(fftn(arrs=input_vecs, fft_N=fft_N, wf=wf))

    if label is not None:
        return input_vecs, [label]*sample_cnt
    return input_vecs

def _random_idx_gen(n):
    """要素が0からnまでの重複のないランダム値を返すジェネレータ"""
    vacant_idx = range(n)
    for i in xrange(n):
        r = np.random.randint(0, len(vacant_idx))
        yield vacant_idx[r]
        del vacant_idx[r]

def random_input_iter(inputs, labels):
    """入力ベクトルとラベルをシャッフルしてイテレート"""

    assert len(inputs) == len(labels)
    r_gen = _random_idx_gen(len(inputs))
    for r in r_gen:
        yield inputs[r], labels[r]

if __name__ == '__main__':
    pass