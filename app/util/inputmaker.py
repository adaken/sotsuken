# coding: utf-8

import numpy as np
import itertools as it
from random import randint
from excelwrapper import ExcelWrapper
from normalize import standardize, scale_zero_one
from fft import fftn
from util import random_idx_gen
from app import L

def _sample_xlsx(xlsx, sample_cnt, sheetnames, col, min_row, exfs, read_N, fft_N,
                 overlap, log):
    """Excelを順にサンプリング"""

    wb = ExcelWrapper(xlsx) # ワークブック
    if sheetnames is None:
        sheetnames = wb.sheetnames # すべてのシート
    sheets = [wb[s] for s in sheetnames] # ワークシート
    input_vecs = [] # 戻り値
    is_full = False # sample_cntだけサンプリングできたかどうか
    vec_cnt = 0

    if sample_cnt is None:
        sample_cnt = 0
        if overlap:
            for s in sheets:
                max_ = (s.ws.max_row - min_row - (fft_N - overlap)) \
                / (fft_N - overlap)
                sample_cnt += max_
        else:
            for s in sheets:
                max_ = (s.ws.max_row - min_row) / fft_N
                sample_cnt += max_

    def iter_alt(iter1, iter2):
        """交互にイテレート"""
        for i1, i2 in it.izip_longest(iter1, iter2):
            if i1 is not None:
                yield i1
            if i2 is not None:
                yield i2

    for ws in sheets:
        if col is None:
            col, _ = ws.find_letter_by_header('Magnitude Vector')

        vec_iter = ws.iter_part_col(col, exfs, (min_row, None), log=log)

        if overlap:
            _vec_iter = ws.iter_part_col(col, fft_N,
                                         (min_row + fft_N - overlap, None),
                                         log=log)
            _iter = iter_alt(vec_iter, _vec_iter)
        else:
            _iter = vec_iter

        def cut(vec, n):
            m = len(vec) / 2
            n = n / 2
            return vec[m-n:m+n]

        for vec in _iter:
            #input_vecs.append(vec[:read_N])
            v = cut(vec, read_N)
            input_vecs.append(v)
            vec_cnt += 1
            if vec_cnt == sample_cnt:
                is_full = True
                break
        if is_full:
            break
    else:
        if not vec_cnt == sample_cnt:
            raise ValueError("指定したサンプル回数に対してデータが足りません: {}/{}".format(vec_cnt, sample_cnt))
        else:
            raise RuntimeError
    return np.array(input_vecs)

def _sample_xlsx_random(xlsx, sample_cnt, sheetnames, col, min_row, read_N,
                        fft_N, overlap, log):
    """Excelをランダムサンプリング"""

    wb = ExcelWrapper(xlsx)
    ws = [wb.get_sheet(s) for s in sheetnames]
    n_ws = len(ws) - 1
    begin_limits = [s.ws.max_row - read_N for s in ws] # 読み込み開始行の限界

    if sheetnames is None:
        sheetnames = wb.sheetnames

    if col is None:
        _, col = ws.find_letter_by_header('Magnitude Vector')

    input_vecs = []
    for r in (randint(0, n_ws) for i in xrange(sample_cnt)):
        begin = randint(min_row, begin_limits[r])
        vec = ws[r].get_col(col, (begin, begin + read_N - 1), log=log)
        input_vecs.append(vec)

    return input_vecs


def make_input(xlsx, sample_cnt, sheetnames=None, col=None, min_row=2,
               exfs=128, read_N=None, fft_N=None, label=None, wf='hanning',
               normalizing=None, sampling='std', overlap=0, log=False):

    """Excelファイルから入力ベクトルを作成

    sheetnamesはリストで、サンプル回数に対して足りないデータはリストの次のシート
    から読み込む

    :param xlsx : str
        加速度のExcelファイルのパス

    :param sample_cnt : int or None, default: None
        欲しい入力ベクトルの数

    :param sheetnames : iterable of str or None
        読み込み可能なシート名のiterable
        Noneを指定ですべてのシート

    :param col : str, default: None
        読み込む列
        None指定で'Magnitude Vector'の列を自動で検索

    :param min_row : int, default: 2
        読み込み開始行

    :param exfs : int, default: 128
        Excelのサンプリング周波数

    :param read_N : int or None, default: None
        FFTに使う1つのベクトルの長さ
        = xlsxからループごとに読み込む行数
        Noneの場合はexfsと同じになる

    :param fft_N : int, default: None
        FFTのポイント数
        Noneの場合read_Nと同じ

    :param label : str or int, default: None
        指定した場合は長さsample_cntのラベルのリストも返す

    :param wf : str, default: 'hanning'
        FFTで使う窓関数
        'hanning', 'hamming', 'blackman'

    :param normalizing : str, default: '01'
        入力ベクトルの正規化方法
        '01'  -> 各ベクトルの要素を0-1の間に丸める
        'std' -> 各ベクトルの要素を平均0、分散1にする
        None  -> 正規化しない

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
        長さfft_N/2の配列の配列
        またはtuple(input_vectors, labels)
    """

    assert normalizing in ('01', 'std', None)
    assert 0 <= overlap < fft_N

    if read_N is None:
        read_N = exfs
    if fft_N is None:
        fft_N = read_N
    args = (xlsx, sample_cnt, sheetnames, col, min_row, exfs, read_N, fft_N)
    kwargs = {'overlap': overlap, 'log': log}

    if sampling == 'std':
        input_vecs = _sample_xlsx(*args, **kwargs)
    elif sampling == 'rand':
        input_vecs = _sample_xlsx_random(*args, **kwargs)
    else:
        raise ValueError

    normalizer = scale_zero_one if normalizing=='01' \
    else standardize if normalizing=='std' \
    else lambda a, axis: a
    input_vecs /= np.max(input_vecs, axis=1)[:, np.newaxis] # 最大値で割る
    input_vecs = normalizer(fftn(arrs=input_vecs, fft_N=fft_N, wf=wf, fs=100), axis=None)
    """
    inp, freq = fftn(arrs=input_vecs, fft_N=fft_N, wf=wf, fs=100, freq=True)
    inp = normalizer(inp, axis=None)
    import matplotlib.pyplot as plt
    plt.hold(True)
    plt.plot(freq, inp[0])
    for i in xrange(400, 700):
        plt.plot(inp[i])
    plt.show()
    """

    if label is not None:
        return input_vecs, [label]*sample_cnt
    return input_vecs

def random_input_iter(inputs, labels):
    """入力ベクトルとラベルをシャッフルしてイテレート"""

    assert len(inputs) == len(labels)
    r_gen = random_idx_gen(len(inputs))
    for r in r_gen:
        yield inputs[r], labels[r]

if __name__ == '__main__':
    pass
