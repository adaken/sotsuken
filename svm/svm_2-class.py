# coding: utf-8

from libsvm.python.svm import *
from libsvm.python.svmutil import *
import numpy as np
from util.excelwrapper import ExcelWrapper
from fft import fft

def normalize_standard(arr):
    """
    正規化方法1
    (Xi - "Xの平均") / "Xの標準偏差" で平均0分散1にする
    """
    if isinstance(arr, list): arr = np.array(arr)
    return (arr - np.mean(arr)) / np.std(arr)

def normalize_scale(arr):
    """
    正規化方法2
    (Xi - Xmin) / (Xmax - Xmin) で0<Xi<1にする
    """
    if isinstance(arr, list): arr = np.array(arr)
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def xlsx_sample_gen(ws, col, read_range, fft_N, overlap, sample_cnt, log):
    """
    Excelファイルからサンプリング
    """
    assert ws.ws.max_row > (fft_N - overlap) * sample_cnt, "サンプリングのためのデータが足りません"
    begin = read_range[0]
    for i in xrange(sample_cnt):
        end = begin + fft_N - 1
        yield ws.select_column(col, begin, end, log)
        begin = end - overlap

def xlsx_random_sample_gen(ws, col, read_range, fft_N, overlap, sample_cnt, log):
    """
    Excelファイルからランダムにサンプリング
    オーバーラップは行わない
    """
    read_stop = read_range[1] if read_range[1] is not None else ws.ws.max_row
    for i in xrange(sample_cnt):
        rb = np.random.randint(low=read_range[0], high=read_stop - fft_N + 2)
        end = rb + fft_N - 1
        yield ws.select_column(col, rb, end, log)

def make_input_from_xlsx(filename,
                         sheetname='Sheet1',
                         col=None,
                         read_range=(1, None),
                         sampling='std',
                         sample_cnt=None,
                         overlap=0,
                         fft_N=128,
                         normalizing='01',
                         label=None,
                         log=False):
    """
    Excelファイルから入力ベクトル列を作成

    Parameters
    ----------
    filename : str
        読み込みたいExcelファイルのパス

    sheetname : str default: 'Sheet1'
        読み込みたいExcelファイルのシート名

    col : str
        Excelの読み込みたい列を指定
        ex) 'A', 'B', 'C'...

    read_range : tuple, default: (1, None)
        読み込む行の範囲
        tuple(開始行, 終了行)で指定
        終了行をNoneで指定すると最後まで読み込む

    sampling : str, default: 'std'
        サンプリングの方式
        'std' : 開始行から順に読み込む
        'rand': ランダムな行から読み込む

    sample_cnt : int
        サンプリングを行う回数

    overlap : int, default: 0
        重複して読み込む行数
        samplingに'rand'を指定した場合は使用されない

    fft_N : int, default: 128
        FFTする際のポイント数
        1回のサンプリングでこの値の分だけ行を読み込む

    normalizing : str, default: '01'
        入力ベクトルの正規化方法
        'std': ベクトルの成分を平均0、分散1で正規化
        '01' : ベクトルの成分を0<Xi<1で正規化

    label : any
        入力ベクトルに対するラベル
        文字列以外でもよい

    log : bool, default: False
        標準出力にログを出力するかどうか

    Return
    ------
    vectors : list
        入力ベクトルのリスト
        ベクトルの次元はfft_Nの値の半分になる
        ラベルを指定した場合は[[label, vector]...]
    """

    assert sampling in ('std', 'rand')
    assert normalizing in ('std', '01')
    sample_gen = xlsx_sample_gen if sampling == 'std' else xlsx_random_sample_gen
    normalize = normalize_standard if normalizing == 'std' else normalize_scale
    args = (ExcelWrapper(filename, sheetname), col, read_range, fft_N, overlap, sample_cnt, log)
    if label is not None:
        return [[label, normalize(fftdata)] for fftdata in (fft(data, fft_N) for data in sample_gen(*args))]
    else:
        return [normalize(fftdata) for fftdata in (fft(data, fft_N) for data in sample_gen(*args))]

if __name__ == '__main__':
    xls = r"E:\work\data\new_run.xlsx"
    input_vec = make_input_from_xlsx(filename=xls, sheetname='Sheet4', col='F', read_range=(2, None),
                                     sampling='rand', sample_cnt=20, overlap=0,
                                     fft_N=128, normalizing='01', label=1, log=False)
    #print >> file(r'D:\home\desk\log.txt', 'w'), input_vec
    #label = [1]*len(input_vec)
    labels = [vec[0] for vec in input_vec]
    vecs = [list(vec[1]) for vec in input_vec]
    print "input finish"
    print labels, vecs

    prob = svm_problem(labels, vecs)    # 教師データ (XOR)
    param = svm_parameter('-s 0 -t 2 -c 1')    # パラメータ (C-SVC, RBF カーネル, C=1)
    machine = svm_train(prob, param)    # 学習

    p_labels, p_acc, p_vals = svm_predict(labels,vecs,machine)    # テストデータ
    print(p_labels)    # 識別結果を表示
