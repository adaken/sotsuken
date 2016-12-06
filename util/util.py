# coding: utf-8

import numpy as np
from fft import fft
import random
import time
from util.fft import fftn
from util.excelwrapper import ExcelWrapper

def timecounter(func):
    """
    関数の処理時間を計測して標準出力に出力するデコレータ
    """
    def wrapper(*args):
        start = time.time()
        ret = func(*args)
        elapsed = time.time() - start
        print "elapsed time for {}(): {}sec".format(func.__name__, float(elapsed))
        return ret
    return wrapper

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
        #yield ws.select_column(col, (begin, end), log=log)
        yield ws.get_col(col, (begin, end), log=log)
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
        #yield ws.select_column(col, (rb, end), log=log)
        yield ws.get_col(col, (rb, end), log=log)

def make_input_from_xlsx(filename,
                         sheetname='Sheet1',
                         col=None,
                         read_range=(1, None),
                         sampling='std',
                         sample_cnt=None,
                         overlap=0,
                         fft_N=128,
                         fft_wf='hunning',
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

    fft_wf : str, default: 'hunning'
        FFTの際に使用する窓関数
        'hunning' : ハニング窓
        'humming' : ハミング窓
        'blackman': ブラックマン窓

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
        入力ベクトル(list)のリスト
        ベクトルの次元はfft_Nの値の半分になる
        ラベルを指定した場合は[[label, vector]...]
    """

    assert sampling in ('std', 'rand')
    assert normalizing in ('std', '01')
    assert fft_wf in ('hunning', 'humming', 'blackman')
    from excelwrapper import ExcelWrapper
    sample_gen = xlsx_sample_gen if sampling == 'std' else xlsx_random_sample_gen
    normalize = normalize_standard if normalizing == 'std' else normalize_scale
    args = (ExcelWrapper(filename).get_sheet(sheetname), col, read_range, fft_N, overlap, sample_cnt, log)
    if label is not None:
        return [[label, list(normalize(fftdata))] for fftdata in (fft(data, fft_N, fft_wf) for data in sample_gen(*args))]
    else:
        return [list(normalize(fftdata)) for fftdata in (fft(data, fft_N) for data in sample_gen(*args))]


def drow_circle(rgb, size, savepath):
    """
    円の画像(png)を作成

    Parameters
    ----------
    rgb : tuple or list
        円の色
    size : tuple or list
        円のサイズ
    savepath : str
        画像を保存するパス
    """
    from PIL import Image
    from PIL import ImageDraw
    import os
    assert isinstance(rgb, (tuple, list))
    assert isinstance(size, (tuple, list))
    assert isinstance(savepath, str)
    if os.path.exists(savepath): print "すでにファイルが存在するため上書きします: {}".format(savepath)
    if isinstance(rgb, list): rgb = tuple(rgb)
    if isinstance(size, list): size = tuple(size)
    im= Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.ellipse(((1, 1), size), outline=None, fill=rgb)
    del draw
    im.save(savepath)
    return savepath

def make_input_data(xlsx, sheetname, col, begin_row, sample_cnt, fft_N, log):
    """入力ベクトルを作成

    """

    row_range = (begin_row, fft_N * sample_cnt)
    ws = ExcelWrapper(xlsx).get_sheet(sheetname)
    acces = ws.iter_part_col(col=col, length=fft_N, row_range=row_range,log=log)
    fftdata = fftn(arrs=acces, fft_N=fft_N)
    fftdata = map(normalize_scale, fftdata)
    return fftdata

def drow_random_color_circle(size, savepath):
    """
    ランダムな色の円(png)を作成

    Parameters
    ----------
    size : tuple or list
        円のサイズ
    savepath : str
        画像を保存するパス
    """
    rgb = tuple([random.randint(0, 255) for i in range(3)])
    return drow_circle(rgb, size, savepath)

if __name__ == '__main__':
    label, xls = 1, r"E:\work\data\run.xlsx"

    @timecounter
    def main(label, xls):
        # こんな感じで使う
        input_vec = make_input_from_xlsx(filename=xls, sheetname='Sheet4', col='F', read_range=(2, None),
                                         sampling='std', sample_cnt=20, overlap=0,
                                         fft_N=128, normalizing='01', label=label, log=True)
        print >> file(r'E:\log.txt', 'w'), input_vec
        print "finish"
        from som.modsom import SOM
        som = SOM(shape=(30, 45), input_data=input_vec, display='gray_scale')
        map_, label_coord = som.train(30)
        import matplotlib.pyplot as plt
        plt.imshow(map_, interpolation='nearest')
        for l, c in label_coord:
            plt.text(c[0], c[1], l, color='red')
        plt.show()