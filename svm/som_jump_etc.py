# coding: utf-8

from util.excelwrapper import ExcelWrapper
from collections import namedtuple
from fft import fft
import numpy as np
from util.util import modsom
import matplotlib.pyplot as plt
from util.util import normalize_scale

def sample_at_random(ws, col, N, ol):
    start = 500
    rb = np.random.randint(start, ws.ws.max_row - N)
    re = rb + N - 1
    if rb - ol < start:
        return ws.select_column(column_letter=col, begin_row=rb, end_row=re, log=True), None
    else:
        return (ws.select_column(column_letter=col, begin_row=rb, end_row=re, log=True),
                ws.select_column(column_letter=col, begin_row=rb-ol, end_row=re-ol, log=True))
        
def main():
    Xls = namedtuple('Xls', 'label, path, sheets')
    xls = [Xls('r', r'E:\work\data\new_run.xlsx', ['Sheet4', 'Sheet5', 'Sheet6']),
           Xls('b', r'E:\work\data\brisk_walk.xlsx', ['Sheet4', 'Sheet5', 'Sheet6']),
           Xls('j', r'E:\work\data\jump.xlsx', ['Sheet4', 'Sheet5', 'Sheet6'])]

    colors = {'r':'red',
              'b':'green',
              'j':'deeppink'}

    COLUMN_LETTER = 'F'
    FFT_N = 128
    SAMPLE_CNT = 20     # xlシート1つのサンプリング回数
    OVERLAP = 50
    START_ROW = 1000
    MAP_SIZE = (40, 60) # 表示するマップの大きさ
    TRAIN_CNT = 100     # 学習ループの回数

    normalize_scale(arr)
    input_vec = []
    for x_ in xls:
        for sheet in x_.sheets:
            ws = ExcelWrapper(filename=x_.path, sheetname=sheet)
            for i in xrange(SAMPLE_CNT):
                arr1, arr2 = sample_at_random(ws=ws, col=COLUMN_LETTER, N=FFT_N, ol=OVERLAP)
                fftdata1 = fft(arr1, FFT_N)
                input_vec.append((x_.label, fftdata1))
                if arr2 is not None:
                    fftdata2 = fft(arr2, FFT_N)
                    input_vec.append((x_.label, fftdata2))

    print "input_vector_size:", len(input_vec)
    som = modsom.SOM(shape=MAP_SIZE, input_data=input_vec, display='gray_scale')
    som.set_parameter(neighbor=0.2, learning_rate=0.3, input_length_ratio=0.25)
    map_, label_coord = som.train(TRAIN_CNT)
    plt.imshow(map_, interpolation='nearest')
    for label, coord in label_coord:
        x, y = coord
        s = label
        plt.text(x, y, s, color=colors[label])
    plt.show()

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(map_, interpolation='nearest')
    for label, coord in label_coord:
        x, y = coord
        ax.text(x=x, y=y, s=label, color=colors[label])
    fig.savefig(r"E:\work\fig\jumpetc\fig_%d" % (i+1))
    """
    
if __name__ == '__main__':
    main()
