# coding: utf-8
from collections import namedtuple
from util.excelwrapper import ExcelWrapper
from util.util import make_input_data, normalize_scale
from util.fft import fftn
from modsom import SOM
import matplotlib.pyplot as plt

def som_action():
    sample_cnt = 90
    min_row = 2
    N = 128
    Xl = namedtuple('Xl', 'path, sheet, col')
    xls = {'S':Xl(r'E:\work\data\acc_stop_1206.xlsx', 'Sheet4', 'F'),
           'B':Xl(r'E:\work\data\walk.xlsx', 'Sheet4', 'F'),
           'R':Xl(r'E:\work\data\run_1122.xlsx', 'Sheet4', 'F'),
           'J':Xl(r'E:\work\data\jump_128p_174data_fixed.xlsx', 'Sheet', 'A')}

    colors = {'S':'blue',
              'B':'green',
              'R':'red',
              'J':'deeppink'}

    in_vecs = []
    for key, xl in xls.items():
        wb = ExcelWrapper(xl.path)
        vecs = make_input_data(xlsx=wb, sheetname=xl.sheet, col=xl.col, min_row=min_row,
                              sample_cnt=sample_cnt, fft_N=N, log=True)
        #vecs = normalize_scale(fftn(vecs, N))
        vecs = [[key, vec] for vec in vecs]

        in_vecs += vecs

    som = SOM(shape=(40, 60), input_data=in_vecs, display='gray_scale')
    map_, labels = som.train(200)
    plt.imshow(map_, interpolation='nearest')
    for label, coord in labels:
        x, y = coord
        plt.text(x, y, label, color=colors[label])
    plt.show()

if __name__ == '__main__':
    som_action()